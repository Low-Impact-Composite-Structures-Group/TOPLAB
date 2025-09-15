"""
Scenario-based Missions for stops_model integration with HFT framework.

This module implements the mission classes specifically designed for the
stops_model approach:
- DischargeMission: Handles discharge scenarios with configurable parameters
- RefuelMission: Handles refueling scenarios with cryopump modeling
- DormancyMission: Handles dormancy/storage scenarios
- IsochoricMissionAnalysis: Analysis wrapper for isochoric missions

These missions are designed to replace the function-based execution of
stops_model with the class-based approach used in the HFT framework.

Integration with HFT Framework:
Victor Kees Poorte, 2025
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Union, Optional, Callable
import numpy as np

from src.mission.mission import Mission
from src.mission.mission_sections import MissionSection, OutFlow, InFlow
from src.thermodynamics.isochoric_thermal_model import IsochoricThermalModel
from src.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher
from src.multistep_methods.linear_multistep_methods import ScipyMethod
from src.thermodynamics.tank_states import (
    IsochoricTankState,
    IsochoricInitialState,
    IsochoricTankStates
)


@dataclass
class IsochoricMissionParameters:
    """
    Parameters for isochoric mission analysis.

    These parameters define the operational conditions, tank characteristics,
    and analysis settings for stops_model missions.
    """
    # Tank parameters
    tank_volume: float = 0.5  # [m³]
    p_min: float = 15e5  # Minimum pressure threshold [Pa]
    p_vent: float = 450e5  # Venting pressure threshold [Pa]

    # Initial conditions
    initial_mass: float = 10.0  # [kg]
    initial_temperature: float = 20.0  # [K]
    initial_solid_temperature: float = 288.15  # [K]

    # Environmental conditions
    ambient_temperature: float = 288.15  # [K]

    # Analysis parameters
    time_step: float = 1.0  # [s]
    rtol: float = 1e-6  # Relative tolerance for scipy solver
    atol: float = 1e-9  # Absolute tolerance for scipy solver


class IsochoricMission(Mission):
    """
    Base class for isochoric missions.

    Extends the standard HFT Mission class to support the isochoric
    (stops_model) approach with [m, T, Ts] state vectors and specialized
    dynamic/thermal models.
    """

    def __init__(self,
                 sections: list[MissionSection],
                 parameters: IsochoricMissionParameters,
                 scenario: str = "DISCHARGE"):
        """
        Initialize isochoric mission.

        Args:
            sections: List of mission sections
            parameters: Mission parameters
            scenario: Scenario name ("DISCHARGE", "REFUEL", "DORMANCY")
        """
        super().__init__(sections)
        self.parameters = parameters
        self.scenario = scenario

        # Create specialized models for isochoric analysis
        self.dynamic_model_switcher = IsochoricModelSwitcher(
            scenario=scenario,
            p_min=parameters.p_min,
            p_vent=parameters.p_vent,
            tank_volume=parameters.tank_volume
        )

        # Integration method
        self.integration_method = ScipyMethod(
            timestep=parameters.time_step,
            rtol=parameters.rtol,
            atol=parameters.atol
        )

    def set_thermal_model(self, thermal_model: IsochoricThermalModel):
        """Set the thermal model for coupled analysis"""
        self.thermal_model = thermal_model

    def create_initial_state(self) -> IsochoricInitialState:
        """Create initial state for this mission"""
        return IsochoricInitialState(
            fuel_mass=self.parameters.initial_mass,
            temperature=self.parameters.initial_temperature,
            solid_temperature=self.parameters.initial_solid_temperature,
            scenario=self.scenario
        )

    def get_fuel_flow_function(self, section: MissionSection) -> Callable[[float], float]:
        """
        Create fuel flow function for this mission section.

        Args:
            section: Mission section

        Returns:
            Callable: Function that returns fuel inflow rate [kg/s] at given time
        """
        def fuel_flow_func(time: float) -> float:
            # For isochoric missions, positive flow is inflow (refueling)
            total_inflow = 0.0
            for flow in section.fuel_flows:
                if isinstance(flow, InFlow):
                    if isinstance(flow.mass_flow, list):
                        # Linear interpolation for time-varying flow
                        if len(flow.mass_flow) >= 2:
                            # Simple linear interpolation
                            flow_rate = flow.mass_flow[0] + (flow.mass_flow[-1] - flow.mass_flow[0]) * (time / section.duration)
                        else:
                            flow_rate = flow.mass_flow[0]
                    else:
                        flow_rate = flow.mass_flow
                    total_inflow += flow_rate
            return total_inflow

        return fuel_flow_func

    def get_discharge_flow_function(self, section: MissionSection) -> Callable[[float], float]:
        """
        Create discharge flow function for this mission section.

        Args:
            section: Mission section

        Returns:
            Callable: Function that returns discharge outflow rate [kg/s] at given time
        """
        def discharge_flow_func(time: float) -> float:
            # For isochoric missions, positive flow is outflow (discharge)
            total_outflow = 0.0
            for flow in section.fuel_flows:
                if isinstance(flow, OutFlow):
                    if isinstance(flow.mass_flow, list):
                        # Linear interpolation for time-varying flow
                        if len(flow.mass_flow) >= 2:
                            flow_rate = flow.mass_flow[0] + (flow.mass_flow[-1] - flow.mass_flow[0]) * (time / section.duration)
                        else:
                            flow_rate = flow.mass_flow[0]
                    else:
                        flow_rate = flow.mass_flow
                    # Convert negative outflow to positive discharge
                    total_outflow += abs(flow_rate)
            return total_outflow

        return discharge_flow_func


class DischargeMission(IsochoricMission):
    """
    Discharge mission for hydrogen fuel tank analysis.

    This mission handles discharge scenarios where hydrogen is consumed
    from the tank at specified rates with coupled thermal effects.
    """

    def __init__(self,
                 discharge_rate: float,
                 duration: float,
                 parameters: IsochoricMissionParameters = None,
                 altitude: float = 0.0,
                 mach_number: float = 0.0):
        """
        Initialize discharge mission.

        Args:
            discharge_rate: Discharge mass flow rate [kg/s] (positive)
            duration: Mission duration [s]
            parameters: Mission parameters
            altitude: Mission altitude [m]
            mach_number: Mission Mach number
        """
        if parameters is None:
            parameters = IsochoricMissionParameters()

        # Create discharge section
        discharge_section = MissionSection(
            duration=duration,
            fuel_flows=[OutFlow(-discharge_rate, "gas")],  # Negative for outflow
            altitude=altitude,
            mach_number=mach_number
        )

        super().__init__([discharge_section], parameters, "DISCHARGE")
        self.discharge_rate = discharge_rate

    @classmethod
    def constant_discharge(cls,
                          discharge_rate: float,
                          duration: float,
                          initial_mass: float = 10.0,
                          initial_temperature: float = 20.0) -> DischargeMission:
        """
        Create a constant discharge mission.

        Args:
            discharge_rate: Constant discharge rate [kg/s]
            duration: Discharge duration [s]
            initial_mass: Initial hydrogen mass [kg]
            initial_temperature: Initial temperature [K]

        Returns:
            DischargeMission: Configured discharge mission
        """
        parameters = IsochoricMissionParameters(
            initial_mass=initial_mass,
            initial_temperature=initial_temperature
        )

        return cls(discharge_rate, duration, parameters)

    @classmethod
    def time_varying_discharge(cls,
                              discharge_profile: list[float],
                              duration: float,
                              initial_mass: float = 10.0,
                              initial_temperature: float = 20.0) -> DischargeMission:
        """
        Create a time-varying discharge mission.

        Args:
            discharge_profile: List of discharge rates [kg/s] over time
            duration: Total duration [s]
            initial_mass: Initial hydrogen mass [kg]
            initial_temperature: Initial temperature [K]

        Returns:
            DischargeMission: Configured discharge mission
        """
        parameters = IsochoricMissionParameters(
            initial_mass=initial_mass,
            initial_temperature=initial_temperature
        )

        # Create section with time-varying flow
        discharge_section = MissionSection(
            duration=duration,
            fuel_flows=[OutFlow([-rate for rate in discharge_profile], "gas")],
            altitude=0.0,
            mach_number=0.0
        )

        mission = cls.__new__(cls)  # Create without calling __init__
        IsochoricMission.__init__(mission, [discharge_section], parameters, "DISCHARGE")
        return mission


class RefuelMission(IsochoricMission):
    """
    Refueling mission for hydrogen fuel tank analysis.

    This mission handles refueling scenarios where hydrogen is added
    to the tank through a cryogenic pump with associated enthalpy effects.
    """

    def __init__(self,
                 refuel_rate: float,
                 duration: float,
                 parameters: IsochoricMissionParameters = None,
                 dewar_pressure: float = 3e5,
                 pump_efficiency: float = 0.78):
        """
        Initialize refuel mission.

        Args:
            refuel_rate: Refuel mass flow rate [kg/s] (positive)
            duration: Mission duration [s]
            parameters: Mission parameters
            dewar_pressure: Source dewar pressure [Pa]
            pump_efficiency: Cryogenic pump isentropic efficiency
        """
        if parameters is None:
            parameters = IsochoricMissionParameters()

        # Create refuel section
        # Create a dummy hydrogen object for InFlow compatibility
        class DummyHydrogen:
            pass

        refuel_section = MissionSection(
            duration=duration,
            fuel_flows=[InFlow(refuel_rate, DummyHydrogen())],  # Positive for inflow
            altitude=0.0,
            mach_number=0.0
        )

        super().__init__([refuel_section], parameters, "REFUEL")
        self.refuel_rate = refuel_rate
        self.dewar_pressure = dewar_pressure
        self.pump_efficiency = pump_efficiency

    @classmethod
    def constant_refuel(cls,
                       refuel_rate: float,
                       duration: float,
                       target_mass: float = 20.0,
                       initial_temperature: float = 20.0) -> RefuelMission:
        """
        Create a constant refuel mission.

        Args:
            refuel_rate: Constant refuel rate [kg/s]
            duration: Refuel duration [s]
            target_mass: Target final mass [kg]
            initial_temperature: Initial temperature [K]

        Returns:
            RefuelMission: Configured refuel mission
        """
        # Calculate initial mass based on target and refuel rate
        initial_mass = max(1.0, target_mass - refuel_rate * duration)

        parameters = IsochoricMissionParameters(
            initial_mass=initial_mass,
            initial_temperature=initial_temperature
        )

        return cls(refuel_rate, duration, parameters)


class DormancyMission(IsochoricMission):
    """
    Dormancy mission for hydrogen fuel tank analysis.

    This mission handles storage/dormancy scenarios where no fuel flows
    occur but thermal effects continue to evolve the tank state.
    """

    def __init__(self,
                 duration: float,
                 parameters: IsochoricMissionParameters = None,
                 ambient_temperature: float = 288.15):
        """
        Initialize dormancy mission.

        Args:
            duration: Dormancy duration [s]
            parameters: Mission parameters
            ambient_temperature: Ambient temperature [K]
        """
        if parameters is None:
            parameters = IsochoricMissionParameters()

        # Update ambient temperature
        parameters.ambient_temperature = ambient_temperature

        # Create dormancy section with no fuel flows
        dormancy_section = MissionSection(
            duration=duration,
            fuel_flows=[],  # No fuel flows during dormancy
            altitude=0.0,
            mach_number=0.0
        )

        super().__init__([dormancy_section], parameters, "DORMANCY")

    @classmethod
    def long_term_storage(cls,
                         duration: float,
                         initial_mass: float = 10.0,
                         initial_temperature: float = 20.0,
                         ambient_temperature: float = 288.15) -> DormancyMission:
        """
        Create a long-term storage mission.

        Args:
            duration: Storage duration [s]
            initial_mass: Initial hydrogen mass [kg]
            initial_temperature: Initial temperature [K]
            ambient_temperature: Ambient temperature [K]

        Returns:
            DormancyMission: Configured dormancy mission
        """
        parameters = IsochoricMissionParameters(
            initial_mass=initial_mass,
            initial_temperature=initial_temperature,
            ambient_temperature=ambient_temperature
        )

        return cls(duration, parameters, ambient_temperature)


class IsochoricMissionAnalysis:
    """
    Analysis wrapper for isochoric missions.

    This class provides the analysis framework for executing isochoric
    missions using the stops_model approach integrated with HFT patterns.
    """

    def __init__(self,
                 mission: IsochoricMission,
                 thermal_model: IsochoricThermalModel):
        """
        Initialize mission analysis.

        Args:
            mission: Isochoric mission to analyze
            thermal_model: Thermal model for coupled analysis
        """
        self.mission = mission
        self.thermal_model = thermal_model
        self.mission.set_thermal_model(thermal_model)

        # Results storage
        self.results = None
        self.analysis_complete = False

    def run_analysis(self) -> IsochoricTankStates:
        """
        Run the complete mission analysis.

        Returns:
            IsochoricTankStates: Analysis results
        """
        print("🔧 Creating initial state...")
        # Create initial state
        initial_state = self.mission.create_initial_state()
        print(f"✅ Initial state created: m={initial_state.fuel_mass:.2f}kg, T={initial_state.temperature:.2f}K")

        # Initialize results storage
        all_times = []
        all_states = []

        # Process each mission section
        current_time = 0.0

        # Create a minimal tank-like object for IsochoricTankState
        class MinimalTank:
            def __init__(self, volume):
                self.volume = volume

        minimal_tank = MinimalTank(self.mission.parameters.tank_volume)
        current_state = IsochoricTankState(
            tank=minimal_tank,
            fuel_mass=initial_state.fuel_mass,
            temperature=initial_state.temperature,
            solid_temperature=initial_state.solid_temperature,
            scenario=initial_state.scenario
        )

        for section in self.mission.sections:
            # Get flow functions for this section
            fuel_flow_func = self.mission.get_fuel_flow_function(section)
            discharge_flow_func = self.mission.get_discharge_flow_function(section)

            # Create ODE system for this section
            def ode_system(t, y):
                """ODE system function for scipy integration"""
                # Debug: Print integration progress every 5000 seconds
                if t % 5000 < 0.1:
                    rho = y[0] / 0.5  # density = mass / volume
                    try:
                        from CoolProp.CoolProp import PropsSI
                        p = PropsSI("P", "T", y[1], "Dmass", rho, "hydrogen") / 1e5  # Convert to bar
                        print(f"🔧 ODE Step: t={t:.1f}s, m={y[0]:.2f}kg, T={y[1]:.2f}K, Ts={y[2]:.2f}K, P={p:.1f}bar, ρ={rho:.1f}kg/m³")
                    except:
                        print(f"🔧 ODE Step: t={t:.1f}s, m={y[0]:.2f}kg, T={y[1]:.2f}K, Ts={y[2]:.2f}K, P=?bar, ρ={rho:.1f}kg/m³")

                # Validate state vector
                if y[0] <= 0 or y[1] <= 0 or y[2] <= 0:
                    rho = max(y[0], 0.001) / 0.5  # Avoid division by zero
                    print(f"⚠️ Invalid state at t={t:.1f}s: m={y[0]:.2f}kg, T={y[1]:.2f}K, Ts={y[2]:.2f}K, ρ={rho:.1f}kg/m³")
                    return np.array([0.0, 0.0, 0.0])

                # Convert state vector to IsochoricTankState
                try:
                    state = IsochoricTankState(
                        tank=minimal_tank,
                        fuel_mass=y[0],
                        temperature=y[1],
                        solid_temperature=y[2]
                    )
                except Exception as e:
                    print(f"❌ State creation failed at t={t:.1f}s: {str(e)[:100]}")
                    return np.array([0.0, 0.0, 0.0])

                # Compute thermal coupling
                Q_solid = self.thermal_model.compute_heat_flux(current_time + t, state)
                dTs_dt = self.thermal_model.compute_solid_temperature_derivative(current_time + t, state)

                # Compute dynamic model derivatives
                try:
                    derivatives = self.mission.dynamic_model_switcher.compute_state_derivatives(
                        current_time + t,
                        state,
                        fuel_flow_func,
                        discharge_flow_func,
                        Q_solid=Q_solid,
                        dTs_dt=dTs_dt
                    )

                    # Debug: Print derivatives occasionally
                    if t % 10000 < 0.1:
                        print(f"📊 Derivatives at t={t:.1f}s: dm/dt={derivatives.fuel_mass_derivative:.4f}, dT/dt={derivatives.temperature_derivative:.4f}, dTs/dt={derivatives.solid_temperature_derivative:.4f}")

                    return [
                        derivatives.fuel_mass_derivative,
                        derivatives.temperature_derivative,
                        derivatives.solid_temperature_derivative
                    ]
                except Exception as e:
                    print(f"❌ Derivative computation failed at t={t:.1f}s: {str(e)[:100]}")
                    return np.array([0.0, 0.0, 0.0])

            # Initial conditions for this section
            y0 = [current_state.fuel_mass, current_state.temperature, current_state.solid_temperature]

            # Time span for this section
            t_span = (0.0, section.duration)
            t_eval = np.arange(0.0, section.duration, self.mission.parameters.time_step)
            # Ensure final time is included
            if t_eval[-1] < section.duration:
                t_eval = np.append(t_eval, section.duration)

            # Create density-based stopping event based on scenario
            def density_stopping_event(t, y):
                """Stop integration when density reaches target value"""
                mass = y[0]
                if mass <= 0:
                    return 0.0  # Stop if mass goes negative

                # Access tank volume from mission parameters
                tank_volume = self.mission.parameters.tank_volume if hasattr(self.mission.parameters, 'tank_volume') else 0.5
                density = mass / tank_volume

                # Set target density and direction based on scenario
                if self.mission.scenario == "DISCHARGE":
                    target_density = 5.8  # kg/m³ from stops_model DISCHARGE
                elif self.mission.scenario == "REFUEL":
                    target_density = 78.0  # kg/m³ from stops_model REFUEL
                elif self.mission.scenario == "DORMANCY":
                    target_density = 70.0  # kg/m³ from stops_model DORMANCY
                else:
                    target_density = 5.8  # Default to discharge

                return density - target_density  # Event triggers when this equals 0

            density_stopping_event.terminal = True  # Stop integration when event occurs

            # Set direction based on scenario (REFUEL increases density, others decrease)
            if self.mission.scenario == "REFUEL":
                density_stopping_event.direction = 1   # Trigger when increasing for refuel
                target_for_log = 78.0
            else:
                density_stopping_event.direction = -1   # Trigger when decreasing for discharge/dormancy
                target_for_log = 5.8 if self.mission.scenario == "DISCHARGE" else 70.0

            # Set the ODE function and integrate this section
            print(f"🔧 Setting up integration for section: t_span={t_span}, y0={y0}")
            print(f"🎯 Adding density stopping event: target = {target_for_log} kg/m³")
            self.mission.integration_method.set_ode_function(ode_system)
            print("🚀 Starting ODE integration...")
            section_results = self.mission.integration_method.integrate_full(
                t_span, y0, t_eval, events=density_stopping_event
            )
            print(f"✅ Integration completed! Final time: {section_results.t[-1]:.1f}s")

            # Store results
            section_times = current_time + section_results.t
            section_states = []

            for i, t in enumerate(section_results.t):
                state = IsochoricTankState(
                    tank=minimal_tank,
                    fuel_mass=section_results.y[0][i],
                    temperature=section_results.y[1][i],
                    solid_temperature=section_results.y[2][i]
                )
                section_states.append(state)

            all_times.extend(section_times)
            all_states.extend(section_states)

            # Update current state for next section
            current_time = section_times[-1]
            current_state = section_states[-1]

        # Create results object
        self.results = IsochoricTankStates(
            states=all_states,
            timestep=self.mission.parameters.time_step
        )

        self.analysis_complete = True
        return self.results

    def get_results(self) -> Optional[IsochoricTankStates]:
        """Get analysis results (if analysis has been run)"""
        return self.results if self.analysis_complete else None


def main():
    pass


if __name__ == "__main__":
    main()


# End