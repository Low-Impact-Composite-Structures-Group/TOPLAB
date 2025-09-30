"""
Isochoric Dynamic Models for stops_model integration with HFT framework.

This module implements the stops_model approach for hydrogen tank dynamics:
- Uses [m, T, Ts] state vector (mass, fluid temperature, solid temperature)
- Implements configuration switching (A/B/C) based on pressure thresholds
- Handles single-phase and two-phase behavior through IsochoricHydrogen
- Supports scenario-based behavior (DISCHARGE, REFUEL, DORMANCY)
- Uses coupled heat transfer between solid and fluid

The models in this file do NOT inherit from the standard HFT DynamicModel
classes because they have a fundamentally different state structure and
ODE system.

Integration with HFT Framework:
Victor Kees Poorte, 2025
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Callable, Optional, Union
import numpy as np

from CoolProp.CoolProp import PropsSI

from src.thermodynamics.tank_states import (
    IsochoricTankState,
    IsochoricStateDerivatives,
    IsochoricTankStates
)
from src.fluids.convective_mediums import IsochoricHydrogen
from src.fluids.hydrogen_retrievers import IsochoricHydrogenRequester

# Global variable to hold heat flow data reference
_heat_flow_data = None

def set_heat_flow_data_collector(data_dict):
    """Set the global heat flow data collector reference."""
    global _heat_flow_data
    _heat_flow_data = data_dict


class IsochoricDynamicModel(ABC):
    """
    Base class for isochoric dynamic models.

    This class defines the interface for computing state derivatives
    in the stops_model approach. Unlike HFT's DynamicModel, this works
    with [m, T, Ts] state vectors and IsochoricTankState objects.
    """

    @abstractmethod
    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs
    ) -> IsochoricStateDerivatives:
        """
        Compute state derivatives for the isochoric ODE system.

        Args:
            time: Current time [s]
            state: Current tank state with [m, T, Ts]
            fuel_flow_func: Function returning fuel inflow rate [kg/s]
            discharge_flow_func: Function returning discharge outflow rate [kg/s]
            **kwargs: Additional model-specific parameters

        Returns:
            IsochoricStateDerivatives: Computed derivatives [dm/dt, dT/dt, dTs/dt]
        """
        pass

    @abstractmethod
    def is_applicable(self, state: IsochoricTankState) -> bool:
        """
        Check if this model is applicable for the given state.

        Args:
            state: Current tank state

        Returns:
            bool: True if model should be used for this state
        """
        pass


class SinglePhaseIsochoricModel(IsochoricDynamicModel):
    """
    Single-phase isochoric dynamic model.

    Implements the single-phase energy balance from stops_model:
    - Uses cv for temperature derivative calculation
    - Handles scenario-specific enthalpy calculations (e.g., cryopump for refueling)
    - Includes PV work terms and configuration-dependent flows
    """

    def __init__(self,
                 scenario: str = "DISCHARGE",
                 p_min: float = 15e5,
                 p_vent: float = 450e5,
                 tank_volume: float = 0.5):
        """
        Initialize single-phase isochoric model.

        Args:
            scenario: Scenario name ("DISCHARGE", "REFUEL", "DORMANCY")
            p_min: Minimum pressure threshold for configuration B [Pa]
            p_vent: Venting pressure threshold for configuration C [Pa]
            tank_volume: Tank volume [m³]
        """
        self.scenario = scenario
        self.p_min = p_min
        self.p_vent = p_vent
        self.tank_volume = tank_volume

    def is_applicable(self, state: IsochoricTankState) -> bool:
        """Single-phase model applies when not near saturation"""
        if state.hydrogen is None:
            return True  # Default assumption
        return not state.hydrogen.is_near_saturation

    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs
    ) -> IsochoricStateDerivatives:
        """
        Compute single-phase state derivatives.
        """
        # Extract state variables
        m = state.fuel_mass
        T = state.temperature
        Ts = state.solid_temperature
        rho = state.density
        p = state.pressure


        # Ensure minimum values for stability
        m = max(m, 1e-12)
        T = max(T, 1.0)

        # Get hydrogen properties
        if state.hydrogen is None:
            requester = IsochoricHydrogenRequester()
            hydrogen = requester.get_hydrogen_properties(p, T, rho)
        else:
            hydrogen = state.hydrogen

        # Get thermodynamic properties
        try:
            h = PropsSI("Hmass", "T", T, "Dmass", rho, "hydrogen")
            c_v = PropsSI("Cvmass", "T", T, "Dmass", rho, "hydrogen")
            dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, "hydrogen")
        except:
            # Fallback values
            h = 0.0
            c_v = 14000.0  # Approximate for hydrogen
            dp_dT_rho = p / T  # Ideal gas approximation

        # Determine current configuration
        config = self._determine_configuration(p)
        state.configuration = config

        # Get mass flow rates (configuration-dependent)
        mdot_fuel = fuel_flow_func(time)
        mdot_discharge = discharge_flow_func(time)
        mdot_vent = self._get_vent_flow_rate(config, T, rho, p, time, discharge_flow_func, Ts)

        # Scenario-specific fuel enthalpy
        h_fuel = self._compute_fuel_enthalpy(p, T)

        # Check for coupling enthalpy override
        coupling_enthalpy = kwargs.get('coupling_enthalpy', None)
        if coupling_enthalpy is not None and coupling_enthalpy != 0.0:
            # Use coupling enthalpy for inflow when coupling flows are present
            h_fuel = coupling_enthalpy

        # Discharge and vent enthalpies
        h_discharge = h
        h_vent = h

        # Energy balance terms
        h_term = mdot_fuel * (h_fuel - h) - mdot_discharge * (h_discharge - h) - mdot_vent * (h_vent - h)

        # PV work term
        net_mass_flow = mdot_fuel - mdot_discharge - mdot_vent
        work_term = (T / rho) * dp_dT_rho * net_mass_flow

        # Heat transfer terms (from kwargs or computed)
        Q_solid = kwargs.get('Q_solid', 0.0)  # Heat from solid to fluid [W]
        Q_discharge = kwargs.get('Q_discharge', 0.0)  # Fallback only - should not be needed

        # Configuration-dependent discharge heat calculation
        dm_dt = net_mass_flow

        # Initialize discharge heat for this timestep
        actual_qdot_disch = 0.0

        if config == "B" and self.scenario != "REFUEL":
            # Configuration B: Use special discharge heat calculation from stops_model
            # Q_disch = M_disch · [T/ρ·(∂p/∂T)_ρ - ρ·cv·(∂T/∂ρ)_p] - Q_s
            # This enforces pressure constraint indirectly through the energy balance
            # NOTE: Configuration B is DISABLED during REFUEL scenarios

            try:
                # Use constrained pressure p_min for Configuration B
                p_constrained = self.p_min

                # Calculate partial derivatives following stops_model exactly
                # Term 1: T/ρ · (∂p/∂T)_ρ using real gas relationship
                dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, 'hydrogen')
                term1 = (T / rho) * dp_dT_rho

                # Term 2: ρ·cv·(∂T/∂ρ)_p using constrained pressure
                dT_drho_p = PropsSI('d(T)/d(D)|P', 'P', p_constrained, 'T', T, 'hydrogen')
                term2 = rho * c_v * dT_drho_p

                # Configuration B discharge heat (stops_model equation)
                qdot_disch_B = mdot_discharge * (term1 - term2) - Q_solid
                actual_qdot_disch = qdot_disch_B

                # Use Configuration B discharge heat instead of normal Q_discharge
                dT_dt = (h_term + work_term + Q_solid + qdot_disch_B) / (m * c_v)

                # Debug print for Configuration B activation
                # if time % 100 < 0.1:  # Print occasionally
                #     print(f"🔧 Configuration B: t={time:.1f}s, P={p/1e5:.1f}→{p_constrained/1e5:.1f}bar, qdot_B={qdot_disch_B/1000:.1f}kW")

            except Exception as e:
                # Fallback to normal calculation if CoolProp fails
                print(f"⚠️  Configuration B CoolProp error: {e}, using normal calculation")
                dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v)
                actual_qdot_disch = 0.0  # No special discharge heat in fallback
        else:
            # Configuration A or C: No discharge heat required (qdot_disch = 0)
            dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v)
            actual_qdot_disch = 0.0

        # Capture heat flow data for plotting (for ALL configurations)
        if _heat_flow_data is not None:
            _heat_flow_data['t'].append(time)
            _heat_flow_data['qdot_disch'].append(actual_qdot_disch)
            _heat_flow_data['qdot_ohex'].append(0.0)  # Will be calculated in post-processing
            _heat_flow_data['mdot_disch'].append(mdot_discharge)
            _heat_flow_data['T'].append(T)
            _heat_flow_data['rho'].append(rho)

        # Solid temperature derivative (computed by thermal model)
        dTs_dt = kwargs.get('dTs_dt', 0.0)

        return IsochoricStateDerivatives(
            fuel_mass_derivative=dm_dt,
            temperature_derivative=dT_dt,
            solid_temperature_derivative=dTs_dt,
            heat_flux=Q_solid,
            discharge_heat_flux=Q_discharge
        )

    def _determine_configuration(self, pressure: float) -> str:
        """Determine configuration based on pressure thresholds"""
        if pressure >= self.p_vent:
            return "C"  # Venting configuration
        elif pressure <= self.p_min:
            return "B"  # Minimum pressure configuration
        else:
            return "A"  # Normal configuration

    def _get_vent_flow_rate(self, config: str, T: float, rho: float, p: float,
                           time: float, discharge_func: Callable, Ts: float) -> float:
        """Get venting mass flow rate based on configuration"""
        if config == "C":
            # Configuration C: venting occurs
            # Simplified venting model - could be made more sophisticated
            return max(0.0, (p - self.p_vent) * 1e-8)  # Simple proportional venting
        else:
            return 0.0  # No venting in configurations A and B

    def _compute_fuel_enthalpy(self, pressure: float, temperature: float) -> float:
        """Compute fuel enthalpy based on scenario"""
        if self.scenario == "REFUEL":
            # Use cryopump model for refueling
            return self._compute_cryopump_enthalpy(pressure, temperature)
        else:
            # For discharge and dormancy, fuel enthalpy equals current enthalpy
            try:
                return PropsSI("Hmass", "P", pressure, "T", temperature, "hydrogen")
            except:
                return 0.0

    def _compute_cryopump_enthalpy(self, tank_pressure: float, tank_temperature: float) -> float:
        """
        Compute hydrogen enthalpy after cryogenic pump compression.

        This implements the same logic as compute_pump_outlet_hydrogen in stops_model.
        """
        P1 = 3e5  # Dewar pressure [Pa]
        P2 = tank_pressure  # Target pressure [Pa]
        eta_p = 0.78  # Pump isentropic efficiency

        try:
            # Inlet state: saturated liquid at P1
            h1 = PropsSI("H", "P", P1, "Q", 0, "hydrogen")
            s1 = PropsSI("S", "P", P1, "Q", 0, "hydrogen")

            # Ideal isentropic outlet at P2
            h2s = PropsSI("H", "P", P2, "S", s1, "hydrogen")

            # Actual outlet enthalpy with efficiency
            h2 = h1 + (h2s - h1) / eta_p

            return h2
        except:
            return 0.0


class TwoPhaseIsochoricModel(IsochoricDynamicModel):
    """
    Two-phase isochoric dynamic model.

    Implements the two-phase energy balance from stops_model:
    - Uses c_v2P (two-phase specific heat capacity)
    - Handles Clausius-Clapeyron relations for dp/dT
    - Includes latent heat effects in energy balance
    """

    def __init__(self,
                 scenario: str = "DISCHARGE",
                 p_min: float = 15e5,
                 p_vent: float = 450e5,
                 tank_volume: float = 0.5):
        """
        Initialize two-phase isochoric model.

        Args:
            scenario: Scenario name ("DISCHARGE", "REFUEL", "DORMANCY")
            p_min: Minimum pressure threshold for configuration B [Pa]
            p_vent: Venting pressure threshold for configuration C [Pa]
            tank_volume: Tank volume [m³]
        """
        self.scenario = scenario
        self.p_min = p_min
        self.p_vent = p_vent
        self.tank_volume = tank_volume

    def is_applicable(self, state: IsochoricTankState) -> bool:
        """Two-phase model applies when near saturation"""
        if state.hydrogen is None:
            return False
        return state.hydrogen.is_near_saturation

    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs
    ) -> IsochoricStateDerivatives:
        """
        Compute two-phase state derivatives.
        """
        # Extract state variables
        m = state.fuel_mass
        T = state.temperature
        Ts = state.solid_temperature
        rho = state.density

        # Ensure minimum values for stability
        m = max(m, 1e-12)
        T = max(T, 1.0)

        # Get hydrogen properties
        if state.hydrogen is None:
            requester = IsochoricHydrogenRequester()
            hydrogen = requester.get_hydrogen_properties(state.pressure, T, rho)
        else:
            hydrogen = state.hydrogen

        # Get saturated phase properties
        try:
            p_sat = PropsSI("P", "T", T, "Q", 0, "hydrogen")
            h = PropsSI("Hmass", "T", T, "Dmass", rho, "hydrogen")

            # Vapor fraction
            x = hydrogen.vapor_fraction if hydrogen.vapor_fraction is not None else 0.0

            # Saturated phase properties
            c_v_liquid = PropsSI("Cvmass", "T", T, "Q", 0, "hydrogen")
            c_v_vapor = PropsSI("Cvmass", "T", T, "Q", 1, "hydrogen")

            # Two-phase specific heat capacity
            c_v2P = x * c_v_vapor + (1.0 - x) * c_v_liquid

            # Clausius-Clapeyron derivative
            h_vapor = PropsSI("Hmass", "T", T, "Q", 1, "hydrogen")
            h_liquid = PropsSI("Hmass", "T", T, "Q", 0, "hydrogen")
            rho_vapor = PropsSI("Dmass", "T", T, "Q", 1, "hydrogen")
            rho_liquid = PropsSI("Dmass", "T", T, "Q", 0, "hydrogen")

            L_v = h_vapor - h_liquid  # Latent heat
            delta_v = (1.0/rho_vapor) - (1.0/rho_liquid)  # Specific volume difference
            dp_sat_dT = L_v / (T * delta_v)

        except:
            # Fallback values
            p_sat = state.pressure
            h = 0.0
            c_v2P = 14000.0
            dp_sat_dT = p_sat / T

        # Determine current configuration
        config = self._determine_configuration(p_sat)
        state.configuration = config
        state.pressure = p_sat  # Update pressure to saturation pressure

        # Get mass flow rates
        mdot_fuel = fuel_flow_func(time)
        mdot_discharge = discharge_flow_func(time)
        mdot_vent = self._get_vent_flow_rate(config, T, rho, p_sat, time, discharge_flow_func, Ts)

        # Scenario-specific fuel enthalpy
        h_fuel = self._compute_fuel_enthalpy(p_sat, T)

        # Discharge and vent enthalpies
        h_discharge = h
        h_vent = h

        # Energy balance terms
        h_term = mdot_fuel * (h_fuel - h) - mdot_discharge * (h_discharge - h) - mdot_vent * (h_vent - h)

        # PV work term for two-phase
        net_mass_flow = mdot_fuel - mdot_discharge - mdot_vent
        work_term = (T / rho) * dp_sat_dT * net_mass_flow

        # Heat transfer terms
        Q_solid = kwargs.get('Q_solid', 0.0)
        Q_discharge = kwargs.get('Q_discharge', 0.0)  # Fallback only

        # Configuration-dependent discharge heat calculation
        dm_dt = net_mass_flow

        # Initialize discharge heat for this timestep
        actual_qdot_disch = 0.0

        if config == "B" and self.scenario != "REFUEL":
            # Configuration B: Use special discharge heat calculation from stops_model
            # Q_disch = M_disch · [T/ρ·(dp_sat/dT) + h_disch - h] - Q_s
            # This enforces pressure constraint indirectly through the energy balance
            # NOTE: Configuration B is DISABLED during REFUEL scenarios

            try:
                # For two-phase: T/ρ term uses saturation pressure derivative
                term1 = (T / rho) * dp_sat_dT

                # Enthalpy difference term (discharge - current)
                h_disch = h  # Discharge at current state
                term2 = h_disch - h  # This is zero, but kept for clarity with formula

                # Configuration B discharge heat (stops_model equation for two-phase)
                qdot_disch_B = mdot_discharge * (term1 + term2) - Q_solid
                actual_qdot_disch = qdot_disch_B

                # Use Configuration B discharge heat instead of normal Q_discharge
                dT_dt = (h_term + work_term + Q_solid + qdot_disch_B) / (m * c_v2P)

                # Debug print for Configuration B activation
                # if time % 100 < 0.1:  # Print occasionally
                #     print(f"🔧 Two-Phase Configuration B: t={time:.1f}s, P={p_sat/1e5:.1f}bar, qdot_B={qdot_disch_B/1000:.1f}kW")

            except Exception as e:
                # Fallback to normal calculation if CoolProp fails
                print(f"⚠️  Two-Phase Configuration B error: {e}, using normal calculation")
                dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v2P)
                actual_qdot_disch = 0.0  # No special discharge heat in fallback
        else:
            # Configuration A or C: No discharge heat required (qdot_disch = 0)
            dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v2P)
            actual_qdot_disch = 0.0

        # Capture heat flow data for plotting (for ALL configurations)
        if _heat_flow_data is not None:
            _heat_flow_data['t'].append(time)
            _heat_flow_data['qdot_disch'].append(actual_qdot_disch)
            _heat_flow_data['qdot_ohex'].append(0.0)  # Will be calculated in post-processing
            _heat_flow_data['mdot_disch'].append(mdot_discharge)
            _heat_flow_data['T'].append(T)
            _heat_flow_data['rho'].append(rho)

        # Solid temperature derivative
        dTs_dt = kwargs.get('dTs_dt', 0.0)

        return IsochoricStateDerivatives(
            fuel_mass_derivative=dm_dt,
            temperature_derivative=dT_dt,
            solid_temperature_derivative=dTs_dt,
            heat_flux=Q_solid,
            discharge_heat_flux=Q_discharge
        )

    def _determine_configuration(self, pressure: float) -> str:
        """Determine configuration based on pressure thresholds"""
        if pressure >= self.p_vent:
            return "C"
        elif pressure <= self.p_min:
            return "B"
        else:
            return "A"

    def _get_vent_flow_rate(self, config: str, T: float, rho: float, p: float,
                           time: float, discharge_func: Callable, Ts: float) -> float:
        """Get venting mass flow rate based on configuration"""
        if config == "C":
            return max(0.0, (p - self.p_vent) * 1e-7)
        else:
            return 0.0

    def _compute_fuel_enthalpy(self, pressure: float, temperature: float) -> float:
        """Compute fuel enthalpy based on scenario with robust error handling"""
        if self.scenario == "REFUEL":
            return self._compute_cryopump_enthalpy(pressure, temperature)
        else:
            try:
                return PropsSI("Hmass", "P", pressure, "T", temperature, "hydrogen")
            except Exception as e:
                # Handle CoolProp saturation boundary issues gracefully
                error_msg = str(e).lower()
                if "saturation" in error_msg or "within" in error_msg:
                    # Near saturation boundary - use fallback calculation
                    try:
                        # Try using quality-based calculation near saturation
                        return PropsSI("Hmass", "P", pressure, "Q", 1.0, "hydrogen")  # Saturated vapor
                    except:
                        # Ultimate fallback - estimate from ideal gas
                        R_specific = 4124.0  # J/kg·K for hydrogen
                        cp = 14300.0  # J/kg·K approximate cp for hydrogen
                        return cp * temperature
                else:
                    # For other errors, use ideal gas approximation
                    R_specific = 4124.0  # J/kg·K for hydrogen
                    cp = 14300.0  # J/kg·K approximate cp for hydrogen
                    return cp * temperature

    def _compute_cryopump_enthalpy(self, tank_pressure: float, tank_temperature: float) -> float:
        """Compute hydrogen enthalpy after cryogenic pump compression"""
        P1 = 3e5  # Dewar pressure [Pa]
        P2 = tank_pressure  # Target pressure [Pa]
        eta_p = 0.78  # Pump isentropic efficiency

        try:
            h1 = PropsSI("H", "P", P1, "Q", 0, "hydrogen")
            s1 = PropsSI("S", "P", P1, "Q", 0, "hydrogen")
            h2s = PropsSI("H", "P", P2, "S", s1, "hydrogen")
            h2 = h1 + (h2s - h1) / eta_p
            return h2
        except:
            return 0.0


class IsochoricModelSwitcher:
    """
    Model switcher for isochoric dynamic models.

    This class handles switching between single-phase and two-phase models
    based on the current state conditions, similar to the ModelSwitcher
    in stops_model.
    """

    def __init__(self,
                 scenario: str = "DISCHARGE",
                 p_min: float = 15e5,
                 p_vent: float = 450e5,
                 tank_volume: float = 0.5):
        """
        Initialize model switcher.

        Args:
            scenario: Scenario name
            p_min: Minimum pressure threshold [Pa]
            p_vent: Venting pressure threshold [Pa]
            tank_volume: Tank volume [m³]
        """
        self.single_phase_model = SinglePhaseIsochoricModel(scenario, p_min, p_vent, tank_volume)
        self.two_phase_model = TwoPhaseIsochoricModel(scenario, p_min, p_vent, tank_volume)

        # Storage for model selection history
        self.model_history = []

    def select_model(self, state: IsochoricTankState) -> IsochoricDynamicModel:
        """
        Select appropriate model based on current state.

        Args:
            state: Current tank state

        Returns:
            IsochoricDynamicModel: Selected model
        """
        if self.two_phase_model.is_applicable(state):
            selected_model = self.two_phase_model
            model_name = "two_phase"
        else:
            selected_model = self.single_phase_model
            model_name = "single_phase"

        # Track model selection
        self.model_history.append(model_name)

        return selected_model

    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs
    ) -> IsochoricStateDerivatives:
        """
        Compute state derivatives using appropriate model.

        Args:
            time: Current time [s]
            state: Current tank state
            fuel_flow_func: Fuel inflow function
            discharge_flow_func: Discharge outflow function
            **kwargs: Additional parameters

        Returns:
            IsochoricStateDerivatives: Computed derivatives
        """
        model = self.select_model(state)
        return model.compute_state_derivatives(time, state, fuel_flow_func, discharge_flow_func, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()


# End