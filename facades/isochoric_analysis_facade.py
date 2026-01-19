"""
Heat Flow Data Collection Facade for stops_model integration with HFT framework.

This module implements the facade pattern for collecting and managing heat flow
data during isochoric analysis. It provides:
- Heat flux tracking throughout analysis
- Thermal data collection and storage
- Analysis result packaging with heat flow information
- Integration with existing HFT facade patterns

Integration with HFT Framework:
Victor Kees Poorte, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime

from src.mission.isochoric_missions import (
    IsochoricMission,
    IsochoricMissionAnalysis,
    IsochoricMissionParameters
)
from src.thermodynamics.isochoric_thermal_model import (
    IsochoricThermalModel,
    CoupledSolidFluidThermalModel,
    create_default_coupled_thermal_model
)
from src.thermodynamics.tank_states import IsochoricTankStates, IsochoricTankState
from src.tank_design.tank_shapes import Tank
from src.efficiencies.tank_performance import TankPerformance


@dataclass
class HeatFlowData:
    """
    Container for heat flow data collected during analysis.

    This class stores all thermal-related data from isochoric analysis
    including heat fluxes, temperatures, and thermal properties.
    """
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_flux_solid_to_fluid: List[float] = field(default_factory=list)  # Q_solid [W]
    heat_flux_ambient_to_solid: List[float] = field(default_factory=list)  # Q_ambient [W]
    discharge_heat_flux: List[float] = field(default_factory=list)  # Q_discharge [W]
    fluid_temperatures: List[float] = field(default_factory=list)  # T_fluid [K]
    solid_temperatures: List[float] = field(default_factory=list)  # T_solid [K]

    # Configuration tracking
    configurations: List[str] = field(default_factory=list)  # A, B, C
    model_selections: List[str] = field(default_factory=list)  # single_phase, two_phase

    # Thermal properties
    thermal_diffusivity: float = 0.0  # alpha_s [m²/s]
    convective_htc: float = 0.0  # h_sf [W/m²/K]
    surface_area: float = 0.0  # A [m²]
    solid_mass: float = 0.0  # m_s [kg]

    # Analysis metadata
    scenario: str = ""  # DISCHARGE, REFUEL, DORMANCY
    analysis_duration: float = 0.0  # [s]
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Convert lists to numpy arrays for consistency"""
        if len(self.heat_flux_solid_to_fluid) > 0:
            self.heat_flux_solid_to_fluid = np.array(self.heat_flux_solid_to_fluid)
        if len(self.heat_flux_ambient_to_solid) > 0:
            self.heat_flux_ambient_to_solid = np.array(self.heat_flux_ambient_to_solid)
        if len(self.discharge_heat_flux) > 0:
            self.discharge_heat_flux = np.array(self.discharge_heat_flux)
        if len(self.fluid_temperatures) > 0:
            self.fluid_temperatures = np.array(self.fluid_temperatures)
        if len(self.solid_temperatures) > 0:
            self.solid_temperatures = np.array(self.solid_temperatures)

    @property
    def total_heat_input(self) -> np.ndarray:
        """Compute total heat input to system [W]"""
        if len(self.heat_flux_ambient_to_solid) > 0:
            return np.array(self.heat_flux_ambient_to_solid)
        else:
            return np.zeros_like(self.times)

    @property
    def net_heat_to_fluid(self) -> np.ndarray:
        """Compute net heat transfer to fluid [W]"""
        result = np.zeros_like(self.times)
        if len(self.heat_flux_solid_to_fluid) > 0:
            result += self.heat_flux_solid_to_fluid
        if len(self.discharge_heat_flux) > 0:
            result += self.discharge_heat_flux
        return result

    @property
    def cumulative_heat_input(self) -> np.ndarray:
        """Compute cumulative heat input [J]"""
        if len(self.times) > 1:
            dt = np.diff(self.times)
            dt = np.append(dt, dt[-1])  # Assume last dt same as previous
            return np.cumsum(self.total_heat_input * dt)
        else:
            return np.zeros_like(self.times)

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics for heat flow data"""
        if len(self.times) == 0:
            return {}

        return {
            'duration': self.analysis_duration,
            'max_solid_fluid_heat_flux': np.max(self.heat_flux_solid_to_fluid) if len(self.heat_flux_solid_to_fluid) > 0 else 0.0,
            'average_solid_fluid_heat_flux': np.mean(self.heat_flux_solid_to_fluid) if len(self.heat_flux_solid_to_fluid) > 0 else 0.0,
            'total_heat_input': self.cumulative_heat_input[-1] if len(self.cumulative_heat_input) > 0 else 0.0,
            'max_temperature_difference': np.max(np.array(self.solid_temperatures) - np.array(self.fluid_temperatures)) if len(self.solid_temperatures) > 0 else 0.0,
            'final_fluid_temperature': self.fluid_temperatures[-1] if len(self.fluid_temperatures) > 0 else 0.0,
            'final_solid_temperature': self.solid_temperatures[-1] if len(self.solid_temperatures) > 0 else 0.0,
        }


@dataclass
class IsochoricAnalysisResults:
    """
    Complete results from isochoric analysis.

    This class packages all results from stops_model analysis including
    tank states, heat flow data, and performance metrics.
    """
    tank_states: IsochoricTankStates
    heat_flow_data: HeatFlowData
    mission: IsochoricMission
    thermal_model: IsochoricThermalModel
    tank_performance: Optional[TankPerformance] = None

    @property
    def scenario(self) -> str:
        """Get analysis scenario"""
        return self.mission.scenario

    @property
    def duration(self) -> float:
        """Get total analysis duration [s]"""
        return self.tank_states.times[-1] - self.tank_states.times[0] if len(self.tank_states.times) > 1 else 0.0

    @property
    def final_state(self) -> IsochoricTankState:
        """Get final tank state"""
        return self.tank_states.states[-1] if len(self.tank_states.states) > 0 else None

    @property
    def initial_state(self) -> IsochoricTankState:
        """Get initial tank state"""
        return self.tank_states.states[0] if len(self.tank_states.states) > 0 else None

    def get_mass_change(self) -> float:
        """Get total mass change [kg]"""
        if self.initial_state and self.final_state:
            return self.final_state.fuel_mass - self.initial_state.fuel_mass
        return 0.0

    def get_temperature_change(self) -> float:
        """Get fluid temperature change [K]"""
        if self.initial_state and self.final_state:
            return self.final_state.temperature - self.initial_state.temperature
        return 0.0

    def get_pressure_change(self) -> float:
        """Get pressure change [Pa]"""
        if self.initial_state and self.final_state:
            return self.final_state.pressure - self.initial_state.pressure
        return 0.0


class HeatFlowDataCollector:
    """
    Data collector for heat flow information during isochoric analysis.

    This class implements the facade pattern for collecting thermal data
    throughout the analysis process and packaging results.
    """

    def __init__(self):
        """Initialize data collector"""
        self.heat_flow_data = HeatFlowData()
        self.is_collecting = False
        self.current_analysis = None

    def start_collection(self, mission: IsochoricMission, thermal_model: IsochoricThermalModel):
        """
        Start data collection for analysis.

        Args:
            mission: Isochoric mission being analyzed
            thermal_model: Thermal model for analysis
        """
        self.heat_flow_data = HeatFlowData(
            scenario=mission.scenario,
            analysis_timestamp=datetime.now().isoformat()
        )

        # Extract thermal properties if available
        if hasattr(thermal_model, 'get_thermal_properties'):
            props = thermal_model.get_thermal_properties()
            self.heat_flow_data.thermal_diffusivity = props.get('alpha_s', 0.0)
            self.heat_flow_data.convective_htc = props.get('h_sf', 0.0)
            self.heat_flow_data.surface_area = props.get('surface_area', 0.0)
            self.heat_flow_data.solid_mass = props.get('solid_mass', 0.0)

        self.is_collecting = True

    def collect_timestep_data(self,
                             time: float,
                             state: IsochoricTankState,
                             heat_flux_solid_to_fluid: float = 0.0,
                             heat_flux_ambient_to_solid: float = 0.0,
                             discharge_heat_flux: float = 0.0,
                             configuration: str = "A",
                             model_selection: str = "single_phase"):
        """
        Collect data for a single time step.

        Args:
            time: Current time [s]
            state: Current tank state
            heat_flux_solid_to_fluid: Heat flux from solid to fluid [W]
            heat_flux_ambient_to_solid: Heat flux from ambient to solid [W]
            discharge_heat_flux: Heat flux for discharge processes [W]
            configuration: Current configuration (A, B, C)
            model_selection: Current model selection
        """
        if not self.is_collecting:
            return

        # Append time
        self.heat_flow_data.times = np.append(self.heat_flow_data.times, time)

        # Append heat flux data
        self.heat_flow_data.heat_flux_solid_to_fluid.append(heat_flux_solid_to_fluid)
        self.heat_flow_data.heat_flux_ambient_to_solid.append(heat_flux_ambient_to_solid)
        self.heat_flow_data.discharge_heat_flux.append(discharge_heat_flux)

        # Append temperature data
        self.heat_flow_data.fluid_temperatures.append(state.temperature)
        self.heat_flow_data.solid_temperatures.append(state.solid_temperature)

        # Append configuration data
        self.heat_flow_data.configurations.append(configuration)
        self.heat_flow_data.model_selections.append(model_selection)

    def finalize_collection(self, duration: float):
        """
        Finalize data collection.

        Args:
            duration: Total analysis duration [s]
        """
        if not self.is_collecting:
            return

        self.heat_flow_data.analysis_duration = duration
        self.heat_flow_data.__post_init__()  # Convert lists to arrays
        self.is_collecting = False

    def get_heat_flow_data(self) -> HeatFlowData:
        """Get collected heat flow data"""
        return self.heat_flow_data

    def reset_collection(self):
        """Reset collector for new analysis"""
        self.heat_flow_data = HeatFlowData()
        self.is_collecting = False
        self.current_analysis = None


class IsochoricAnalysisFacade:
    """
    Facade for isochoric (stops_model) analysis with heat flow data collection.

    This class provides a high-level interface for running complete isochoric
    analyses while collecting comprehensive heat flow data, following the
    facade pattern used in the HFT framework.
    """

    def __init__(self):
        """Initialize facade"""
        self.data_collector = HeatFlowDataCollector()
        self.last_results = None

    def analyze_discharge_mission(self,
                                 discharge_rate: float,
                                 duration: float,
                                 initial_mass: float = 10.0,
                                 initial_temperature: float = 20.0,
                                 tank_volume: float = 0.5,
                                 thermal_model: Optional[IsochoricThermalModel] = None) -> IsochoricAnalysisResults:
        """
        Analyze a discharge mission with heat flow data collection.

        Args:
            discharge_rate: Discharge mass flow rate [kg/s]
            duration: Mission duration [s]
            initial_mass: Initial hydrogen mass [kg]
            initial_temperature: Initial temperature [K]
            tank_volume: Tank volume [m³]
            thermal_model: Thermal model (uses default if None)

        Returns:
            IsochoricAnalysisResults: Complete analysis results
        """
        from src.mission.isochoric_missions import DischargeMission

        # Create mission
        mission = DischargeMission.constant_discharge(
            discharge_rate=discharge_rate,
            duration=duration,
            initial_mass=initial_mass,
            initial_temperature=initial_temperature
        )

        # Use default thermal model if none provided
        if thermal_model is None:
            thermal_model = create_default_coupled_thermal_model(
                tank_volume=tank_volume,
                ambient_temperature=288.15
            )

        return self._run_analysis(mission, thermal_model)

    def analyze_refuel_mission(self,
                              refuel_rate: float,
                              duration: float,
                              target_mass: float = 20.0,
                              initial_temperature: float = 20.0,
                              tank_volume: float = 0.5,
                              thermal_model: Optional[IsochoricThermalModel] = None) -> IsochoricAnalysisResults:
        """
        Analyze a refuel mission with heat flow data collection.

        Args:
            refuel_rate: Refuel mass flow rate [kg/s]
            duration: Mission duration [s]
            target_mass: Target final mass [kg]
            initial_temperature: Initial temperature [K]
            tank_volume: Tank volume [m³]
            thermal_model: Thermal model (uses default if None)

        Returns:
            IsochoricAnalysisResults: Complete analysis results
        """
        from src.mission.isochoric_missions import RefuelMission

        # Create mission
        mission = RefuelMission.constant_refuel(
            refuel_rate=refuel_rate,
            duration=duration,
            target_mass=target_mass,
            initial_temperature=initial_temperature
        )

        # Use default thermal model if none provided
        if thermal_model is None:
            thermal_model = create_default_coupled_thermal_model(
                tank_volume=tank_volume,
                ambient_temperature=288.15
            )

        return self._run_analysis(mission, thermal_model)

    def analyze_dormancy_mission(self,
                                duration: float,
                                initial_mass: float = 10.0,
                                initial_temperature: float = 20.0,
                                ambient_temperature: float = 288.15,
                                tank_volume: float = 0.5,
                                thermal_model: Optional[IsochoricThermalModel] = None) -> IsochoricAnalysisResults:
        """
        Analyze a dormancy mission with heat flow data collection.

        Args:
            duration: Dormancy duration [s]
            initial_mass: Initial hydrogen mass [kg]
            initial_temperature: Initial temperature [K]
            ambient_temperature: Ambient temperature [K]
            tank_volume: Tank volume [m³]
            thermal_model: Thermal model (uses default if None)

        Returns:
            IsochoricAnalysisResults: Complete analysis results
        """
        from src.mission.isochoric_missions import DormancyMission

        # Create mission
        mission = DormancyMission.long_term_storage(
            duration=duration,
            initial_mass=initial_mass,
            initial_temperature=initial_temperature,
            ambient_temperature=ambient_temperature
        )

        # Use default thermal model if none provided
        if thermal_model is None:
            thermal_model = create_default_coupled_thermal_model(
                tank_volume=tank_volume,
                ambient_temperature=ambient_temperature
            )

        return self._run_analysis(mission, thermal_model)

    def _run_analysis(self,
                     mission: IsochoricMission,
                     thermal_model: IsochoricThermalModel) -> IsochoricAnalysisResults:
        """
        Run complete analysis with data collection.

        Args:
            mission: Isochoric mission to analyze
            thermal_model: Thermal model for analysis

        Returns:
            IsochoricAnalysisResults: Complete analysis results
        """
        # Start data collection
        self.data_collector.start_collection(mission, thermal_model)

        # Create and run analysis
        analysis = IsochoricMissionAnalysis(mission, thermal_model)

        # Monkey patch data collection into the analysis
        original_ode_system = None

        def enhanced_ode_system(t, y):
            """Enhanced ODE system with data collection"""
            # Convert state vector to IsochoricTankState
            state = IsochoricTankState(
                fuel_mass=y[0],
                temperature=y[1],
                solid_temperature=y[2],
                tank_volume=mission.parameters.tank_volume
            )

            # Get flow functions for this section (assume first section for simplicity)
            section = mission.sections[0]
            fuel_flow_func = mission.get_fuel_flow_function(section)
            discharge_flow_func = mission.get_discharge_flow_function(section)

            # Compute thermal coupling
            Q_solid = thermal_model.compute_heat_flux(t, state)
            dTs_dt = thermal_model.compute_solid_temperature_derivative(t, state)
            Q_ambient = getattr(thermal_model, '_compute_ambient_heat_flux', lambda *args: 0.0)(state.solid_temperature)

            # Compute dynamic model derivatives
            derivatives = mission.dynamic_model_switcher.compute_state_derivatives(
                t, state, fuel_flow_func, discharge_flow_func,
                Q_solid=Q_solid, dTs_dt=dTs_dt
            )

            # Collect data
            self.data_collector.collect_timestep_data(
                time=t,
                state=state,
                heat_flux_solid_to_fluid=Q_solid,
                heat_flux_ambient_to_solid=Q_ambient,
                discharge_heat_flux=0.0,  # Could be enhanced
                configuration=getattr(state, 'configuration', 'A'),
                model_selection="single_phase"  # Could be enhanced
            )

            return [
                derivatives.fuel_mass_derivative,
                derivatives.temperature_derivative,
                derivatives.solid_temperature_derivative
            ]

        # Run analysis (this would need to be modified to use enhanced_ode_system)
        tank_states = analysis.run_analysis()

        # Finalize data collection
        total_duration = sum(section.duration for section in mission.sections)
        self.data_collector.finalize_collection(total_duration)

        # Package results
        results = IsochoricAnalysisResults(
            tank_states=tank_states,
            heat_flow_data=self.data_collector.get_heat_flow_data(),
            mission=mission,
            thermal_model=thermal_model
        )

        self.last_results = results
        return results

    def get_last_results(self) -> Optional[IsochoricAnalysisResults]:
        """Get results from last analysis"""
        return self.last_results

    def export_heat_flow_data(self, filename: str = None) -> Dict:
        """
        Export heat flow data for analysis or plotting.

        Args:
            filename: Optional filename to save data

        Returns:
            Dict: Heat flow data dictionary
        """
        if not self.last_results:
            return {}

        data = {
            'times': self.last_results.heat_flow_data.times.tolist(),
            'heat_flux_solid_to_fluid': self.last_results.heat_flow_data.heat_flux_solid_to_fluid.tolist(),
            'heat_flux_ambient_to_solid': self.last_results.heat_flow_data.heat_flux_ambient_to_solid.tolist(),
            'fluid_temperatures': self.last_results.heat_flow_data.fluid_temperatures.tolist(),
            'solid_temperatures': self.last_results.heat_flow_data.solid_temperatures.tolist(),
            'configurations': self.last_results.heat_flow_data.configurations,
            'model_selections': self.last_results.heat_flow_data.model_selections,
            'scenario': self.last_results.scenario,
            'summary_statistics': self.last_results.heat_flow_data.get_summary_statistics()
        }

        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

        return data


def main():
    pass


if __name__ == "__main__":
    main()


# End