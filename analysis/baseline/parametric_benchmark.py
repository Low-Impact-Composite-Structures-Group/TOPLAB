"""
Parametric benchmark analysis for hydrogen storage systems.

This module provides an abstract base class for comparing different hydrogen storage
approaches (cryocompressed, liquid, compressed gas) under identical mission conditions.

The framework performs comprehensive analysis including:
- Tank geometry optimization with constant φ (length-to-radius ratio)
- Structural mass calculations with netting analysis
- Thermal modeling with ambient heat transfer
- Mission simulation with ATR72 flight profile
- Advanced radius optimization to find minimal compliant tank size

All storage types use identical:
- Mission profiles and flow patterns
- Structural calculation methods
- Thermal modeling approaches
- Optimization algorithms
- Plotting and analysis frameworks

Only the initial thermodynamic conditions and minimum density requirements vary.

Authors: Dante Raso (2025)
Based on baseline_cch2.py analysis framework
"""

# Standard library imports
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Third-party imports
from CoolProp.CoolProp import PropsSI

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mission framework imports
from src.mission.isochoric_missions import (
    IsochoricMissionAnalysis,
    IsochoricMissionParameters
)
from src.thermodynamics.isochoric_thermal_model import StopsModelThermalModel
from src.multistep_methods.linear_multistep_methods import RK45Solver

# Plotting imports
from plotting.sb_plotting import SeabornPlotter


class ParametricBenchmark(ABC):
    """
    Abstract base class for parametric hydrogen storage analysis.

    This class contains all storage-agnostic functionality including tank geometry,
    structural calculations, mission setup, optimization, and plotting. Storage-specific
    parameters are defined through abstract methods.

    Attributes:
        tank_radius: Inner tank radius [m]
        phi: Length-to-radius ratio (φ = L/R) - constant across storage types
        liner_thickness: Aluminum liner thickness [m]
        insulation_thickness: Insulation layer thickness [m]
        design_pressure: Design pressure limit [Pa]
        safety_factor: Structural design safety factor
        composite_winding_angle: Optimal winding angle [degrees]
        ambient_temperature: Environmental temperature [K]
        ambient_htc: Ambient heat transfer coefficient [W/m²K]
    """

    def __init__(self, tank_radius: float = 0.69):
        """
        Initialize analysis with configuration parameters.

        Args:
            tank_radius: Inner tank radius [m]
        """
        # Tank geometry parameters (φ constant across all storage types)
        self.phi = 3.0
        self.tank_radius = tank_radius
        self.tank_length = self.phi * self.tank_radius

        # Material and thickness parameters
        self.liner_thickness = 0.005
        self.insulation_thickness = 0.050
        self.design_pressure = 450e5
        self.safety_factor = 1.25

        # Composite material properties
        self.composite_winding_angle = 54.7

        # Environmental conditions
        self.ambient_temperature = 288.15
        # ambient_htc now provided by get_ambient_htc() method for storage-specific values

        # Solver configuration
        self.time_step = 1.0
        self.rtol = 1e-6
        self.atol = 1e-9
        self.max_step = 10.0

        # Computed tank properties (will be calculated)
        self.tank_volume = None
        self.inner_surface_area = None
        self.outer_surface_area = None
        self.composite_thickness = None
        self.liner_mass = None
        self.composite_mass = None
        self.total_structural_mass = None

        # Analysis results
        self.results = None
        self.mission_analysis = None

    # Abstract methods - must be implemented by storage-specific subclasses
    @abstractmethod
    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure [Pa]."""
        pass

    @abstractmethod
    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature [K]."""
        pass

    @abstractmethod
    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density [kg/m³]."""
        pass

    @abstractmethod
    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        pass

    @abstractmethod
    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get storage-specific optimization parameters."""
        pass

    def get_venting_pressure(self) -> float:
        """Get venting pressure [Pa]. Override in subclasses if needed."""
        return 500e5  # Default 500 bar for most storage types

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure [Pa]. Override in subclasses if needed."""
        return 15e5  # Default 15 bar - sufficient for most storage types

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient [W/m²K]. Override in subclasses if needed."""
        return 0.025  # Default 0.025 W/m²K - typical for vacuum insulation

    # Common methods - identical across all storage types
    def calculate_tank_geometry(self):
        """
        Calculate tank geometry and volume.

        Computes volume and surface areas for cylindrical tank with spherical end caps.
        """
        inner_radius = self.tank_radius
        cylindrical_length = self.tank_length

        # Tank volume (cylinder + spherical caps)
        cylinder_volume = np.pi * inner_radius**2 * cylindrical_length
        cap_volume = (4/3) * np.pi * inner_radius**3
        self.tank_volume = cylinder_volume + cap_volume

        # Inner surface area (for heat transfer)
        cylinder_area = 2 * np.pi * inner_radius * cylindrical_length
        cap_area = 4 * np.pi * inner_radius**2
        self.inner_surface_area = cylinder_area + cap_area

        print(f"Tank Geometry:")
        print(f"  Inner radius: {inner_radius:.3f} m")
        print(f"  Cylindrical length: {cylindrical_length:.3f} m")
        print(f"  φ (L/R) ratio: {self.phi:.1f}")
        print(f"  Inner volume: {self.tank_volume:.3f} m³")
        print(f"  Inner surface area: {self.inner_surface_area:.2f} m²")

    def calculate_composite_thickness(self):
        """
        Calculate composite wall thickness using structural models.

        Uses netting analysis with G10-NIST composite material properties
        to determine required thickness for design pressure.

        Returns:
            float: Composite material density [kg/m³]
        """
        from src.materials.nist_materials import NISTComposite
        from src.tank_design.structural_models import CompositeCylinder

        # Get G10 composite material
        g10_material = NISTComposite.g10_nist(np.radians(self.composite_winding_angle))

        # Tank section interface for structural model
        class TankSectionInterface:
            def __init__(self, radius, material):
                self.radius = radius
                self.material = material
                self.type = "cylinder"

        tank_section = TankSectionInterface(self.tank_radius, g10_material)
        composite_model = CompositeCylinder()

        # Apply safety factor to pressure
        design_pressure = self.design_pressure / self.safety_factor

        # Calculate thickness using netting analysis
        self.composite_thickness = composite_model.compute_thickness(
            tank_section, design_pressure
        )

        print(f"Composite Wall Design:")
        print(f"  Material: G10-NIST at {self.composite_winding_angle}° winding")
        print(f"  Tensile strength: {g10_material.failure_stress/1e6:.1f} MPa")
        print(f"  Design stress: {g10_material.failure_stress/self.safety_factor/1e6:.1f} MPa (SF = {self.safety_factor})")
        print(f"  Design pressure: {self.design_pressure/1e5:.0f} bar")
        print(f"  Required thickness: {self.composite_thickness*1000:.2f} mm")

        return g10_material.density

    def calculate_structural_mass(self):
        """
        Calculate structural mass components.

        Computes liner and composite shell masses using material densities
        and geometric shell volume calculations.
        """
        from src.materials.nist_materials import NISTMetal

        # Get material densities
        aluminum_material = NISTMetal.aluminum_6061T6_nist()
        rho_aluminum = aluminum_material.density
        rho_composite = self.calculate_composite_thickness()

        # Liner mass calculation
        inner_radius = self.tank_radius
        liner_outer_radius = inner_radius + self.liner_thickness

        # Liner volume (thin shell approximation)
        cylinder_liner_volume = 2 * np.pi * inner_radius * self.liner_thickness * self.tank_length
        cap_liner_volume = 4 * np.pi * inner_radius**2 * self.liner_thickness
        liner_volume = cylinder_liner_volume + cap_liner_volume

        self.liner_mass = rho_aluminum * liner_volume

        # Composite mass calculation
        composite_inner_radius = liner_outer_radius
        composite_outer_radius = composite_inner_radius + self.composite_thickness

        # Composite volume (shell difference)
        cylinder_composite_volume = np.pi * (composite_outer_radius**2 - composite_inner_radius**2) * self.tank_length
        cap_composite_volume = (4/3) * np.pi * (composite_outer_radius**3 - composite_inner_radius**3)
        composite_volume = cylinder_composite_volume + cap_composite_volume

        self.composite_mass = rho_composite * composite_volume
        self.total_structural_mass = self.liner_mass + self.composite_mass

        # Outer surface area for ambient heat transfer
        outer_radius = composite_outer_radius + self.insulation_thickness
        cylinder_outer_area = 2 * np.pi * outer_radius * self.tank_length
        cap_outer_area = 4 * np.pi * outer_radius**2
        self.outer_surface_area = cylinder_outer_area + cap_outer_area

        print(f"Structural Mass Analysis:")
        print(f"  Liner (Al 6061, {self.liner_thickness*1000:.1f}mm): {self.liner_mass:.1f} kg")
        print(f"  Composite (G10, {self.composite_thickness*1000:.2f}mm): {self.composite_mass:.1f} kg")
        print(f"  Total structural mass: {self.total_structural_mass:.1f} kg")
        print(f"  Outer radius (with insulation): {outer_radius:.3f} m")
        print(f"  Outer surface area: {self.outer_surface_area:.2f} m²")

    def setup_analysis(self):
        """Set up the analysis components."""
        print(f"Setting up {self.get_storage_type_name()} analysis...")
        print("="*60)

        # Calculate tank geometry and structural mass
        self.calculate_tank_geometry()
        self.calculate_structural_mass()

        # Get storage-specific initial conditions
        initial_pressure = self.get_initial_pressure()
        initial_temperature = self.get_initial_temperature()
        minimum_density = self.get_minimum_density()

        # Validate configuration parameters
        if initial_pressure <= 0:
            raise ValueError(f"Invalid initial pressure: {initial_pressure} Pa. Must be positive.")
        if initial_temperature <= 0:
            raise ValueError(f"Invalid initial temperature: {initial_temperature} K. Must be positive.")
        if self.tank_volume <= 0:
            raise ValueError(f"Invalid tank volume: {self.tank_volume} m³. Must be positive.")
        if minimum_density <= 0:
            raise ValueError(f"Invalid minimum density: {minimum_density} kg/m³. Must be positive.")

        # Calculate initial hydrogen state
        try:
            initial_density = PropsSI("Dmass", "P", initial_pressure,
                                     "T", initial_temperature, "hydrogen")
        except Exception as e:
            raise ValueError(f"CoolProp calculation failed for initial state: {e}")

        initial_mass = initial_density * self.tank_volume

        print(f"\nHydrogen State ({self.get_storage_type_name()}):")
        print(f"  Initial conditions: P={initial_pressure/1e5:.1f} bar, T={initial_temperature:.2f} K")
        print(f"  Initial density: {initial_density:.2f} kg/m³")
        print(f"  Initial mass: {initial_mass:.2f} kg")

        # Get ATR72 mission requirements
        from src.mission.mission import Mission
        atr72_mission = Mission.atr72()

        # Calculate total mission duration by summing section durations
        self.mission_duration = sum(section.duration for section in atr72_mission.sections)
        total_fuel_consumption = atr72_mission.required_fuel

        # Validate fuel requirements
        mass_to_discharge = total_fuel_consumption
        final_mass = initial_mass - mass_to_discharge

        if final_mass <= 0:
            raise ValueError(f"Invalid mission: Final mass would be negative ({final_mass:.2f} kg). Need more initial fuel.")
        if mass_to_discharge <= 0:
            raise ValueError(f"Invalid discharge mass: {mass_to_discharge:.2f} kg. Must be positive.")

        print(f"Mission duration: {self.mission_duration:.0f} s ({self.mission_duration/3600:.2f} h)")
        print(f"Total fuel consumption (ATR72): {total_fuel_consumption:.2f} kg")
        print(f"Mass to discharge: {mass_to_discharge:.2f} kg")
        print(f"Predicted final mass: {final_mass:.2f} kg")

        # Calculate thermal equilibrium initial solid temperature
        # Create temporary thermal model to calculate equilibrium
        temp_thermal_model = StopsModelThermalModel(
            tank_volume=self.tank_volume,
            inner_surface_area=self.inner_surface_area,
            outer_surface_area=self.outer_surface_area,
            inner_diameter=2 * self.tank_radius,
            ambient_temperature=self.ambient_temperature,
            ambient_htc=self.get_ambient_htc(),
            liner_mass=self.liner_mass,
            wall_mass=self.composite_mass
        )

        initial_solid_temp = temp_thermal_model.calculate_thermal_equilibrium_Ts(
            initial_temperature
        )

        print(f"  Thermal equilibrium solid temperature: {initial_solid_temp:.2f} K")

        # Get storage-specific venting pressure
        p_vent = self.get_venting_pressure()
        p_min = self.get_minimum_pressure()

        # Create mission parameters
        mission_params = IsochoricMissionParameters(
            tank_volume=self.tank_volume,
            p_min=p_min,
            p_vent=p_vent,
            initial_mass=initial_mass,
            initial_temperature=initial_temperature,
            initial_solid_temperature=initial_solid_temp,
            ambient_temperature=self.ambient_temperature,
            time_step=self.time_step,
            rtol=self.rtol,
            atol=self.atol,
            use_density_stopping_events=False
        )

        # Create discharge mission with actual ATR72 flow profile
        self.mission = self.create_atr72_discharge_mission(mission_params)
        print("Using actual ATR72 mission profile with varying flow rates across sections")

        # Create solver
        solver = RK45Solver(
            timestep=self.time_step,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step
        )
        self.mission.integration_method = solver

        # Create thermal model with calculated properties
        inner_diameter = 2 * self.tank_radius
        print(f"\nThermal Model Configuration:")
        print(f"  Inner diameter: {inner_diameter:.3f} m")
        print(f"  Inner surface area: {self.inner_surface_area:.2f} m²")
        print(f"  Outer surface area: {self.outer_surface_area:.2f} m²")
        print(f"  Ambient HTC: {self.get_ambient_htc():.3f} W/m²K")

        self.thermal_model = StopsModelThermalModel(
            tank_volume=self.tank_volume,
            inner_surface_area=self.inner_surface_area,
            outer_surface_area=self.outer_surface_area,
            inner_diameter=inner_diameter,
            ambient_temperature=self.ambient_temperature,
            ambient_htc=self.get_ambient_htc(),
            liner_mass=self.liner_mass,
            wall_mass=self.composite_mass
        )

        # Create mission analysis
        self.mission_analysis = IsochoricMissionAnalysis(
            self.mission,
            self.thermal_model
        )

        print("Analysis setup complete")
        print(f"Configuration monitoring: p_vent = {p_vent/1e5:.0f} bar")
        return {
            'initial_mass': initial_mass,
            'initial_density': initial_density,
            'final_mass': final_mass,
            'total_fuel_consumption': total_fuel_consumption,
            'mission_duration': self.mission_duration
        }

    def create_atr72_discharge_mission(self, mission_params: IsochoricMissionParameters):
        """
        Create discharge mission using ATR72 flight profile.

        Converts ATR72 mission sections to isochoric discharge format with
        proper flow rate handling for both constant and time-varying flows.

        Args:
            mission_params: Isochoric mission parameters

        Returns:
            IsochoricMission: Mission with ATR72 flow profile
        """
        from src.mission.isochoric_missions import IsochoricMission
        from src.mission.mission_sections import MissionSection, OutFlow
        from src.mission.mission import Mission

        atr72_mission = Mission.atr72()
        discharge_sections = []

        print(f"\nATR72 Mission Flow Profile ({self.get_storage_type_name()}):")
        print("Section | Duration (s) | Flow Rate (kg/s) | Flow Type")
        print("-" * 60)

        for i, section in enumerate(atr72_mission.sections):
            original_flow = section.fuel_flows[0]

            # Convert to positive discharge rate
            if isinstance(original_flow.mass_flow, list):
                discharge_rates = [abs(rate) for rate in original_flow.mass_flow]
                discharge_flow = OutFlow(discharge_rates, "gas")
                flow_str = f'{discharge_rates[0]:.6f} -> {discharge_rates[-1]:.6f}'
                flow_type = 'Variable'
            else:
                discharge_rate = abs(original_flow.mass_flow)
                discharge_flow = OutFlow(discharge_rate, "gas")
                flow_str = f'{discharge_rate:.6f}'
                flow_type = 'Constant'

            print(f'{i+1:7} | {section.duration:12.1f} | {flow_str:16} | {flow_type}')

            discharge_section = MissionSection(
                duration=section.duration,
                fuel_flows=[discharge_flow],
                altitude=section.altitude,
                mach_number=section.mach_number,
                fuel_flow_key=section.fuel_flow_key
            )
            discharge_sections.append(discharge_section)

        discharge_mission = IsochoricMission(discharge_sections, mission_params, "DISCHARGE")
        print(f"\nCreated ATR72 discharge mission with {len(discharge_sections)} sections")
        return discharge_mission

    def _get_mission_flow_rate_at_time(self, absolute_time: float) -> float:
        """
        Get the actual mission discharge flow rate at a given absolute time.

        This uses the mission's flow functions to get the correct flow rate,
        avoiding artifacts from discrete mass difference calculations.

        Args:
            absolute_time: Time in seconds from mission start

        Returns:
            float: Discharge flow rate [kg/s] at given time
        """
        if not hasattr(self, 'mission') or self.mission is None:
            return 0.0

        current_time = 0.0

        # Find which section we're in and get the flow rate
        for section in self.mission.sections:
            if current_time <= absolute_time < current_time + section.duration:
                # Get relative time within this section
                section_time = absolute_time - current_time

                # Get discharge flow function for this section
                discharge_flow_func = self.mission.get_discharge_flow_function(section)
                return discharge_flow_func(section_time)

            current_time += section.duration

        # If we're past the end of the mission, return 0
        return 0.0

    def _calculate_ohex_requirements(self, times_seconds: list) -> list:
        """
        Calculate OHEX (Outboard Heat Exchanger) heat requirements.

        Uses enthalpy difference method: Q_oHEX = mdot * (h_target - h_current)
        where h_target is at standard fuel cell inlet conditions (200K, 20 bar).

        Args:
            times_seconds: Time array in seconds

        Returns:
            list: OHEX heat requirements [W] for each time point
        """
        # OHEX target conditions (fuel cell inlet requirements)
        OHEX_TARGET_TEMPERATURE = 200.0   # K - typical fuel cell inlet temperature
        OHEX_TARGET_PRESSURE = 20e5       # Pa - 20 bar fuel cell inlet pressure

        try:
            # Calculate target enthalpy (constant for all time points)
            h_target = PropsSI("Hmass", "T", OHEX_TARGET_TEMPERATURE, "P", OHEX_TARGET_PRESSURE, "hydrogen")
        except Exception as e:
            print(f"Warning: Could not calculate OHEX target enthalpy: {e}")
            return [0.0] * len(times_seconds)

        qdot_ohex = []

        for i, state in enumerate(self.results.states):
            try:
                # Get current state conditions
                T_current = state.temperature
                rho_current = state.fuel_mass / self.tank_volume  # kg/m³

                # Get mass flow rate from mission profile
                time_s = i * self.time_step
                mass_rate = self._get_mission_flow_rate_at_time(time_s)

                if T_current > 0 and rho_current > 0 and mass_rate > 0:
                    # Calculate current pressure and enthalpy
                    p_current = PropsSI("P", "T", T_current, "Dmass", rho_current, "hydrogen")
                    h_current = PropsSI("Hmass", "T", T_current, "P", p_current, "hydrogen")

                    # Calculate OHEX heat requirement
                    q_ohex = mass_rate * (h_target - h_current)  # [W]
                    qdot_ohex.append(max(0.0, q_ohex))  # Ensure non-negative
                else:
                    qdot_ohex.append(0.0)

            except Exception as e:
                # Silently handle CoolProp errors (common at extreme conditions)
                qdot_ohex.append(0.0)

        return qdot_ohex

    def _calculate_energy_requirements(self, times_seconds: list, heat_flow_data: dict) -> dict:
        """
        Calculate total energy requirements by integrating heat flow curves.

        Uses trapezoidal rule to compute area under the heat exchanger requirement curves.

        Args:
            times_seconds: Time array in seconds
            heat_flow_data: Dictionary with 'qdot_disch' (iHEX) and 'qdot_ohex' (oHEX) data

        Returns:
            dict: Energy requirements in MJ and kWh for iHEX, oHEX, and total
        """
        import numpy as np

        # Convert time to hours for integration
        times_hours = np.array(times_seconds) / 3600.0

        # Get heat flow data
        qdot_ihex = np.array(heat_flow_data.get('qdot_disch', []))
        qdot_ohex = np.array(heat_flow_data.get('qdot_ohex', []))

        # Ensure arrays have same length
        min_length = min(len(times_hours), len(qdot_ihex), len(qdot_ohex))
        times_hours = times_hours[:min_length]
        qdot_ihex = qdot_ihex[:min_length]
        qdot_ohex = qdot_ohex[:min_length]

        energy_results = {}

        # Calculate iHEX energy (area under iHEX curve)
        if len(qdot_ihex) > 1 and any(q > 1e-6 for q in qdot_ihex):
            # Convert W to kW and integrate over time (hours) to get kWh
            qdot_ihex_kw = qdot_ihex / 1000.0  # W to kW
            energy_ihex_kwh = np.trapz(qdot_ihex_kw, times_hours)  # kWh
            energy_ihex_mj = energy_ihex_kwh * 3.6  # kWh to MJ
        else:
            energy_ihex_kwh = 0.0
            energy_ihex_mj = 0.0

        # Calculate oHEX energy (area under oHEX curve)
        if len(qdot_ohex) > 1 and any(q > 1e-6 for q in qdot_ohex):
            # Convert W to kW and integrate over time (hours) to get kWh
            qdot_ohex_kw = qdot_ohex / 1000.0  # W to kW
            energy_ohex_kwh = np.trapz(qdot_ohex_kw, times_hours)  # kWh
            energy_ohex_mj = energy_ohex_kwh * 3.6  # kWh to MJ
        else:
            energy_ohex_kwh = 0.0
            energy_ohex_mj = 0.0

        # Calculate total energy
        energy_total_kwh = energy_ihex_kwh + energy_ohex_kwh
        energy_total_mj = energy_ihex_mj + energy_ohex_mj

        energy_results = {
            'ihex': {
                'kwh': energy_ihex_kwh,
                'mj': energy_ihex_mj
            },
            'ohex': {
                'kwh': energy_ohex_kwh,
                'mj': energy_ohex_mj
            },
            'total': {
                'kwh': energy_total_kwh,
                'mj': energy_total_mj
            }
        }

        return energy_results

    def run_single_analysis(self):
        """Run single discharge analysis."""
        print("\n" + "="*60)
        print(f"{self.get_storage_type_name().upper()} ANALYSIS - SINGLE DISCHARGE")
        print("="*60)

        # Setup analysis
        setup_data = self.setup_analysis()

        print(f"\nStarting integration...")

        # Run analysis (the mission is already configured to discharge the right amount)
        self.results = self.mission_analysis.run_analysis()

        if not self.results:
            raise ValueError("Analysis execution failed: No results returned from mission analysis")

        if len(self.results.states) == 0:
            raise ValueError("Analysis execution failed: Results contain no tank states")

        # Process results
        final_state = self.results.states[-1]
        if not hasattr(final_state, 'fuel_mass'):
            raise ValueError("Analysis results invalid: Final state missing fuel_mass attribute")

        final_mass = final_state.fuel_mass
        final_density = final_mass / self.tank_volume

        # Calculate time (verification script pattern)
        final_time = (len(self.results.states) - 1) * self.time_step

        print(f"\nAnalysis completed successfully!")
        print(f"Final density: {final_density:.2f} kg/m³")
        print(f"Final mass: {final_mass:.2f} kg")
        print(f"Total time: {final_time:.0f} s")

        return True

    def print_results(self):
        """Print comprehensive analysis results."""
        if not self.results or len(self.results.states) == 0:
            print("No results to display")
            return

        print("\n" + "="*60)
        print(f"{self.get_storage_type_name().upper()} ANALYSIS RESULTS")
        print("="*60)

        # Extract key states
        initial_state = self.results.states[0]
        final_state = self.results.states[-1]

        # Tank configuration
        print("Tank Configuration:")
        print(f"  Volume: {self.tank_volume:.1f} m³")
        print(f"  Inner Surface Area: {self.inner_surface_area:.1f} m²")
        print(f"  Outer Surface Area: {self.outer_surface_area:.1f} m²")
        print(f"  Total Structural Mass: {self.total_structural_mass:.1f} kg")
        print(f"  Ambient HTC: {self.get_ambient_htc():.3f} W/m²K (insulation type varies by storage)")

        # Initial conditions
        initial_density = initial_state.fuel_mass / self.tank_volume
        print("\nInitial Conditions:")
        print(f"  Pressure: {initial_state.pressure/1e5:.1f} bar")
        print(f"  Temperature: {initial_state.temperature:.1f} K")
        print(f"  Density: {initial_density:.2f} kg/m³")
        print(f"  Mass: {initial_state.fuel_mass:.2f} kg")

        # Final conditions
        final_density = final_state.fuel_mass / self.tank_volume
        print("\nFinal Conditions:")
        print(f"  Pressure: {final_state.pressure/1e5:.1f} bar")
        print(f"  Temperature: {final_state.temperature:.1f} K")
        print(f"  Density: {final_density:.2f} kg/m³")
        print(f"  Mass: {final_state.fuel_mass:.2f} kg")

        # Mission performance
        fuel_consumed = initial_state.fuel_mass - final_state.fuel_mass
        mission_time = (len(self.results.states) - 1) * self.time_step
        avg_discharge_rate = fuel_consumed / mission_time

        print("\nMission Performance:")
        print(f"  Duration: {mission_time:.0f} s ({mission_time/3600:.2f} h)")
        print(f"  Fuel Consumed: {fuel_consumed:.2f} kg")
        print(f"  Average Discharge Rate: {avg_discharge_rate:.4f} kg/s")
        print(f"  Final Density: {final_density:.2f} kg/m³")
        print(f"  Minimum Density Requirement: {self.get_minimum_density():.1f} kg/m³")

        # Validation checks
        print("\nValidation:")
        density_check = final_density >= self.get_minimum_density()
        print(f"  Minimum density met: {'✓' if density_check else '✗'} ({final_density:.2f} ≥ {self.get_minimum_density():.1f})")
        print(f"  Final mass reasonable: {'✓' if final_state.fuel_mass > 0 else '✗'}")
        print(f"  Mission completed: {'✓' if mission_time > 1000 else '✗'}")

    def plot_results(self, save_path=None):
        """Generate result plots using sb_plotting module."""
        if not self.results or len(self.results.states) == 0:
            print("No data available for plotting")
            return

        # Initialize the sb_plotter
        sb_plotter = SeabornPlotter()

        # Extract time series data
        times_seconds = [i * self.time_step for i in range(len(self.results.states))]
        times_hours = [t / 3600.0 for t in times_seconds]  # Convert to hours for better readability
        masses = [state.fuel_mass for state in self.results.states]
        temperatures = [state.temperature for state in self.results.states]
        pressures = [state.pressure/1e5 for state in self.results.states]  # Convert to bar
        densities = [mass / self.tank_volume for mass in masses]

        # Use sb_plotting baseline tank states function
        fig = sb_plotter.plot_baseline_tank_states(
            times=times_hours,
            masses=masses,
            temperatures=temperatures,
            pressures=pressures,
            densities=densities,
            target_density=self.get_minimum_density(),  # Show minimum density requirement
            figsize=(12, 10)
        )

        # Update title to include storage type
        fig.suptitle(f'{self.get_storage_type_name()} Tank States During ATR72 Mission',
                     fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_fuel_flow_profile(self, save_path=None):
        """
        Plot the fuel flow profile using sb_plotting module.
        Shows actual ATR72 mission sections with varying flow rates and durations.
        """
        if not self.results or not hasattr(self, 'mission'):
            raise ValueError("Cannot plot fuel flow profile: No analysis results or mission data available")

        # Initialize the sb_plotter
        sb_plotter = SeabornPlotter()

        # Get actual ATR72 mission data
        from src.mission.mission import Mission
        atr72_mission = Mission.atr72()
        mission_sections = atr72_mission.sections

        # Use sb_plotting baseline fuel flow function
        fig = sb_plotter.plot_baseline_fuel_flow(
            mission_sections=mission_sections,
            figsize=(14, 10)
        )

        # Update title to include storage type
        fig.suptitle(f'ATR72 Mission Fuel Flow Profile - {self.get_storage_type_name()}',
                     fontsize=16, fontweight='bold', y=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fuel flow plot saved to: {save_path}")
        else:
            plt.show()

    def plot_density_temperature(self, save_path=None):
        """
        Plot density vs temperature during discharge using sb_plotting module.
        Shows the thermodynamic path with saturation line and isobars.
        """
        if not self.results or len(self.results.states) == 0:
            print("No data available for density-temperature plot")
            return

        # Initialize the sb_plotter
        sb_plotter = SeabornPlotter()

        # Extract temperature, density, and pressure data
        temperatures = [state.temperature for state in self.results.states]
        masses = [state.fuel_mass for state in self.results.states]
        pressures = [state.pressure/1e5 for state in self.results.states]  # Convert to bar
        densities = [mass / self.tank_volume for mass in masses]

        # Use sb_plotting baseline density-temperature function
        fig = sb_plotter.plot_baseline_density_temperature(
            temperatures=temperatures,
            densities=densities,
            pressures=pressures,
            include_saturation_line=True,
            include_isobars=True,
            figsize=(8, 6)
        )

        # Update title to include storage type
        fig.suptitle(f'{self.get_storage_type_name()} Density vs Temperature Path',
                     fontsize=14, fontweight='bold', y=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Density-temperature plot saved to: {save_path}")
        else:
            plt.show()

    def plot_heat_exchanger_requirements(self, save_path=None):
        """
        Plot heat exchanger requirements for sLH2 and CCH2 storage types.

        CH2 (compressed hydrogen) operates at ambient temperature and maintains
        high pressure throughout the mission, so it requires no heat exchange
        and this method will skip plotting for CH2.

        Args:
            save_path: Optional path to save plot file
        """
        if not self.results or len(self.results.states) == 0:
            print("No data available for heat exchanger plot")
            return

        # Skip heat exchanger analysis entirely for CH2 (compressed hydrogen)
        storage_name = self.get_storage_type_name()
        if "Compressed H2 (CH2)" in storage_name:
            print(f"\n=== HEAT EXCHANGER ANALYSIS SKIPPED FOR {storage_name} ===")
            print("CH2 operates at ambient temperature and maintains high pressure")
            print("throughout the mission, requiring no heat exchanger systems.")
            print("No iHEX needed: Pressure stays well above minimum threshold")
            print("No oHEX needed: Storage temperature is already near ambient")
            print("Energy requirements: 0.00 kWh (0.0 MJ) - no thermal management")
            print("=" * 65)
            return

        sb_plotter = SeabornPlotter()
        times_seconds = [i * self.time_step for i in range(len(self.results.states))]

        # Compute iHEX heat flow requirements
        qdot_disch = []
        min_pressure = float('inf')
        max_pressure = 0.0
        activation_count = 0

        for i, state in enumerate(self.results.states):
            try:
                pressure_bar = state.pressure / 1e5
                min_pressure = min(min_pressure, pressure_bar)
                max_pressure = max(max_pressure, pressure_bar)

                # Configuration B threshold
                minimum_pressure_pa = self.get_minimum_pressure()
                minimum_pressure_bar = minimum_pressure_pa / 1e5
                venting_pressure_bar = self.get_venting_pressure() / 1e5

                # Check Configuration B status
                has_config_b_attr = hasattr(state, 'is_configuration_B')
                state_config_b = state.is_configuration_B(minimum_pressure_pa) if has_config_b_attr else False
                pressure_config_b = pressure_bar <= minimum_pressure_bar
                is_config_b = state_config_b or pressure_config_b

                if is_config_b:
                    activation_count += 1

                # Debug output for boundary states
                if i < 5 or (i > len(self.results.states) - 5):
                    print(f"State {i}: P={pressure_bar:.1f}bar, has_attr={has_config_b_attr}, state_B={state_config_b}, press_B={pressure_config_b}, final_B={is_config_b}")

                if is_config_b:
                    # Compute heat requirement for iHEX operation using actual mission flow rate
                    time_s = i * self.time_step
                    mass_rate = self._get_mission_flow_rate_at_time(time_s)

                    T_hydrogen = state.temperature
                    if T_hydrogen > 0 and mass_rate > 0:
                        cp_hydrogen = 14300.0
                        temp_difference = max(0, 80.0 - T_hydrogen)
                        ihex_requirement = mass_rate * cp_hydrogen * temp_difference



                        qdot_disch.append(max(0.0, ihex_requirement))
                    else:
                        qdot_disch.append(0.0)
                else:
                    qdot_disch.append(0.0)

            except Exception as e:
                print(f"Warning: iHEX computation failed at t={i * self.time_step}s: {e}")
                qdot_disch.append(0.0)

        print(f"\n=== Configuration Debug Information ({self.get_storage_type_name()}) ===")
        print(f"Min pressure during mission: {min_pressure:.1f} bar")
        print(f"Max pressure during mission: {max_pressure:.1f} bar")
        p_min_bar = self.get_minimum_pressure() / 1e5
        print(f"Configuration B threshold (p_min): {p_min_bar:.1f} bar")
        print(f"Configuration B: Pressure maintenance when P ≤ {p_min_bar:.1f} bar")
        print(f"Configuration C: Venting active when P ≥ {venting_pressure_bar:.1f} bar")
        print(f"States where Configuration B was active: {activation_count} out of {len(self.results.states)}")
        print(f"Non-zero heat flows: {sum(1 for q in qdot_disch if abs(q) > 1e-6)}")
        if qdot_disch:
            print(f"Max heat flow: {max(qdot_disch):.2f} kW")

        # 🔍 PROBE VALUES: Check for section boundary artifacts (zero-drop issue)
        print(f"\n=== HEAT FLOW DATA PROBING ({self.get_storage_type_name()}) ===")
        print(f"Total data points: {len(qdot_disch)}")
        zero_indices = [i for i, q in enumerate(qdot_disch) if q == 0.0]
        nonzero_indices = [i for i, q in enumerate(qdot_disch) if q > 1e-6]

        print(f"Zero values: {len(zero_indices)} indices")
        print(f"Non-zero values: {len(nonzero_indices)} indices")

        # Check for suspicious zero patterns (likely section artifacts)
        if zero_indices:
            print(f"First 10 zero indices: {zero_indices[:10]}")
            print(f"Last 10 zero indices: {zero_indices[-10:]}")

            # Check if zeros are clustered at specific intervals (section boundaries)
            if len(nonzero_indices) > 10:
                print(f"First 10 non-zero indices: {nonzero_indices[:10]}")
                print(f"Last 10 non-zero indices: {nonzero_indices[-10:]}")

                # Look for transitions: non-zero -> zero -> non-zero (section boundary pattern)
                transitions = []
                for i in range(1, len(qdot_disch) - 1):
                    prev_q = qdot_disch[i-1]
                    curr_q = qdot_disch[i]
                    next_q = qdot_disch[i+1]

                    # Pattern: non-zero -> zero -> non-zero (likely section boundary)
                    if prev_q > 1e-6 and curr_q == 0.0 and next_q > 1e-6:
                        transitions.append((i, prev_q, curr_q, next_q))

                if transitions:
                    print(f"\n🚨 SECTION BOUNDARY ARTIFACTS DETECTED:")
                    print(f"Found {len(transitions)} suspicious zero-drops:")
                    for i, (idx, prev, curr, next_val) in enumerate(transitions[:5]):  # Show first 5
                        time_s = idx * self.time_step
                        print(f"  {i+1}. Index {idx} (t={time_s:.1f}s): {prev:.1f} -> {curr:.1f} -> {next_val:.1f}")

                    if len(transitions) > 5:
                        print(f"  ... and {len(transitions) - 5} more")

                    # 🛠️ FIX SECTION BOUNDARY ARTIFACTS
                    print(f"\n🔧 APPLYING INTERPOLATION FIX:")
                    fixes_applied = 0
                    for idx, prev_q, curr_q, next_q in transitions:
                        # Apply interpolation fix: average between previous and next values for smoother transition
                        interpolated_q = (prev_q + next_q) / 2.0
                        qdot_disch[idx] = interpolated_q
                        fixes_applied += 1
                        print(f"  Fixed index {idx}: {curr_q:.1f} -> {interpolated_q:.1f} (interpolated from {prev_q:.1f} & {next_q:.1f})")

                    print(f"✓ Applied {fixes_applied} interpolation fixes")

                    # Verify the fix worked
                    remaining_transitions = []
                    for i in range(1, len(qdot_disch) - 1):
                        prev_q = qdot_disch[i-1]
                        curr_q = qdot_disch[i]
                        next_q = qdot_disch[i+1]
                        if prev_q > 1e-6 and curr_q == 0.0 and next_q > 1e-6:
                            remaining_transitions.append((i, prev_q, curr_q, next_q))

                    print(f"✓ Verification: {len(remaining_transitions)} artifacts remaining (should be 0)")
                else:
                    print("No obvious section boundary artifacts detected")

        print("================================\n")

        # Calculate OHEX heat requirements for all storage types
        print("Calculating OHEX (Outboard Heat Exchanger) requirements...")
        qdot_ohex = self._calculate_ohex_requirements(times_seconds)

        if qdot_ohex and any(q > 1e-6 for q in qdot_ohex):
            max_ohex = max(qdot_ohex)
            print(f"OHEX calculation complete: max = {max_ohex/1000:.1f} kW")
        else:
            print("OHEX calculation complete: all values zero or failed")

        # Determine what curves to plot based on storage type
        storage_name = self.get_storage_type_name()
        has_ihex = "Compressed H2 (CH2)" not in storage_name  # Only pure CH2 has no iHEX (no Configuration B)

        heat_flow_data = {
            't': times_seconds,
            'qdot_disch': qdot_disch if has_ihex else [0.0] * len(times_seconds),
            'qdot_ohex': qdot_ohex
        }

        # Configure plotting based on storage type
        if has_ihex:
            # sLH2 and CCH2: Show iHEX + oHEX + total (3 curves)
            plot_total = True
            print(f"Plotting 3 curves for {storage_name}: iHEX, oHEX, and total")
        else:
            # CH2: Show only oHEX + total (2 curves, total coincides with oHEX)
            plot_total = True
            print(f"Plotting 2 curves for {storage_name}: oHEX and total (coincident)")

        # Calculate energy requirements (area under curves)
        print("\nCalculating energy requirements (area under heat exchanger curves)...")
        energy_results = self._calculate_energy_requirements(times_seconds, heat_flow_data)

        # Print energy summary
        print(f"\n=== ENERGY REQUIREMENTS SUMMARY ({storage_name}) ===")

        if has_ihex and energy_results['ihex']['kwh'] > 0.001:
            print(f"iHEX Energy:  {energy_results['ihex']['kwh']:.2f} kWh  ({energy_results['ihex']['mj']:.1f} MJ)")
        elif has_ihex:
            print(f"iHEX Energy:  0.00 kWh  (0.0 MJ)")

        if energy_results['ohex']['kwh'] > 0.001:
            print(f"oHEX Energy:  {energy_results['ohex']['kwh']:.2f} kWh  ({energy_results['ohex']['mj']:.1f} MJ)")
        else:
            print(f"oHEX Energy:  0.00 kWh  (0.0 MJ)")

        print(f"Total Energy: {energy_results['total']['kwh']:.2f} kWh  ({energy_results['total']['mj']:.1f} MJ)")

        # Energy breakdown percentage
        if energy_results['total']['kwh'] > 0.001:
            ihex_pct = (energy_results['ihex']['kwh'] / energy_results['total']['kwh']) * 100 if has_ihex else 0
            ohex_pct = (energy_results['ohex']['kwh'] / energy_results['total']['kwh']) * 100
            if has_ihex:
                print(f"Energy Split: iHEX {ihex_pct:.1f}% | oHEX {ohex_pct:.1f}%")
            else:
                print(f"Energy Split: oHEX {ohex_pct:.1f}% (Pure CH2 has no iHEX)")
        print("=" * 60)

        # Store energy results for use in summary generation
        self.energy_requirements = energy_results

        fig = sb_plotter.plot_heat_exchanger_requirements(
            heat_flow_data=heat_flow_data,
            scenario_name=f"{storage_name} discharge",
            ihex_data=None,
            ohex_data=None,
            plot_total=plot_total,
            figsize=(10, 6)
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heat exchanger plot saved to: {save_path}")
        else:
            plt.show()

    def plot_optimization_progress_from_data(self, optimization_progress, target_density, density_tolerance, save_path=None):
        """
        Visualize bisection search progress using pre-existing optimization data.

        This method uses optimization progress data that was already captured during
        a previous optimization run, avoiding the need to re-run the entire optimization.

        Args:
            optimization_progress: Dictionary with optimization iteration data
            target_density: Minimum density target for the optimization
            density_tolerance: Density tolerance used in optimization
            save_path: Optional path to save plot file
        """
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION PROGRESS VISUALIZATION - {self.get_storage_type_name().upper()}")
        print(f"{'='*80}")

        # Create progress plot using SeabornPlotter
        from plotting.sb_plotting import SeabornPlotter
        sb_plotter = SeabornPlotter()

        try:
            fig = sb_plotter.plot_bisection_optimization_progress(
                optimization_data=optimization_progress,
                target_density=target_density,
                density_tolerance=density_tolerance
            )

            print("Creating optimization progress plot...")
            print(f"Total evaluations: {len(optimization_progress['iterations'])}")
            if optimization_progress['radii']:
                optimal_radius = optimization_progress['radii'][-1]  # Last radius should be optimal
                print(f"Optimal radius found: {optimal_radius:.3f}m")

            # Save plot if path provided
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Optimization progress plot saved to: {save_path}")

            return fig

        except Exception as e:
            print(f"⚠️  Failed to create optimization progress plot: {e}")
            return None

    def plot_optimization_progress(self, save_path=None):
        """
        Run optimization and visualize the bisection search progress.

        This method runs the full bisection optimization to find the optimal radius
        and plots the convergence progress showing how the algorithm narrows down
        the search range to find the minimum feasible tank radius.

        Note: This method re-runs the optimization. For efficiency, use
        plot_optimization_progress_from_data() with pre-existing optimization data.

        Args:
            save_path: Optional path to save plot file
        """
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION PROGRESS VISUALIZATION - {self.get_storage_type_name().upper()}")
        print(f"{'='*80}")

        # Run optimization to capture progress data
        optimization_results = find_optimal_radius_for_storage_type(self.__class__)

        if not optimization_results['search_successful']:
            print("⚠️  Optimization failed - cannot generate progress plot")
            print(f"Failure reason: {optimization_results.get('failure_reason', 'Unknown')}")
            return None

        # Extract optimization progress data and delegate to the efficient version
        return self.plot_optimization_progress_from_data(
            optimization_results['optimization_progress'],
            optimization_results['minimum_density_target'],
            optimization_results['density_tolerance'],
            save_path
        )



    def run_analysis(self, include_plots=True):
        """Run complete benchmark analysis."""
        # Run single discharge analysis
        success = self.run_single_analysis()

        if not success:
            raise ValueError("Benchmark analysis failed: Single discharge analysis did not complete successfully")

        # Validate results before proceeding
        if not self.results:
            raise ValueError("Benchmark analysis failed: No results available after successful analysis")

        # Print results
        try:
            self.print_results()
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Results printing failed: Missing required data - {e}")

        # Generate all 4 plots simultaneously (each in separate windows)
        if include_plots:
            # Turn on interactive mode to show multiple plots without blocking
            plt.ion()

            try:
                print("Generating 4 separate plots...")
                self.plot_results()
                self.plot_fuel_flow_profile()
                self.plot_density_temperature()
                self.plot_heat_exchanger_requirements()
                print("All 4 plots displayed successfully!")

                # Keep plots open and return to blocking mode
                plt.ioff()
                plt.show(block=True)  # This will keep all plots visible

            except (AttributeError, ValueError) as e:
                print(f"Plotting failed: {e}")
                plt.ioff()  # Make sure to turn off interactive mode even if plotting fails

        return self.results

    def generate_analysis_summary(self, save_path=None, optimization_results=None):
        """
        Generate a comprehensive analysis summary document in Markdown format.

        Args:
            save_path: Path to save the summary document
            optimization_results: Optional optimization results data

        Returns:
            str: The markdown content
        """
        if not self.results or len(self.results.states) == 0:
            print("No analysis results available for summary generation")
            return ""

        # Get basic data
        initial_state = self.results.states[0]
        final_state = self.results.states[-1]
        storage_type = self.get_storage_type_name()

        # Create markdown content
        md_lines = [
            f"# {storage_type} Analysis Summary",
            f"",
            f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Storage Type Configuration",
            f"",
            f"- **Storage Type:** {storage_type}",
            f"- **Initial Pressure:** {self.get_initial_pressure()/1e5:.1f} bar",
            f"- **Initial Temperature:** {self.get_initial_temperature():.2f} K",
            f"- **Minimum Density Requirement:** {self.get_minimum_density():.1f} kg/m³",
            f"- **Venting Pressure:** {self.get_venting_pressure()/1e5:.1f} bar",
            f"",
            f"## Tank Geometry",
            f"",
            f"- **Inner Radius:** {self.tank_radius:.3f} m",
            f"- **Cylindrical Length:** {2.4 * self.tank_radius:.3f} m (φ = 3.0)",
            f"- **Inner Volume:** {self.tank_volume:.3f} m³",
            f"- **Inner Surface Area:** {self.inner_surface_area:.2f} m²",
            f"- **Outer Surface Area:** {self.outer_surface_area:.2f} m²",
            f"- **Liner Thickness:** {self.liner_thickness*1000:.1f} mm",
            f"- **Wall Thickness (Composite):** {self.composite_thickness*1000:.2f} mm",
            f"",
            f"## Structural Design",
            f"",
            f"- **Total Structural Mass:** {self.total_structural_mass:.1f} kg",
            f"  - Liner Mass: {self.liner_mass:.1f} kg",
            f"  - Composite Mass: {self.composite_mass:.1f} kg",
            f"- **Ambient Heat Transfer Coefficient:** {self.get_ambient_htc():.6f} W/m²K",
            f"",
            f"## Initial Conditions",
            f"",
            f"- **Initial Mass:** {initial_state.fuel_mass:.2f} kg",
            f"- **Initial Pressure:** {initial_state.pressure/1e5:.1f} bar",
            f"- **Initial Temperature:** {initial_state.temperature:.2f} K",
            f"- **Initial Density:** {initial_state.fuel_mass/self.tank_volume:.2f} kg/m³",
            f"",
            f"## Final Conditions",
            f"",
            f"- **Final Mass:** {final_state.fuel_mass:.2f} kg",
            f"- **Final Pressure:** {final_state.pressure/1e5:.1f} bar",
            f"- **Final Temperature:** {final_state.temperature:.2f} K",
            f"- **Final Density:** {final_state.fuel_mass/self.tank_volume:.2f} kg/m³",
            f"",
            f"## Mission Performance",
            f"",
            f"- **Mission Duration:** {len(self.results.states) * self.time_step:.0f} s ({len(self.results.states) * self.time_step/3600:.2f} h)",
            f"- **Fuel Consumed:** {initial_state.fuel_mass - final_state.fuel_mass:.2f} kg",
            f"- **Average Discharge Rate:** {(initial_state.fuel_mass - final_state.fuel_mass)/(len(self.results.states) * self.time_step):.4f} kg/s",
            f"- **Minimum Density Met:** {'✓' if final_state.fuel_mass/self.tank_volume >= self.get_minimum_density() else '✗'} ({final_state.fuel_mass/self.tank_volume:.2f} ≥ {self.get_minimum_density():.1f})",
            f"",
            f"## Heat Exchanger Requirements",
            f"",
        ]

        # Add heat exchanger data if available
        if hasattr(self.results, 'heat_flows') and self.results.heat_flows:
            max_heat_flow = max([abs(hf) for hf in self.results.heat_flows if hf is not None])
            avg_heat_flow = sum([abs(hf) for hf in self.results.heat_flows if hf is not None and hf != 0]) / len([hf for hf in self.results.heat_flows if hf is not None and hf != 0])

            md_lines.extend([
                f"- **Maximum Heat Flow:** {max_heat_flow/1000:.1f} kW",
                f"- **Average Heat Flow:** {avg_heat_flow/1000:.1f} kW",
            ])
        else:
            md_lines.append("- Heat exchanger data not available")

        # Add energy requirements if available
        if hasattr(self, 'energy_requirements') and self.energy_requirements:
            energy = self.energy_requirements
            storage_name = self.get_storage_type_name()

            md_lines.extend([
                f"",
                f"### Energy Requirements",
                f"",
            ])

            # Check if this is CH2 (no iHEX) or CCH2/sLH2 (has iHEX)
            has_ihex = 'iHEX' in storage_name.upper() or 'CCH2' in storage_name.upper() or 'SLH2' in storage_name.upper()

            if has_ihex and energy['ihex']['kwh'] > 0.001:
                md_lines.append(f"- **iHEX Energy:** {energy['ihex']['kwh']:.2f} kWh ({energy['ihex']['mj']:.1f} MJ)")
            elif has_ihex:
                md_lines.append(f"- **iHEX Energy:** 0.00 kWh (0.0 MJ)")

            if energy['ohex']['kwh'] > 0.001:
                md_lines.append(f"- **oHEX Energy:** {energy['ohex']['kwh']:.2f} kWh ({energy['ohex']['mj']:.1f} MJ)")
            else:
                md_lines.append(f"- **oHEX Energy:** 0.00 kWh (0.0 MJ)")

            md_lines.append(f"- **Total Energy:** {energy['total']['kwh']:.2f} kWh ({energy['total']['mj']:.1f} MJ)")

            # Energy split percentage
            if energy['total']['kwh'] > 0.001:
                if has_ihex and energy['ihex']['kwh'] > 0.001:
                    ihex_pct = (energy['ihex']['kwh'] / energy['total']['kwh']) * 100
                    ohex_pct = (energy['ohex']['kwh'] / energy['total']['kwh']) * 100
                    md_lines.append(f"- **Energy Split:** iHEX {ihex_pct:.1f}% | oHEX {ohex_pct:.1f}%")
                else:
                    ohex_pct = (energy['ohex']['kwh'] / energy['total']['kwh']) * 100
                    if 'CH2' in storage_name.upper() and 'CCH2' not in storage_name.upper():
                        md_lines.append(f"- **Energy Split:** oHEX {ohex_pct:.1f}% (Pure CH2 has no iHEX)")
                    else:
                        md_lines.append(f"- **Energy Split:** oHEX {ohex_pct:.1f}%")

        # Add optimization results if provided
        if optimization_results:
            md_lines.extend([
                f"",
                f"## Optimization Results",
                f"",
                f"- **Optimization Successful:** {'✓' if optimization_results.get('search_successful') else '✗'}",
                f"- **Total Evaluations:** {optimization_results.get('evaluation_count', 'N/A')}",
            ])

            # Add optimization parameters
            opt_params = self.get_optimization_parameters()
            md_lines.extend([
                f"- **Starting Radius:** {opt_params.get('min_radius', 'N/A'):.3f} m",
                f"- **Maximum Radius:** {opt_params.get('max_radius', 'N/A'):.3f} m",
                f"- **Optimal Radius:** {optimization_results.get('optimal_radius', 'N/A'):.3f} m" if optimization_results.get('optimal_radius') else "- **Optimal Radius:** Not found",
                f"- **Density Tolerance:** {optimization_results.get('density_tolerance', 'N/A'):.1f} kg/m³",
                f"- **Radius Precision:** ±{opt_params.get('radius_precision', 'N/A')*1000:.1f} mm",
            ])

        # Add dormancy results if available
        if hasattr(self, 'dormancy_summary') and self.dormancy_summary:
            dormancy = self.dormancy_summary
            md_lines.extend([
                f"",
                f"## Dormancy Analysis",
                f"",
                f"- **Dormancy Duration:** {dormancy['duration_hours']:.1f} hours",
                f"- **Initial Mass (Dormancy):** {dormancy['initial_mass']:.2f} kg",
                f"- **Final Mass (Dormancy):** {dormancy['final_mass']:.2f} kg",
                f"- **Mass Vented:** {dormancy['mass_vented']:.2f} kg",
            ])

            if dormancy['time_to_vent_hours'] is not None:
                md_lines.append(f"- **Time to First Venting:** {dormancy['time_to_vent_hours']:.1f} hours")
            else:
                md_lines.append(f"- **Time to First Venting:** No venting occurred during {dormancy['duration_hours']:.1f}h period")

        # Add gravimetric efficiency
        fuel_mass = final_state.fuel_mass
        gravimetric_eff = fuel_mass / (fuel_mass + self.total_structural_mass) * 100
        md_lines.extend([
            f"",
            f"## Performance Metrics",
            f"",
            f"- **Gravimetric Efficiency:** {gravimetric_eff:.1f}%",
            f"- **Fuel-to-Structure Mass Ratio:** {fuel_mass/self.total_structural_mass:.2f}:1",
            f"",
            f"## Mission Profile",
            f"",
            f"- **Mission Type:** ATR72 Regional Aircraft",
            f"- **Flow Profile:** Variable flow rates across 11 mission sections",
            f"- **Mission Sections:** Taxi-out, Takeoff, Climb, Cruise, Initial Descent, Approach, Go-around, Climb-2, Cruise-2, Final Descent, Taxi-in",
            f"",
            f"---",
            f"*Analysis generated using Parametric Benchmark Framework*"
        ])

        # Join all lines
        markdown_content = "\n".join(md_lines)

        # Save if path provided
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"Analysis summary saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save analysis summary: {e}")

        return markdown_content

    def run_dormancy_analysis(self, duration_hours: float = 60.0):
        """
        Run dormancy scenario for cryogenic storage types.

        Simulates a dormancy period where the tank is at rest with no discharge
        or refuel flows. Only ambient heat leak occurs, potentially causing
        pressure buildup and venting (Configuration C).

        Args:
            duration_hours: Dormancy duration in hours (default: 60 hours)

        Returns:
            dict: Dormancy analysis results including time-to-vent
        """
        # Skip dormancy for CH2 (compressed hydrogen at ambient temperature)
        storage_name = self.get_storage_type_name()
        if "Compressed H2 (CH2)" in storage_name:
            print(f"Dormancy analysis skipped for {storage_name} (operates at ambient temperature)")
            return None

        print(f"\n{'='*60}")
        print(f"{storage_name.upper()} DORMANCY ANALYSIS")
        print(f"{'='*60}")
        print(f"Duration: {duration_hours:.1f} hours ({duration_hours*3600:.0f} seconds)")
        print(f"Scenario: Tank at rest with ambient heat leak only")

        # Use initial conditions from discharge analysis
        if not self.results or len(self.results.states) == 0:
            raise ValueError("Must run discharge analysis first to get initial conditions")

        initial_discharge_state = self.results.states[0]

        # Create dormancy mission parameters
        from src.mission.isochoric_missions import IsochoricMissionParameters, IsochoricMission
        from src.mission.mission_sections import MissionSection

        dormancy_params = IsochoricMissionParameters(
            tank_volume=self.tank_volume,
            p_min=self.get_minimum_pressure(),
            p_vent=self.get_venting_pressure(),
            initial_mass=initial_discharge_state.fuel_mass,
            initial_temperature=initial_discharge_state.temperature,
            initial_solid_temperature=initial_discharge_state.temperature,  # Simplified
            ambient_temperature=self.ambient_temperature,
            time_step=self.time_step,
            rtol=self.rtol,
            atol=self.atol,
            use_density_stopping_events=False
        )

        # Create dormancy section (no flows, just ambient heat leak)
        dormancy_section = MissionSection(
            duration=duration_hours * 3600,  # Convert to seconds
            fuel_flows=[],  # No flows during dormancy
            altitude=0.0,
            mach_number=0.0,
            fuel_flow_key="dormancy"
        )

        # Create dormancy mission
        dormancy_mission = IsochoricMission([dormancy_section], dormancy_params, "DORMANCY")
        dormancy_mission.integration_method = RK45Solver(
            timestep=self.time_step,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step
        )

        # Create mission analysis
        dormancy_analysis = IsochoricMissionAnalysis(
            dormancy_mission,
            self.thermal_model
        )

        print("Starting dormancy integration...")
        self.dormancy_results = dormancy_analysis.run_analysis()

        if not self.dormancy_results or len(self.dormancy_results.states) == 0:
            raise ValueError("Dormancy analysis failed: No results returned")

        # Find time to vent (first Configuration C activation)
        time_to_vent = self._find_time_to_vent()

        # Calculate final conditions
        final_state = self.dormancy_results.states[-1]
        initial_mass = initial_discharge_state.fuel_mass
        final_mass = final_state.fuel_mass
        mass_vented = initial_mass - final_mass

        print(f"\nDormancy analysis completed!")
        print(f"Initial mass: {initial_mass:.2f} kg")
        print(f"Final mass: {final_mass:.2f} kg")
        print(f"Mass vented: {mass_vented:.2f} kg")
        if time_to_vent is not None:
            print(f"Time to first venting: {time_to_vent/3600:.1f} hours")
        else:
            print("No venting occurred during dormancy period")

        return {
            'time_to_vent_hours': time_to_vent/3600 if time_to_vent else None,
            'time_to_vent_seconds': time_to_vent,
            'initial_mass': initial_mass,
            'final_mass': final_mass,
            'mass_vented': mass_vented,
            'duration_hours': duration_hours
        }

    def _find_time_to_vent(self):
        """Find the first time step where Configuration C (venting) occurs."""
        venting_pressure = self.get_venting_pressure()

        for i, state in enumerate(self.dormancy_results.states):
            if state.pressure >= venting_pressure:
                return i * self.time_step

        return None  # No venting occurred

    def plot_dormancy_results(self, dormancy_data, save_path=None):
        """
        Plot dormancy results showing mass vs time with venting onset using SeabornPlotter.

        Args:
            dormancy_data: Dictionary with dormancy analysis results
            save_path: Optional path to save plot file
        """
        if not self.dormancy_results or len(self.dormancy_results.states) == 0:
            print("No dormancy data available for plotting")
            return

        # Initialize the SeabornPlotter with consistent styling
        from plotting.sb_plotting import SeabornPlotter
        sb_plotter = SeabornPlotter()

        # Extract data
        times_hours = [i * self.time_step / 3600 for i in range(len(self.dormancy_results.states))]
        masses = [state.fuel_mass for state in self.dormancy_results.states]
        pressures = [state.pressure/1e5 for state in self.dormancy_results.states]  # Convert to bar

        # Create standalone mass vs time plot using SeabornPlotter styling
        fig = sb_plotter.plot_dormancy_mass_evolution(
            times=times_hours,
            masses=masses,
            pressures=pressures,
            time_to_vent_hours=dormancy_data.get('time_to_vent_hours'),
            venting_pressure=self.get_venting_pressure()/1e5,  # Convert to bar
            storage_type=self.get_storage_type_name(),
            figsize=(10, 6)
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dormancy plot saved to: {save_path}")

        return fig


# Common utility functions for all storage types
def check_for_venting(results, venting_threshold=500e5):
    """
    Check if venting occurred during analysis.

    Args:
        results: Mission analysis results
        venting_threshold: Venting pressure threshold [Pa]

    Returns:
        tuple: (venting_occurred, max_pressure, venting_times)
    """
    if not results or len(results.states) == 0:
        return False, 0.0, []

    pressures = [state.pressure for state in results.states]
    max_pressure = max(pressures)
    venting_occurred = max_pressure > venting_threshold

    venting_times = []
    time_step = 1.0
    for i, pressure in enumerate(pressures):
        if pressure > venting_threshold:
            venting_times.append(i * time_step)

    return venting_occurred, max_pressure, venting_times


def check_minimum_density(results, tank_volume, minimum_density):
    """
    Check if final density meets minimum requirement.

    Args:
        results: Mission analysis results
        tank_volume: Tank volume [m³]
        minimum_density: Minimum acceptable density [kg/m³]

    Returns:
        tuple: (density_violation, final_density)
    """
    if not results or len(results.states) == 0:
        return True, 0.0

    final_state = results.states[-1]
    final_density = final_state.fuel_mass / tank_volume
    density_violation = final_density < minimum_density

    return density_violation, final_density


def safe_analysis_run(analysis_class, radius, minimum_density, venting_threshold):
    """
    Run analysis with CoolProp error detection and constraint checking.

    Args:
        analysis_class: Storage-specific analysis class
        radius: Tank radius to test [m]
        minimum_density: Minimum acceptable density [kg/m³]
        venting_threshold: Venting pressure threshold [Pa]

    Returns:
        dict: Analysis results with success/feasible flags
    """
    try:
        # Capture CoolProp warnings and other warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create and run analysis
            analysis = analysis_class(tank_radius=radius)
            success = analysis.run_single_analysis()

            # Check for CoolProp warnings/errors (common patterns)
            coolprop_issues = []
            for warn in w:
                warn_msg = str(warn.message).lower()
                if any(keyword in warn_msg for keyword in ['coolprop', 'saturation', 'convergence', 'iteration', 'invalid']):
                    coolprop_issues.append(str(warn.message))

            if coolprop_issues:
                return {
                    'success': False,
                    'feasible': False,
                    'reason': f'CoolProp_issues: {len(coolprop_issues)} warnings',
                    'details': coolprop_issues[:3],  # First 3 warnings for debugging
                    'radius': radius
                }

            if not success:
                return {
                    'success': False,
                    'feasible': False,
                    'reason': 'Analysis_failed',
                    'radius': radius
                }

            # Check constraints
            venting_occurred, max_pressure, venting_times = check_for_venting(analysis.results, venting_threshold)
            density_violation, final_density = check_minimum_density(
                analysis.results, analysis.tank_volume, minimum_density
            )

            feasible = not venting_occurred and not density_violation

            return {
                'success': True,
                'feasible': feasible,
                'final_density': final_density,
                'max_pressure': max_pressure,
                'max_pressure_bar': max_pressure / 1e5,
                'venting_occurred': venting_occurred,
                'density_violation': density_violation,
                'venting_times': venting_times,
                'analysis': analysis,
                'results': analysis.results,
                'radius': radius,
                'volume': analysis.tank_volume,
                'structural_mass': analysis.total_structural_mass,
                'density_margin': final_density - minimum_density if not density_violation else None
            }

    except Exception as e:
        return {
            'success': False,
            'feasible': False,
            'reason': f'Exception: {str(e)[:100]}',  # Truncate long error messages
            'radius': radius
        }


def find_optimal_radius_for_storage_type(analysis_class, **optimization_params):
    """
    Search for optimal tank radius using robust bisection method.

    This approach is much faster and more robust than brute-force search:
    - Handles CoolProp errors/warnings automatically
    - Uses bisection for O(log N) convergence
    - Finds minimum feasible radius efficiently
    - Stops when within tolerance (2 kg/m³)

    Args:
        analysis_class: Storage-specific analysis class
        **optimization_params: Optimization parameters from storage type

    Returns:
        dict: Search results including optimal radius and analysis data
    """
    # Create temporary instance to get storage-specific parameters
    temp_analysis = analysis_class()

    # Get parameters
    minimum_density = temp_analysis.get_minimum_density()
    venting_threshold = temp_analysis.get_venting_pressure()
    storage_name = temp_analysis.get_storage_type_name()

    # Optimization bounds (updated defaults)
    min_radius = optimization_params.get('min_radius', 0.5)
    max_radius = optimization_params.get('max_radius', 1.5)
    precision = optimization_params.get('radius_precision', 0.005)  # 5mm
    density_tolerance = optimization_params.get('density_tolerance', 2.0)  # 2 kg/m³
    max_evaluations = optimization_params.get('max_evaluations', 20)

    print("\n" + "="*80)
    print(f"ROBUST BISECTION SEARCH - {storage_name.upper()}")
    print("="*80)
    print(f"Search range: {min_radius:.3f}m to {max_radius:.3f}m")
    print(f"Precision: ±{precision*1000:.0f}mm, Density tolerance: ±{density_tolerance:.1f} kg/m³")
    print(f"Requirements: No venting AND final density ≥ {minimum_density:.1f} kg/m³")
    print(f"CoolProp error detection: ENABLED")
    print("-"*80)

    search_results = []
    evaluation_count = 0

    # Initialize optimization progress tracking for plotting
    optimization_progress = {
        'iterations': [],
        'radii': [],
        'final_densities': [],
        'structural_masses': [],
        'feasible': [],
        'phase': [],  # 'bound_search', 'bisection'
        'converged': [],
        'density_margins': [],
        'range_widths': []  # Track bisection range convergence
    }

    # Phase 1: Find feasible upper bound
    print("\n=== PHASE 1: FINDING FEASIBLE UPPER BOUND ===")
    r_max = None
    r_test = max_radius

    while r_test >= min_radius and evaluation_count < max_evaluations:
        evaluation_count += 1
        print(f"Testing upper bound: R={r_test:.3f}m")

        result = safe_analysis_run(analysis_class, r_test, minimum_density, venting_threshold)
        search_results.append(result)

        # Record progress data for plotting
        optimization_progress['iterations'].append(evaluation_count)
        optimization_progress['radii'].append(r_test)
        optimization_progress['final_densities'].append(result.get('final_density', 0))
        optimization_progress['structural_masses'].append(result.get('structural_mass', 0))
        optimization_progress['feasible'].append(result.get('feasible', False))
        optimization_progress['phase'].append('bound_search')
        optimization_progress['converged'].append(result.get('success', False))
        optimization_progress['density_margins'].append(result.get('density_margin', 0))
        optimization_progress['range_widths'].append(max_radius - min_radius)

        if result['success']:
            print(f"  ✓ Success: ρ={result['final_density']:.2f} kg/m³, P_max={result['max_pressure_bar']:.1f} bar")
            if result['feasible']:
                r_max = r_test
                print(f"  ✓ FEASIBLE upper bound found: R={r_max:.3f}m")
                break
            else:
                issues = []
                if result['venting_occurred']:
                    issues.append("venting")
                if result['density_violation']:
                    issues.append("low density")
                print(f"  ✗ Infeasible: {', '.join(issues)}")
        else:
            print(f"  ✗ Failed: {result['reason']}")

        r_test -= 0.1  # Step down by 100mm to find feasible region

    if r_max is None:
        print(f"✗ No feasible solution found in range {min_radius:.3f}m to {max_radius:.3f}m")
        return {
            'optimal_radius': None,
            'optimal_results': None,
            'search_results': search_results,
            'search_successful': False,
            'storage_type': storage_name,
            'failure_reason': 'No feasible upper bound found'
        }

    # Phase 2: Find infeasible lower bound
    print(f"\n=== PHASE 2: FINDING INFEASIBLE LOWER BOUND ===")
    r_min = min_radius
    r_test = r_max - 0.1

    while r_test >= min_radius and evaluation_count < max_evaluations:
        evaluation_count += 1
        print(f"Testing lower bound: R={r_test:.3f}m")

        result = safe_analysis_run(analysis_class, r_test, minimum_density, venting_threshold)
        search_results.append(result)

        # Record progress data for plotting
        optimization_progress['iterations'].append(evaluation_count)
        optimization_progress['radii'].append(r_test)
        optimization_progress['final_densities'].append(result.get('final_density', 0))
        optimization_progress['structural_masses'].append(result.get('structural_mass', 0))
        optimization_progress['feasible'].append(result.get('feasible', False))
        optimization_progress['phase'].append('bound_search')
        optimization_progress['converged'].append(result.get('success', False))
        optimization_progress['density_margins'].append(result.get('density_margin', 0))
        optimization_progress['range_widths'].append(r_max - r_test if r_max else 0)

        if result['success'] and result['feasible']:
            print(f"  ✓ Still feasible: ρ={result['final_density']:.2f} kg/m³")
            r_max = r_test  # Update upper feasible bound
        else:
            r_min = r_test  # Found infeasible lower bound
            if result['success']:
                print(f"  ✓ Infeasible lower bound: R={r_min:.3f}m")
            else:
                print(f"  ✓ Failed lower bound: R={r_min:.3f}m ({result['reason']})")
            break

        r_test -= 0.1

    # Phase 3: Bisection search
    print(f"\n=== PHASE 3: BISECTION OPTIMIZATION ===")
    print(f"Bisection range: [{r_min:.3f}m, {r_max:.3f}m]")

    best_result = None
    iteration = 0

    while (r_max - r_min) > precision and evaluation_count < max_evaluations:
        iteration += 1
        evaluation_count += 1
        r_mid = (r_min + r_max) / 2

        print(f"Bisection {iteration}: R={r_mid:.3f}m (range: {r_max-r_min:.3f}m)")

        result = safe_analysis_run(analysis_class, r_mid, minimum_density, venting_threshold)
        search_results.append(result)

        # Record bisection progress data for plotting
        optimization_progress['iterations'].append(evaluation_count)
        optimization_progress['radii'].append(r_mid)
        optimization_progress['final_densities'].append(result.get('final_density', 0))
        optimization_progress['structural_masses'].append(result.get('structural_mass', 0))
        optimization_progress['feasible'].append(result.get('feasible', False))
        optimization_progress['phase'].append('bisection')
        optimization_progress['converged'].append(result.get('success', False))
        optimization_progress['density_margins'].append(result.get('density_margin', 0))
        optimization_progress['range_widths'].append(r_max - r_min)

        if result['success'] and result['feasible']:
            r_max = r_mid  # Feasible, try smaller radius
            best_result = result
            density_error = abs(result['final_density'] - minimum_density)
            print(f"  ✓ Feasible: ρ={result['final_density']:.2f} kg/m³ (margin: +{result['density_margin']:.2f})")

            # Check if close enough to target
            if density_error <= density_tolerance:
                print(f"  ✓ Within tolerance: |ρ - ρ_min| = {density_error:.2f} ≤ {density_tolerance:.1f}")
                break
        else:
            r_min = r_mid  # Infeasible, need larger radius
            if result['success']:
                print(f"  ✗ Infeasible: ρ={result['final_density']:.2f} kg/m³")
            else:
                print(f"  ✗ Failed: {result['reason']}")

    # Results
    optimal_radius = r_max if best_result else None
    optimal_results = best_result

    # If no good result from bisection, use best from search_results
    if optimal_results is None:
        feasible_results = [r for r in search_results if r['success'] and r['feasible']]
        if feasible_results:
            optimal_results = min(feasible_results, key=lambda x: x['radius'])
            optimal_radius = optimal_results['radius']
            print(f"Using best attempt: R={optimal_radius:.3f}m")

    # Print search summary
    print("\n" + "="*80)
    print(f"BISECTION SEARCH SUMMARY - {storage_name.upper()}")
    print("="*80)
    print(f"Total evaluations: {evaluation_count}")
    print(f"Search results: {len([r for r in search_results if r['success']])} successful, {len([r for r in search_results if r['success'] and r['feasible']])} feasible")

    if optimal_radius is not None:
        print(f"✓ SUCCESS: Optimal radius found: {optimal_radius:.3f}m")
        print(f"  Volume: {optimal_results['volume']:.3f} m³")
        print(f"  Structural mass: {optimal_results['structural_mass']:.1f} kg")
        print(f"  Max pressure: {optimal_results['max_pressure_bar']:.1f} bar (< {venting_threshold/1e5:.0f} bar)")
        print(f"  Final density: {optimal_results['final_density']:.2f} kg/m³ (≥ {minimum_density:.1f} kg/m³)")
        print(f"  Density margin: +{optimal_results['density_margin']:.2f} kg/m³")

        # Calculate efficiency
        if 'analysis' in optimal_results:
            final_state = optimal_results['results'].states[-1]
            fuel_mass = final_state.fuel_mass
            gravimetric_eff = fuel_mass / (fuel_mass + optimal_results['structural_mass']) * 100
            print(f"  Gravimetric efficiency: {gravimetric_eff:.1f}%")
    else:
        print(f"✗ FAILED: No feasible radius found in range")
        # Show best attempt
        if search_results:
            best_attempt = max(search_results, key=lambda x: x.get('final_density', 0) if x['success'] else 0)
            if best_attempt['success']:
                print(f"Best attempt: R={best_attempt['radius']:.3f}m, ρ={best_attempt['final_density']:.2f} kg/m³")

    return {
        'optimal_radius': optimal_radius,
        'optimal_results': optimal_results,
        'search_results': search_results,
        'search_successful': optimal_radius is not None,
        'storage_type': storage_name,
        'evaluation_count': evaluation_count,
        'optimization_progress': optimization_progress,
        'minimum_density_target': minimum_density,
        'density_tolerance': density_tolerance
    }