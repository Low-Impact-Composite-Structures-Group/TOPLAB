"""
Baseline analysis for cryocompressed hydrogen (CCH2) storage using ATR72 mission profile.

This module performs comprehensive hydrogen tank analysis with advanced radius optimization
to find the minimal tank size that meets all mission requirements without venting.

Key Features:
- Two-phase radius optimization algorithm
- Full ATR72 mission profile with realistic fuel flow patterns
- Preliminary structural mass calculations with netting analysis
- Heat exchanger requirements analysis with Configuration B detection
- Tank state visualizations

User Input:
- Mission profile (configurable in the Mission class)
- Materials and thicknesses for tank structure (material properties configurable in NIST material classes)
- Radius-to-length ratio (φ) for tank geometry
- Starting conditions
- Operating conditions (max and min pressures, min density)
- Stopping criteria for optimization
- Vacuum insulation heat transfer coefficient
- Safety factors for structural design



Authors: Dante Raso (2025)
Based on isochoric framework from verification_cch2.py
"""

# Standard library imports
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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


class BaselineCCH2Analysis:
    """
    Baseline cryocompressed hydrogen analysis for ATR72 mission.

    This class performs comprehensive analysis including structural design,
    thermal modeling, and mission simulation with advanced optimization capabilities.

    Attributes:
        tank_radius: Inner tank radius [m]
        phi: Length-to-radius ratio (φ = L/R)
        liner_thickness: Aluminum liner thickness [m]
        insulation_thickness: Insulation layer thickness [m]
        design_pressure: Design pressure limit [Pa]
        safety_factor: Structural design safety factor
        composite_winding_angle: Optimal winding angle [degrees]
        initial_pressure: Initial hydrogen pressure [Pa]
        initial_temperature: Initial hydrogen temperature [K]
        minimum_density: Minimum acceptable final density [kg/m³]
        ambient_temperature: Environmental temperature [K]
        ambient_htc: Ambient heat transfer coefficient [W/m²K]
    """

    def __init__(self, tank_radius: float = 0.69):
        """
        Initialize analysis with configuration parameters.

        Args:
            tank_radius: Inner tank radius [m]
        """
        # Tank geometry parameters
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

        # Initial hydrogen conditions
        self.initial_pressure = 400e5
        self.initial_temperature = 55.0

        # Minimum density requirement
        self.minimum_density = 5.0

        # Environmental conditions
        self.ambient_temperature = 288.15
        self.ambient_htc = 0.025

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
        print("Setting up baseline CCH2 analysis...")
        print("="*60)

        # Calculate tank geometry and structural mass
        self.calculate_tank_geometry()
        self.calculate_structural_mass()

        # Validate configuration parameters
        if self.initial_pressure <= 0:
            raise ValueError(f"Invalid initial pressure: {self.initial_pressure} Pa. Must be positive.")
        if self.initial_temperature <= 0:
            raise ValueError(f"Invalid initial temperature: {self.initial_temperature} K. Must be positive.")
        if self.tank_volume <= 0:
            raise ValueError(f"Invalid tank volume: {self.tank_volume} m³. Must be positive.")
        if self.minimum_density <= 0:
            raise ValueError(f"Invalid minimum density: {self.minimum_density} kg/m³. Must be positive.")

        # Calculate initial hydrogen state
        try:
            initial_density = PropsSI("Dmass", "P", self.initial_pressure,
                                     "T", self.initial_temperature, "hydrogen")
        except Exception as e:
            raise ValueError(f"CoolProP calculation failed for initial state: {e}")

        initial_mass = initial_density * self.tank_volume

        print(f"\nHydrogen State:")
        print(f"  Initial conditions: P={self.initial_pressure/1e5:.1f} bar, T={self.initial_temperature:.2f} K")
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
            ambient_htc=self.ambient_htc,
            liner_mass=self.liner_mass,
            wall_mass=self.composite_mass
        )

        initial_solid_temp = temp_thermal_model.calculate_thermal_equilibrium_Ts(
            self.initial_temperature
        )

        print(f"  Thermal equilibrium solid temperature: {initial_solid_temp:.2f} K")

        # Create mission parameters
        mission_params = IsochoricMissionParameters(
            tank_volume=self.tank_volume,
            p_min=15e5,
            p_vent=500e5,
            initial_mass=initial_mass,
            initial_temperature=self.initial_temperature,
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
        print(f"  Ambient HTC: {self.ambient_htc:.3f} W/m²K")

        self.thermal_model = StopsModelThermalModel(
            tank_volume=self.tank_volume,
            inner_surface_area=self.inner_surface_area,
            outer_surface_area=self.outer_surface_area,
            inner_diameter=inner_diameter,
            ambient_temperature=self.ambient_temperature,
            ambient_htc=self.ambient_htc,
            liner_mass=self.liner_mass,
            wall_mass=self.composite_mass
        )

        # Create mission analysis
        self.mission_analysis = IsochoricMissionAnalysis(
            self.mission,
            self.thermal_model
        )

        print("Analysis setup complete")
        print(f"Configuration monitoring: p_vent = {mission_params.p_vent/1e5:.0f} bar")
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

        print("\nATR72 Mission Flow Profile:")
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

    def run_single_analysis(self):
        """Run single discharge analysis."""
        print("\n" + "="*60)
        print("BASELINE CCH2 ANALYSIS - SINGLE DISCHARGE")
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
        print("BASELINE CCH2 ANALYSIS RESULTS")
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
        print(f"  Ambient HTC: {self.ambient_htc:.3f} W/m²K (vacuum insulation)")

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
        print(f"  Minimum Density Requirement: {self.minimum_density:.1f} kg/m³")

        # Validation checks
        print("\nValidation:")
        density_check = final_density >= self.minimum_density
        print(f"  Minimum density met: {'✓' if density_check else '✗'} ({final_density:.2f} ≥ {self.minimum_density:.1f})")
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
            target_density=self.minimum_density,  # Show minimum density requirement
            figsize=(12, 10)
        )

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

        # Extract temperature and density data
        temperatures = [state.temperature for state in self.results.states]
        masses = [state.fuel_mass for state in self.results.states]
        densities = [mass / self.tank_volume for mass in masses]

        # Use sb_plotting baseline density-temperature function
        fig = sb_plotter.plot_baseline_density_temperature(
            temperatures=temperatures,
            densities=densities,
            include_saturation_line=True,
            include_isobars=True,
            figsize=(8, 6)
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Density-temperature plot saved to: {save_path}")
        else:
            plt.show()

    def plot_heat_exchanger_requirements(self, save_path=None):
        """
        Plot iHEX heat exchanger requirements.

        Computes and visualizes heat flow requirements for maintaining hydrogen
        temperature during discharge when Configuration B is active.

        Args:
            save_path: Optional path to save plot file
        """
        if not self.results or len(self.results.states) == 0:
            print("No data available for heat exchanger plot")
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
                minimum_pressure_bar = 15.0
                minimum_pressure_pa = minimum_pressure_bar * 1e5

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
                    # Compute heat requirement for iHEX operation
                    if i > 0:
                        mass_rate = (self.results.states[i-1].fuel_mass - state.fuel_mass) / self.time_step
                    else:
                        mass_rate = 0.0367

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

        print(f"\n=== iHEX Debug Information ===")
        print(f"Min pressure during mission: {min_pressure:.1f} bar")
        print(f"Max pressure during mission: {max_pressure:.1f} bar")
        print(f"iHEX activation threshold (p_min): 15.0 bar")
        print(f"Configuration B: iHEX active when P ≤ 15.0 bar")
        print(f"Configuration C: Venting active when P ≥ 500.0 bar")
        print(f"States where iHEX was activated (Config B): {activation_count} out of {len(self.results.states)}")
        print(f"Non-zero heat flows: {sum(1 for q in qdot_disch if abs(q) > 1e-6)}")
        if qdot_disch:
            print(f"Max heat flow: {max(qdot_disch):.2f} kW")
        print("================================\n")

        heat_flow_data = {
            't': times_seconds,
            'qdot_disch': qdot_disch,
            'qdot_ohex': [0.0] * len(times_seconds)
        }

        fig = sb_plotter.plot_heat_exchanger_requirements(
            heat_flow_data=heat_flow_data,
            scenario_name="Baseline CCH2 Discharge",
            ihex_data=None,
            ohex_data=None,
            plot_total=False,
            figsize=(10, 6)
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heat exchanger plot saved to: {save_path}")
        else:
            plt.show()

    def run_analysis(self, include_plots=True):
        """Run complete baseline analysis."""
        # Run single discharge analysis
        success = self.run_single_analysis()

        if not success:
            raise ValueError("Baseline analysis failed: Single discharge analysis did not complete successfully")

        # Validate results before proceeding
        if not self.results:
            raise ValueError("Baseline analysis failed: No results available after successful analysis")

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


def check_for_venting(results):
    """
    Check if venting occurred during analysis.

    Args:
        results: Mission analysis results

    Returns:
        tuple: (venting_occurred, max_pressure, venting_times)
    """
    if not results or len(results.states) == 0:
        return False, 0.0, []

    pressures = [state.pressure for state in results.states]
    max_pressure = max(pressures)
    venting_threshold = 500e5
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


def find_optimal_radius(initial_radius=0.5, max_radius=2.0, radius_increment=0.05, max_iterations=50, minimum_density=4.0, target_density_margin=0.5):
    """
    Search for optimal tank radius using two-phase optimization.

    Phase 1: Coarse search to identify feasible range
    Phase 2: Fine search to optimize within feasible range

    The algorithm finds the smallest tank that meets all requirements:
    - No venting (pressure stays below limit)
    - Final density above minimum threshold
    - Target density close to minimum + margin for efficiency

    Args:
        initial_radius: Starting radius for search [m]
        max_radius: Maximum radius to try [m]
        radius_increment: Radius increment for coarse search [m]
        max_iterations: Maximum number of iterations
        minimum_density: Minimum acceptable final density [kg/m³]
        target_density_margin: Target margin above minimum density [kg/m³]

    Returns:
        dict: Search results including optimal radius and analysis data
    """
    print("\n" + "="*80)
    print("OPTIMAL RADIUS SEARCH - ADVANCED OPTIMIZATION")
    print("="*80)
    print(f"Starting search from radius {initial_radius:.3f}m to {max_radius:.3f}m")
    print(f"Coarse increment: {radius_increment:.3f}m, Max iterations: {max_iterations}")
    print(f"Requirements: No venting AND final density ≥ {minimum_density:.1f} kg/m³")
    print(f"Target: Final density = {minimum_density + target_density_margin:.1f} kg/m³ (optimal efficiency)")
    print("-"*80)

    search_results = []
    current_radius = initial_radius
    iteration = 0
    feasible_solutions = []

    # Phase 1: Coarse search to identify feasible range
    print("\n=== PHASE 1: COARSE SEARCH ===")

    while current_radius <= max_radius and iteration < max_iterations:
        iteration += 1

        print(f"\nIteration {iteration}: Testing radius {current_radius:.3f}m")
        print("-" * 50)

        try:
            # Create and run analysis
            analysis = BaselineCCH2Analysis(tank_radius=current_radius)
            results = analysis.run_single_analysis()

            if results:
                # Check for venting
                venting_occurred, max_pressure, venting_times = check_for_venting(analysis.results)

                # Check for minimum density violation
                density_violation, final_density = check_minimum_density(
                    analysis.results, analysis.tank_volume, minimum_density
                )

                # Store results
                iteration_data = {
                    'radius': current_radius,
                    'volume': analysis.tank_volume,
                    'structural_mass': analysis.total_structural_mass,
                    'venting_occurred': venting_occurred,
                    'max_pressure': max_pressure,
                    'max_pressure_bar': max_pressure / 1e5,
                    'venting_times': venting_times,
                    'density_violation': density_violation,
                    'final_density': final_density,
                    'analysis': analysis,
                    'results': analysis.results,
                    'density_margin': final_density - minimum_density  # How much above minimum
                }
                search_results.append(iteration_data)

                # Print iteration results
                print(f"  Volume: {analysis.tank_volume:.3f} m³")
                print(f"  Structural mass: {analysis.total_structural_mass:.1f} kg")
                print(f"  Max pressure: {max_pressure/1e5:.1f} bar")
                print(f"  Final density: {final_density:.2f} kg/m³ (margin: +{final_density - minimum_density:.2f})")
                print(f"  Venting occurred: {'YES' if venting_occurred else 'NO'}")
                print(f"  Density violation: {'YES' if density_violation else 'NO'}")
                if venting_times:
                    print(f"  Venting duration: {len(venting_times)} seconds ({len(venting_times):.1f}% of mission)")

                # Check if this is a feasible solution (no venting AND no density violation)
                if not venting_occurred and not density_violation:
                    feasible_solutions.append(iteration_data)
                    print(f"  ✓ FEASIBLE: Added to candidate list (margin: +{final_density - minimum_density:.2f} kg/m³)")
                else:
                    issues = []
                    if venting_occurred:
                        issues.append("venting detected")
                    if density_violation:
                        issues.append(f"density too low ({final_density:.2f} < {minimum_density:.1f})")
                    print(f"  ✗ Issues: {', '.join(issues)}")

            else:
                print(f"  ✗ Analysis failed for radius {current_radius:.3f}m")

        except Exception as e:
            print(f"  ✗ Error at radius {current_radius:.3f}m: {e}")

        current_radius += radius_increment

    # Phase 2: Fine search within feasible range if we found multiple solutions
    optimal_radius = None
    optimal_results = None

    if len(feasible_solutions) >= 2:
        print(f"\n=== PHASE 2: FINE SEARCH OPTIMIZATION ===")
        print(f"Found {len(feasible_solutions)} feasible solutions")
        print("Performing fine search to find optimal density...")

        # Find the range for fine search (between smallest feasible and next radius)
        feasible_solutions.sort(key=lambda x: x['radius'])
        smallest_feasible = feasible_solutions[0]

        # Search between the smallest feasible radius and a slightly smaller radius
        fine_start = max(initial_radius, smallest_feasible['radius'] - radius_increment)
        fine_end = smallest_feasible['radius']
        fine_increment = 0.005  # 5mm precision

        print(f"Fine search range: {fine_start:.3f}m to {fine_end:.3f}m (increment: {fine_increment:.3f}m)")

        best_candidate = None
        fine_radius = fine_start
        fine_iteration = 0
        max_fine_iterations = int((fine_end - fine_start) / fine_increment) + 1

        while fine_radius <= fine_end and fine_iteration < max_fine_iterations:
            fine_iteration += 1

            try:
                analysis = BaselineCCH2Analysis(tank_radius=fine_radius)
                results = analysis.run_single_analysis()

                if results:
                    venting_occurred, max_pressure, venting_times = check_for_venting(analysis.results)
                    density_violation, final_density = check_minimum_density(
                        analysis.results, analysis.tank_volume, minimum_density
                    )

                    if not venting_occurred and not density_violation:
                        candidate = {
                            'radius': fine_radius,
                            'volume': analysis.tank_volume,
                            'structural_mass': analysis.total_structural_mass,
                            'venting_occurred': venting_occurred,
                            'max_pressure': max_pressure,
                            'max_pressure_bar': max_pressure / 1e5,
                            'venting_times': venting_times,
                            'density_violation': density_violation,
                            'final_density': final_density,
                            'analysis': analysis,
                            'results': analysis.results,
                            'density_margin': final_density - minimum_density
                        }

                        # Select candidate closest to target density (minimum + margin)
                        target_density = minimum_density + target_density_margin
                        if best_candidate is None or abs(final_density - target_density) < abs(best_candidate['final_density'] - target_density):
                            best_candidate = candidate
                            print(f"  New best: R={fine_radius:.3f}m, ρ={final_density:.2f} kg/m³ (target: {target_density:.1f})")

            except Exception as e:
                pass  # Skip failed analyses in fine search

            fine_radius += fine_increment

        optimal_results = best_candidate if best_candidate else smallest_feasible
        optimal_radius = optimal_results['radius']

    elif len(feasible_solutions) == 1:
        # Only one feasible solution found
        optimal_results = feasible_solutions[0]
        optimal_radius = optimal_results['radius']
        print(f"\nOnly one feasible solution found at radius {optimal_radius:.3f}m")

    else:
        # No feasible solutions found
        print(f"\nNo feasible solutions found in search range")

    # Print search summary
    print("\n" + "="*80)
    print("RADIUS SEARCH SUMMARY")
    print("="*80)

    if optimal_radius is not None:
        print(f"SUCCESS: Optimal radius found: {optimal_radius:.3f}m")
        print(f"Volume: {optimal_results['volume']:.3f} m³")
        print(f"Structural mass: {optimal_results['structural_mass']:.1f} kg")
        print(f"Max pressure: {optimal_results['max_pressure_bar']:.1f} bar (< 500 bar)")
        print(f"Final density: {optimal_results['final_density']:.2f} kg/m³ (≥ {minimum_density:.1f} kg/m³)")

        # Calculate efficiency
        final_state = optimal_results['results'].states[-1]
        fuel_mass = final_state.fuel_mass
        gravimetric_eff = fuel_mass / (fuel_mass + optimal_results['structural_mass']) * 100
        print(f"Gravimetric efficiency: {gravimetric_eff:.1f}%")

    else:
        print(f"FAILED: No radius found up to {max_radius:.3f}m that meets both requirements")
        if search_results:
            # Find best result (prioritize no venting, then highest density)
            venting_free_results = [r for r in search_results if not r['venting_occurred']]
            if venting_free_results:
                best_result = max(venting_free_results, key=lambda x: x['final_density'])
                print(f"Best venting-free attempt: {best_result['radius']:.3f}m with density {best_result['final_density']:.2f} kg/m³")
            else:
                best_result = min(search_results, key=lambda x: x['max_pressure'])
                print(f"Best pressure attempt: {best_result['radius']:.3f}m with {best_result['max_pressure_bar']:.1f} bar")

    print("\nDetailed Results:")
    print(f"{'Radius (m)':<12} {'Volume (m³)':<12} {'Mass (kg)':<12} {'Max P (bar)':<12} {'Density (kg/m³)':<15} {'Venting':<10} {'Density OK':<12}")
    print("-" * 95)
    for result in search_results:
        venting_status = "YES" if result['venting_occurred'] else "NO"
        density_status = "NO" if result['density_violation'] else "YES"
        print(f"{result['radius']:<12.3f} {result['volume']:<12.3f} {result['structural_mass']:<12.1f} "
              f"{result['max_pressure_bar']:<12.1f} {result['final_density']:<15.2f} {venting_status:<10} {density_status:<12}")

    return {
        'optimal_radius': optimal_radius,
        'optimal_results': optimal_results,
        'search_results': search_results,
        'search_successful': optimal_radius is not None
    }


def main():
    """Main execution function."""
    print("Starting Baseline CCH2 Analysis with No-Venting Radius Search")
    print("="*80)

    # Run optimal radius search
    search_results = find_optimal_radius(
        initial_radius=0.4,
        max_radius=1.0,
        radius_increment=0.05,
        max_iterations=30,
        minimum_density=5.0,
        target_density_margin=0.3
    )

    if search_results['search_successful']:
        print(f"\n" + "="*80)
        print("FINAL ANALYSIS WITH OPTIMAL RADIUS")
        print("="*80)

        # Use the optimal configuration
        optimal_data = search_results['optimal_results']
        analysis = optimal_data['analysis']

        # Print comprehensive results
        analysis.print_results()

        # Generate plots for optimal configuration (optional)
        try:
            print("\nGenerating 4 separate plots for optimal configuration...")

            # Turn on interactive mode to show multiple plots without blocking
            plt.ion()

            analysis.plot_results()
            analysis.plot_fuel_flow_profile()
            analysis.plot_density_temperature()
            analysis.plot_heat_exchanger_requirements()

            print("All 4 plots displayed successfully!")

            # Keep plots open and return to blocking mode
            plt.ioff()
            plt.show(block=True)  # This will keep all plots visible

        except Exception as e:
            print(f"Plotting failed (this is non-critical): {e}")
            plt.ioff()  # Make sure to turn off interactive mode even if plotting fails

    else:
        print(f"\n" + "="*80)
        print("SEARCH FAILED - ANALYZING BEST ATTEMPT")
        print("="*80)

        if search_results['search_results']:
            # Use the best attempt (lowest maximum pressure)
            best_result = min(search_results['search_results'], key=lambda x: x['max_pressure'])
            analysis = best_result['analysis']

            print(f"Using radius {best_result['radius']:.3f}m (max pressure: {best_result['max_pressure_bar']:.1f} bar, final density: {best_result['final_density']:.2f} kg/m³)")
            analysis.print_results()

            try:
                print("\nGenerating 4 separate plots for best attempt...")

                # Turn on interactive mode to show multiple plots without blocking
                plt.ion()

                analysis.plot_results()
                analysis.plot_fuel_flow_profile()
                analysis.plot_density_temperature()
                analysis.plot_heat_exchanger_requirements()

                print("All 4 plots displayed successfully!")

                # Keep plots open and return to blocking mode
                plt.ioff()
                plt.show(block=True)  # This will keep all plots visible

            except Exception as e:
                print(f"Plotting failed (this is non-critical): {e}")
                plt.ioff()  # Make sure to turn off interactive mode even if plotting fails
        else:
            print("No valid results to analyze.")


if __name__ == "__main__":
    main()