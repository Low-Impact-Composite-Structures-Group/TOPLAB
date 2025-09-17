"""
Baseline analysis for cryocompressed hydrogen (CCH2) storage using the ATR72 mission profile.

This script performs a single discharge analysis to validate the approach before iterative sizing.
Uses the same framework as verification_cch2.py but with ATR72 mission fuel flow profile.

Mission Requirements:
- ATR72 mission profile with realistic fuel flow requirements
- Initial conditions: 40 MPa (400 bar), 70K, gas-phase hydrogen
- Stopping condition: 5.8 kg/m³ density
- Materials: Same thermal model as verification_cch2.py

Authors: Dante Raso (2025)
Based on isochoric framework from verification_cch2.py
"""

# Standard library imports
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Third-party imports
from CoolProp.CoolProp import PropsSI

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mission framework imports
# from src.mission.mission_profiles import Mission
from src.mission.isochoric_missions import (
    DischargeMission,
    IsochoricMissionAnalysis,
    IsochoricMissionParameters
)
from src.thermodynamics.isochoric_thermal_model import StopsModelThermalModel
from src.multistep_methods.linear_multistep_methods import RK45Solver

# Plotting imports
from plotting.sb_plotting import SeabornPlotter
from plotting.plot_style_sb import configure_plot_style


class BaselineCCH2Analysis:
    """
    Baseline cryocompressed hydrogen analysis for ATR72 mission.

    Single discharge case to validate approach before iterative sizing.
    """

    def __init__(self):
        """Initialize analysis with configuration parameters."""
        # Use similar tank size as verification (scale up for ATR72)
        self.tank_volume = 5.0           # m³ (10x verification tank)
        self.tank_surface_area = 20.0    # m² (approximate)

        # Initial hydrogen conditions (cryocompressed)
        # Pressures for analysis
        self.initial_pressure = 400e5    # Pa (400 bar) - initial storage pressure
        self.max_allowable_pressure = 450e5  # Pa (450 bar) - maximum pressure for structural design (pvent)
        self.initial_temperature = 70.0  # K - cryogenic temperature

        # Stopping condition
        self.stopping_density = 5.8      # kg/m³ - same as verification

        # Environmental conditions
        self.ambient_temperature = 288.15  # K (15°C)
        self.ambient_htc = 0.025          # W/m²K (vacuum insulation)

        # Solver configuration
        self.time_step = 1.0             # s
        self.rtol = 1e-6
        self.atol = 1e-9
        self.max_step = 10.0

        # Analysis results
        self.results = None
        self.mission_analysis = None

# Standard library imports
import sys
import time
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Analysis framework imports
from src.mission.isochoric_missions import (
    DischargeMission,
    IsochoricMission,
    IsochoricMissionAnalysis,
    IsochoricMissionParameters
)
from src.mission.mission import Mission
from src.thermodynamics.isochoric_thermal_model import StopsModelThermalModel
from src.multistep_methods.linear_multistep_methods import (
    LSODASolver,
    RK45Solver,
    RadauSolver,
    DOP853Solver,
    BDFSolver
)

# Tank geometry and materials imports
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.tank_design.liner import Liner
from src.materials.nist_materials import NISTMetal, NISTComposite
from src.materials.materials import Metal

# Plotting imports
from plotting.sb_plotting import SeabornPlotter
from plotting.plot_style_sb import BORDEAUX, KONINGSBLAUW, BOSGROEN, DONKERGRIJS, ORANJE
from plotting.plot_style_sb import configure_plot_style

# Dynamic models imports
from src.dynamics.isochoric_dynamic_models import set_heat_flow_data_collector


class BaselineCCH2Analysis:
    """Baseline analysis for cryocompressed hydrogen storage tank."""

    def __init__(self):
        """Initialize analysis parameters."""
        # Mission parameters
        self.mission = Mission.atr72()
        self.volume_margin = 1.25  # Factor of safety

        # Initial conditions (same as verification_cch2)
        self.initial_pressure = 40e6  # Pa (400 bar) - initial storage pressure
        self.max_allowable_pressure = 450e5  # Pa (450 bar) - maximum pressure for structural design (pvent)
        self.initial_temperature = 70.0  # K
        self.initial_mass_fraction = 0.0  # Pure gas behavior

        # Target conditions
        self.target_density = 5.8  # kg/m³ (stopping condition)
        self.mass_tolerance = 0.5  # kg (convergence tolerance)

        # Thermal model parameters
        self.ambient_temperature = 288.15  # K (reduced to avoid extreme temperature differences)
        self.ambient_htc = 0.01  # W/m²K (very low heat transfer for stability)

        # Tank material parameters
        self.liner_density = 2700.0  # kg/m³ (aluminum)
        self.wall_density = 1800.0  # kg/m³ (G10 composite)
        self.liner_thickness = 0.002  # m (2mm aluminum liner)
        self.wall_thickness = 0.05  # m (50mm G10 wall)

        # Iteration parameters - start with smaller tanks
        self.max_iterations = 20  # Reduce iterations to avoid extreme cases
        self.volume_scaling_factor = 0.9  # Factor for adjusting total tank volume
        self.initial_phi = 0.5  # Initial dimensionless ratio phi = radius/body_length
        self.phi_adjustment_factor = 0.05  # How much to adjust phi each iteration

    def calculate_initial_guess(self) -> tuple[float, float, float]:
        """Calculate initial tank size guess based on mission fuel requirements.

        Returns:
            tuple: (initial_volume [m³], phi [dimensionless], mass [kg])
        """
        # Get mission fuel requirements
        required_fuel_mass = self.mission.required_fuel * self.volume_margin
        print(f"Mission required fuel mass: {self.mission.required_fuel:.2f} kg")
        print(f"With volume margin ({self.volume_margin}): {required_fuel_mass:.2f} kg")

        # Get hydrogen properties at initial conditions
        try:
            density = PropsSI('D', 'P', self.initial_pressure, 'T', self.initial_temperature, 'PARAHYDROGEN')
        except Exception as e:
            print(f"CoolProp error, using fallback density: {e}")
            density = 70.0  # kg/m³ fallback

        print(f"Initial hydrogen density: {density:.2f} kg/m³")

        # Calculate required volume for cylindrical tank with hemispherical caps
        required_volume = required_fuel_mass / density

        # Start with a smaller, more conservative tank size for numerical stability
        initial_volume = required_volume * 0.6  # 40% smaller for stability
        phi = self.initial_phi  # phi = radius/body_length

        print(f"Initial volume estimate: {required_volume:.4f} m³")
        print(f"Initial conservative volume: {initial_volume:.4f} m³")
        print(f"Initial phi (r/L): {phi:.3f}")

        return initial_volume, phi, required_fuel_mass

    def volume_to_dimensions(self, volume: float, phi: float) -> tuple[float, float]:
        """Convert volume and phi to radius and body_length.

        For cylindrical tank with spherical caps: V = π*r²*L_body + (4/3)*π*r³
        With phi = r/L_body, we get: V = π*r²*(r/phi) + (4/3)*π*r³ = π*r³*(1/phi + 4/3)

        Args:
            volume: Tank volume [m³]
            phi: Dimensionless ratio r/L_body

        Returns:
            tuple: (radius [m], body_length [m])
        """
        # Solve for radius: r³ = V / (π * (1/phi + 4/3))
        radius = (volume / (np.pi * (1/phi + 4/3))) ** (1/3)
        body_length = radius / phi

        return radius, body_length

    def create_tank(self, radius: float, body_length: float) -> CylindricalTankSphericalCaps:
        """Create cylindrical tank with hemispherical caps.

        Args:
            radius: Tank radius [m]
            body_length: Cylindrical body length [m]

        Returns:
            CylindricalTankSphericalCaps: Tank object
        """
        # Create NIST composite material for tank structure
        # Note: The tank geometry uses this for structural calculations
        # The thermal model uses separate NIST materials internally for temperature-dependent cp
        wall_material = NISTComposite.g10_nist(np.radians(54.7))  # G10 composite with NIST properties (optimal winding angle)

        # Total length includes the hemispherical caps
        total_length = body_length + 2 * radius

        # Create tank - use max allowable pressure (450 bar) for structural design
        # This ensures the netting analysis computes thickness for maximum expected pressure
        tank = CylindricalTankSphericalCaps(
            radius=radius,
            total_length=total_length,
            material=wall_material,
            operating_pressure=self.max_allowable_pressure  # 450 bar for structural design
        )

        return tank

    def create_thermal_model(self, tank: CylindricalTankSphericalCaps) -> StopsModelThermalModel:
        """Create thermal model for given tank.

        Args:
            tank: Tank object with geometry and materials

        Returns:
            StopsModelThermalModel: Configured thermal model
        """
        # Use tank properties for geometry
        volume = tank.volume
        surface_area = tank.surface_area
        diameter = tank.diameter

        # === LINER MASS CALCULATION (5mm aluminum, load-bearing assumption: no load) ===
        liner_thickness = 0.005  # 5mm thickness as specified by user
        liner = Liner.from_thickness(
            thickness=liner_thickness,
            tank=tank,
            material=Metal.aluminum()
        )
        liner_mass = liner.mass

        # === WALL MASS CALCULATION (G10 composite from netting analysis) ===
        # The tank.structural_mass comes from netting theory using:
        # 1. Maximum allowable pressure (450 bar pvent) - NOT initial pressure (400 bar)
        # 2. G10 composite material properties (failure strength, density)
        # 3. 54.7° winding angle (optimal for pressure vessels)
        # 4. Helical + Hoop thickness calculations:
        #    - Helical thickness: (meridional_stress) / (failure_stress * cos²(winding_angle))
        #    - Hoop thickness: (hoop_stress - meridional_stress*tan²(winding_angle)) / failure_stress
        #    - Total thickness = helical_thickness + hoop_thickness
        # 5. Mass = surface_area × total_thickness × G10_density
        # 6. Assumptions: Liner takes no structural load, insulation takes no structural load
        wall_mass = tank.structural_mass

        # Print mass calculation details for verification
        print(f"Mass calculations (physics-based netting analysis):")
        print(f"  Design pressure: {self.max_allowable_pressure/1e5:.0f} bar (pvent)")
        print(f"  Liner (5mm aluminum, non-load-bearing): {liner_mass:.2f} kg")
        print(f"  Wall (G10 composite, 54.7° winding, netting analysis): {wall_mass:.2f} kg")
        print(f"  Total structural mass: {liner_mass + wall_mass:.2f} kg")

        # Create thermal model
        thermal_model = StopsModelThermalModel(
            tank_volume=volume,
            inner_surface_area=surface_area,
            outer_surface_area=surface_area,  # Assume same for outer surface
            inner_diameter=diameter,
            ambient_temperature=self.ambient_temperature,
            ambient_htc=self.ambient_htc,
            liner_mass=liner_mass,
            wall_mass=wall_mass
        )

        return thermal_model

    def create_atr72_discharge_mission(self, parameters: IsochoricMissionParameters) -> IsochoricMission:
        """Create discharge mission based on ATR72 fuel flow profile.

        Args:
            parameters: Mission parameters

        Returns:
            IsochoricMission: Configured discharge mission
        """
        from src.mission.mission_sections import MissionSection, OutFlow

        # Get ATR72 mission sections
        atr72_sections = self.mission.sections

        # Convert to isochoric mission sections with discharge flows
        discharge_sections = []

        for section in atr72_sections:
            # Extract fuel flow from the section
            if section.fuel_flows and len(section.fuel_flows) > 0:
                flow = section.fuel_flows[0]  # Take first flow

                if isinstance(flow.mass_flow, list):
                    # Time-varying flow - convert to positive discharge rates
                    discharge_rates = [abs(rate) for rate in flow.mass_flow]
                    discharge_flow = OutFlow(discharge_rates, "gas")
                else:
                    # Constant flow - convert to positive discharge rate
                    discharge_rate = abs(flow.mass_flow)
                    discharge_flow = OutFlow(discharge_rate, "gas")

                # Create discharge section
                discharge_section = MissionSection(
                    duration=section.duration,
                    fuel_flows=[discharge_flow],
                    altitude=section.altitude,
                    mach_number=section.mach_number
                )
                discharge_sections.append(discharge_section)

        # Create discharge mission using IsochoricMission directly
        discharge_mission = IsochoricMission(discharge_sections, parameters, "DISCHARGE")

        return discharge_mission

    def run_discharge_simulation(self, radius: float, body_length: float) -> tuple[any, float, float]:
        """Run discharge simulation for given tank geometry.

        Args:
            radius: Tank radius [m]
            body_length: Tank body length [m]

        Returns:
            tuple: (tank_states, final_mass, final_density)
        """
        # Create tank object
        tank = self.create_tank(radius, body_length)

        # Create thermal model
        thermal_model = self.create_thermal_model(tank)

        # Tank parameters
        volume = tank.volume

        # Use target mass for sizing, not volume-based mass
        # This ensures we test whether the tank can properly discharge the required fuel
        initial_mass = self.mission.required_fuel * self.volume_margin

        # Validate that tank can physically hold this mass
        try:
            initial_density = PropsSI('D', 'P', self.initial_pressure, 'T', self.initial_temperature, 'PARAHYDROGEN')
        except Exception as e:
            raise ValueError(f"CoolProp calculation failed for P={self.initial_pressure:.0f} Pa, T={self.initial_temperature:.1f} K: {e}")

        required_volume = initial_mass / initial_density
        if required_volume > volume:
            raise ValueError(f"Tank too small: requires {required_volume:.3f} m³ but only has {volume:.3f} m³")

        # Mission parameters for isochoric discharge with tighter tolerances for stability
        mission_parameters = IsochoricMissionParameters(
            tank_volume=volume,
            initial_mass=initial_mass,
            initial_temperature=self.initial_temperature,
            ambient_temperature=self.ambient_temperature,
            atol=1e-12,  # Much tighter tolerances
            rtol=1e-9,   # Much tighter tolerances
            time_step=0.1  # Smaller time steps for stability
        )

        # Create discharge mission using ATR72 fuel flow profile
        discharge_mission = self.create_atr72_discharge_mission(mission_parameters)

        # Create analysis
        analysis = IsochoricMissionAnalysis(
            mission=discharge_mission,
            thermal_model=thermal_model
        )        # Run analysis with additional error handling
        try:
            tank_states = analysis.run_analysis()
        except Exception as e:
            error_msg = str(e)
            if "PropsSI failed ungracefully" in error_msg:
                raise ValueError(f"CoolProp thermodynamic failure for radius {radius:.4f} m - tank too large causing extreme conditions")
            elif "Invalid state" in error_msg:
                raise ValueError(f"Numerical instability for radius {radius:.4f} m - simulation reached invalid physical state")
            else:
                raise ValueError(f"Simulation failed for radius {radius:.4f} m: {error_msg}")

        # Validate results
        if not tank_states:
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: No results returned")

        if not tank_states.states:
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: Results contain no tank states")

        # Get final mass and density
        final_mass = tank_states.states[-1].fuel_mass
        final_density = final_mass / volume

        if final_mass < 0:
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: Invalid final mass {final_mass:.2f} kg")

        # Check for numerical instabilities in temperature
        final_temperature = tank_states.states[-1].temperature
        if final_temperature > 1000.0:  # Temperature too high indicates numerical issues
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: Numerical instability detected (T={final_temperature:.1f}K)")

        if final_temperature < 10.0:  # Temperature too low is also problematic
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: Unphysical temperature detected (T={final_temperature:.1f}K)")

        return tank_states, final_mass, final_density

    def iterative_sizing(self) -> tuple[float, float, any]:
        """Perform iterative tank sizing exploring phi parameter space to find optimal design.

        Returns:
            tuple: (optimized_radius, optimized_body_length, final_tank_states)
        """
        print("\n=== Starting Iterative Tank Sizing with Phi Exploration ===")

        # Get initial guess
        initial_volume, phi, target_mass = self.calculate_initial_guess()

        best_radius = None
        best_body_length = None
        best_states = None
        best_volume = float('inf')  # Track best (smallest) volume

        # Track optimization progress for plotting
        self.optimization_data = {
            'iterations': [],
            'phi_values': [],
            'fuel_consumed': [],
            'final_density': [],
            'converged': [],
            'volume': []
        }

        # Explore different phi values systematically
        phi_range = np.linspace(0.3, 0.7, 5)  # Test 5 different aspect ratios
        print(f"Exploring phi values: {phi_range}")

        iteration_count = 0

        # Test each phi value systematically
        for phi_idx, current_phi in enumerate(phi_range):
            current_volume = initial_volume  # Reset volume for each phi
            print(f"\n--- Testing Phi = {current_phi:.3f} ({phi_idx+1}/{len(phi_range)}) ---")

            # Try to converge at this phi value
            max_volume_iterations = 15
            phi_converged = False

            for vol_iter in range(max_volume_iterations):
                iteration_count += 1

                # Convert volume and phi to radius and body_length
                current_radius, current_body_length = self.volume_to_dimensions(current_volume, current_phi)
                print(f"Iteration {iteration_count}: Vol={current_volume:.4f} m³, phi={current_phi:.3f}")
                print(f"  -> radius: {current_radius:.4f} m, body length: {current_body_length:.4f} m")

                # Run simulation
                try:
                    tank_states, final_mass, final_density = self.run_discharge_simulation(current_radius, current_body_length)
                except ValueError as e:
                    if "Tank too small" in str(e):
                        print(f"Tank too small, increasing volume")
                        current_volume *= 1.2  # Increase volume by 20%
                        continue
                    else:
                        print(f"Simulation failed: {e}")
                        current_volume *= 1.2  # Increase volume by 20%
                        continue

                if tank_states is None:
                    print(f"Simulation failed, increasing tank size")
                    current_volume *= 1.2  # Increase volume by 20%
                    continue

                print(f"  Final mass: {final_mass:.2f} kg, Final density: {final_density:.2f} kg/m³")

                # Record optimization data
                fuel_consumed = self.mission.required_fuel * self.volume_margin - final_mass
                self.optimization_data['iterations'].append(iteration_count)
                self.optimization_data['phi_values'].append(current_phi)
                self.optimization_data['fuel_consumed'].append(fuel_consumed)
                self.optimization_data['final_density'].append(final_density)
                self.optimization_data['volume'].append(current_radius**2 * np.pi * current_body_length)

                # Check convergence based on density target with tolerance
                density_tolerance = 0.1  # kg/m³ tolerance for convergence
                density_diff = final_density - self.target_density

                if final_mass > 1.0 and abs(density_diff) <= density_tolerance:
                    print(f"✅ Converged at phi={current_phi:.3f}! Density {final_density:.2f} kg/m³ (target: {self.target_density:.2f})")
                    self.optimization_data['converged'].append(True)
                    phi_converged = True

                    # Check if this is our best design (smallest volume)
                    current_tank_volume = current_radius**2 * np.pi * current_body_length
                    if current_tank_volume < best_volume:
                        print(f"🎯 New best design! Volume: {current_tank_volume:.3f} m³ (previous best: {best_volume:.3f} m³)")
                        best_volume = current_tank_volume
                        best_radius = current_radius
                        best_body_length = current_body_length
                        best_states = tank_states

                    break
                elif final_mass <= 1.0:
                    # Tank too large - simulation consumed all fuel
                    print(f"  Tank too large (final mass {final_mass:.2f} kg), reducing volume")
                    self.optimization_data['converged'].append(False)
                    current_volume *= 0.8  # Reduce volume by 20%
                elif density_diff > density_tolerance:
                    # Density too high, tank too small
                    print(f"  Density too high ({final_density:.2f} > {self.target_density:.2f}), increasing volume")
                    self.optimization_data['converged'].append(False)
                    current_volume *= 1.3  # Increase volume more aggressively
                else:
                    # Density close enough
                    print(f"✅ Close convergence at phi={current_phi:.3f}! Density {final_density:.2f} kg/m³")
                    self.optimization_data['converged'].append(True)
                    phi_converged = True

                    current_tank_volume = current_radius**2 * np.pi * current_body_length
                    if current_tank_volume < best_volume:
                        print(f"🎯 New best design! Volume: {current_tank_volume:.3f} m³")
                        best_volume = current_tank_volume
                        best_radius = current_radius
                        best_body_length = current_body_length
                        best_states = tank_states
                    break

            if not phi_converged:
                print(f"❌ Phi {current_phi:.3f} did not converge within {max_volume_iterations} iterations")

        if best_radius is None:
            print("❌ No converged solution found across all phi values - using last attempt")
            best_radius = current_radius
            best_body_length = current_body_length
            best_states = tank_states
        else:
            print(f"\n🏆 OPTIMAL DESIGN FOUND:")
            print(f"   Phi: {best_radius/best_body_length:.3f}")
            print(f"   Radius: {best_radius:.3f} m")
            print(f"   Body Length: {best_body_length:.3f} m")
            print(f"   Volume: {best_volume:.3f} m³")

        return best_radius, best_body_length, best_states

    def print_results(self, radius: float, body_length: float, tank_states: any):
        """Print analysis results.

        Args:
            radius: Optimized tank radius [m]
            body_length: Optimized tank body length [m]
            tank_states: Final tank states
        """
        print("\n" + "="*60)
        print("BASELINE CCH2 ANALYSIS RESULTS")
        print("="*60)

        # Create tank for geometry calculations
        tank = self.create_tank(radius, body_length)

        # Tank geometry
        volume = tank.volume
        surface_area = tank.surface_area
        diameter = tank.diameter
        total_length = tank.total_length

        print(f"Tank Geometry:")
        print(f"  Radius: {radius:.4f} m")
        print(f"  Diameter: {diameter:.4f} m")
        print(f"  Body Length: {body_length:.4f} m")
        print(f"  Total Length: {total_length:.4f} m")
        print(f"  Volume: {volume:.4f} m³")
        print(f"  Surface Area: {surface_area:.2f} m²")
        print(f"  L/D Ratio: {body_length/diameter:.2f}")

        # Tank masses
        total_structural_mass = tank.structural_mass

        print(f"\nTank Masses:")
        print(f"  Total Structural: {total_structural_mass:.2f} kg")
        print(f"  (Carbon Fiber Composite)")

        # Mission performance
        if tank_states and tank_states.states:
            initial_mass = tank_states.states[0].fuel_mass
            final_mass = tank_states.states[-1].fuel_mass
            consumed_mass = initial_mass - final_mass
            mission_duration = tank_states.times[-1] if tank_states.times else 0.0

            print(f"\nMission Performance:")
            print(f"  Initial H2 Mass: {initial_mass:.2f} kg")
            print(f"  Final H2 Mass: {final_mass:.2f} kg")
            print(f"  Consumed H2 Mass: {consumed_mass:.2f} kg")
            print(f"  Mission Duration: {mission_duration/3600:.2f} hours")
            print(f"  Required H2 Mass: {self.mission.required_fuel:.2f} kg")

            # Performance metrics
            utilization = consumed_mass / initial_mass * 100
            gravimetric_efficiency = consumed_mass / (consumed_mass + total_structural_mass) * 100

            print(f"\nPerformance Metrics:")
            print(f"  H2 Utilization: {utilization:.1f}%")
            print(f"  Gravimetric Efficiency: {gravimetric_efficiency:.1f}%")

    def plot_fuel_flow_profile(self, save_path=None):
        """
        Plot the fuel flow profile driving the discharge analysis.
        Shows actual ATR72 mission sections with varying flow rates and durations.
        Includes linear interpolation for ramping sections matching MissionSectionAnalysis logic.
        """
        if not hasattr(self, 'mission') or not self.mission:
            raise ValueError("Cannot plot fuel flow profile: No mission data available")

        configure_plot_style()

        # Get actual ATR72 mission data
        atr72_mission = self.mission  # self.mission is already Mission.atr72()

        # Extract mission profile data from the actual mission sections
        mission_times = []
        mission_flows = []
        analysis_times = []
        analysis_flows = []

        current_time = 0.0
        analysis_time = 0.0
        section_names = ['Taxi-out', 'Takeoff', 'Climb', 'Cruise', 'Initial Descent', 'Approach', 'Go-around', 'Climb-2', 'Cruise-2', 'Final Descent', 'Taxi-in']

        for i, section in enumerate(atr72_mission.sections):
            duration = section.duration  # already in seconds
            section_start = current_time
            section_end = current_time + duration

            # Extract fuel flow from section
            fuel_flow = section.fuel_flows[0].mass_flow  # Get the OutFlow mass_flow

            if isinstance(fuel_flow, list):
                # Linear ramp section - extract from list created by OutFlow
                # The OutFlow constructor creates [-throttle * flow for flow in fuel_flow]
                # So we need to reverse this to get original flows
                start_flow = abs(fuel_flow[0])
                end_flow = abs(fuel_flow[1])

                # Mission profile (step representation)
                mission_times.extend([section_start, section_end])
                mission_flows.extend([start_flow, end_flow])

                # Analysis resolution with interpolation (matching MissionSectionAnalysis logic)
                timestep = 1.0  # Use 1 second timestep for analysis resolution
                section_steps = int(duration / timestep)
                for step in range(section_steps):
                    # Linear interpolation within section (same as interpolate_mass_flows)
                    interpolated_flow = start_flow + (end_flow - start_flow) * step / section_steps
                    analysis_times.append(analysis_time)
                    analysis_flows.append(interpolated_flow)
                    analysis_time += timestep

            else:
                # Constant flow section
                flow_rate = abs(fuel_flow)

                # Mission profile
                mission_times.extend([section_start, section_end])
                mission_flows.extend([flow_rate, flow_rate])

                # Analysis resolution
                timestep = 1.0
                section_steps = int(duration / timestep)
                for step in range(section_steps):
                    analysis_times.append(analysis_time)
                    analysis_flows.append(flow_rate)
                    analysis_time += timestep

            current_time += duration

        # Convert to hours for plotting
        mission_times_hr = np.array(mission_times) / 3600.0
        analysis_times_hr = np.array(analysis_times) / 3600.0
        total_time_hours = current_time / 3600.0

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Top panel: ATR72 Mission Profile
        ax1.plot(mission_times_hr, mission_flows, linewidth=3, color='#003f7f', label='ATR72 Mission Profile')
        ax1.fill_between(mission_times_hr, 0, mission_flows, alpha=0.3, color='#003f7f')

        # Add section labels
        cumulative_time = 0
        for i, (section, name) in enumerate(zip(atr72_mission.sections, section_names)):
            section_center = (cumulative_time + section.duration / 2) / 3600.0
            if i % 2 == 0:  # Alternate label heights to avoid overlap
                ax1.annotate(name, xy=(section_center, max(mission_flows) * 0.9),
                           ha='center', va='bottom', rotation=45, fontsize=10, color='darkred')
            else:
                ax1.annotate(name, xy=(section_center, max(mission_flows) * 0.7),
                           ha='center', va='bottom', rotation=45, fontsize=10, color='darkred')
            cumulative_time += section.duration

        ax1.set_xlabel('Time [h]')
        ax1.set_ylabel('Discharge Rate [kg/s]')
        ax1.set_title('ATR72 Mission Fuel Flow Profile (11 Sections)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, total_time_hours)

        # Bottom panel: Analysis Resolution with Linear Interpolation
        ax2.plot(analysis_times_hr, analysis_flows, linewidth=1, color='#c41e3a',
                label='Analysis Resolution (Δt = 1s)')
        ax2.set_xlabel('Time [h]')
        ax2.set_ylabel('Discharge Rate [kg/s]')
        ax2.set_title('Interpolated Flow Profile (Analysis Timesteps)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, total_time_hours)

        # Add summary statistics
        total_fuel = atr72_mission.required_fuel

        fig.suptitle(f'ATR72 Mission Profile Analysis\nTotal Duration: {total_time_hours:.2f}h, Total Fuel: {total_fuel:.1f}kg',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fuel flow plot saved to: {save_path}")
        else:
            plt.show()

    def create_plots(self, tank_states: any, radius: float, body_length: float):
        """Create analysis plots.

        Args:
            tank_states: Analysis results
            radius: Tank radius [m]
            body_length: Tank body length [m]
        """
        if not tank_states or not tank_states.states:
            print("No data available for plotting")
            return

        # Configure plotting style
        configure_plot_style()
        plotter = SeabornPlotter(font="Cambria", palette="deep")

        # Extract data for plotting
        times = np.array(tank_states.times) / 3600  # Convert to hours

        # Create tank for volume calculation
        tank = self.create_tank(radius, body_length)

        masses = [state.fuel_mass for state in tank_states.states]
        temperatures = [state.temperature for state in tank_states.states]
        pressures = [state.pressure / 1e5 for state in tank_states.states]  # Convert to bar
        densities = [state.fuel_mass / tank.volume for state in tank_states.states]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Baseline CCH2 Discharge Analysis', fontsize=16, fontweight='bold')

        # Mass vs time
        axes[0, 0].plot(times, masses, color=BORDEAUX, linewidth=2)
        axes[0, 0].set_xlabel('Time [h]')
        axes[0, 0].set_ylabel('Mass [kg]')
        axes[0, 0].set_title('H₂ Mass Evolution')
        axes[0, 0].grid(True, alpha=0.3)

        # Temperature vs time
        axes[0, 1].plot(times, temperatures, color=KONINGSBLAUW, linewidth=2)
        axes[0, 1].set_xlabel('Time [h]')
        axes[0, 1].set_ylabel('Temperature [K]')
        axes[0, 1].set_title('Temperature Evolution')
        axes[0, 1].grid(True, alpha=0.3)

        # Pressure vs time
        axes[1, 0].plot(times, pressures, color=BOSGROEN, linewidth=2)
        axes[1, 0].set_xlabel('Time [h]')
        axes[1, 0].set_ylabel('Pressure [bar]')
        axes[1, 0].set_title('Pressure Evolution')
        axes[1, 0].grid(True, alpha=0.3)

        # Density vs time
        axes[1, 1].plot(times, densities, color=ORANJE, linewidth=2)
        axes[1, 1].axhline(y=self.target_density, color='red', linestyle='--',
                          label=f'Target: {self.target_density} kg/m³')
        axes[1, 1].set_xlabel('Time [h]')
        axes[1, 1].set_ylabel('Density [kg/m³]')
        axes[1, 1].set_title('Density Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Temperature-Density plot (swapped axes: T on x-axis, density on y-axis)
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.plot(temperatures, densities, color=DONKERGRIJS, linewidth=2, marker='o', markersize=3, label='Discharge Process')

        # Add continuous hydrogen saturation line using sb_plotting method
        try:
            fluid = 'PARAHYDROGEN'
            T_triple = PropsSI('Ttriple', fluid)
            T_crit = PropsSI('Tcrit', fluid)

            # Create temperature range from triple point to critical point
            temps = np.linspace(T_triple, T_crit, 100)

            # Create complete continuous saturation curve
            sat_temps = []
            sat_densities = []

            # First add the saturated liquid branch (low to high temperature)
            for temp in temps:
                try:
                    # Get saturated liquid density (Q=0)
                    density = PropsSI('D', 'T', temp, 'Q', 0, fluid)
                    sat_temps.append(temp)
                    sat_densities.append(density)
                except Exception:
                    pass

            # Then add the saturated vapor branch (high to low temperature)
            for temp in reversed(temps):
                try:
                    # Get saturated vapor density (Q=1)
                    density = PropsSI('D', 'T', temp, 'Q', 1, fluid)
                    sat_temps.append(temp)
                    sat_densities.append(density)
                except Exception:
                    pass

            # Plot the complete continuous saturation dome
            if len(sat_temps) > 0:
                ax2.plot(sat_temps, sat_densities, '--', color=ORANJE,
                        linewidth=1.5, label="Saturation Line", alpha=0.8)

        except Exception as e:
            print(f"Warning: Could not add saturation line: {e}")

        ax2.set_xlabel('Temperature [K]')
        ax2.set_ylabel('Density [kg/m³]')
        ax2.set_title('Density vs Temperature - Discharge Process')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add annotations for start and end points (swapped coordinates)
        if len(densities) > 0 and len(temperatures) > 0:
            ax2.annotate('Start', xy=(temperatures[0], densities[0]),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'))
            ax2.annotate('End', xy=(temperatures[-1], densities[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()
        plt.show()

    def plot_optimization_progress(self):
        """Plot fuel mass consumed vs phi to show optimization progress."""
        if not hasattr(self, 'optimization_data') or not self.optimization_data['iterations']:
            print("No optimization data available for plotting")
            return

        # Create optimization plot
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))

        # Plot all iterations
        iterations = self.optimization_data['iterations']
        fuel_consumed = self.optimization_data['fuel_consumed']
        phi_values = self.optimization_data['phi_values']
        converged = self.optimization_data['converged']

        # Plot non-converged points
        non_converged_fuel = [f for i, f in enumerate(fuel_consumed) if not converged[i]]
        non_converged_phi = [p for i, p in enumerate(phi_values) if not converged[i]]

        if non_converged_fuel:
            ax3.scatter(non_converged_phi, non_converged_fuel,
                       color='red', alpha=0.6, s=50, label='Non-converged', marker='x')

        # Plot converged points
        converged_fuel = [f for i, f in enumerate(fuel_consumed) if converged[i]]
        converged_phi = [p for i, p in enumerate(phi_values) if converged[i]]

        if converged_fuel:
            ax3.scatter(converged_phi, converged_fuel,
                       color='green', alpha=0.8, s=80, label='Converged', marker='o')

        # Connect points with iteration order
        ax3.plot(phi_values, fuel_consumed, 'k--', alpha=0.3, linewidth=1, label='Iteration Path')

        # Add iteration numbers as annotations
        for i, (phi, fuel, iteration) in enumerate(zip(phi_values, fuel_consumed, iterations)):
            ax3.annotate(f'{iteration}', (phi, fuel),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

        # Add target fuel requirement line
        target_fuel = self.mission.required_fuel
        ax3.axhline(y=target_fuel, color=ORANJE, linestyle='--',
                   linewidth=2, label=f'Required Fuel: {target_fuel:.1f} kg')

        ax3.set_xlabel('Phi (radius/body_length) [-]')
        ax3.set_ylabel('Fuel Consumed [kg]')
        ax3.set_title('Optimization Progress: Fuel Mass vs Phi Parameter')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def plot_phi_optimization_results(self):
        """Plot phi optimization results using SeabornPlotter."""
        if not hasattr(self, 'optimization_data') or not self.optimization_data['iterations']:
            print("No optimization data available for phi optimization plotting")
            return

        # Extract unique phi values and their corresponding volumes
        phi_values = []
        volumes = []
        converged = []

        seen_phi = set()
        for i, phi in enumerate(self.optimization_data['phi_values']):
            if phi not in seen_phi and self.optimization_data['converged'][i]:
                seen_phi.add(phi)
                phi_values.append(phi)
                volumes.append(self.optimization_data['volume'][i])
                converged.append(self.optimization_data['converged'][i])

        if len(phi_values) < 2:
            print("Insufficient phi optimization data for plotting")
            return

        # Create plotter and generate plot
        from plotting.sb_plotting import SeabornPlotter
        plotter = SeabornPlotter()

        # Generate the plot with save path
        save_path = Path(__file__).parent / 'phi_optimization_results.png'
        fig = plotter.plot_phi_optimization_results(phi_values, volumes, converged, str(save_path))

        plt.show()

    def run_analysis(self):
        """Run complete baseline CCH2 analysis."""
        print("Starting Baseline CCH2 Analysis")
        print("="*50)

        start_time = time.time()

        # Perform iterative sizing
        optimized_radius, optimized_body_length, tank_states = self.iterative_sizing()

        # Print results
        self.print_results(optimized_radius, optimized_body_length, tank_states)

        # Create plots
        self.create_plots(tank_states, optimized_radius, optimized_body_length)

        # Plot optimization progress
        self.plot_optimization_progress()

        # Plot phi optimization results
        self.plot_phi_optimization_results()

        # Plot fuel flow profile
        self.plot_fuel_flow_profile()

        analysis_time = time.time() - start_time
        print(f"\nAnalysis completed in {analysis_time:.2f} seconds")


def main():
    """Main execution function."""
    analysis = BaselineCCH2Analysis()
    analysis.run_analysis()


if __name__ == "__main__":
    main()
