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
        self.initial_pressure = 400e5    # Pa (400 bar) - same as verification
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
        self.initial_pressure = 40e6  # Pa (400 bar)
        self.initial_temperature = 70.0  # K
        self.initial_mass_fraction = 0.0  # Pure gas behavior

        # Target conditions
        self.target_density = 5.8  # kg/m³ (stopping condition)
        self.mass_tolerance = 0.5  # kg (convergence tolerance)

        # Thermal model parameters
        self.ambient_temperature = 298.15  # K
        self.ambient_htc = 0.025  # W/m²K (vacuum insulation equivalent)

        # Tank material parameters
        self.liner_density = 2700.0  # kg/m³ (aluminum)
        self.wall_density = 1800.0  # kg/m³ (G10 composite)
        self.liner_thickness = 0.002  # m (2mm aluminum liner)
        self.wall_thickness = 0.05  # m (50mm G10 wall)

        # Iteration parameters
        self.max_iterations = 20
        self.radius_reduction_factor = 0.95  # Reduce radius by 5% each iteration

    def calculate_initial_guess(self) -> tuple[float, float]:
        """Calculate initial tank size guess based on mission fuel requirements.

        Returns:
            tuple: (radius [m], mass [kg])
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

        # Calculate required volume and radius (assuming spherical tank)
        required_volume = required_fuel_mass / density
        radius = (3.0 * required_volume / (4.0 * np.pi)) ** (1.0 / 3.0)

        print(f"Initial volume estimate: {required_volume:.4f} m³")
        print(f"Initial radius estimate: {radius:.4f} m")

        return radius, required_fuel_mass

    def calculate_tank_masses(self, radius: float) -> tuple[float, float]:
        """Calculate liner and wall masses based on tank geometry.

        Args:
            radius: Tank radius [m]

        Returns:
            tuple: (liner_mass [kg], wall_mass [kg])
        """
        # Surface area for spherical tank
        surface_area = 4.0 * np.pi * radius**2

        # Liner mass (thin shell)
        liner_volume = surface_area * self.liner_thickness
        liner_mass = liner_volume * self.liner_density

        # Wall mass (thick shell)
        wall_volume = surface_area * self.wall_thickness
        wall_mass = wall_volume * self.wall_density

        return liner_mass, wall_mass

    def create_thermal_model(self, radius: float) -> StopsModelThermalModel:
        """Create thermal model for given tank geometry.

        Args:
            radius: Tank radius [m]

        Returns:
            StopsModelThermalModel: Configured thermal model
        """
        # Tank geometry
        volume = (4.0 / 3.0) * np.pi * radius**3
        surface_area = 4.0 * np.pi * radius**2
        diameter = 2.0 * radius

        # Calculate masses
        liner_mass, wall_mass = self.calculate_tank_masses(radius)

        # Create thermal model
        thermal_model = StopsModelThermalModel(
            tank_volume=volume,
            inner_surface_area=surface_area,
            outer_surface_area=surface_area,  # Same for spherical tank
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

    def run_discharge_simulation(self, radius: float) -> tuple[any, float]:
        """Run discharge simulation for given tank radius.

        Args:
            radius: Tank radius [m]

        Returns:
            tuple: (tank_states, final_mass)
        """
        # Create thermal model
        thermal_model = self.create_thermal_model(radius)

        # Tank parameters
        volume = (4.0 / 3.0) * np.pi * radius**3

        # Calculate initial mass based on tank volume and initial conditions
        try:
            initial_density = PropsSI('D', 'P', self.initial_pressure, 'T', self.initial_temperature, 'PARAHYDROGEN')
        except Exception as e:
            raise ValueError(f"CoolProp calculation failed for P={self.initial_pressure:.0f} Pa, T={self.initial_temperature:.1f} K: {e}")

        initial_mass = initial_density * volume

        # Mission parameters for isochoric discharge
        mission_parameters = IsochoricMissionParameters(
            tank_volume=volume,
            initial_mass=initial_mass,
            initial_temperature=self.initial_temperature,
            ambient_temperature=self.ambient_temperature,
            atol=1e-9,
            rtol=1e-6
        )

        # Create discharge mission using ATR72 fuel flow profile
        discharge_mission = self.create_atr72_discharge_mission(mission_parameters)

        # Create analysis
        analysis = IsochoricMissionAnalysis(
            mission=discharge_mission,
            thermal_model=thermal_model
        )        # Run analysis
        tank_states = analysis.run_analysis()

        # Validate results
        if not tank_states:
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: No results returned")

        if not tank_states.states:
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: Results contain no tank states")

        # Get final mass
        final_mass = tank_states.states[-1].fuel_mass

        if final_mass < 0:
            raise ValueError(f"Analysis failed for radius {radius:.4f} m: Invalid final mass {final_mass:.2f} kg")

        return tank_states, final_mass

    def iterative_sizing(self) -> tuple[float, any]:
        """Perform iterative tank sizing to minimize volume.

        Returns:
            tuple: (optimized_radius, final_tank_states)
        """
        print("\n=== Starting Iterative Tank Sizing ===")

        # Get initial guess
        initial_radius, target_mass = self.calculate_initial_guess()
        current_radius = initial_radius

        best_radius = None
        best_states = None

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Testing radius: {current_radius:.4f} m")

            # Run simulation
            tank_states, final_mass = self.run_discharge_simulation(current_radius)

            if tank_states is None:
                print(f"Simulation failed for radius {current_radius:.4f} m, increasing radius")
                current_radius *= 1.05  # Increase radius by 5%
                if current_radius > self.max_radius * 2:  # Safety check
                    raise ValueError(f"Iterative sizing failed: Radius exceeded reasonable bounds ({current_radius:.4f} m)")
                continue

            print(f"Final mass: {final_mass:.2f} kg")
            print(f"Mass residual: {final_mass:.2f} kg")

            # Check convergence
            if abs(final_mass) <= self.mass_tolerance:
                print(f"Converged! Final mass within tolerance ({self.mass_tolerance} kg)")
                best_radius = current_radius
                best_states = tank_states
                break
            elif final_mass > self.mass_tolerance:
                # Too much mass remaining, tank too small
                print("Tank too small, increasing radius")
                current_radius *= 1.05  # Increase radius by 5%
            else:
                # Mass below tolerance (good), try smaller tank
                print("Mass below tolerance, trying smaller tank")
                best_radius = current_radius
                best_states = tank_states
                current_radius *= self.radius_reduction_factor

        if best_radius is None:
            raise ValueError(f"Iterative sizing failed: No convergence achieved after {iteration} iterations. Last radius tried: {current_radius:.4f} m")

        if best_states is None:
            raise ValueError(f"Iterative sizing failed: No valid tank states found for optimal radius {best_radius:.4f} m")

        return best_radius, best_states

    def print_results(self, radius: float, tank_states: any):
        """Print analysis results.

        Args:
            radius: Optimized tank radius [m]
            tank_states: Final tank states
        """
        print("\n" + "="*60)
        print("BASELINE CCH2 ANALYSIS RESULTS")
        print("="*60)

        # Tank geometry
        volume = (4.0 / 3.0) * np.pi * radius**3
        surface_area = 4.0 * np.pi * radius**2
        diameter = 2.0 * radius

        print(f"Tank Geometry:")
        print(f"  Radius: {radius:.4f} m")
        print(f"  Diameter: {diameter:.4f} m")
        print(f"  Volume: {volume:.4f} m³")
        print(f"  Surface Area: {surface_area:.2f} m²")

        # Tank masses
        liner_mass, wall_mass = self.calculate_tank_masses(radius)
        total_structural_mass = liner_mass + wall_mass

        print(f"\nTank Masses:")
        print(f"  Aluminum Liner: {liner_mass:.2f} kg")
        print(f"  G10 Wall: {wall_mass:.2f} kg")
        print(f"  Total Structural: {total_structural_mass:.2f} kg")

        # Mission performance
        if tank_states and tank_states.states:
            initial_mass = tank_states.states[0].mass
            final_mass = tank_states.states[-1].mass
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
            fuel_flow = abs(fuel_flow)  # Make positive for plotting

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
                           ha='center', va='bottom', rotation=45, fontsize=8)
            else:
                ax1.annotate(name, xy=(section_center, max(mission_flows) * 0.7),
                           ha='center', va='bottom', rotation=45, fontsize=8)
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

    def create_plots(self, tank_states: any, radius: float):
        """Create analysis plots.

        Args:
            tank_states: Analysis results
            radius: Tank radius [m]
        """
        if not tank_states or not tank_states.states:
            print("No data available for plotting")
            return

        # Configure plotting style
        configure_plot_style()
        plotter = SeabornPlotter(font="Cambria", palette="deep")

        # Extract data for plotting
        times = np.array(tank_states.times) / 3600  # Convert to hours

        masses = [state.fuel_mass for state in tank_states.states]
        temperatures = [state.temperature for state in tank_states.states]
        pressures = [state.pressure / 1e5 for state in tank_states.states]  # Convert to bar
        densities = [state.fuel_mass / ((4.0/3.0) * np.pi * radius**3) for state in tank_states.states]

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

        # Temperature-Density plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.plot(densities, temperatures, color=DONKERGRIJS, linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Density [kg/m³]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Temperature vs Density - Discharge Process')
        ax2.grid(True, alpha=0.3)

        # Add annotations for start and end points
        if len(densities) > 0 and len(temperatures) > 0:
            ax2.annotate('Start', xy=(densities[0], temperatures[0]),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'))
            ax2.annotate('End', xy=(densities[-1], temperatures[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        """Run complete baseline CCH2 analysis."""
        print("Starting Baseline CCH2 Analysis")
        print("="*50)

        start_time = time.time()

        # Perform iterative sizing
        optimized_radius, tank_states = self.iterative_sizing()

        # Print results
        self.print_results(optimized_radius, tank_states)

        # Create plots
        self.create_plots(tank_states, optimized_radius)

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
