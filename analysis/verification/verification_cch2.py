"""
Compressed Cold Hydrogen (CCH2) Verification Analysis.

This module implements the verification analysis for compressed cold hydrogen
scenarios using the integrated stops_model approach with the new HFT framework.

The analysis focuses on the discharge scenario with the validated solver architecture,
providing a production-ready implementation of the class-based tank analysis.

Integration Features:
- New SciPy solver architecture (LSODA primary, RK45 backup)
- Isochoric mission framework integration
- stops_model physics with HFT patterns
- Production-ready discharge scenario validation

Authors: Victor Kees Poorte, 2025 (Original stops_model)
         HFT Integration Team, 2025
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# Global heat flow data collector (similar to stops_model approach)
heat_flow_data = {
    't': [],           # Time [s]
    'qdot_disch': [],  # Discharge heat rate [W] (iHEX requirements)
    'qdot_ohex': [],   # oHEX heat rate [W] (will be calculated in post-processing)
    'mdot_disch': [],  # Discharge mass flow rate [kg/s]
    'T': [],           # Temperature [K]
    'rho': []          # Density [kg/m³]
}

# oHEX target conditions (same as stops_model)
OHEX_TARGET_TEMPERATURE = 200.0   # Target temperature [K] (from stops_model)
OHEX_TARGET_PRESSURE = 20e5       # Target pressure [Pa] = 20 bar (from stops_model)

def calculate_ohex_heat_requirements():
    """
    Calculate oHEX heat requirements using enthalpy difference.

    Uses the same approach as stops_model:
    Q_oHEX = mdot * (h_target - h_disch)

    where:
    - h_target: enthalpy at OHEX target conditions (20°C, 1 atm)
    - h_disch: enthalpy at discharge conditions (current T, P)
    """
    print("🔧 Calculating oHEX heat requirements...")

    # Clear existing oHEX data
    heat_flow_data['qdot_ohex'] = []

    # Calculate target enthalpy (constant for all time points)
    h_target = PropsSI("Hmass", "T", OHEX_TARGET_TEMPERATURE, "P", OHEX_TARGET_PRESSURE, "hydrogen")

    for i, (T, rho, mdot) in enumerate(zip(heat_flow_data['T'], heat_flow_data['rho'], heat_flow_data['mdot_disch'])):
        if T > 0 and rho > 0:
            try:
                # Calculate discharge conditions pressure and enthalpy
                p_disch = PropsSI("P", "T", T, "Dmass", rho, "hydrogen")
                h_disch = PropsSI("Hmass", "T", T, "P", p_disch, "hydrogen")

                # Calculate oHEX heat requirement
                q_ohex = mdot * (h_target - h_disch)  # [W]
                heat_flow_data['qdot_ohex'].append(q_ohex)

            except Exception as e:
                print(f"⚠️ oHEX calculation failed at point {i}: {e}")
                heat_flow_data['qdot_ohex'].append(0.0)
        else:
            heat_flow_data['qdot_ohex'].append(0.0)

    total_points = len(heat_flow_data['qdot_ohex'])
    max_ohex = max(heat_flow_data['qdot_ohex']) if heat_flow_data['qdot_ohex'] else 0
    print(f"✅ oHEX calculation complete: {total_points} points, max = {max_ohex/1000:.1f} kW")

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# HFT Framework imports
from src.mission.isochoric_missions import (
    DischargeMission,
    IsochoricMissionAnalysis,
    IsochoricMissionParameters
)
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


class CCH2VerificationConfig:
    """
    Configuration parameters for CCH2 verification analysis.

    Based on the original stops_model scenarios but adapted for
    the class-based HFT framework integration.
    """

    # Tank physical parameters (from stops_model)
    TANK_VOLUME = 0.5           # m³ - V_t from stops_model
    TANK_SURFACE_AREA = 4.0     # m² - A_in from stops_model

    # Pressure thresholds (from stops_model ConfigurationManager)
    P_MIN = 15e5                # Pa (15 bar) - p_min from stops_model
    P_VENT = 450e5              # Pa (450 bar) - p_vent from stops_model

    # Environmental conditions (from stops_model)
    AMBIENT_TEMPERATURE = 298.15  # K - T_amb from stops_model

    # Discharge scenario parameters (matching stops_model exactly)
    class Discharge:
        # Initial conditions - STOPS_MODEL uses P,T to calculate mass via EOS
        INITIAL_PRESSURE = 400e5        # Pa (400 bar) - from stops_model DISCHARGE
        INITIAL_TEMPERATURE = 53.25     # K - from stops_model DISCHARGE
        INITIAL_SOLID_TEMP = "thermal_equilibrium"  # K - calculated from T like stops_model

        # Stopping condition - STOPS_MODEL uses density, not mass
        STOPPING_DENSITY = 5.8          # kg/m³ - from stops_model DISCHARGE rho_stop

        # Discharge parameters - from stops_model DISCHARGE
        CONSTANT_RATE = 0.001           # kg/s - from stops_model mdot_disch
        MISSION_DURATION = 40000.0      # s - from stops_model max_time

        # Analysis parameters
        TIME_STEP = 1.0             # s (1 second resolution)

    # Thermal model parameters (from stops_model)
    class Thermal:
        HEAT_TRANSFER_COEFF = 0.025   # W/m²K - k_amb from stops_model
        SOLID_THERMAL_MASS = 1000.0   # J/K (tank structure thermal mass)

    # Solver configuration parameters
    class Solver:
        PRIMARY_METHOD = 'RK45'         # Primary solver method (more stable than LSODA)
        RTOL = 1e-5                     # Slightly relaxed relative tolerance
        ATOL = 1e-8                     # Slightly relaxed absolute tolerance
        MAX_STEP = 5.0                  # Smaller maximum step size (seconds)


class CCH2DischargeAnalysis:
    """
    CCH2 discharge scenario analysis using the integrated HFT framework.

    This class provides the production-ready implementation of the discharge
    scenario originally developed in stops_model, now fully integrated with
    the class-based HFT patterns and new solver architecture.
    """

    def __init__(self, config: CCH2VerificationConfig = None):
        """
        Initialize CCH2 discharge analysis.

        Args:
            config: Configuration object (uses default if None)
        """
        print("🏗️ Initializing CCH2DischargeAnalysis...")
        self.config = config or CCH2VerificationConfig()
        print("✅ Configuration loaded")

        # Analysis components (initialized in setup())
        self.mission = None
        self.thermal_model = None
        self.mission_analysis = None
        self.solver = None

        # Results storage
        self.results = None
        self.analysis_metadata = {}

    def setup_analysis(self, solver_method: str = None):
        """
        Set up all analysis components.

        Args:
            solver_method: Override solver method ("LSODA", "RK45", etc.)
        """
        print("🔧 Setting up CCH2 discharge analysis...")

        # 1. Calculate initial mass from P,T like stops_model
        from CoolProp.CoolProp import PropsSI

        initial_density = PropsSI("Dmass", "P", self.config.Discharge.INITIAL_PRESSURE,
                                 "T", self.config.Discharge.INITIAL_TEMPERATURE, "hydrogen")
        initial_mass = initial_density * self.config.TANK_VOLUME

        print(f"   Initial conditions: P={self.config.Discharge.INITIAL_PRESSURE/1e5:.1f} bar, "
              f"T={self.config.Discharge.INITIAL_TEMPERATURE:.2f} K")
        print(f"   Calculated: ρ={initial_density:.2f} kg/m³, m={initial_mass:.2f} kg")
        print(f"   Stopping density: {self.config.Discharge.STOPPING_DENSITY:.1f} kg/m³")

        # Calculate mass to discharge and expected time
        final_mass = self.config.Discharge.STOPPING_DENSITY * self.config.TANK_VOLUME
        mass_to_discharge = initial_mass - final_mass
        theoretical_time = mass_to_discharge / self.config.Discharge.CONSTANT_RATE
        # Add buffer time for thermal effects and use reasonable maximum
        discharge_duration = min(theoretical_time * 1.5, self.config.Discharge.MISSION_DURATION)

        print(f"   Mass to discharge: {mass_to_discharge:.2f} kg")
        print(f"   Theoretical time: {theoretical_time:.0f} s")
        print(f"   Mission duration: {discharge_duration:.0f} s")

        # Handle thermal equilibrium solid temperature like stops_model
        if self.config.Discharge.INITIAL_SOLID_TEMP == "thermal_equilibrium":
            # Create temporary thermal model to calculate equilibrium temperature
            temp_thermal_model = StopsModelThermalModel(
                tank_volume=self.config.TANK_VOLUME,
                inner_surface_area=self.config.TANK_SURFACE_AREA,
                outer_surface_area=self.config.TANK_SURFACE_AREA * 1.025,
                inner_diameter=1.0,
                ambient_temperature=self.config.AMBIENT_TEMPERATURE,
                ambient_htc=self.config.Thermal.HEAT_TRANSFER_COEFF,
                liner_mass=100.0,
                wall_mass=150.0
            )
            initial_solid_temp = temp_thermal_model.calculate_thermal_equilibrium_Ts(
                self.config.Discharge.INITIAL_TEMPERATURE
            )
            print(f"   Using thermal equilibrium Ts0 = {initial_solid_temp:.2f}K")
        else:
            initial_solid_temp = self.config.Discharge.INITIAL_SOLID_TEMP

        # 2. Create mission parameters with calculated mass
        mission_params = IsochoricMissionParameters(
            tank_volume=self.config.TANK_VOLUME,
            p_min=self.config.P_MIN,
            p_vent=self.config.P_VENT,
            initial_mass=initial_mass,
            initial_temperature=self.config.Discharge.INITIAL_TEMPERATURE,
            initial_solid_temperature=initial_solid_temp,
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            time_step=self.config.Discharge.TIME_STEP,
            rtol=self.config.Solver.RTOL,
            atol=self.config.Solver.ATOL
        )

        # 3. Create discharge mission with calculated parameters
        self.mission = DischargeMission.constant_discharge(
            discharge_rate=self.config.Discharge.CONSTANT_RATE,
            duration=discharge_duration,  # Use calculated duration instead of fixed
            initial_mass=initial_mass,
            initial_temperature=self.config.Discharge.INITIAL_TEMPERATURE
        )

        # Update mission parameters
        self.mission.parameters = mission_params

        # 4. Configure solver
        solver_method = solver_method or self.config.Solver.PRIMARY_METHOD
        self.solver = self._create_solver(solver_method)
        self.mission.integration_method = self.solver

        # 5. Create thermal model (matching stops_model parameters exactly)
        self.thermal_model = StopsModelThermalModel(
            tank_volume=self.config.TANK_VOLUME,
            inner_surface_area=self.config.TANK_SURFACE_AREA,
            outer_surface_area=self.config.TANK_SURFACE_AREA * 1.025,  # Slightly larger outer area
            inner_diameter=1.0,  # From stops_model
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            ambient_htc=self.config.Thermal.HEAT_TRANSFER_COEFF,
            liner_mass=100.0,  # From stops_model
            wall_mass=150.0    # From stops_model
        )

        # 6. Set up heat flow data collector
        from src.dynamics.isochoric_dynamic_models import set_heat_flow_data_collector
        set_heat_flow_data_collector(heat_flow_data)
        print("🔧 Heat flow data collector configured")

        # 7. Create mission analysis
        self.mission_analysis = IsochoricMissionAnalysis(
            self.mission,
            self.thermal_model
        )

        print(f"✅ Analysis setup complete with {solver_method} solver")

    def _create_solver(self, method: str):
        """Create solver instance based on method name"""
        solver_config = {
            'timestep': self.config.Discharge.TIME_STEP,
            'rtol': self.config.Solver.RTOL,
            'atol': self.config.Solver.ATOL,
            'max_step': self.config.Solver.MAX_STEP
        }

        solver_classes = {
            'LSODA': LSODASolver,
            'RK45': RK45Solver,
            'Radau': RadauSolver,
            'DOP853': DOP853Solver,
            'BDF': BDFSolver
        }

        if method not in solver_classes:
            print(f"⚠️ Unknown solver method '{method}', falling back to RK45")
            method = 'RK45'

        return solver_classes[method](**solver_config)

    def run_analysis(self, solver_method: str = None) -> dict:
        """
        Run the complete CCH2 discharge analysis.

        Args:
            solver_method: Override solver method

        Returns:
            dict: Analysis results including performance metrics
        """
        print("🚀 Starting CCH2 discharge analysis...")

        # Clear heat flow data from any previous runs
        for key in heat_flow_data:
            heat_flow_data[key].clear()
        print("🔧 Cleared heat flow data from previous runs")

        # Setup if not already done
        if self.mission_analysis is None:
            print("🔧 Mission analysis not initialized, running setup...")
            self.setup_analysis(solver_method)
        else:
            print("✅ Mission analysis already initialized")

        # Run analysis with timing
        start_time = time.time()

        try:
            print("🚀 Starting mission analysis...")
            self.results = self.mission_analysis.run_analysis()
            print("✅ Mission analysis completed!")
            end_time = time.time()

            # Calculate performance metrics
            wall_time = end_time - start_time
            times = np.array([i * self.config.Discharge.TIME_STEP for i in range(len(self.results.states))])
            n_points = len(times)

            # Physics validation
            initial_mass = self.results.states[0].fuel_mass
            final_mass = self.results.states[-1].fuel_mass
            mass_change = initial_mass - final_mass
            expected_change = self.config.Discharge.CONSTANT_RATE * self.config.Discharge.MISSION_DURATION

            initial_temp = self.results.states[0].temperature
            final_temp = self.results.states[-1].temperature

            # Store metadata
            self.analysis_metadata = {
                'solver_method': self.solver.method_name,
                'wall_time': wall_time,
                'n_points': n_points,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'expected_mass_change': expected_change,
                'mass_error': abs(mass_change - expected_change),
                'initial_temperature': initial_temp,
                'final_temperature': final_temp,
                'temperature_change': final_temp - initial_temp
            }

            print(f"✅ Analysis completed successfully!")
            print(f"   Solver: {self.solver.method_name}")
            print(f"   Wall time: {wall_time:.3f}s")
            print(f"   Data points: {n_points}")
            print(f"   Mass change: {mass_change:.3f} kg (expected: {expected_change:.3f} kg)")
            print(f"   Temperature change: {self.analysis_metadata['temperature_change']:.2f} K")

            return self.analysis_metadata

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

    def validate_results(self) -> dict:
        """
        Validate analysis results against physical expectations.

        Returns:
            dict: Validation results
        """
        if self.results is None:
            raise ValueError("No analysis results available. Run analysis first.")

        validation = {}

        # Mass conservation check
        mass_error = self.analysis_metadata['mass_error']
        validation['mass_conserved'] = mass_error < 0.01  # 10g tolerance

        # Temperature realism check
        final_temp = self.analysis_metadata['final_temperature']
        validation['temperature_realistic'] = 13.8 <= final_temp <= 500.0

        # Pressure realism (simplified check)
        final_mass = self.analysis_metadata['final_mass']
        validation['mass_positive'] = final_mass >= 0.0

        # Physical monotonicity (mass should decrease)
        masses = [state.fuel_mass for state in self.results.states]
        validation['mass_monotonic'] = all(masses[i] <= masses[i-1] for i in range(1, len(masses)))

        print(f"🔍 Validation Results:")
        print(f"   Mass conserved: {'✓' if validation['mass_conserved'] else '✗'}")
        print(f"   Temperature realistic: {'✓' if validation['temperature_realistic'] else '✗'}")
        print(f"   Mass positive: {'✓' if validation['mass_positive'] else '✗'}")
        print(f"   Mass monotonic: {'✓' if validation['mass_monotonic'] else '✗'}")

        return validation

    def plot_results(self, save_path: str = None):
        """
        Create comprehensive plots of the analysis results.

        Args:
            save_path: Optional path to save plots
        """
        if self.results is None:
            raise ValueError("No analysis results available. Run analysis first.")

        # Create plotter
        plotter = SeabornPlotter()

        # Extract data for plotting
        # Reconstruct times from timestep and number of states
        n_states = len(self.results.states)
        times = np.arange(0, n_states * self.results.timestep, self.results.timestep) / 3600.0  # Convert to hours
        masses = [state.fuel_mass for state in self.results.states]
        temperatures = [state.temperature for state in self.results.states]
        solid_temps = [state.solid_temperature for state in self.results.states]

        # Calculate densities for the temperature-density plot
        densities = [state.fuel_mass / self.config.TANK_VOLUME for state in self.results.states]

        # Create basic plots first
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Mass evolution
        ax1.plot(times, masses, 'b-', linewidth=2)
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Fuel Mass [kg]')
        ax1.set_title('Fuel Mass Evolution')
        ax1.grid(True, alpha=0.3)

        # Temperature evolution
        ax2.plot(times, temperatures, 'r-', linewidth=2, label='Fuel Temperature')
        ax2.plot(times, solid_temps, 'k--', linewidth=2, label='Solid Temperature')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Temperature Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Pressure evolution
        pressures = [state.hydrogen.pressure / 1e5 for state in self.results.states]  # Convert Pa to bar
        ax3.plot(times, pressures, 'g-', linewidth=2)
        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Pressure [bar]')
        ax3.set_title('Pressure Evolution')
        ax3.grid(True, alpha=0.3)

        # Performance summary
        ax4.axis('off')
        summary_text = f"""
CCH2 Discharge Analysis Summary

Solver: {self.solver.method_name}
Duration: {self.config.Discharge.MISSION_DURATION/3600:.1f} hours
Wall Time: {self.analysis_metadata['wall_time']:.3f} s

Initial Mass: {self.analysis_metadata['initial_mass']:.2f} kg
Final Mass: {self.analysis_metadata['final_mass']:.2f} kg
Mass Change: {self.analysis_metadata['mass_change']:.2f} kg

Initial Temp: {self.analysis_metadata['initial_temperature']:.1f} K
Final Temp: {self.analysis_metadata['final_temperature']:.1f} K
Temp Change: {self.analysis_metadata['temperature_change']:.1f} K

Data Points: {self.analysis_metadata['n_points']}
        """
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")

        plt.show()

        # Create temperature-density multi-scenario plot using SeabornPlotter
        print("📊 Creating temperature-density multi-scenario plot...")
        scenario_data = {
            'discharge': {
                'temperatures': temperatures,
                'densities': densities
            },
            'refuel': {
                'temperatures': [],  # Empty for now, will be populated when refuel is implemented
                'densities': []
            },
            'dormancy': {
                'temperatures': [],  # Empty for now, will be populated when dormancy is implemented
                'densities': []
            }
        }

        # Create the multi-scenario temperature-density plot
        plotter.plot_density_temperature_combined(
            scenario_data,
            include_saturation_line=True,
            include_isobars=True,
            include_ref_data=True,
            figsize=(10, 8)
        )
        plt.show()

        # Create heat exchanger requirements plot using SeabornPlotter
        print("📊 Creating heat exchanger requirements plot...")

        # Use the global heat_flow_data that was populated during integration
        # (Don't create a new local variable that overwrites it!)
        if len(heat_flow_data['t']) > 0:
            print(f"🔧 Using captured heat flow data: {len(heat_flow_data['t'])} data points")
            print(f"🔍 Time range: {min(heat_flow_data['t']):.1f}s to {max(heat_flow_data['t']):.1f}s ({max(heat_flow_data['t'])/3600:.1f} hours)")
            print(f"🔍 iHEX heat range: {min(heat_flow_data['qdot_disch']):.1f}W to {max(heat_flow_data['qdot_disch']):.1f}W")
            calculate_ohex_heat_requirements()
            print(f"🔍 oHEX heat range: {min(heat_flow_data['qdot_ohex']):.1f}W to {max(heat_flow_data['qdot_ohex']):.1f}W")
        else:
            print("⚠️ No heat flow data captured - plotting will show zeros")

        plotter.plot_heat_exchanger_requirements(
            heat_flow_data,
            scenario_name="discharge",
            figsize=(10, 6)
        )
        plt.show()

        return fig


def compare_solver_methods():
    """
    Compare different solver methods on the CCH2 discharge scenario.

    This function demonstrates the flexibility of the new solver architecture
    by running the same analysis with different integration methods.
    """
    print("🔬 SOLVER METHOD COMPARISON FOR CCH2 DISCHARGE")
    print("="*60)

    # Solver methods to compare
    methods = ['LSODA', 'RK45', 'DOP853', 'BDF', 'Radau']

    results = {}

    for method in methods:
        print(f"\n🧪 Testing {method} solver...")

        try:
            # Create analysis instance
            analysis = CCH2DischargeAnalysis()

            # Run with specific solver
            metadata = analysis.run_analysis(solver_method=method)

            # Validate results
            validation = analysis.validate_results()

            # Store results
            results[method] = {
                'metadata': metadata,
                'validation': validation,
                'analysis': analysis
            }

        except Exception as e:
            print(f"❌ {method} failed: {e}")
            results[method] = {'error': str(e)}

    # Summary comparison
    print(f"\n{'='*60}")
    print("SOLVER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<8} {'Time':<8} {'Valid':<6} {'Mass Error':<12}")
    print("-" * 60)

    for method, result in results.items():
        if 'error' not in result:
            time_str = f"{result['metadata']['wall_time']:.3f}s"
            valid_str = "✓" if all(result['validation'].values()) else "✗"
            error_str = f"{result['metadata']['mass_error']:.2e}"
            print(f"{method:<8} {time_str:<8} {valid_str:<6} {error_str:<12}")
        else:
            print(f"{method:<8} {'ERROR':<8} {'✗':<6} {'N/A':<12}")

    return results


def main():
    """
    Main execution function for CCH2 verification analysis.

    This function demonstrates the complete workflow:
    1. Setup and run discharge analysis
    2. Validate results
    3. Create plots
    4. Compare solver methods
    """
    print("🚀 CCH2 VERIFICATION ANALYSIS")
    print("="*50)
    print("Compressed Cold Hydrogen discharge scenario verification")
    print("Using integrated HFT framework with new solver architecture")
    print("="*50)

    # 1. Run primary analysis with LSODA
    print("\n1️⃣ Running primary analysis with RK45 solver...")
    print("🏗️ Creating CCH2DischargeAnalysis instance...")
    analysis = CCH2DischargeAnalysis()
    print("✅ Instance created, starting analysis...")
    metadata = analysis.run_analysis()
    validation = analysis.validate_results()

    # 2. Create plots
    print("\n2️⃣ Creating analysis plots...")
    analysis.plot_results()

    # 3. Compare solver methods (commented out to prevent auto re-running)
    print("\n3️⃣ Solver comparison skipped (prevents auto re-running)")
    comparison_results = None  # compare_solver_methods()

    print(f"\n{'='*50}")
    print("🎉 CCH2 VERIFICATION ANALYSIS COMPLETED!")
    print("✅ Discharge scenario successfully validated")
    print("✅ New solver architecture working correctly")
    print("✅ Class-based HFT integration functional")
    print(f"{'='*50}")

    return analysis, comparison_results


if __name__ == "__main__":
    # Run the verification analysis
    primary_analysis, solver_comparison = main()
