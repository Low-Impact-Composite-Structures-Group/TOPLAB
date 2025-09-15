"""
Full cycle analysis for a cryocompressed storage tank.

Includes discharge, refuel, and dormancy scenarios for a single tank. Used to verify
the functionality of the thermodynamic modelling/sizing tool developed for the
TRIATHLON project.

Authors: Dante Raso (2025)

Based on numerical framework detailed in "Generalized thermodynamic modeling of
hydrogen storage tanks for truck applications" (Stops et al., 2024)
DOI: 10.1016/j.cryogenics.2024.103826
"""

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

# HFT Framework imports
from src.mission.isochoric_missions import (
    DischargeMission,
    RefuelMission,
    DormancyMission,
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

# =================== GLOBAL CONFIGURATION ===================

# Global heat flow data collector for computing oHEX requirements
heat_flow_data = {
    't': [],           # Time [s]
    'qdot_disch': [],  # Discharge heat rate [W] (iHEX requirements)
    'qdot_ohex': [],   # oHEX heat rate [W] (calculated in post-processing)
    'mdot_disch': [],  # Discharge mass flow rate [kg/s]
    'T': [],           # Temperature [K]
    'rho': []          # Density [kg/m³]
}

# oHEX target conditions (what temp/pressure to condition to flowing out of the tank)
OHEX_TARGET_TEMPERATURE = 200.0   # Target temperature [K]
OHEX_TARGET_PRESSURE = 20e5       # Target pressure [Pa] = 20 bar

# =================== UTILITY FUNCTIONS ===================

def calculate_ohex_heat_requirements():
    """
    Calculate oHEX heat requirements using enthalpy difference.

    Uses the same approach as stops_model:
    Q_oHEX = mdot * (h_target - h_disch)

    where:
    - h_target: enthalpy at OHEX target conditions (20°C, 1 atm)
    - h_disch: enthalpy at discharge conditions (current T, P)
    """
    print("Calculating oHEX heat requirements...")

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
                print(f"WARNING: oHEX calculation failed at point {i}: {e}")
                heat_flow_data['qdot_ohex'].append(0.0)
        else:
            heat_flow_data['qdot_ohex'].append(0.0)

    total_points = len(heat_flow_data['qdot_ohex'])
    max_ohex = max(heat_flow_data['qdot_ohex']) if heat_flow_data['qdot_ohex'] else 0
    print(f"oHEX calculation complete: {total_points} points, max = {max_ohex/1000:.1f} kW")


class CCH2VerificationConfig:
    """
    Configuration parameters for CCH2 verification analysis.

    Matching the parameters reported in the verification paper of Stops et al. (2024).

    """

    # Tank physical parameters
    TANK_VOLUME = 0.5           # m³ - V_t
    TANK_SURFACE_AREA = 4.0     # m² - A_in

    # Pressure thresholds
    P_MIN = 15e5                # Pa (15 bar) - p_min
    P_VENT = 450e5              # Pa (450 bar) - p_vent

    # Environmental conditions
    AMBIENT_TEMPERATURE = 298.15  # K - T_amb

    # Discharge scenario parameters
    class Discharge:
        # Initial conditions
        INITIAL_PRESSURE = 400e5        # Pa (400 bar)
        INITIAL_TEMPERATURE = 53.25     # K
        INITIAL_SOLID_TEMP = "thermal_equilibrium"  # K

        # Stopping condition
        STOPPING_DENSITY = 5.8          # kg/m³

        # Discharge parameters
        CONSTANT_RATE = 0.001           # kg/s
        MISSION_DURATION = 40000.0      # s

        # Analysis parameters
        TIME_STEP = 1.0             # s

    # Refuel scenario parameters (matching stops_model exactly)
    class Refuel:
        # Initial conditions
        INITIAL_PRESSURE = 15.3e5       # Pa (15.3 bar)
        INITIAL_TEMPERATURE = 65.5      # K
        INITIAL_SOLID_TEMP = "thermal_equilibrium"  # K

        # Stopping condition
        STOPPING_DENSITY = 78.0         # kg/m³

        # Refuel parameters - from stops_model REFUEL
        CONSTANT_RATE = 0.07            # kg/s
        MISSION_DURATION = 700.0        # s

        # Cryopump parameters - from stops_model compute_pump_outlet_hydrogen
        DEWAR_PRESSURE = 3e5            # Pa (3 bar)
        PUMP_EFFICIENCY = 0.78          # isentropic efficiency - eta_p

        # Analysis parameters
        TIME_STEP = 0.1                 # s (higher resolution for fast refuel dynamics)

    # Dormancy scenario parameters
    class Dormancy:
        # Initial conditions - STOPS_MODEL DORMANCY scenario
        INITIAL_PRESSURE = 400e5        # Pa (400 bar)
        INITIAL_TEMPERATURE = 53.25     # K
        INITIAL_SOLID_TEMP = "thermal_equilibrium"  # K - calculated

        # Stopping condition
        STOPPING_DENSITY = 70.0         # kg/m³

        # Dormancy parameters - all mass flows zero, venting handled by Config C
        FUEL_FLOW_RATE = 0.0            # kg/s - no fuel input during dormancy
        DISCHARGE_FLOW_RATE = 0.0       # kg/s - no discharge during dormancy
        MISSION_DURATION = 216000.0     # s (60 hours)

        # Analysis parameters
        TIME_STEP = 10.0                # s (lower resolution for long dormancy period)

    # Thermal model parameters (from stops_model)
    class Thermal:
        HEAT_TRANSFER_COEFF = 0.025   # W/m²K

    # Solver configuration parameters
    class Solver:
        PRIMARY_METHOD = 'RK45'         # Primary solver method
        RTOL = 1e-5                     # Slightly relaxed relative tolerance
        ATOL = 1e-8                     # Slightly relaxed absolute tolerance
        MAX_STEP = 5.0                # Maximum step size (seconds) among all scenarios


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
        print("Initializing CCH2DischargeAnalysis...")
        self.config = config or CCH2VerificationConfig()
        print("Configuration loaded")

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
        print("\n" + "="*80)
        print("SETTING UP CCH2 DISCHARGE ANALYSIS")
        print("="*80)
        print(f"Scenario: DISCHARGE")
        print(f"Description: Constant discharge rate until target density reached")
        print(f"Tank Parameters:")
        print(f"   • Volume: {self.config.TANK_VOLUME:.1f} m³")
        print(f"   • Surface Area: {self.config.TANK_SURFACE_AREA:.1f} m²")
        print(f"   • Min Pressure: {self.config.P_MIN/1e5:.1f} bar")
        print(f"   • Vent Pressure: {self.config.P_VENT/1e5:.1f} bar")
        print(f"   • Ambient Temperature: {self.config.AMBIENT_TEMPERATURE:.1f} K")
        print(f"Mission Parameters:")
        print(f"   • Initial Pressure: {self.config.Discharge.INITIAL_PRESSURE/1e5:.1f} bar")
        print(f"   • Initial Temperature: {self.config.Discharge.INITIAL_TEMPERATURE:.2f} K")
        print(f"   • Discharge Rate: {self.config.Discharge.CONSTANT_RATE:.3f} kg/s")
        print(f"   • Target Density: {self.config.Discharge.STOPPING_DENSITY:.1f} kg/m³")
        print(f"   • Max Duration: {self.config.Discharge.MISSION_DURATION/3600:.1f} hours")
        print(f"   • Time Step: {self.config.Discharge.TIME_STEP:.1f} s")
        solver_method = solver_method or self.config.Solver.PRIMARY_METHOD
        print(f"Solver Configuration:")
        print(f"   • Method: {solver_method}")
        print(f"   • Relative Tolerance: {self.config.Solver.RTOL:.0e}")
        print(f"   • Absolute Tolerance: {self.config.Solver.ATOL:.0e}")
        print(f"   • Max Step: {self.config.Solver.MAX_STEP:.1f} s")
        print(f"Expected Behavior:")
        print(f"   • Configuration A (normal operation) initially")
        print(f"   • Mass decreases linearly during discharge")
        print(f"   • Temperature increases due to expansion and ambient heat")
        print(f"   • Pressure drops following equation of state")
        print(f"   • Heat exchanger requirements captured for sizing")
        print("="*80)


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

        # 5. Create thermal model
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
        print("Heat flow data collector configured")

        # 7. Create mission analysis
        self.mission_analysis = IsochoricMissionAnalysis(
            self.mission,
            self.thermal_model
        )

        print(f"Analysis setup complete with {solver_method} solver")

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
            print(f"Unknown solver method '{method}', falling back to RK45")
            method = 'RK45'

        return solver_classes[method](**solver_config)

    def _display_key_steps(self):
        """Display key analysis steps in stops_model format"""
        if not self.results or len(self.results.states) < 2:
            print("WARNING: Insufficient data for step display")
            return

        print(f"\nKey Analysis Steps (DISCHARGE):")
        print("-" * 120)
        print("Step     |   Time   |   Mass   | Density  |   Temp   | Ts_solid | Pressure | Config   | Phase       | Notes")
        print("-" * 120)

        n_states = len(self.results.states)
        # Display start, key milestones, and end
        step_indices = [0, n_states//4, n_states//2, 3*n_states//4, n_states-1]

        def is_near_saturation(T, p_pa):
            """Check if state is near saturation (two-phase)"""
            try:
                p_sat = PropsSI("P", "T", T, "Q", 0, "hydrogen")
                tolerance = 1e-4
                return abs(p_pa - p_sat) < tolerance * p_sat
            except:
                return False

        for i, step_idx in enumerate(step_indices):
            if step_idx >= n_states:
                continue

            state = self.results.states[step_idx]
            t = step_idx * self.config.Discharge.TIME_STEP
            m = state.fuel_mass
            rho = m / self.config.TANK_VOLUME
            T = state.temperature
            Ts = state.solid_temperature

            # Calculate pressure and phase
            try:
                p_pa = PropsSI("P", "T", T, "Dmass", rho, "hydrogen")
                p = p_pa / 1e5  # Convert to bar
                phase = "two-phase" if is_near_saturation(T, p_pa) else "single-phase"
            except:
                p = 0.0
                phase = "unknown"

            # Determine configuration
            if p >= self.config.P_VENT/1e5:
                config = "Config C"
            elif p <= self.config.P_MIN/1e5:
                config = "Config B"
            else:
                config = "Config A"

            # Add notes for key milestones
            notes = ""
            if i == 0:
                notes = "START"
            elif i == len(step_indices)-1:
                notes = "END"
            elif i == len(step_indices)//2:
                notes = "MIDPOINT"

            print(f"{step_idx:4d}     | {t:7.1f}s | {m:6.2f}kg | {rho:6.2f}kg/m³ | "
                  f"{T:6.2f}K | {Ts:6.2f}K | {p:6.2f}bar | {config:8s} | {phase:11s} | {notes}")

        print("-" * 120)

    def run_analysis(self, solver_method: str = None) -> dict:
        """
        Run the complete CCH2 discharge analysis.

        Args:
            solver_method: Override solver method

        Returns:
            dict: Analysis results including performance metrics
        """
        print("Starting CCH2 discharge analysis...")

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

            # Display key steps (similar to stops_model style)
            self._display_key_steps()

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
        Create comprehensive plots of the analysis results using SeabornPlotter.

        Args:
            save_path: Optional path to save plots
        """
        if self.results is None:
            raise ValueError("No analysis results available. Run analysis first.")

        print("📊 Creating discharge analysis plots using SeabornPlotter...")

        # Extract data for plotting
        times = np.array([i * self.config.Discharge.TIME_STEP for i in range(len(self.results.states))])
        masses = np.array([state.fuel_mass for state in self.results.states])
        temperatures = np.array([state.temperature for state in self.results.states])
        densities = masses / self.config.TANK_VOLUME

        # Calculate pressures
        pressures = []
        for i, state in enumerate(self.results.states):
            try:
                p = PropsSI("P", "T", state.temperature, "Dmass", densities[i], "hydrogen")
                pressures.append(p / 1e5)  # Convert to bar
            except:
                pressures.append(400.0)  # Fallback to initial pressure in bar

        pressures = np.array(pressures)

        # Create SeabornPlotter instance
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Create discharge analysis plots using SeabornPlotter
        initial_conditions = {
            'pressure': pressures[0],
            'temperature': temperatures[0],
            'density': densities[0]
        }

        # Use the refuel analysis plotting method (works for discharge too with proper title)
        fig = plotter.plot_refuel_analysis(
            times=times,
            masses=masses,
            temperatures=temperatures,
            densities=densities,
            pressures=pressures,
            initial_conditions=initial_conditions,
            figsize=(14, 10)
        )

        # Update the title for discharge scenario
        if fig:
            fig.suptitle(f"CCH2 Discharge Analysis - {self.solver.method_name}", fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Discharge analysis plot saved to: {save_path}")

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


class CCH2RefuelAnalysis:
    """
    CCH2 refuel scenario analysis using the integrated HFT framework.

    This class provides the refuel scenario implementation matching stops_model
    physics, with proper twophase handling and Configuration B exclusion logic.
    """

    def __init__(self, config: CCH2VerificationConfig = None):
        """
        Initialize CCH2 refuel analysis.

        Args:
            config: Configuration object (uses default if None)
        """
        print("🏗️ Initializing CCH2RefuelAnalysis...")
        self.config = config or CCH2VerificationConfig()
        self.mission = None
        self.mission_analysis = None
        self.thermal_model = None
        self.solver = None
        self.results = None
        self.analysis_metadata = {}
        print("✅ Configuration loaded")

    def setup_analysis(self, solver_method: str = None):
        """
        Set up all analysis components for refuel scenario.

        Args:
            solver_method: Override solver method ("LSODA", "RK45", etc.)
        """
        print("\n" + "="*80)
        print("🔧 SETTING UP CCH2 REFUEL ANALYSIS")
        print("="*80)
        print(f"📋 Scenario: REFUEL")
        print(f"📝 Description: Constant refuel rate until target density reached")
        print(f"📊 Tank Parameters:")
        print(f"   • Volume: {self.config.TANK_VOLUME:.1f} m³")
        print(f"   • Surface Area: {self.config.TANK_SURFACE_AREA:.1f} m²")
        print(f"   • Min Pressure: {self.config.P_MIN/1e5:.1f} bar")
        print(f"   • Vent Pressure: {self.config.P_VENT/1e5:.1f} bar")
        print(f"   • Ambient Temperature: {self.config.AMBIENT_TEMPERATURE:.1f} K")
        print(f"🎯 Mission Parameters:")
        print(f"   • Initial Pressure: {self.config.Refuel.INITIAL_PRESSURE/1e5:.1f} bar")
        print(f"   • Initial Temperature: {self.config.Refuel.INITIAL_TEMPERATURE:.2f} K")
        print(f"   • Refuel Rate: {self.config.Refuel.CONSTANT_RATE:.3f} kg/s")
        print(f"   • Target Density: {self.config.Refuel.STOPPING_DENSITY:.1f} kg/m³")
        print(f"   • Max Duration: {self.config.Refuel.MISSION_DURATION/60:.1f} minutes")
        print(f"   • Time Step: {self.config.Refuel.TIME_STEP:.1f} s")
        solver_method = solver_method or self.config.Solver.PRIMARY_METHOD
        print(f"⚙️ Solver Configuration:")
        print(f"   • Method: {solver_method}")
        print(f"   • Relative Tolerance: {self.config.Solver.RTOL:.0e}")
        print(f"   • Absolute Tolerance: {self.config.Solver.ATOL:.0e}")
        print(f"   • Max Step: {self.config.Solver.MAX_STEP:.1f} s")
        print(f"🔄 Expected Behavior:")
        print(f"   • Configuration A (normal operation) throughout")
        print(f"   • Configuration B disabled during refuel (matches stops_model)")
        print(f"   • Mass increases linearly during refuel")
        print(f"   • Temperature increases due to compression")
        print(f"   • Pressure rises following equation of state")
        print(f"   • Two-phase conditions possible at low temperatures")
        print("="*80)

        # 1. Calculate initial mass from P,T like stops_model
        from CoolProp.CoolProp import PropsSI

        initial_density = PropsSI("Dmass", "P", self.config.Refuel.INITIAL_PRESSURE,
                                 "T", self.config.Refuel.INITIAL_TEMPERATURE, "hydrogen")
        initial_mass = initial_density * self.config.TANK_VOLUME

        print(f"   Initial conditions: P={self.config.Refuel.INITIAL_PRESSURE/1e5:.1f} bar, "
              f"T={self.config.Refuel.INITIAL_TEMPERATURE:.2f} K")
        print(f"   Calculated: ρ={initial_density:.2f} kg/m³, m={initial_mass:.2f} kg")
        print(f"   Stopping density: {self.config.Refuel.STOPPING_DENSITY:.1f} kg/m³")

        # Calculate mass to add and expected time
        final_mass = self.config.Refuel.STOPPING_DENSITY * self.config.TANK_VOLUME
        mass_to_add = final_mass - initial_mass
        theoretical_time = mass_to_add / self.config.Refuel.CONSTANT_RATE
        # Use reasonable maximum duration from stops_model
        refuel_duration = min(theoretical_time * 1.2, self.config.Refuel.MISSION_DURATION)

        print(f"   Mass to add: {mass_to_add:.2f} kg")
        print(f"   Theoretical time: {theoretical_time:.0f} s")
        print(f"   Mission duration: {refuel_duration:.0f} s")
        print(f"   NOTE: Configuration B disabled during REFUEL (matches stops_model)")

        # Handle thermal equilibrium solid temperature like stops_model
        if self.config.Refuel.INITIAL_SOLID_TEMP == "thermal_equilibrium":
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
                self.config.Refuel.INITIAL_TEMPERATURE
            )
            print(f"   Using thermal equilibrium Ts0 = {initial_solid_temp:.2f}K")
        else:
            initial_solid_temp = self.config.Refuel.INITIAL_SOLID_TEMP

        # 2. Create mission parameters with calculated mass
        mission_params = IsochoricMissionParameters(
            tank_volume=self.config.TANK_VOLUME,
            p_min=self.config.P_MIN,
            p_vent=self.config.P_VENT,
            initial_mass=initial_mass,
            initial_temperature=self.config.Refuel.INITIAL_TEMPERATURE,
            initial_solid_temperature=initial_solid_temp,
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            time_step=self.config.Refuel.TIME_STEP,
            rtol=self.config.Solver.RTOL,
            atol=self.config.Solver.ATOL
        )

        # 3. Create refuel mission with calculated parameters
        self.mission = RefuelMission.constant_refuel(
            refuel_rate=self.config.Refuel.CONSTANT_RATE,
            duration=refuel_duration,
            target_mass=final_mass,
            initial_temperature=self.config.Refuel.INITIAL_TEMPERATURE
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

        # 6. Set up heat flow data collector (though not used for refuel plotting)
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
            'timestep': self.config.Refuel.TIME_STEP,
            'rtol': self.config.Solver.RTOL,
            'atol': self.config.Solver.ATOL,
            'max_step': self.config.Solver.MAX_STEP
        }

        solver_classes = {
            'LSODA': LSODASolver,
            'RK45': RK45Solver,
            'DOP853': DOP853Solver,
            'BDF': BDFSolver,
            'Radau': RadauSolver
        }

        if method not in solver_classes:
            print(f"⚠️ Unknown solver method: {method}, using RK45")
            method = 'RK45'

        return solver_classes[method](**solver_config)

    def _display_key_steps(self):
        """Display key analysis steps in stops_model format"""
        if not self.results or len(self.results.states) < 2:
            print("WARNING: Insufficient data for step display")
            return

        print(f"\nKey Analysis Steps (REFUEL):")
        print("-" * 120)
        print("Step     |   Time   |   Mass   | Density  |   Temp   | Ts_solid | Pressure | Config   | Phase       | Notes")
        print("-" * 120)

        n_states = len(self.results.states)
        # Display start, key milestones, and end
        step_indices = [0, n_states//4, n_states//2, 3*n_states//4, n_states-1]

        def is_near_saturation(T, p_pa):
            """Check if state is near saturation (two-phase)"""
            try:
                p_sat = PropsSI("P", "T", T, "Q", 0, "hydrogen")
                tolerance = 1e-6
                return abs(p_pa - p_sat) < tolerance * p_sat
            except:
                return False

        for i, step_idx in enumerate(step_indices):
            if step_idx >= n_states:
                continue

            state = self.results.states[step_idx]
            t = step_idx * self.config.Refuel.TIME_STEP
            m = state.fuel_mass
            rho = m / self.config.TANK_VOLUME
            T = state.temperature
            Ts = state.solid_temperature

            # Calculate pressure and phase
            try:
                p_pa = PropsSI("P", "T", T, "Dmass", rho, "hydrogen")
                p = p_pa / 1e5  # Convert to bar
                phase = "two-phase" if is_near_saturation(T, p_pa) else "single-phase"
            except:
                p = 0.0
                phase = "unknown"

            # Determine configuration (Config B disabled during refuel)
            if p >= self.config.P_VENT/1e5:
                config = "Config C"
            else:
                config = "Config A"  # Config B disabled during refuel

            # Add notes for key milestones
            notes = ""
            if i == 0:
                notes = "START"
            elif i == len(step_indices)-1:
                notes = "END"
            elif i == len(step_indices)//2:
                notes = "MIDPOINT"

            print(f"{step_idx:4d}     | {t:7.1f}s | {m:6.2f}kg | {rho:6.2f}kg/m³ | "
                  f"{T:6.2f}K | {Ts:6.2f}K | {p:6.2f}bar | {config:8s} | {phase:11s} | {notes}")

        print("-" * 120)

    def run_analysis(self, solver_method: str = None):
        """
        Run the complete refuel analysis.

        Args:
            solver_method: Override solver method

        Returns:
            dict: Analysis results including performance metrics
        """
        print("Starting CCH2 refuel analysis...")

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
            times = np.array([i * self.config.Refuel.TIME_STEP for i in range(len(self.results.states))])
            n_points = len(times)

            # Display key steps (similar to stops_model style)
            self._display_key_steps()

            # Physics validation
            initial_mass = self.results.states[0].fuel_mass
            final_mass = self.results.states[-1].fuel_mass
            mass_change = final_mass - initial_mass  # Positive for refuel
            expected_change = self.config.Refuel.CONSTANT_RATE * self.config.Refuel.MISSION_DURATION

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

        # Check mass conservation
        mass_error = self.analysis_metadata['mass_error']
        validation['mass_conservation'] = mass_error < 0.01  # Within 1% of expected

        # Check temperature evolution (refuel should cause heating due to compression)
        temp_change = self.analysis_metadata['temperature_change']
        validation['temperature_physics'] = temp_change > 0  # Should increase during refuel

        # Check final density reached target within tolerance
        final_density = self.results.states[-1].fuel_mass / self.config.TANK_VOLUME
        target_density = self.config.Refuel.STOPPING_DENSITY
        density_error = abs(final_density - target_density) / target_density
        validation['density_target'] = density_error < 0.05  # Within 5%

        # Solver stability check
        validation['solver_stability'] = len(self.results.states) > 10  # Reasonable number of points

        print(f"📊 Validation Results:")
        for test, passed in validation.items():
            status = "✅" if passed else "❌"
            print(f"   {test}: {status}")

        return validation

    def plot_results(self, save_path: str = None):
        """
        Create comprehensive plots for refuel analysis results.

        Args:
            save_path: Optional path to save the figure
        """
        if self.results is None:
            raise ValueError("No analysis results available. Run analysis first.")

        print("📊 Creating refuel analysis plots...")

        # Create time array and extract data
        times = np.array([i * self.config.Refuel.TIME_STEP for i in range(len(self.results.states))])
        masses = np.array([state.fuel_mass for state in self.results.states])
        temperatures = np.array([state.temperature for state in self.results.states])
        densities = masses / self.config.TANK_VOLUME

        # Calculate pressures
        pressures = []
        for i, state in enumerate(self.results.states):
            try:
                from CoolProp.CoolProp import PropsSI
                p = PropsSI("P", "T", state.temperature, "Dmass", densities[i], "hydrogen")
                pressures.append(p / 1e5)  # Convert to bar
            except:
                pressures.append(15.3)  # Fallback to initial pressure in bar

        pressures = np.array(pressures)

        # Create SeabornPlotter instance for consistent styling
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Set up initial conditions for plotting context
        initial_conditions = {
            'pressure': pressures[0],
            'temperature': temperatures[0],
            'density': densities[0]
        }

        # Create the refuel plot
        fig = plotter.plot_refuel_analysis(
            times=times,
            masses=masses,
            temperatures=temperatures,
            densities=densities,
            pressures=pressures,
            initial_conditions=initial_conditions,
            figsize=(14, 10)
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Figure saved to: {save_path}")

        plt.show()

        return fig



class CCH2DormancyAnalysis:
    """
    CCH2 dormancy scenario analysis using the integrated HFT framework.

    This class implements the dormancy scenario from stops_model, where the tank
    is at rest with no fuel or discharge flows. The scenario triggers Configuration C
    when pressure exceeds the venting threshold (450 bar), causing automatic venting
    to maintain pressure control.

    Key features:
    - Initial conditions: 400 bar, 53.25K (same as discharge start)
    - No fuel input or discharge output (all zero mass flows)
    - Configuration C venting activated at high pressure
    - Long duration: 60 hours (216,000 seconds)
    - Stopping condition: density reaches 70.0 kg/m³
    """

    def __init__(self, config: CCH2VerificationConfig = None):
        """
        Initialize CCH2 dormancy analysis.

        Args:
            config: Configuration object (uses default if None)
        """
        print("🏗️ Initializing CCH2DormancyAnalysis...")
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
        Set up all analysis components for dormancy scenario.

        Args:
            solver_method: Override solver method ("LSODA", "RK45", etc.)
        """
        print("\n" + "="*80)
        print("🔧 SETTING UP CCH2 DORMANCY ANALYSIS")
        print("="*80)
        print(f"📋 Scenario: DORMANCY")
        print(f"📝 Description: Long-term storage with Configuration C venting")
        print(f"📊 Tank Parameters:")
        print(f"   • Volume: {self.config.TANK_VOLUME:.1f} m³")
        print(f"   • Surface Area: {self.config.TANK_SURFACE_AREA:.1f} m²")
        print(f"   • Min Pressure: {self.config.P_MIN/1e5:.1f} bar")
        print(f"   • Vent Pressure: {self.config.P_VENT/1e5:.1f} bar (Configuration C trigger)")
        print(f"   • Ambient Temperature: {self.config.AMBIENT_TEMPERATURE:.1f} K")
        print(f"🎯 Mission Parameters:")
        print(f"   • Initial Pressure: {self.config.Dormancy.INITIAL_PRESSURE/1e5:.1f} bar")
        print(f"   • Initial Temperature: {self.config.Dormancy.INITIAL_TEMPERATURE:.2f} K")
        print(f"   • Fuel Flow Rate: {self.config.Dormancy.FUEL_FLOW_RATE:.3f} kg/s (none)")
        print(f"   • Discharge Flow Rate: {self.config.Dormancy.DISCHARGE_FLOW_RATE:.3f} kg/s (none)")
        print(f"   • Target Density: {self.config.Dormancy.STOPPING_DENSITY:.1f} kg/m³")
        print(f"   • Max Duration: {self.config.Dormancy.MISSION_DURATION/3600:.1f} hours")
        print(f"   • Time Step: {self.config.Dormancy.TIME_STEP:.1f} s")
        solver_method = solver_method or self.config.Solver.PRIMARY_METHOD
        print(f"⚙️ Solver Configuration:")
        print(f"   • Method: {solver_method}")
        print(f"   • Relative Tolerance: {self.config.Solver.RTOL:.0e}")
        print(f"   • Absolute Tolerance: {self.config.Solver.ATOL:.0e}")
        print(f"   • Max Step: {self.config.Solver.MAX_STEP * 10:.1f} s (extended for dormancy)")
        print(f"🔄 Expected Behavior:")
        print(f"   • Configuration A initially (P < {self.config.P_VENT/1e5:.0f} bar)")
        print(f"   • Configuration C activates when P ≥ {self.config.P_VENT/1e5:.0f} bar")
        print(f"   • Automatic venting maintains pressure control")
        print(f"   • Temperature rises due to ambient heat input")
        print(f"   • Mass decreases through venting to reach target density")
        print(f"   • Long timescale dominated by thermal effects")
        print("="*80)

        # 1. Calculate initial mass from P,T like stops_model
        from CoolProp.CoolProp import PropsSI

        initial_density = PropsSI("Dmass", "P", self.config.Dormancy.INITIAL_PRESSURE,
                                 "T", self.config.Dormancy.INITIAL_TEMPERATURE, "hydrogen")
        initial_mass = initial_density * self.config.TANK_VOLUME

        print(f"   Initial conditions: P={self.config.Dormancy.INITIAL_PRESSURE/1e5:.1f} bar, "
              f"T={self.config.Dormancy.INITIAL_TEMPERATURE:.2f} K")
        print(f"   Calculated: ρ={initial_density:.2f} kg/m³, m={initial_mass:.2f} kg")
        print(f"   Stopping density: {self.config.Dormancy.STOPPING_DENSITY:.1f} kg/m³")

        # Calculate expected mass and time for dormancy scenario
        final_mass = self.config.Dormancy.STOPPING_DENSITY * self.config.TANK_VOLUME
        mass_change = initial_mass - final_mass  # Expected mass reduction due to venting
        theoretical_time = self.config.Dormancy.MISSION_DURATION

        print(f"   Expected mass change: {mass_change:.2f} kg (due to venting)")
        print(f"   Mission duration: {theoretical_time/3600:.1f} hours")
        print(f"   NOTE: Configuration C venting will activate if P ≥ {self.config.P_VENT/1e5:.0f} bar")

        # Handle thermal equilibrium solid temperature like stops_model
        if self.config.Dormancy.INITIAL_SOLID_TEMP == "thermal_equilibrium":
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
                self.config.Dormancy.INITIAL_TEMPERATURE
            )
            print(f"   Using thermal equilibrium Ts0 = {initial_solid_temp:.2f}K")
        else:
            initial_solid_temp = self.config.Dormancy.INITIAL_SOLID_TEMP

        # 2. Create dormancy mission with zero mass flows
        mission_params = IsochoricMissionParameters(
            tank_volume=self.config.TANK_VOLUME,
            p_min=self.config.P_MIN,
            p_vent=self.config.P_VENT,
            initial_mass=initial_mass,
            initial_temperature=self.config.Dormancy.INITIAL_TEMPERATURE,
            initial_solid_temperature=initial_solid_temp,
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            time_step=self.config.Dormancy.TIME_STEP,
            rtol=self.config.Solver.RTOL,
            atol=self.config.Solver.ATOL
        )

        # Note: Dormancy missions don't need the DischargeMission or RefuelMission classes
        # They're handled directly through IsochoricMissionParameters
        print("🔧 Heat flow data collector configured")

        # 3. Create thermal model (stops_model integration)
        self.thermal_model = StopsModelThermalModel(
            tank_volume=self.config.TANK_VOLUME,
            inner_surface_area=self.config.TANK_SURFACE_AREA,
            outer_surface_area=self.config.TANK_SURFACE_AREA * 1.025,
            inner_diameter=1.0,
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            ambient_htc=self.config.Thermal.HEAT_TRANSFER_COEFF,
            liner_mass=100.0,
            wall_mass=150.0
        )

        # 4. Configure solver
        solver_method = solver_method or self.config.Solver.PRIMARY_METHOD
        self.solver = self._create_solver(solver_method)

        # 5. Create dormancy mission with calculated parameters
        self.mission = DormancyMission.long_term_storage(
            duration=self.config.Dormancy.MISSION_DURATION,
            initial_mass=initial_mass,
            initial_temperature=self.config.Dormancy.INITIAL_TEMPERATURE,
            ambient_temperature=self.config.AMBIENT_TEMPERATURE
        )

        # Update mission parameters
        self.mission.parameters = mission_params

        # 6. Configure solver
        self.mission.integration_method = self.solver

        # 7. Create mission analysis (HFT framework integration)
        self.mission_analysis = IsochoricMissionAnalysis(
            self.mission,
            self.thermal_model
        )

        print(f"✅ Analysis setup complete with {solver_method} solver")

    def _create_solver(self, method: str):
        """Create solver instance based on method name"""
        solver_config = {
            'timestep': 10.0,  # Larger timestep for dormancy (10s)
            'rtol': self.config.Solver.RTOL,
            'atol': self.config.Solver.ATOL,
            'max_step': self.config.Solver.MAX_STEP * 10  # Larger steps for dormancy
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

    def _display_key_steps(self):
        """Display key analysis steps in stops_model format"""
        if not self.results or len(self.results.states) < 2:
            print("WARNING: Insufficient data for step display")
            return

        print(f"\nKey Analysis Steps (DORMANCY):")
        print("-" * 125)
        print("Step     |   Time   |   Mass   | Density  |   Temp   | Ts_solid | Pressure | Config   | Phase       | Notes")
        print("-" * 125)

        n_states = len(self.results.states)
        # Display start, key milestones, and end
        step_indices = [0, n_states//4, n_states//2, 3*n_states//4, n_states-1]

        def is_near_saturation(T, p_pa):
            """Check if state is near saturation (two-phase)"""
            try:
                p_sat = PropsSI("P", "T", T, "Q", 0, "hydrogen")
                tolerance = 1e-6
                return abs(p_pa - p_sat) < tolerance * p_sat
            except:
                return False

        for i, step_idx in enumerate(step_indices):
            if step_idx >= n_states:
                continue

            state = self.results.states[step_idx]
            t = step_idx * self.config.Dormancy.TIME_STEP
            m = state.fuel_mass
            rho = m / self.config.TANK_VOLUME
            T = state.temperature
            Ts = state.solid_temperature

            # Calculate pressure and phase
            try:
                p_pa = PropsSI("P", "T", T, "Dmass", rho, "hydrogen")
                p = p_pa / 1e5  # Convert to bar
                phase = "two-phase" if is_near_saturation(T, p_pa) else "single-phase"
            except:
                p = 0.0
                phase = "unknown"

            # Determine configuration (dormancy scenario)
            if p >= self.config.P_VENT/1e5:
                config = "Config C"  # Venting active
                notes_suffix = "VENTING"
            elif p <= self.config.P_MIN/1e5:
                config = "Config B"
                notes_suffix = ""
            else:
                config = "Config A"
                notes_suffix = ""

            # Add notes for key milestones
            notes = ""
            if i == 0:
                notes = "START"
            elif i == len(step_indices)-1:
                notes = "END"
            elif i == len(step_indices)//2:
                notes = "MIDPOINT"

            if notes and notes_suffix:
                notes = f"{notes} {notes_suffix}"
            elif notes_suffix:
                notes = notes_suffix

            print(f"{step_idx:4d}     | {t/3600:5.1f}h | {m:6.2f}kg | {rho:6.2f}kg/m³ | "
                  f"{T:6.2f}K | {Ts:6.2f}K | {p:6.2f}bar | {config:8s} | {phase:11s} | {notes}")

        print("-" * 125)

    def run_analysis(self, solver_method: str = None) -> dict:
        """
        Execute the dormancy analysis.

        Args:
            solver_method: Override solver method

        Returns:
            dict: Analysis results and metadata
        """
        print("🚀 Starting CCH2 dormancy analysis...")

        # Clear heat flow data from previous runs
        for key in heat_flow_data:
            heat_flow_data[key].clear()
        print(" Cleared heat flow data from previous runs")

        # Setup if not already done
        if self.mission_analysis is None:
            print("🔧 Mission analysis not initialized, running setup...")
            self.setup_analysis(solver_method)
        else:
            print(" Mission analysis already initialized")

        # Run analysis with timing
        start_time = time.time()

        try:
            print(" Starting mission analysis...")
            self.results = self.mission_analysis.run_analysis()
            print("✅ Mission analysis completed!")
            end_time = time.time()

            # Calculate performance metrics
            wall_time = end_time - start_time

            # Calculate performance metrics (matching other classes)
            times = np.array([i * self.config.Dormancy.TIME_STEP for i in range(len(self.results.states))])
            n_points = len(times)

            # Display key steps (similar to stops_model style)
            self._display_key_steps()

            # Physics validation
            initial_mass = self.results.states[0].fuel_mass
            final_mass = self.results.states[-1].fuel_mass
            mass_change = final_mass - initial_mass  # Should be negative due to venting
            expected_change = 0.0  # No expected mass flow in dormancy (venting is dynamic)

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

            print(f" Analysis completed successfully!")
            print(f"   Solver: {self.solver.method_name}")
            print(f"   Wall time: {wall_time:.3f}s")
            print(f"   Data points: {n_points}")
            print(f"   Mass change: {mass_change:.3f} kg (expected: negative due to venting)")
            print(f"   Temperature change: {self.analysis_metadata['temperature_change']:.2f} K")

            return self.analysis_metadata

        except Exception as e:
            wall_time = time.time() - start_time
            print(f"❌ Analysis failed after {wall_time:.3f}s: {e}")
            raise

    def validate_results(self) -> dict:
        """
        Validate the dormancy analysis results.

        Returns:
            dict: Validation check results
        """
        if self.results is None:
            raise ValueError("No analysis results available. Run analysis first.")

        validation = self._validate_results()

        print(f" Validation Results:")
        for test, passed in validation.items():
            status = "✅" if passed else "❌"
            print(f"   {test}: {status}")

        return validation

    def _validate_results(self):
        """
        Validate the dormancy analysis results.

        Returns:
            dict: Validation check results
        """
        if not self.results or len(self.results.states) == 0:
            return {'no_data': False}

        initial_state = self.results.states[0]
        final_state = self.results.states[-1]

        # Validation checks specific to dormancy scenario
        validation_checks = {}

        # 1. Check if Configuration C was activated (look for venting behavior)
        mass_decreased = final_state.fuel_mass < initial_state.fuel_mass
        validation_checks['venting_occurred'] = mass_decreased

        # 2. Check temperature physics (should increase due to ambient heat)
        temperature_increased = final_state.temperature > initial_state.temperature
        validation_checks['temperature_physics'] = temperature_increased

        # 3. Check density target achievement
        final_density = final_state.fuel_mass / self.config.TANK_VOLUME
        target_density = self.config.Dormancy.STOPPING_DENSITY
        density_close = abs(final_density - target_density) < 5.0  # Within 5 kg/m³
        validation_checks['density_target'] = density_close

        # 4. Check solver stability (no NaN or negative masses)
        all_masses = [state.fuel_mass for state in self.results.states]
        all_temperatures = [state.temperature for state in self.results.states]
        stability_check = all(m > 0 and not np.isnan(m) for m in all_masses)
        stability_check &= all(T > 0 and not np.isnan(T) for T in all_temperatures)
        validation_checks['solver_stability'] = stability_check

        return validation_checks


def main():
    """
    Main execution function for CCH2 verification analysis.

    This function demonstrates the complete workflow:
    1. Setup and run discharge analysis
    2. Setup and run refuel analysis
    3. Validate results
    4. Create plots including combined scenarios
    5. Compare solver methods
    """
    print("CCH2 VERIFICATION ANALYSIS")
    print("="*60)
    print("Compressed Cold Hydrogen discharge & refuel scenario verification")
    print("Using integrated HFT framework with new solver architecture")
    print("="*60)

    # 1. Run discharge analysis
    print("\n Running DISCHARGE analysis...")
    print(" Creating CCH2DischargeAnalysis instance...")
    discharge_analysis = CCH2DischargeAnalysis()
    print(" Instance created, starting analysis...")
    discharge_metadata = discharge_analysis.run_analysis()
    discharge_validation = discharge_analysis.validate_results()

    # Store discharge heat flow data before refuel analysis overwrites it
    discharge_heat_flow_data = {
        't': heat_flow_data['t'].copy(),
        'qdot_disch': heat_flow_data['qdot_disch'].copy(),
        'qdot_ohex': heat_flow_data['qdot_ohex'].copy(),
        'mdot_disch': heat_flow_data['mdot_disch'].copy(),
        'T': heat_flow_data['T'].copy(),
        'rho': heat_flow_data['rho'].copy()
    }

    # 2. Run refuel analysis
    print("\n Running REFUEL analysis...")
    print("Creating CCH2RefuelAnalysis instance...")
    refuel_analysis = CCH2RefuelAnalysis()
    print("Instance created, starting analysis...")
    refuel_metadata = refuel_analysis.run_analysis()
    refuel_validation = refuel_analysis.validate_results()

    # 3. Run dormancy analysis
    print("\n Running DORMANCY analysis with RK45 solver...")
    print("Creating CCH2DormancyAnalysis instance...")
    dormancy_analysis = CCH2DormancyAnalysis()
    print("Instance created, starting analysis...")
    dormancy_metadata = dormancy_analysis.run_analysis()
    dormancy_validation = dormancy_analysis.validate_results()

    # 4. Create consolidated plots (exactly 3 figures)
    print("\n Creating consolidated plots...")

    # Restore discharge heat flow data for HEX plotting
    for key in discharge_heat_flow_data:
        heat_flow_data[key] = discharge_heat_flow_data[key]

    create_consolidated_plots(discharge_analysis, refuel_analysis, dormancy_analysis)

    #show all returned plots
    plt.show()


    print(f"\n{'='*60}")
    print("CCH2 VERIFICATION ANALYSIS COMPLETED!")
    print(f"{'='*60}")

    return {
        'discharge': (discharge_analysis, discharge_metadata, discharge_validation),
        'refuel': (refuel_analysis, refuel_metadata, refuel_validation),
        'dormancy': (dormancy_analysis, dormancy_metadata, dormancy_validation),
    }


def create_consolidated_plots(discharge_analysis, refuel_analysis, dormancy_analysis):
    """
    Create exactly 3 consolidated plots:
    1. Combined pressure/temperature/mass vs time (discharge + refuel + dormancy)
    2. Combined temperature-density plot (discharge + refuel + dormancy)
    3. Heat exchanger requirements (discharge only)
    """
    print(" Creating consolidated plots...")

    # Extract data from all three analyses
    discharge_times = np.array([i * discharge_analysis.config.Discharge.TIME_STEP for i in range(len(discharge_analysis.results.states))])
    discharge_masses = np.array([state.fuel_mass for state in discharge_analysis.results.states])
    discharge_temperatures = np.array([state.temperature for state in discharge_analysis.results.states])
    discharge_densities = discharge_masses / discharge_analysis.config.TANK_VOLUME

    refuel_times = np.array([i * refuel_analysis.config.Refuel.TIME_STEP for i in range(len(refuel_analysis.results.states))])
    refuel_masses = np.array([state.fuel_mass for state in refuel_analysis.results.states])
    refuel_temperatures = np.array([state.temperature for state in refuel_analysis.results.states])
    refuel_densities = refuel_masses / refuel_analysis.config.TANK_VOLUME

    dormancy_times = np.array([i * dormancy_analysis.config.Dormancy.TIME_STEP for i in range(len(dormancy_analysis.results.states))])
    dormancy_masses = np.array([state.fuel_mass for state in dormancy_analysis.results.states])
    dormancy_temperatures = np.array([state.temperature for state in dormancy_analysis.results.states])
    dormancy_densities = dormancy_masses / dormancy_analysis.config.TANK_VOLUME

    # Calculate pressures for all three scenarios
    discharge_pressures = []
    refuel_pressures = []
    dormancy_pressures = []

    for i, state in enumerate(discharge_analysis.results.states):
        try:
            p = PropsSI("P", "T", state.temperature, "Dmass", discharge_densities[i], "hydrogen")
            discharge_pressures.append(p / 1e5)  # Convert to bar
        except:
            discharge_pressures.append(400.0)  # Fallback to initial pressure

    for i, state in enumerate(refuel_analysis.results.states):
        try:
            p = PropsSI("P", "T", state.temperature, "Dmass", refuel_densities[i], "hydrogen")
            refuel_pressures.append(p / 1e5)  # Convert to bar
        except:
            refuel_pressures.append(15.3)  # Fallback to initial pressure

    for i, state in enumerate(dormancy_analysis.results.states):
        try:
            p = PropsSI("P", "T", state.temperature, "Dmass", dormancy_densities[i], "hydrogen")
            dormancy_pressures.append(p / 1e5)  # Convert to bar
        except:
            dormancy_pressures.append(400.0)  # Fallback to initial pressure

    discharge_pressures = np.array(discharge_pressures)
    refuel_pressures = np.array(refuel_pressures)
    dormancy_pressures = np.array(dormancy_pressures)

    # Create SeabornPlotter instance
    plotter = SeabornPlotter(font="Cambria", palette="delft")

    # =================== FIGURE 1: Combined Time Series Using SeabornPlotter ===================
    print("\n" + "="*80)
    print(" Figure 1: Combined transient analysis (3 separate plots)...")

    # Import colors from SeabornPlotter
    from plotting.plot_style_sb import BORDEAUX, KONINGSBLAUW, BOSGROEN, DONKERGRIJS, ORANJE
    from plotting.plot_style_sb import configure_plot_style

    # Apply consistent styling
    configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

    # Create separate plots for each scenario using SeabornPlotter styling
    # 1. Discharge Analysis Plot
    print("  Creating discharge analysis subplot...")
    discharge_initial_conditions = {
        'pressure': discharge_pressures[0],
        'temperature': discharge_temperatures[0],
        'density': discharge_densities[0]
    }

    fig1 = plotter.plot_refuel_analysis(
        times=discharge_times,
        masses=discharge_masses,
        temperatures=discharge_temperatures,
        densities=discharge_densities,
        pressures=discharge_pressures,
        initial_conditions=discharge_initial_conditions,
        figsize=(14, 10)
    )
    if fig1:
        fig1.suptitle("CCH2 Discharge Analysis - Complete Scenario", fontsize=16, fontweight='bold')

    # 2. Refuel Analysis Plot
    print(" Creating refuel analysis subplot...")
    refuel_initial_conditions = {
        'pressure': refuel_pressures[0],
        'temperature': refuel_temperatures[0],
        'density': refuel_densities[0]
    }

    fig2 = plotter.plot_refuel_analysis(
        times=refuel_times,
        masses=refuel_masses,
        temperatures=refuel_temperatures,
        densities=refuel_densities,
        pressures=refuel_pressures,
        initial_conditions=refuel_initial_conditions,
        figsize=(14, 10)
    )

    # 3. Dormancy Analysis Plot
    print(" Creating dormancy analysis subplot...")
    dormancy_initial_conditions = {
        'pressure': dormancy_pressures[0],
        'temperature': dormancy_temperatures[0],
        'density': dormancy_densities[0]
    }

    fig3 = plotter.plot_refuel_analysis(
        times=dormancy_times,
        masses=dormancy_masses,
        temperatures=dormancy_temperatures,
        densities=dormancy_densities,
        pressures=dormancy_pressures,
        initial_conditions=dormancy_initial_conditions,
        figsize=(14, 10)
    )
    if fig3:
        fig3.suptitle("CCH2 Dormancy Analysis - Long Duration Scenario", fontsize=16, fontweight='bold')

    # =================== FIGURE 2: Temperature-Density Plot ===================
    print(" Figure 2: Combined temperature-density comparison...")
    scenario_data = {
        'discharge': {
            'temperatures': list(discharge_temperatures),
            'densities': list(discharge_densities),
            'pressures': list(discharge_pressures)
        },
        'refuel': {
            'temperatures': list(refuel_temperatures),
            'densities': list(refuel_densities),
            'pressures': list(refuel_pressures)
        },
        'dormancy': {
            'temperatures': list(dormancy_temperatures),
            'densities': list(dormancy_densities),
            'pressures': list(dormancy_pressures)
        }
    }

    fig2 = plotter.plot_density_temperature_combined(
        scenario_data=scenario_data,
        include_saturation_line=True,
        include_isobars=True,
        include_ref_data=True,
        figsize=(12, 8),
        temperature_range=(20, 80),
        density_range=(0, 80)
    )

    # =================== FIGURE 3: Heat Exchanger Requirements ===================
    print(" Figure 3: Heat exchanger requirements (discharge only)...")

    # Use the discharge heat flow data
    if len(heat_flow_data['t']) > 0:
        print(f"🔧 Using captured heat flow data: {len(heat_flow_data['t'])} data points")
        print(f"🔍 Time range: {min(heat_flow_data['t']):.1f}s to {max(heat_flow_data['t']):.1f}s ({max(heat_flow_data['t'])/3600:.1f} hours)")
        print(f"🔍 iHEX heat range: {min(heat_flow_data['qdot_disch']):.1f}W to {max(heat_flow_data['qdot_disch']):.1f}W")
        calculate_ohex_heat_requirements()
        print(f"🔍 oHEX heat range: {min(heat_flow_data['qdot_ohex']):.1f}W to {max(heat_flow_data['qdot_ohex']):.1f}W")
    else:
        print("⚠️ No heat flow data captured - plotting will show zeros")

    fig3 = plotter.plot_heat_exchanger_requirements(
        heat_flow_data,
        scenario_name="discharge",
        figsize=(10, 6)
    )

    print("✅ All consolidated plots created successfully!")
    return fig1, fig2, fig3


if __name__ == "__main__":
    # Run the verification analysis
    main()
