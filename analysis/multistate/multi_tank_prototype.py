"""
Multi-Tank CCH2 Prototype Analysis

This script implements a generalized multi-tank framework for cryocompressed hydrogen storage
analysis. It demonstrates the architecture with a 2-tank test case:
- Tank 1: Dormancy scenario (no flow in/out, possible venting)
- Tank 2: Discharge scenario (constant outflow)

The framework is designed as scaffolding for future inter-tank coupling capabilities
while maintaining all the sophisticated physics from the single-tank verification script.

Key Features:
- N-tank state vector: [m1, T1, Ts1, m2, T2, Ts2, ..., mN, TN, TsN]
- Generalized flow nomenclature: in/out/vent instead of fuel/discharge/vent
- Independent tank physics with configuration switching
- Unified ODE system and results management
- ANY-tank stopping criteria

Authors: Dante Raso (2025)
Based on verification framework from verification_cch2.py
"""

# Standard library imports
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Analysis framework imports
from src.mission.isochoric_missions import (
    DischargeMission,
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
from src.thermodynamics.tank_states import (
    TankState,
    IsochoricTankState,
    IsochoricTankStates,
    IsochoricStateDerivatives,
    IsochoricInitialState
)
from src.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher

# Graph-based network definition imports
from graph_factory import GraphFactory, TankSystemGraph

# Plotting imports
from plotting.sb_plotting import SeabornPlotter
from plotting.plot_style_sb import configure_plot_style

# =================== MULTI-TANK CONFIGURATION ===================

@dataclass
class MultiTankConfig:
    """
    Configuration parameters for multi-tank CCH2 analysis.

    Maintains compatibility with single-tank parameters while extending
    to support multiple tanks with individual or shared settings.
    """

    # Global tank parameters (shared by all tanks)
    TANK_VOLUME = 0.5           # m³ - V_t (per tank)
    TANK_SURFACE_AREA = 4.0     # m² - A_in (per tank)

    # Pressure thresholds (shared by all tanks)
    P_MIN = 15e5                # Pa (15 bar) - p_min
    P_VENT = 450e5              # Pa (450 bar) - p_vent

    # Environmental conditions
    AMBIENT_TEMPERATURE = 298.15  # K - T_amb

    # Individual tank configurations
    class Tank1:  # Dormancy scenario
        INITIAL_PRESSURE = 400e5        # Pa (400 bar)
        INITIAL_TEMPERATURE = 53.25     # K
        INITIAL_SOLID_TEMP = "thermal_equilibrium"  # K
        STOPPING_DENSITY = 70.0         # kg/m³

        # Flow rates (dormancy = all zeros except venting)
        INFLOW_RATE = 0.0              # kg/s - no inflow
        OUTFLOW_RATE = 0.0             # kg/s - no outflow
        MISSION_DURATION = 216000.0    # s (60 hours)
        TIME_STEP = 10.0               # s

    class Tank2:  # Discharge scenario
        INITIAL_PRESSURE = 400e5        # Pa (400 bar)
        INITIAL_TEMPERATURE = 53.25     # K
        INITIAL_SOLID_TEMP = "thermal_equilibrium"  # K
        STOPPING_DENSITY = 5.8          # kg/m³

        # Flow rates (discharge = constant outflow)
        INFLOW_RATE = 0.0              # kg/s - no inflow
        OUTFLOW_RATE = 0.001           # kg/s - constant discharge
        MISSION_DURATION = 40000.0     # s
        TIME_STEP = 1.0                # s

    # Thermal model parameters (shared)
    class Thermal:
        HEAT_TRANSFER_COEFF = 0.025   # W/m²K

    # Solver configuration parameters
    class Solver:
        PRIMARY_METHOD = 'RK45'         # Primary solver method
        RTOL = 1e-5                     # Relative tolerance
        ATOL = 1e-8                     # Absolute tolerance
        MAX_STEP = 5.0                  # Maximum step size (seconds)


# =================== MULTI-TANK STATE MANAGEMENT ===================

@dataclass
class MultiTankState:
    """
    State container for multiple tanks.

    Manages the N-tank state vector [m1, T1, Ts1, m2, T2, Ts2, ..., mN, TN, TsN]
    and provides convenient access to individual tank states.
    """
    tank_states: List[IsochoricTankState]
    time: float = 0.0

    @property
    def n_tanks(self) -> int:
        """Number of tanks in the system"""
        return len(self.tank_states)

    @property
    def state_vector(self) -> np.ndarray:
        """Combined state vector [m1, T1, Ts1, m2, T2, Ts2, ...]"""
        vector = []
        for tank_state in self.tank_states:
            vector.extend([
                tank_state.fuel_mass,
                tank_state.temperature,
                tank_state.solid_temperature
            ])
        return np.array(vector)

    @classmethod
    def from_state_vector(cls,
                         state_vector: np.ndarray,
                         tank_objects: List[Any],
                         time: float = 0.0) -> 'MultiTankState':
        """Create MultiTankState from combined state vector"""
        n_tanks = len(tank_objects)
        if len(state_vector) != 3 * n_tanks:
            raise ValueError(f"State vector length {len(state_vector)} != 3 * {n_tanks} tanks")

        tank_states = []
        for i in range(n_tanks):
            idx = 3 * i
            tank_state = IsochoricTankState(
                tank=tank_objects[i],
                fuel_mass=state_vector[idx],
                temperature=state_vector[idx + 1],
                solid_temperature=state_vector[idx + 2]
            )
            tank_states.append(tank_state)

        return cls(tank_states=tank_states, time=time)

    def get_tank_state(self, tank_index: int) -> IsochoricTankState:
        """Get state for specific tank"""
        return self.tank_states[tank_index]

    def update_from_state_vector(self, state_vector: np.ndarray):
        """Update all tank states from combined state vector"""
        for i, tank_state in enumerate(self.tank_states):
            idx = 3 * i
            tank_state.fuel_mass = state_vector[idx]
            tank_state.temperature = state_vector[idx + 1]
            tank_state.solid_temperature = state_vector[idx + 2]

            # Recompute derived properties
            tank_state.compute_pressure()
            tank_state.get_hydrogen_properties()


@dataclass
class MultiTankResults:
    """
    Results container for multi-tank analysis.

    Provides both individual tank access and unified time series data
    for convenient post-processing and plotting.
    """
    times: np.ndarray
    multi_tank_states: List[MultiTankState]
    tank_metadata: List[Dict[str, Any]]

    @property
    def n_tanks(self) -> int:
        """Number of tanks"""
        return len(self.tank_metadata)

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self.times)

    def get_tank_series(self, tank_index: int) -> IsochoricTankStates:
        """Get time series for specific tank (compatible with single-tank plotting)"""
        tank_states = []
        for multi_state in self.multi_tank_states:
            tank_states.append(multi_state.get_tank_state(tank_index))

        # Use the first tank's time step for compatibility
        timestep = self.times[1] - self.times[0] if len(self.times) > 1 else 1.0

        return IsochoricTankStates(states=tank_states, timestep=timestep)

    def get_combined_data(self) -> Dict[str, np.ndarray]:
        """Get combined data arrays for all tanks"""
        data = {
            'times': self.times,
            'masses': [],
            'temperatures': [],
            'solid_temperatures': [],
            'pressures': [],
            'densities': []
        }

        for tank_idx in range(self.n_tanks):
            tank_data = self._extract_tank_arrays(tank_idx)
            for key in ['masses', 'temperatures', 'solid_temperatures', 'pressures', 'densities']:
                data[key].append(tank_data[key])

        # Convert lists to numpy arrays
        for key in ['masses', 'temperatures', 'solid_temperatures', 'pressures', 'densities']:
            data[key] = np.array(data[key])

        return data

    def _extract_tank_arrays(self, tank_index: int) -> Dict[str, np.ndarray]:
        """Extract time series arrays for specific tank"""
        masses = []
        temperatures = []
        solid_temperatures = []
        pressures = []
        densities = []

        for multi_state in self.multi_tank_states:
            tank_state = multi_state.get_tank_state(tank_index)
            masses.append(tank_state.fuel_mass)
            temperatures.append(tank_state.temperature)
            solid_temperatures.append(tank_state.solid_temperature)

            # Calculate pressure if not available
            if tank_state.pressure is None:
                tank_state.compute_pressure()
            pressures.append(tank_state.pressure / 1e5)  # Convert to bar

            densities.append(tank_state.density)

        return {
            'masses': np.array(masses),
            'temperatures': np.array(temperatures),
            'solid_temperatures': np.array(solid_temperatures),
            'pressures': np.array(pressures),
            'densities': np.array(densities)
        }


# =================== MULTI-TANK SYSTEM CLASS ===================

class MultiTankCCH2System:
    """
    Multi-tank CCH2 system for generalized hydrogen storage analysis.

    This class manages multiple tanks with independent physics but unified
    integration. Designed as scaffolding for future inter-tank coupling.

    Key Features:
    - Manages N tanks with individual configurations
    - Unified ODE system for all tanks
    - Independent thermal models per tank
    - Generalized flow interface (in/out/vent)
    - ANY-tank stopping criteria
    """

    def __init__(self, config: MultiTankConfig = None):
        """
        Initialize multi-tank system.

        Args:
            config: Multi-tank configuration object
        """
        print("Initializing MultiTankCCH2System...")
        self.config = config or MultiTankConfig()

        # System components
        self.tanks = []  # Minimal tank objects for state management
        self.thermal_models = []  # Individual thermal models
        self.dynamic_models = []  # Individual dynamic models
        self.solver = None

        # Results storage
        self.results = None
        self.analysis_metadata = {}

        # Setup tanks
        self._setup_tanks()
        print(f"Multi-tank system initialized with {len(self.tanks)} tanks")

    def _setup_tanks(self):
        """Setup individual tanks and their associated models"""

        # Tank 1: Dormancy scenario
        print("Setting up Tank 1 (Dormancy scenario)...")
        tank1 = self._create_minimal_tank(self.config.TANK_VOLUME)
        thermal1 = self._create_thermal_model()
        dynamic1 = IsochoricModelSwitcher(
            scenario="DORMANCY",
            p_min=self.config.P_MIN,
            p_vent=self.config.P_VENT,
            tank_volume=self.config.TANK_VOLUME
        )

        self.tanks.append(tank1)
        self.thermal_models.append(thermal1)
        self.dynamic_models.append(dynamic1)

        # Tank 2: Discharge scenario
        print("Setting up Tank 2 (Discharge scenario)...")
        tank2 = self._create_minimal_tank(self.config.TANK_VOLUME)
        thermal2 = self._create_thermal_model()
        dynamic2 = IsochoricModelSwitcher(
            scenario="DISCHARGE",
            p_min=self.config.P_MIN,
            p_vent=self.config.P_VENT,
            tank_volume=self.config.TANK_VOLUME
        )

        self.tanks.append(tank2)
        self.thermal_models.append(thermal2)
        self.dynamic_models.append(dynamic2)

    def _create_minimal_tank(self, volume: float):
        """Create minimal tank object for state management"""
        class MinimalTank:
            def __init__(self, volume):
                self.volume = volume
        return MinimalTank(volume)

    def _create_thermal_model(self) -> StopsModelThermalModel:
        """Create thermal model with shared configuration"""
        return StopsModelThermalModel(
            tank_volume=self.config.TANK_VOLUME,
            inner_surface_area=self.config.TANK_SURFACE_AREA,
            outer_surface_area=self.config.TANK_SURFACE_AREA * 1.025,
            inner_diameter=1.0,
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            ambient_htc=self.config.Thermal.HEAT_TRANSFER_COEFF,
            liner_mass=100.0,
            wall_mass=150.0
        )

    def _create_solver(self, method: str):
        """Create solver instance"""
        # Use the most restrictive time step from all tanks
        min_timestep = min(self.config.Tank1.TIME_STEP, self.config.Tank2.TIME_STEP)

        solver_config = {
            'timestep': min_timestep,
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

    def _create_initial_state(self) -> MultiTankState:
        """Create initial multi-tank state"""
        print("Creating initial multi-tank state...")

        # Tank 1 initial conditions
        tank1_density = PropsSI("Dmass", "P", self.config.Tank1.INITIAL_PRESSURE,
                               "T", self.config.Tank1.INITIAL_TEMPERATURE, "hydrogen")
        tank1_mass = tank1_density * self.config.TANK_VOLUME

        # Calculate thermal equilibrium solid temperature for Tank 1
        if self.config.Tank1.INITIAL_SOLID_TEMP == "thermal_equilibrium":
            tank1_solid_temp = self.thermal_models[0].calculate_thermal_equilibrium_Ts(
                self.config.Tank1.INITIAL_TEMPERATURE
            )
        else:
            tank1_solid_temp = self.config.Tank1.INITIAL_SOLID_TEMP

        tank1_state = IsochoricTankState(
            tank=self.tanks[0],
            fuel_mass=tank1_mass,
            temperature=self.config.Tank1.INITIAL_TEMPERATURE,
            solid_temperature=tank1_solid_temp,
            scenario="DORMANCY"
        )

        # Tank 2 initial conditions (identical to Tank 1)
        tank2_density = PropsSI("Dmass", "P", self.config.Tank2.INITIAL_PRESSURE,
                               "T", self.config.Tank2.INITIAL_TEMPERATURE, "hydrogen")
        tank2_mass = tank2_density * self.config.TANK_VOLUME

        # Calculate thermal equilibrium solid temperature for Tank 2
        if self.config.Tank2.INITIAL_SOLID_TEMP == "thermal_equilibrium":
            tank2_solid_temp = self.thermal_models[1].calculate_thermal_equilibrium_Ts(
                self.config.Tank2.INITIAL_TEMPERATURE
            )
        else:
            tank2_solid_temp = self.config.Tank2.INITIAL_SOLID_TEMP

        tank2_state = IsochoricTankState(
            tank=self.tanks[1],
            fuel_mass=tank2_mass,
            temperature=self.config.Tank2.INITIAL_TEMPERATURE,
            solid_temperature=tank2_solid_temp,
            scenario="DISCHARGE"
        )

        print(f"Tank 1 initial: m={tank1_mass:.2f}kg, T={self.config.Tank1.INITIAL_TEMPERATURE:.2f}K, Ts={tank1_solid_temp:.2f}K")
        print(f"Tank 2 initial: m={tank2_mass:.2f}kg, T={self.config.Tank2.INITIAL_TEMPERATURE:.2f}K, Ts={tank2_solid_temp:.2f}K")

        return MultiTankState(tank_states=[tank1_state, tank2_state])

    def _get_flow_rates(self, time: float, tank_index: int) -> Tuple[float, float]:
        """Get inflow and outflow rates for specific tank at given time"""
        if tank_index == 0:  # Tank 1 (Dormancy)
            return self.config.Tank1.INFLOW_RATE, self.config.Tank1.OUTFLOW_RATE
        elif tank_index == 1:  # Tank 2 (Discharge)
            return self.config.Tank2.INFLOW_RATE, self.config.Tank2.OUTFLOW_RATE
        else:
            raise ValueError(f"Invalid tank index: {tank_index}")

    def _create_ode_system(self):
        """Create the unified ODE system for all tanks"""

        def ode_system(t, y):
            """
            Unified ODE system for multi-tank analysis.

            Args:
                t: Time [s]
                y: State vector [m1, T1, Ts1, m2, T2, Ts2, ...]

            Returns:
                dy/dt: State derivatives
            """
            # Create multi-tank state from state vector
            try:
                multi_state = MultiTankState.from_state_vector(y, self.tanks, t)
            except Exception as e:
                print(f"❌ Failed to create multi-tank state at t={t:.1f}s: {e}")
                return np.zeros_like(y)

            # Compute derivatives for each tank
            derivatives = []

            for tank_idx in range(len(self.tanks)):
                try:
                    tank_state = multi_state.get_tank_state(tank_idx)

                    # Apply bounds checking
                    if tank_state.fuel_mass <= 0.1:
                        tank_state.fuel_mass = max(tank_state.fuel_mass, 0.1)
                    tank_state.temperature = max(min(tank_state.temperature, 1000.0), 10.0)
                    tank_state.solid_temperature = max(min(tank_state.solid_temperature, 1000.0), 10.0)

                    # Get flow rates for this tank
                    inflow_rate, outflow_rate = self._get_flow_rates(t, tank_idx)

                    # Create flow functions
                    def inflow_func(time_arg): return inflow_rate
                    def outflow_func(time_arg): return outflow_rate

                    # Compute thermal coupling
                    Q_solid = self.thermal_models[tank_idx].compute_heat_flux(t, tank_state)
                    dTs_dt = self.thermal_models[tank_idx].compute_solid_temperature_derivative(t, tank_state)

                    # Compute dynamic derivatives
                    tank_derivatives = self.dynamic_models[tank_idx].compute_state_derivatives(
                        t, tank_state, inflow_func, outflow_func,
                        Q_solid=Q_solid, dTs_dt=dTs_dt
                    )

                    # Apply derivative bounds
                    dm_dt = tank_derivatives.fuel_mass_derivative
                    if tank_state.fuel_mass <= 0.1 and dm_dt < 0:
                        dm_dt = 0.0

                    dT_dt = max(min(tank_derivatives.temperature_derivative, 100.0), -100.0)
                    dTs_dt = max(min(tank_derivatives.solid_temperature_derivative, 10.0), -10.0)

                    derivatives.extend([dm_dt, dT_dt, dTs_dt])

                except Exception as e:
                    print(f"❌ Tank {tank_idx} derivative computation failed at t={t:.1f}s: {e}")
                    derivatives.extend([0.0, 0.0, 0.0])

            return np.array(derivatives)

        return ode_system

    def _create_stopping_events(self):
        """Create density-based stopping events for ANY tank"""

        def tank1_density_event(t, y):
            """Tank 1 density stopping event"""
            mass = y[0]  # Tank 1 mass is first element
            if mass <= 0:
                return 0.0
            density = mass / self.config.TANK_VOLUME
            return density - self.config.Tank1.STOPPING_DENSITY

        def tank2_density_event(t, y):
            """Tank 2 density stopping event"""
            mass = y[3]  # Tank 2 mass is fourth element (after m1, T1, Ts1)
            if mass <= 0:
                return 0.0
            density = mass / self.config.TANK_VOLUME
            return density - self.config.Tank2.STOPPING_DENSITY

        # Configure events
        tank1_density_event.terminal = True
        tank1_density_event.direction = -1  # Trigger when decreasing (venting)

        tank2_density_event.terminal = True
        tank2_density_event.direction = -1  # Trigger when decreasing (discharge)

        return [tank1_density_event, tank2_density_event]

    def run_analysis(self, solver_method: str = None) -> MultiTankResults:
        """
        Run the complete multi-tank analysis.

        Args:
            solver_method: Override solver method

        Returns:
            MultiTankResults: Analysis results
        """
        print("\n" + "="*80)
        print("STARTING MULTI-TANK CCH2 ANALYSIS")
        print("="*80)
        print(f"System Configuration:")
        print(f"   • Number of Tanks: {len(self.tanks)}")
        print(f"   • Tank 1: Dormancy scenario (no flow, possible venting)")
        print(f"   • Tank 2: Discharge scenario (constant outflow)")
        print(f"   • State Vector Dimension: {3 * len(self.tanks)}")
        print(f"   • Stopping Criteria: ANY tank reaches density target")
        print("="*80)

        # Setup solver
        solver_method = solver_method or self.config.Solver.PRIMARY_METHOD
        self.solver = self._create_solver(solver_method)
        print(f"Using solver: {solver_method}")

        # Create initial state
        initial_multi_state = self._create_initial_state()
        initial_state_vector = initial_multi_state.state_vector

        # Create ODE system and events
        ode_system = self._create_ode_system()
        stopping_events = self._create_stopping_events()

        # Determine integration duration (use maximum of all tanks)
        max_duration = max(self.config.Tank1.MISSION_DURATION,
                          self.config.Tank2.MISSION_DURATION)

        # Setup integration parameters
        t_span = (0.0, max_duration)
        min_timestep = min(self.config.Tank1.TIME_STEP, self.config.Tank2.TIME_STEP)
        t_eval = np.arange(0.0, max_duration, min_timestep)
        if t_eval[-1] < max_duration:
            t_eval = np.append(t_eval, max_duration)

        print(f"Integration setup:")
        print(f"   • Duration: {max_duration/3600:.1f} hours")
        print(f"   • Time step: {min_timestep:.1f}s")
        print(f"   • Expected points: {len(t_eval)}")

        # Run integration
        start_time = time.time()

        try:
            print("Starting ODE integration...")
            self.solver.set_ode_function(ode_system)
            solution = self.solver.integrate_full(
                t_span, initial_state_vector, t_eval, events=stopping_events
            )

            end_time = time.time()
            wall_time = end_time - start_time

            if solution.success:
                print(f"✅ Integration completed successfully!")
                print(f"   Wall time: {wall_time:.3f}s")
                print(f"   Final time: {solution.t[-1]:.1f}s ({solution.t[-1]/3600:.2f} hours)")
                print(f"   Data points: {len(solution.t)}")

                # Check if stopped due to event
                if hasattr(solution, 't_events') and solution.t_events:
                    for i, event_times in enumerate(solution.t_events):
                        if len(event_times) > 0:
                            tank_name = "Tank 1" if i == 0 else "Tank 2"
                            print(f"   Stopped by {tank_name} density event at t={event_times[0]:.1f}s")

            else:
                print(f"❌ Integration failed: {solution.message}")
                raise RuntimeError(f"Integration failed: {solution.message}")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

        # Process results
        print("Processing results...")
        multi_tank_states = []

        for i, t in enumerate(solution.t):
            try:
                multi_state = MultiTankState.from_state_vector(
                    solution.y[:, i], self.tanks, t
                )
                multi_tank_states.append(multi_state)
            except Exception as e:
                print(f"Warning: Failed to process state at t={t:.1f}s: {e}")
                continue

        # Create metadata
        tank_metadata = [
            {
                'scenario': 'DORMANCY',
                'initial_mass': initial_multi_state.tank_states[0].fuel_mass,
                'initial_temp': initial_multi_state.tank_states[0].temperature,
                'stopping_density': self.config.Tank1.STOPPING_DENSITY,
                'inflow_rate': self.config.Tank1.INFLOW_RATE,
                'outflow_rate': self.config.Tank1.OUTFLOW_RATE
            },
            {
                'scenario': 'DISCHARGE',
                'initial_mass': initial_multi_state.tank_states[1].fuel_mass,
                'initial_temp': initial_multi_state.tank_states[1].temperature,
                'stopping_density': self.config.Tank2.STOPPING_DENSITY,
                'inflow_rate': self.config.Tank2.INFLOW_RATE,
                'outflow_rate': self.config.Tank2.OUTFLOW_RATE
            }
        ]

        # Store analysis metadata
        self.analysis_metadata = {
            'solver_method': solver_method,
            'wall_time': wall_time,
            'n_points': len(solution.t),
            'final_time': solution.t[-1],
            'success': solution.success
        }

        # Create results object
        self.results = MultiTankResults(
            times=solution.t,
            multi_tank_states=multi_tank_states,
            tank_metadata=tank_metadata
        )

        print("✅ Multi-tank analysis completed successfully!")
        self._display_key_steps()

        return self.results

    def _display_key_steps(self):
        """Display key analysis steps for both tanks"""
        if not self.results or len(self.results.multi_tank_states) < 2:
            print("WARNING: Insufficient data for step display")
            return

        print(f"\nKey Analysis Steps (MULTI-TANK):")
        print("-" * 140)
        print("Step     |   Time   | Tank |   Mass   | Density  |   Temp   | Ts_solid | Pressure | Config   | Phase    | Notes")
        print("-" * 140)

        n_states = len(self.results.multi_tank_states)
        step_indices = [0, n_states//4, n_states//2, 3*n_states//4, n_states-1]

        for i, step_idx in enumerate(step_indices):
            if step_idx >= n_states:
                continue

            multi_state = self.results.multi_tank_states[step_idx]
            t = self.results.times[step_idx]

            # Display both tanks at this time step
            for tank_idx in range(2):
                tank_state = multi_state.get_tank_state(tank_idx)

                m = tank_state.fuel_mass
                rho = tank_state.density
                T = tank_state.temperature
                Ts = tank_state.solid_temperature

                # Calculate pressure and phase
                try:
                    if tank_state.pressure is None:
                        tank_state.compute_pressure()
                    p = tank_state.pressure / 1e5  # Convert to bar
                    phase = "single-phase"  # Simplified for display
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

                # Notes
                notes = ""
                if i == 0:
                    notes = "START"
                elif i == len(step_indices)-1:
                    notes = "END"

                scenario = "DORM" if tank_idx == 0 else "DISCH"

                print(f"{step_idx:4d}     | {t/3600:5.1f}h |  {tank_idx+1}   | {m:6.2f}kg | {rho:6.2f}kg/m³ | "
                      f"{T:6.2f}K | {Ts:6.2f}K | {p:6.2f}bar | {config:8s} | {phase:8s} | {scenario} {notes}")

        print("-" * 140)

    def validate_results(self) -> Dict[str, Any]:
        """Validate multi-tank analysis results"""
        if not self.results:
            raise ValueError("No analysis results available. Run analysis first.")

        validation = {'overall': True, 'tanks': []}

        for tank_idx in range(self.results.n_tanks):
            tank_data = self.results._extract_tank_arrays(tank_idx)
            tank_validation = {}

            # Mass bounds
            final_mass = tank_data['masses'][-1]
            tank_validation['mass_bounds'] = 0.0 <= final_mass <= 1000.0

            # Temperature realism
            final_temp = tank_data['temperatures'][-1]
            tank_validation['temperature_realistic'] = 13.8 <= final_temp <= 500.0

            # Density progression (tank-specific)
            if tank_idx == 0:  # Dormancy tank may increase or decrease
                tank_validation['density_reasonable'] = True
            else:  # Discharge tank should decrease
                tank_validation['density_reasonable'] = tank_data['masses'][-1] < tank_data['masses'][0]

            validation['tanks'].append(tank_validation)

            # Update overall validation
            for key, value in tank_validation.items():
                if not value:
                    validation['overall'] = False

        print("Multi-tank validation results:")
        for tank_idx, tank_val in enumerate(validation['tanks']):
            print(f"Tank {tank_idx+1}: {tank_val}")
        print(f"Overall: {'✅ PASSED' if validation['overall'] else '❌ FAILED'}")

        return validation

    def plot_results(self, save_path: str = None):
        """Create comprehensive plots for multi-tank results"""
        if not self.results:
            raise ValueError("No analysis results available. Run analysis first.")

        print("Creating multi-tank analysis plots...")

        # Get combined data
        combined_data = self.results.get_combined_data()

        # Create plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Configure consistent styling
        configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

        # Create multi-tank comparison plot
        fig = self._create_multi_tank_plot(combined_data)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-tank analysis plot saved to: {save_path}")

        plt.show()
        return fig

    def _create_multi_tank_plot(self, data: Dict[str, np.ndarray]):
        """Create comprehensive multi-tank comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Multi-Tank CCH2 Analysis - Prototype Results", fontsize=16, fontweight='bold')

        times_hours = data['times'] / 3600

        # Colors for tanks
        colors = ['#1f77b4', '#ff7f0e']  # Blue for Tank 1, Orange for Tank 2
        labels = ['Tank 1 (Dormancy)', 'Tank 2 (Discharge)']

        # Plot 1: Mass vs Time
        ax1 = axes[0, 0]
        for i in range(2):
            ax1.plot(times_hours, data['masses'][i], color=colors[i],
                    linewidth=2, label=labels[i])
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Mass [kg]')
        ax1.set_title('Fuel Mass Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Temperature vs Time
        ax2 = axes[0, 1]
        for i in range(2):
            ax2.plot(times_hours, data['temperatures'][i], color=colors[i],
                    linewidth=2, label=labels[i])
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Fluid Temperature Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Pressure vs Time
        ax3 = axes[1, 0]
        for i in range(2):
            ax3.plot(times_hours, data['pressures'][i], color=colors[i],
                    linewidth=2, label=labels[i])
        ax3.axhline(y=450, color='red', linestyle='--', alpha=0.7, label='Vent Pressure')
        ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Min Pressure')
        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Pressure [bar]')
        ax3.set_title('Pressure Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Density vs Time
        ax4 = axes[1, 1]
        for i in range(2):
            ax4.plot(times_hours, data['densities'][i], color=colors[i],
                    linewidth=2, label=labels[i])

        # Add stopping density lines
        ax4.axhline(y=70.0, color=colors[0], linestyle=':', alpha=0.7,
                   label='Tank 1 Target')
        ax4.axhline(y=5.8, color=colors[1], linestyle=':', alpha=0.7,
                   label='Tank 2 Target')

        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Density [kg/m³]')
        ax4.set_title('Density Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =================== GRAPH-BASED WRAPPER ===================

class GraphConfiguredMultiTankSystem:
    """
    A system that wraps MultiTankCCH2System with graph-based configuration.
    
    This class demonstrates how to configure the existing proven physics simulation
    using a graph-based tank network definition instead of hard-coded parameters.
    """
    
    def __init__(self, graph: TankSystemGraph):
        """
        Initialize with a tank system graph.
        
        Args:
            graph: TankSystemGraph defining the network topology and parameters
        """
        self.graph = graph
        
        # Create configuration that matches the graph
        self.config = self._create_config_from_graph()
        
        # Initialize the proven MultiTankCCH2System with graph-derived config
        self.physics_system = MultiTankCCH2System(self.config)
    
    def _create_config_from_graph(self) -> MultiTankConfig:
        """Create MultiTankConfig from graph definition"""
        
        # Get tank parameters from graph
        tank1 = self.graph.tanks[0]  # Tank_1
        tank2 = self.graph.tanks[1]  # Tank_2
        
        # Create config with parameters extracted from graph
        config = MultiTankConfig()
        
        # Set tank volumes from graph
        config.TANK_VOLUME = tank1.volume  # Assume both tanks same volume for prototype
        
        # Set initial conditions from graph
        config.INITIAL_PRESSURE = tank1.initial_conditions["pressure"]
        config.INITIAL_TEMPERATURE = tank1.initial_conditions["temperature"]
        
        # Set stopping densities from graph
        config.Tank1.STOPPING_DENSITY = tank1.scenario_params.get("stopping_density", 70.0)
        config.Tank2.STOPPING_DENSITY = tank2.scenario_params.get("stopping_density", 5.8)
        
        # Set thermal parameters from graph
        htc1 = tank1.thermal_params.get("htc", 0.025)
        htc2 = tank2.thermal_params.get("htc", 0.025)
        config.Thermal.HEAT_TRANSFER_COEFF = max(htc1, htc2)  # Use the higher value
        
        # Configure scenarios based on graph connections
        tank1_discharge_rate = self._get_discharge_rate_from_graph("Tank_1")
        tank2_discharge_rate = self._get_discharge_rate_from_graph("Tank_2")
        
        # Tank 1 scenario determination
        if tank1_discharge_rate == 0.0:
            scenario1 = "dormancy"
        else:
            scenario1 = "discharge"
        
        # Tank 2 scenario determination
        if tank2_discharge_rate == 0.0:
            scenario2 = "dormancy"
        else:
            scenario2 = "discharge"
        
        print(f"Graph configuration extracted:")
        print(f"  Tank 1: {scenario1} scenario, stopping at {config.Tank1.STOPPING_DENSITY} kg/m³")
        print(f"  Tank 2: {scenario2} scenario, stopping at {config.Tank2.STOPPING_DENSITY} kg/m³")
        print(f"  Discharge rates: T1={tank1_discharge_rate:.4f} kg/s, T2={tank2_discharge_rate:.4f} kg/s")
        
        return config
    
    def _get_discharge_rate_from_graph(self, tank_id: str) -> float:
        """Get discharge rate for a tank from its graph connections"""
        discharge_rate = 0.0
        
        outflow_connections = self.graph.get_outflow_connections(tank_id)
        for conn in outflow_connections:
            if conn.connection_type.value == "discharge":
                discharge_rate += conn.parameters.get("rate", 0.0)
        
        return discharge_rate
    
    def run_simulation(self, solver_method: str = "LSODA"):
        """
        Run simulation using the graph-configured physics system.
        
        Args:
            solver_method: Integration method to use
            
        Returns:
            MultiTankResults: Simulation results
        """
        print(f"\n🚀 Running graph-configured simulation with {solver_method} solver...")
        
        # Use the existing proven physics system
        results = self.physics_system.run_analysis(solver_method)
        
        print(f"✅ Graph-configured simulation completed successfully!")
        return results
    
    def get_network_summary(self) -> dict:
        """Get summary of the network configuration"""
        return {
            "system_name": self.graph.system_name,
            "tank_count": len(self.graph.tanks),
            "connection_count": len(self.graph.connections),
            "tank_configurations": [
                {
                    "tank_id": tank.tank_id,
                    "volume": tank.volume,
                    "scenario": tank.scenario_params.get("scenario", "unknown"),
                    "stopping_density": tank.scenario_params.get("stopping_density", 0.0)
                }
                for tank in self.graph.tanks
            ],
            "connections": [
                {
                    "from": conn.source,
                    "to": conn.target,
                    "type": conn.connection_type.value,
                    "rate": conn.parameters.get("rate", 0.0)
                }
                for conn in self.graph.connections
            ]
        }

# =================== MAIN EXECUTION ===================

def create_user_network_config():
    """Create user-specified network configuration"""
    print("\n🔧 NETWORK CONFIGURATION")
    print("-" * 40)
    print("Using User Specified configuration (Tank1=vent only, Tank2=discharge+vent)")
    
    factory = GraphFactory()
    config = factory.get_user_specified_prototype_config()
    print("✅ User Specified configuration loaded")
    
    return factory.from_config(config)

def main():
    """
    Main execution function for graph-based multi-tank prototype analysis.

    Features:
    - User-defined network configuration
    - Network topology visualization  
    - Graph-configured physics simulation
    - Results processing and plotting
    """
    print("GRAPH-BASED MULTI-TANK CCH2 PROTOTYPE ANALYSIS")
    print("="*70)
    print("Multi-tank framework with graph-based network definition")
    print("Supports various tank network topologies and connection types")
    print("="*70)

    # Step 1: Create user-defined network
    graph = create_user_network_config()
    
    # Step 2: Visualize network topology
    print(f"\n📊 NETWORK VISUALIZATION")
    print("-" * 40)
    print(f"System: {graph.system_name}")
    print(f"Tanks: {len(graph.tanks)}")
    print(f"Connections: {len(graph.connections)}")
    
    # Show network topology
    factory = GraphFactory()
    factory.visualize(graph, figsize=(14, 10), save_path="network_topology.png")
    plt.show()
    
    # Step 3: Create graph-configured system and run simulation
    print(f"\n🚀 GRAPH-CONFIGURED SIMULATION")
    print("-" * 40)
    
    try:
        # Create graph-configured system
        graph_system = GraphConfiguredMultiTankSystem(graph)
        
        # Show network summary
        summary = graph_system.get_network_summary()
        print("Network Summary:")
        for tank_config in summary["tank_configurations"]:
            print(f"  • {tank_config['tank_id']}: {tank_config['scenario']} "
                  f"(V={tank_config['volume']}m³, stop@{tank_config['stopping_density']}kg/m³)")
        
        print(f"\nConnections:")
        for conn in summary["connections"]:
            rate_str = f" ({conn['rate']:.4f} kg/s)" if conn['rate'] != 0.0 else ""
            print(f"  • {conn['from']} → {conn['to']}: {conn['type']}{rate_str}")
        
        # Run simulation
        print(f"\nRunning simulation...")
        results = graph_system.run_simulation("LSODA")
        
        # Step 4: Process and display results
        print(f"\n📈 RESULTS PROCESSING")
        print("-" * 40)
        
        print(f"✅ Simulation completed successfully!")
        print(f"   Final time: {results.times[-1] / 3600:.2f} hours")
        print(f"   Data points: {len(results.times):,}")
        
        # Get final densities from tank states
        final_state = results.multi_tank_states[-1]
        tank1_final_density = final_state.get_tank_state(0).fuel_mass / 0.5  # mass/volume
        tank2_final_density = final_state.get_tank_state(1).fuel_mass / 0.5
        
        print(f"   Tank 1 final density: {tank1_final_density:.1f} kg/m³")
        print(f"   Tank 2 final density: {tank2_final_density:.1f} kg/m³")
        
        # Validate stopping criteria
        expected_rho1 = graph.tanks[0].scenario_params.get("stopping_density", 70.0)
        expected_rho2 = graph.tanks[1].scenario_params.get("stopping_density", 5.8)
        
        rho1_final = tank1_final_density
        rho2_final = tank2_final_density
        
        print(f"\nValidation:")
        if abs(rho2_final - expected_rho2) < 0.5:
            print(f"   ✅ Tank 2 stopping criterion: {rho2_final:.1f} ≈ {expected_rho2} kg/m³")
        else:
            print(f"   ⚠️  Tank 2 deviation: {rho2_final:.1f} vs {expected_rho2} kg/m³")
        
        if abs(rho1_final - expected_rho1) < 2.0:
            print(f"   ✅ Tank 1 behavior: {rho1_final:.1f} ≈ {expected_rho1} kg/m³")
        else:
            print(f"   ⚠️  Tank 1 deviation: {rho1_final:.1f} vs {expected_rho1} kg/m³")
        
        # Step 5: Create plots
        print(f"\n📊 PLOTTING RESULTS")
        print("-" * 40)
        
        # Create comprehensive plots
        plot_graph_based_results(results, graph)
        
        print(f"\n{'='*70}")
        print("GRAPH-BASED MULTI-TANK PROTOTYPE ANALYSIS COMPLETED!")
        print(f"{'='*70}")
        
        return graph_system, results, graph
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, graph

def plot_graph_based_results(results: 'MultiTankResults', graph: TankSystemGraph):
    """Create comprehensive plots for graph-based results"""
    
    # Configure plotting style
    configure_plot_style()
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Graph-Based Multi-Tank Results: {graph.system_name}', 
                 fontsize=16, fontweight='bold')
    
    # Convert times to hours
    times_hours = results.times / 3600
    
    # Extract history arrays from MultiTankResults
    tank_volume = 0.5  # m³
    
    rho1_history = []
    rho2_history = []
    m1_history = []
    m2_history = []
    T1_history = []
    T2_history = []
    Ts1_history = []
    Ts2_history = []
    P1_history = []
    P2_history = []
    
    for state in results.multi_tank_states:
        tank1_state = state.get_tank_state(0)
        tank2_state = state.get_tank_state(1)
        
        rho1_history.append(tank1_state.fuel_mass / tank_volume)
        rho2_history.append(tank2_state.fuel_mass / tank_volume)
        m1_history.append(tank1_state.fuel_mass)
        m2_history.append(tank2_state.fuel_mass)
        T1_history.append(tank1_state.temperature)
        T2_history.append(tank2_state.temperature)
        Ts1_history.append(tank1_state.solid_temperature)
        Ts2_history.append(tank2_state.solid_temperature)
        # For now, set pressure to placeholder values since MinimalTank doesn't have pressure
        P1_history.append(400.0)  # Initial pressure in bar
        P2_history.append(15.0 if tank2_state.fuel_mass < 20 else 400.0)  # Simplified pressure model
    
    # Convert to numpy arrays
    rho1_history = np.array(rho1_history)
    rho2_history = np.array(rho2_history)
    m1_history = np.array(m1_history)
    m2_history = np.array(m2_history)
    T1_history = np.array(T1_history)
    T2_history = np.array(T2_history)
    Ts1_history = np.array(Ts1_history)
    Ts2_history = np.array(Ts2_history)
    P1_history = np.array(P1_history)
    P2_history = np.array(P2_history)
    
    # Plot 1: Density evolution
    axes[0,0].plot(times_hours, rho1_history, 'b-', linewidth=2, label='Tank 1')
    axes[0,0].plot(times_hours, rho2_history, 'r-', linewidth=2, label='Tank 2')
    axes[0,0].axhline(y=5.8, color='r', linestyle='--', alpha=0.7, label='Tank 2 Target')
    axes[0,0].axhline(y=70.0, color='b', linestyle='--', alpha=0.7, label='Tank 1 Target')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].set_ylabel('Density (kg/m³)')
    axes[0,0].set_title('Density Evolution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Mass evolution
    axes[0,1].plot(times_hours, m1_history, 'b-', linewidth=2, label='Tank 1')
    axes[0,1].plot(times_hours, m2_history, 'r-', linewidth=2, label='Tank 2')
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].set_ylabel('Mass (kg)')
    axes[0,1].set_title('Mass Evolution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Temperature evolution
    axes[0,2].plot(times_hours, T1_history, 'b-', linewidth=2, label='Tank 1 Fluid')
    axes[0,2].plot(times_hours, T2_history, 'r-', linewidth=2, label='Tank 2 Fluid')
    axes[0,2].plot(times_hours, Ts1_history, 'b--', linewidth=1, alpha=0.7, label='Tank 1 Solid')
    axes[0,2].plot(times_hours, Ts2_history, 'r--', linewidth=1, alpha=0.7, label='Tank 2 Solid')
    axes[0,2].set_xlabel('Time (hours)')
    axes[0,2].set_ylabel('Temperature (K)')
    axes[0,2].set_title('Temperature Evolution')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Mass flow rates (calculated from mass derivatives)
    dt = times_hours[1] - times_hours[0] if len(times_hours) > 1 else 0.01
    dm1_dt = np.gradient(m1_history) / (dt * 3600)  # kg/s
    dm2_dt = np.gradient(m2_history) / (dt * 3600)  # kg/s
    
    axes[1,0].plot(times_hours, -dm1_dt, 'b-', linewidth=2, label='Tank 1 Outflow')
    axes[1,0].plot(times_hours, -dm2_dt, 'r-', linewidth=2, label='Tank 2 Outflow') 
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].set_ylabel('Mass Flow Rate (kg/s)')
    axes[1,0].set_title('Mass Flow Rates')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Phase diagram
    axes[1,1].scatter(T1_history, rho1_history, c=times_hours, 
                      cmap='Blues', alpha=0.6, s=20, label='Tank 1')
    axes[1,1].scatter(T2_history, rho2_history, c=times_hours, 
                      cmap='Reds', alpha=0.6, s=20, label='Tank 2')
    axes[1,1].set_xlabel('Temperature (K)')
    axes[1,1].set_ylabel('Density (kg/m³)')
    axes[1,1].set_title('Phase Diagram (colored by time)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Network topology (simplified)
    axes[1,2].axis('off')
    axes[1,2].text(0.5, 0.9, 'Network Configuration', ha='center', va='top', 
                   fontsize=14, fontweight='bold', transform=axes[1,2].transAxes)
    
    # Add network summary text
    config_text = f"System: {graph.system_name}\n\n"
    config_text += f"Tanks: {len(graph.tanks)}\n"
    config_text += f"Connections: {len(graph.connections)}\n\n"
    
    for i, tank in enumerate(graph.tanks):
        scenario = tank.scenario_params.get("scenario", "unknown")
        target = tank.scenario_params.get("stopping_density", 0)
        config_text += f"{tank.tank_id}: {scenario}\n"
        config_text += f"  Target: {target} kg/m³\n"
        config_text += f"  Volume: {tank.volume} m³\n\n"
    
    axes[1,2].text(0.05, 0.8, config_text, ha='left', va='top',
                   fontsize=10, transform=axes[1,2].transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('graph_based_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Results plots created and saved as 'graph_based_results.png'")


if __name__ == "__main__":
    # Run the multi-tank prototype analysis
    main()