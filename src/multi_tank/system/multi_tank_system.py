"""
Multi-tank system core engine for hydrogen storage analysis.

This module provides the main MultiTankSystem class that manages multiple tanks
with unified integration and inter-tank coupling capabilities.
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

from src.tank_design.tank_shapes import SphericalTank
from src.thermodynamics.isochoric_thermal_model import StopsModelThermalModel
from src.multistep_methods.linear_multistep_methods import (
    LSODASolver, RK45Solver, RadauSolver, DOP853Solver, BDFSolver
)
from src.thermodynamics.tank_states import IsochoricTankState
from src.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher

from .state_management import MultiTankState, MultiTankResults
from ..coupling.inter_tank_coupling import PressureTriggeredValve


@dataclass
class MultiTankConfig:
    """Configuration parameters for multi-tank analysis."""

    # Global parameters
    AMBIENT_TEMPERATURE = 298.15  # K

    # Example tank configurations
    class HighPressureTank:
        INITIAL_PRESSURE = 700e5
        INITIAL_TEMPERATURE = 330.0
        INITIAL_SOLID_TEMP = "thermal_equilibrium"
        INITIAL_MASS = 25.0
        STOPPING_DENSITY = 5.0

        P_MIN = 100e5
        P_VENT = 800e5

        INFLOW_RATE = 0.0
        OUTFLOW_RATE = 0.0
        MISSION_DURATION = 216000.0
        TIME_STEP = 10.0

    class CryogenicTank:
        INITIAL_PRESSURE = 400e5
        INITIAL_TEMPERATURE = 53.25
        INITIAL_SOLID_TEMP = "thermal_equilibrium"
        STOPPING_DENSITY = 5.8

        P_MIN = 15e5
        P_VENT = 450e5

        INFLOW_RATE = 0.0
        OUTFLOW_RATE = 0.001
        MISSION_DURATION = 40000.0
        TIME_STEP = 1.0

    # Solver configuration
    class Solver:
        PRIMARY_METHOD = 'RK45'
        RTOL = 1e-5
        ATOL = 1e-8
        MAX_STEP = 5.0


class MultiTankSystem:
    """
    Multi-tank system for hydrogen storage analysis.

    Manages multiple tanks with independent physics but unified integration.
    Provides scaffolding for inter-tank coupling capabilities.

    Key Features:
    - N-tank management with individual configurations
    - Unified ODE system integration
    - Independent thermal models per tank
    - Generalized flow interface (inflow/outflow/venting)
    - Configurable stopping criteria
    """

    def __init__(self, config: MultiTankConfig = None, tank_geometries: List[SphericalTank] = None):
        """
        Initialize multi-tank system.

        Args:
            config: Multi-tank configuration object
            tank_geometries: List of SphericalTank objects for system tanks
        """
        print("Initializing MultiTankSystem...")
        self.config = config or MultiTankConfig()
        self.tank_geometries = tank_geometries or []

        # System components
        self.tanks = []
        self.thermal_models = []
        self.dynamic_models = []
        self.coupling_rules = []
        self.solver = None

        # Results storage
        self.results = None
        self.analysis_metadata = {}

        # Configuration
        self.enable_coupling = True

        # Setup system
        self._setup_tanks()
        self._setup_coupling_rules()

        print(f"Multi-tank system initialized with {len(self.tanks)} tanks and {len(self.coupling_rules)} coupling rules")

    def _setup_tanks(self):
        """Setup individual tanks with different geometries and pressure thresholds"""

        print(f"\n🏗️ TANK SETUP:")
        print(f"   High-Pressure Tank: P_VENT={self.config.HighPressureTank.P_VENT/1e5:.0f} bar, P_MIN={self.config.HighPressureTank.P_MIN/1e5:.0f} bar")
        print(f"   Cryogenic Tank: P_VENT={self.config.CryogenicTank.P_VENT/1e5:.0f} bar, P_MIN={self.config.CryogenicTank.P_MIN/1e5:.0f} bar")

        # Setup tanks from provided geometries
        for i, (tank_geom, tank_config, scenario) in enumerate([
            (self.tank_geometries[0] if len(self.tank_geometries) > 0 else None, self.config.HighPressureTank, "DORMANCY"),
            (self.tank_geometries[1] if len(self.tank_geometries) > 1 else None, self.config.CryogenicTank, "DISCHARGE")
        ]):
            tank_name = ["High-Pressure", "Cryogenic"][i]
            print(f"Setting up Tank {i+1} ({tank_name})...")

            if tank_geom is None:
                print(f"   ⚠️  No {tank_name} tank provided, creating default")
                tank_geom = self._create_minimal_tank(0.1)

            tank_properties = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}")
            thermal_model = self._create_thermal_model(tank_properties)
            dynamic_model = IsochoricModelSwitcher(
                scenario=scenario,
                p_min=tank_config.P_MIN,
                p_vent=tank_config.P_VENT,
                tank_volume=tank_properties['volume']
            )

            self.tanks.append(tank_geom)
            self.thermal_models.append(thermal_model)
            self.dynamic_models.append(dynamic_model)

            print(f"   ✅ Tank {i+1}: V={tank_properties['volume']:.4f} m³, A_in={tank_properties['inner_surface_area']:.3f} m²")

    def _setup_coupling_rules(self):
        """Setup inter-tank coupling rules"""
        if not self.enable_coupling:
            print("🔗 Coupling disabled - running in independent tank mode")
            return

        print(f"\n🔗 COUPLING SETUP:")

        # Pressure-triggered valve: High-pressure → Cryogenic tank
        # Opens at 50 bar, closes at 52 bar to trigger BEFORE Configuration B (15 bar)
        # High capacity valve to compete with ATR72 discharge rates (~0.1 kg/s)
        valve = PressureTriggeredValve(
            source_idx=0,
            target_idx=1,
            p_open=17e5,
            p_close=18e5,
            max_flow_rate=0.15,  # 150 g/s - much higher capacity
            orifice_diameter=0.01,  # 10mm diameter for higher flow
            coupling_id="HighPressure→Cryogenic"
        )

        self.coupling_rules.append(valve)

        print(f"   🔗 Valve: Opens at {valve.p_open/1e5:.0f} bar, closes at {valve.p_close/1e5:.0f} bar")
        print(f"      Max flow rate: {valve.max_flow_rate*1000:.1f} g/s")
        print(f"      Orifice diameter: {math.sqrt(4*valve.effective_area/math.pi)*1000:.1f} mm")
        print(f"   ✅ {len(self.coupling_rules)} coupling rules configured")

    def _create_minimal_tank(self, volume: float):
        """Create minimal tank object for state management"""
        class MinimalTank:
            def __init__(self, volume):
                self.volume = volume
        return MinimalTank(volume)

    def _get_tank_properties(self, tank: SphericalTank, tank_id: str = "Unknown"):
        """Calculate tank properties from SphericalTank geometry"""
        if tank is None:
            print(f"   ⚠️  No {tank_id} tank provided, using minimal defaults")
            return {
                'volume': 0.1,
                'inner_surface_area': 1.0,
                'outer_surface_area': 1.05,
                'inner_diameter': 0.6,
                'liner_mass': 50.0,
                'wall_mass': 100.0
            }

        # Calculate real properties from SphericalTank geometry

        # Geometry
        radius = tank.radius  # External radius (includes liner)
        inner_radius = radius - 0.005  # Subtract 5mm liner thickness
        volume = tank.volume

        # Surface areas (spherical: A = 4πr²)
        inner_surface_area = 4 * math.pi * inner_radius**2
        outer_surface_area = 4 * math.pi * radius**2
        inner_diameter = 2 * inner_radius

        # Material masses (simplified calculation)
        liner_material = tank.material  # Aluminum 6061-T6
        liner_thickness = 0.005  # 5mm

        # Liner mass: ρ × V_shell = ρ × 4π × r_avg × thickness × r_outer
        liner_avg_radius = radius - liner_thickness/2
        liner_volume = 4 * math.pi * liner_avg_radius**2 * liner_thickness
        liner_mass = liner_material.density * liner_volume

        # Wall mass (G10 composite, simplified)
        wall_thickness = 0.05  # 50mm insulation/structural
        wall_avg_radius = radius + wall_thickness/2
        wall_volume = 4 * math.pi * wall_avg_radius**2 * wall_thickness
        wall_mass = 1800.0 * wall_volume  # G10 density kg/m³

        properties = {
            'volume': volume,
            'inner_surface_area': inner_surface_area,
            'outer_surface_area': outer_surface_area,
            'inner_diameter': inner_diameter,
            'liner_mass': liner_mass,
            'wall_mass': wall_mass,
            'radius': radius,
            'inner_radius': inner_radius
        }

        print(f"   🔧 {tank_id} properties calculated:")
        print(f"      Volume: {volume:.6f} m³")
        print(f"      Inner surface area: {inner_surface_area:.4f} m²")
        print(f"      Radius: {radius:.3f} m")
        print(f"      Liner mass: {liner_mass:.2f} kg")
        print(f"      Wall mass: {wall_mass:.2f} kg")

        return properties

    def _create_thermal_model(self, tank_properties: dict) -> StopsModelThermalModel:
        """Create thermal model with tank-specific properties"""
        return StopsModelThermalModel(
            tank_volume=tank_properties['volume'],
            inner_surface_area=tank_properties['inner_surface_area'],
            outer_surface_area=tank_properties['outer_surface_area'],
            inner_diameter=tank_properties['inner_diameter'],
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            ambient_htc=0.025,  # Standard heat transfer coefficient
            liner_mass=tank_properties['liner_mass'],
            wall_mass=tank_properties['wall_mass']
        )

    def _create_solver(self, method: str):
        """Create solver instance"""
        # Use the most restrictive time step from all tanks
        min_timestep = min(self.config.HighPressureTank.TIME_STEP, self.config.CryogenicTank.TIME_STEP)

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
        """Create initial multi-tank state with individual tank properties"""
        print("Creating initial multi-tank state...")

        # Tank 1 initial conditions
        tank1_mass = self.config.HighPressureTank.INITIAL_MASS
        tank1_volume = self.tanks[0].volume
        tank1_density = tank1_mass / tank1_volume

        print(f"Tank 1: Using specified mass {tank1_mass:.2f} kg in {tank1_volume:.4f} m³ volume")
        print(f"        Resulting density: {tank1_density:.2f} kg/m³")

        # Calculate thermal equilibrium solid temperature for Tank 1
        if self.config.HighPressureTank.INITIAL_SOLID_TEMP == "thermal_equilibrium":
            tank1_solid_temp = self.thermal_models[0].calculate_thermal_equilibrium_Ts(
                self.config.HighPressureTank.INITIAL_TEMPERATURE
            )
        else:
            tank1_solid_temp = self.config.HighPressureTank.INITIAL_SOLID_TEMP

        tank1_state = IsochoricTankState(
            tank=self.tanks[0],
            fuel_mass=tank1_mass,
            temperature=self.config.HighPressureTank.INITIAL_TEMPERATURE,
            solid_temperature=tank1_solid_temp,
            scenario="DORMANCY"
        )

        # Tank 2 initial conditions (cryogenic tank calculated from P, T, V)
        tank2_volume = self.tanks[1].volume
        tank2_density = PropsSI("Dmass", "P", self.config.CryogenicTank.INITIAL_PRESSURE,
                               "T", self.config.CryogenicTank.INITIAL_TEMPERATURE, "hydrogen")
        tank2_mass = tank2_density * tank2_volume

        print(f"Tank 2: Calculated from P={self.config.CryogenicTank.INITIAL_PRESSURE/1e5:.0f} bar, T={self.config.CryogenicTank.INITIAL_TEMPERATURE:.1f} K")
        print(f"        Density: {tank2_density:.2f} kg/m³, Volume: {tank2_volume:.4f} m³")
        print(f"        Resulting mass: {tank2_mass:.2f} kg")

        # Calculate thermal equilibrium solid temperature for Tank 2
        if self.config.CryogenicTank.INITIAL_SOLID_TEMP == "thermal_equilibrium":
            tank2_solid_temp = self.thermal_models[1].calculate_thermal_equilibrium_Ts(
                self.config.CryogenicTank.INITIAL_TEMPERATURE
            )
        else:
            tank2_solid_temp = self.config.CryogenicTank.INITIAL_SOLID_TEMP

        tank2_state = IsochoricTankState(
            tank=self.tanks[1],
            fuel_mass=tank2_mass,
            temperature=self.config.CryogenicTank.INITIAL_TEMPERATURE,
            solid_temperature=tank2_solid_temp,
            scenario="DISCHARGE"
        )

        print(f"Initial conditions summary:")
        print(f"  Tank 1: m={tank1_mass:.2f}kg, T={self.config.HighPressureTank.INITIAL_TEMPERATURE:.1f}K, Ts={tank1_solid_temp:.1f}K")
        print(f"  Tank 2: m={tank2_mass:.2f}kg, T={self.config.CryogenicTank.INITIAL_TEMPERATURE:.1f}K, Ts={tank2_solid_temp:.1f}K")

        return MultiTankState(tank_states=[tank1_state, tank2_state])

    def _get_flow_rates(self, time: float, tank_index: int) -> Tuple[float, float]:
        """Get inflow and outflow rates for specific tank at given time"""
        if tank_index == 0:  # Tank 1 (High Pressure)
            return self.config.HighPressureTank.INFLOW_RATE, self.config.HighPressureTank.OUTFLOW_RATE
        elif tank_index == 1:  # Tank 2 (Cryogenic)
            return self.config.CryogenicTank.INFLOW_RATE, self.config.CryogenicTank.OUTFLOW_RATE
        else:
            raise ValueError(f"Invalid tank index: {tank_index}")

    def _create_ode_system(self):
        """Create the unified ODE system for all tanks with coupling"""

        # Storage for flow rates during integration
        self._flow_history = []

        def ode_system(t, y):
            """
            Unified ODE system for multi-tank analysis with inter-tank coupling.

            Args:
                t: Time [s]
                y: State vector [m1, T1, Ts1, m2, T2, Ts2, ...]

            Returns:
                dy/dt: State derivatives
            """
            # Create multi-tank state from state vector
            try:
                multi_state = MultiTankState.from_state_vector(y, self.tanks, t)
                tank_states = [multi_state.get_tank_state(i) for i in range(len(self.tanks))]
            except Exception as e:
                print(f"❌ Failed to create multi-tank state at t={t:.1f}s: {e}")
                return np.zeros_like(y)

            # Step 1: Evaluate coupling rules and calculate inter-tank flows
            coupling_flows = {i: 0.0 for i in range(len(self.tanks))}

            if self.enable_coupling and self.coupling_rules:
                try:
                    for rule in self.coupling_rules:
                        # Evaluate if coupling should be active
                        if rule.evaluate(t, tank_states):
                            # Calculate flow rate
                            flow_rate = rule.calculate_flow_rate(t, tank_states)

                            # Debug output for significant flows
                            if flow_rate > 1e-6:  # Only log flows > 1 mg/s
                                if int(t) % 300 == 0:  # Every 5 minutes
                                    p1 = tank_states[rule.source_idx].pressure / 1e5
                                    p2 = tank_states[rule.target_idx].pressure / 1e5
                                    print(f"t={t/3600:.2f}h: Coupling flow {rule.source_idx}→{rule.target_idx}: {flow_rate*1000:.2f} g/s (P1={p1:.1f}→P2={p2:.1f}bar)")

                            # Apply flow: source loses mass, target gains mass
                            coupling_flows[rule.source_idx] -= flow_rate
                            coupling_flows[rule.target_idx] += flow_rate

                except Exception as e:
                    print(f"❌ Coupling evaluation failed at t={t:.1f}s: {e}")
                    # Continue with zero coupling flows

            # Step 2: Compute derivatives for each tank with coupling contributions
            derivatives = []

            for tank_idx in range(len(self.tanks)):
                try:
                    tank_state = tank_states[tank_idx]

                    # Apply bounds checking
                    if tank_state.fuel_mass <= 0.1:
                        tank_state.fuel_mass = max(tank_state.fuel_mass, 0.1)
                    tank_state.temperature = max(min(tank_state.temperature, 1000.0), 10.0)
                    tank_state.solid_temperature = max(min(tank_state.solid_temperature, 1000.0), 10.0)

                    # Get base flow rates for this tank (external flows)
                    inflow_rate, outflow_rate = self._get_flow_rates(t, tank_idx)

                    # Get coupling contribution
                    coupling_dm_dt = coupling_flows[tank_idx]

                    # Store flow rates in tank state for later plotting
                    tank_state.inflow_rate = inflow_rate
                    tank_state.outflow_rate = outflow_rate
                    tank_state.vent_rate = 0.0  # TODO: Add venting logic
                    tank_state.coupling_inflow_rate = max(0.0, coupling_dm_dt)
                    tank_state.coupling_outflow_rate = max(0.0, -coupling_dm_dt)

                    # Store flow data for this tank and time step
                    if not hasattr(self, '_current_flow_data'):
                        self._current_flow_data = []

                    if len(self._current_flow_data) <= tank_idx:
                        self._current_flow_data.extend([{} for _ in range(tank_idx + 1 - len(self._current_flow_data))])

                    self._current_flow_data[tank_idx] = {
                        'inflow_rate': inflow_rate,
                        'outflow_rate': outflow_rate,
                        'vent_rate': 0.0,
                        'coupling_inflow_rate': max(0.0, coupling_dm_dt),
                        'coupling_outflow_rate': max(0.0, -coupling_dm_dt)
                    }

                    # Create flow functions
                    def inflow_func(time_arg): return inflow_rate
                    def outflow_func(time_arg): return outflow_rate

                    # Compute thermal coupling
                    Q_solid = self.thermal_models[tank_idx].compute_heat_flux(t, tank_state)
                    dTs_dt = self.thermal_models[tank_idx].compute_solid_temperature_derivative(t, tank_state)

                    # Create modified flow functions that include coupling effects
                    def coupled_inflow_func(time_arg):
                        base_inflow = inflow_rate
                        coupling_inflow = max(0.0, coupling_dm_dt)  # Only positive coupling flow is inflow
                        return base_inflow + coupling_inflow

                    def coupled_outflow_func(time_arg):
                        base_outflow = outflow_rate
                        coupling_outflow = max(0.0, -coupling_dm_dt)  # Only negative coupling flow is outflow
                        return base_outflow + coupling_outflow

                    # Compute tank dynamics with coupling flows included in temperature calculation
                    tank_derivatives = self.dynamic_models[tank_idx].compute_state_derivatives(
                        t, tank_state, coupled_inflow_func, coupled_outflow_func,
                        Q_solid=Q_solid, dTs_dt=dTs_dt
                    )

                    # Total mass derivative already includes coupling from the flow functions
                    dm_dt = tank_derivatives.fuel_mass_derivative

                    # Temperature derivatives now properly account for coupling flows
                    dT_dt = tank_derivatives.temperature_derivative
                    dTs_dt = tank_derivatives.solid_temperature_derivative

                    # Apply derivative bounds
                    if tank_state.fuel_mass <= 0.1 and dm_dt < 0:
                        dm_dt = 0.0

                    dT_dt = max(min(dT_dt, 100.0), -100.0)
                    dTs_dt = max(min(dTs_dt, 10.0), -10.0)

                    derivatives.extend([dm_dt, dT_dt, dTs_dt])

                except Exception as e:
                    print(f"❌ Tank {tank_idx} derivative computation failed at t={t:.1f}s: {e}")
                    derivatives.extend([0.0, 0.0, 0.0])

            # Store flow data for this time step
            if hasattr(self, '_current_flow_data'):
                self._flow_history.append({
                    'time': t,
                    'flow_data': self._current_flow_data.copy()
                })
                self._current_flow_data = []  # Reset for next evaluation

            return np.array(derivatives)

        return ode_system

    def _create_stopping_events(self):
        """Create stopping events: density-based AND time-based"""

        # Get individual tank volumes
        tank1_volume = self.tanks[0].volume
        tank2_volume = self.tanks[1].volume

        def tank1_density_event(t, y):
            """Tank 1 density stopping event"""
            mass = y[0]  # Tank 1 mass is first element
            if mass <= 0.1:  # Safety check for near-empty tank
                return -1.0  # Force termination
            density = mass / tank1_volume
            return density - self.config.HighPressureTank.STOPPING_DENSITY

        def tank2_density_event(t, y):
            """Tank 2 density stopping event"""
            mass = y[3]  # Tank 2 mass is fourth element (after m1, T1, Ts1)
            if mass <= 0.1:  # Safety check for near-empty tank
                return -1.0  # Force termination
            density = mass / tank2_volume
            return density - self.config.CryogenicTank.STOPPING_DENSITY

        def time_limit_event(t, y):
            """Time-based stopping event - stop after 6 hours for coupling tests"""
            max_time = 5 * 3600.0  # 5 hours in seconds
            return t - max_time

        def mass_safety_event(t, y):
            """Safety event: stop if any tank becomes too empty"""
            min_safe_mass = 0.5  # kg - minimum safe mass
            tank1_mass = y[0]
            tank2_mass = y[3]

            if tank1_mass < min_safe_mass or tank2_mass < min_safe_mass:
                return -1.0  # Force termination
            return 1.0  # Continue

        # Configure events
        tank1_density_event.terminal = True
        tank1_density_event.direction = -1  # Trigger when decreasing (venting)

        tank2_density_event.terminal = True
        tank2_density_event.direction = -1  # Trigger when decreasing (discharge)

        time_limit_event.terminal = True
        time_limit_event.direction = 1   # Trigger when time exceeds limit

        mass_safety_event.terminal = True
        mass_safety_event.direction = -1  # Trigger when masses get too low

        return [tank1_density_event, tank2_density_event, time_limit_event, mass_safety_event]

    def enable_tank_coupling(self, enable: bool = True):
        """Enable or disable inter-tank coupling for debugging"""
        self.enable_coupling = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🔗 Inter-tank coupling {status}")

    def get_coupling_status(self) -> Dict[str, Any]:
        """Get status of all coupling rules"""
        if not self.coupling_rules:
            return {"coupling_enabled": self.enable_coupling, "rules": []}

        rule_status = []
        for rule in self.coupling_rules:
            rule_status.append({
                "rule_id": rule.coupling_id,
                "source_idx": rule.source_idx,
                "target_idx": rule.target_idx,
                "is_active": rule.is_active,
                "type": type(rule).__name__
            })

        return {
            "coupling_enabled": self.enable_coupling,
            "n_rules": len(self.coupling_rules),
            "rules": rule_status
        }

    def run_analysis(self, solver_method: str = None) -> MultiTankResults:
        """
        Run the complete multi-tank analysis.

        Args:
            solver_method: Override solver method

        Returns:
            MultiTankResults: Analysis results
        """
        print("\n" + "="*80)
        print("STARTING MULTI-TANK CCH2 ANALYSIS WITH COUPLING")
        print("="*80)
        print(f"System Configuration:")
        print(f"   • Number of Tanks: {len(self.tanks)}")
        print(f"   • Tank 1: Dormancy scenario (no flow, possible venting)")
        print(f"   • Tank 2: Discharge scenario (constant outflow)")
        print(f"   • State Vector Dimension: {3 * len(self.tanks)}")
        print(f"   • Inter-tank Coupling: {'ENABLED' if self.enable_coupling else 'DISABLED'}")
        print(f"   • Coupling Rules: {len(self.coupling_rules)}")
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
        max_duration = max(self.config.HighPressureTank.MISSION_DURATION,
                          self.config.CryogenicTank.MISSION_DURATION)

        # Setup integration parameters
        t_span = (0.0, max_duration)
        min_timestep = min(self.config.HighPressureTank.TIME_STEP, self.config.CryogenicTank.TIME_STEP)
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
                    event_names = ["Tank 1 density", "Tank 2 density", "Time limit (5h)", "Mass safety"]
                    for i, event_times in enumerate(solution.t_events):
                        if len(event_times) > 0:
                            event_name = event_names[i] if i < len(event_names) else f"Event {i}"
                            print(f"   Stopped by {event_name} event at t={event_times[0]:.1f}s ({event_times[0]/3600:.2f}h)")

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
                # Find matching flow data for this time
                flow_data = None
                if hasattr(self, '_flow_history'):
                    # Find the closest flow data entry
                    closest_flow_entry = None
                    min_time_diff = float('inf')
                    for flow_entry in self._flow_history:
                        time_diff = abs(flow_entry['time'] - t)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_flow_entry = flow_entry

                    if closest_flow_entry and min_time_diff < 1.0:  # Within 1 second tolerance
                        flow_data = closest_flow_entry['flow_data']

                multi_state = MultiTankState.from_state_vector(
                    solution.y[:, i], self.tanks, t, flow_data
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
                'stopping_density': self.config.HighPressureTank.STOPPING_DENSITY,
                'inflow_rate': self.config.HighPressureTank.INFLOW_RATE,
                'outflow_rate': self.config.HighPressureTank.OUTFLOW_RATE
            },
            {
                'scenario': 'DISCHARGE',
                'initial_mass': initial_multi_state.tank_states[1].fuel_mass,
                'initial_temp': initial_multi_state.tank_states[1].temperature,
                'stopping_density': self.config.CryogenicTank.STOPPING_DENSITY,
                'inflow_rate': self.config.CryogenicTank.INFLOW_RATE,
                'outflow_rate': self.config.CryogenicTank.OUTFLOW_RATE
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

                # Determine configuration using tank-specific thresholds
                if tank_idx == 0:  # Tank 1 (High Pressure)
                    p_vent = self.config.HighPressureTank.P_VENT/1e5
                    p_min = self.config.HighPressureTank.P_MIN/1e5
                else:  # Tank 2 (Cryogenic)
                    p_vent = self.config.CryogenicTank.P_VENT/1e5
                    p_min = self.config.CryogenicTank.P_MIN/1e5

                if p >= p_vent:
                    config = "Config C"
                elif p <= p_min:
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

        # Configure consistent styling
        self._configure_plot_style()

        # Create multi-tank comparison plot
        fig = self._create_multi_tank_plot(combined_data)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-tank analysis plot saved to: {save_path}")

        return fig

    def _configure_plot_style(self):
        """Configure plot styling (simplified version)"""
        try:
            # Try to import and use the project's styling
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from plotting.plot_style_sb import configure_plot_style
            configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")
        except ImportError:
            # Fallback to basic matplotlib styling
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                plt.style.use('default')
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.size': 10,
                'axes.labelsize': 11,
                'axes.titlesize': 12,
                'legend.fontsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'figure.figsize': [12, 8]
            })

    def _create_multi_tank_plot(self, data: Dict[str, np.ndarray]):
        """Create comprehensive multi-tank comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Multi-Tank CCH2 Analysis - Prototype Results", fontsize=16, fontweight='bold')

        times_hours = data['times'] / 3600

        # Colors for tanks
        colors = ['#1f77b4', '#ff7f0e']  # Blue for Tank 1, Orange for Tank 2
        labels = ['Tank 1 (CH2)', 'Tank 2 (CCH2)']

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
        ax3.axhline(y=450, color='red', linestyle='--', alpha=0.7, label='CCH2 Vent Pressure')
        ax3.axhline(y=800, color='blue', linestyle='--', alpha=0.7, label='CH2 Vent Pressure')
        ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='CCH2 Min Pressure')
        ax3.axhline(y=100, color='cyan', linestyle='--', alpha=0.7, label='CH2 Min Pressure')
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
        ax4.axhline(y=5.0, color=colors[0], linestyle=':', alpha=0.7,
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