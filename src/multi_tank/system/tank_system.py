"""
General tank system core engine for hydrogen storage analysis.

This module provides the main TankSystem class that can manage any number of tanks
(from 1 to N) with unified integration and inter-tank coupling capabilities.
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

from src.tank_design.tank_shapes import SphericalTank
from src.thermodynamics.isochoric_thermal_model import StopsModelThermalModel
from ..solver import (
    LSODASolver, RK45Solver, RadauSolver, DOP853Solver, BDFSolver
)
from src.thermodynamics.tank_states import IsochoricTankState
from src.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher

from .state_management import MultiTankState, MultiTankResults
from ..coupling.inter_tank_coupling import PressureTriggeredValve


@dataclass
class TankConfig:
    """Configuration parameters for a single tank."""
    P_INIT: float           # Initial pressure [Pa]
    T_INIT: float           # Initial temperature [K]
    P_VENT: float          # Venting pressure [Pa]
    P_MIN: float           # Minimum pressure [Pa]
    MASS_INIT: Optional[float] = None  # Initial mass [kg] - calculated if None
    scenario: str = "DISCHARGE"        # Tank scenario type
    name: str = "Tank"                 # Tank identifier


@dataclass
class TankSystemConfig:
    """Configuration parameters for the tank system."""
    AMBIENT_TEMPERATURE: float = 298.15  # K
    MISSION_DURATION: float = 3600.0     # Mission duration in seconds (default 1 hour)
    tanks: List[TankConfig] = None       # Tank configurations
    mission_profile: Any = None          # Mission profile for flow calculations
    minimum_density: float = 5.8         # Stopping density [kg/m³]

    def __post_init__(self):
        if self.tanks is None:
            self.tanks = []


class TankSystem:
    """
    General tank system for hydrogen storage analysis.

    Can handle any number of tanks (1 to N) with:
    - Unified ODE integration across all tanks
    - Inter-tank coupling through pressure-triggered valves
    - Individual tank scenarios (discharge, refuel, dormancy)
    - Flexible configuration via TankSystemConfig
    """

    def __init__(self,
                 tank_geometries: List[SphericalTank],
                 config: TankSystemConfig,
                 coupling_rules: List[Dict] = None):
        """
        Initialize tank system.

        Args:
            tank_geometries: List of SphericalTank objects (geometry and materials)
            config: TankSystemConfig with tank parameters and scenarios
            coupling_rules: List of inter-tank coupling rules (optional)
        """
        print("Initializing TankSystem...")

        self.tank_geometries = tank_geometries
        self.config = config
        self.coupling_rules = coupling_rules or []

        # Validate configuration
        if len(tank_geometries) != len(config.tanks):
            raise ValueError(f"Number of tank geometries ({len(tank_geometries)}) must match "
                           f"number of tank configs ({len(config.tanks)})")

        # Initialize system components
        self.tanks = []
        self.thermal_models = []
        self.dynamic_models = []
        self.coupling_valves = []

        # Initialize caching system
        self._cached_tank_properties = {}
        self._properties_printed = set()  # Track which tanks have printed properties

        # Setup tanks and coupling
        self._setup_tanks()
        self._setup_coupling_rules()

        print(f"Tank system initialized with {len(self.tanks)} tanks and {len(self.coupling_valves)} coupling rules")

    def _setup_tanks(self):
        """Setup tanks from provided geometries and configurations."""
        print(f"\n🏗️ TANK SETUP:")

        # Print tank configurations
        for i, tank_config in enumerate(self.config.tanks):
            print(f"   Tank {i+1} ({tank_config.name}): P_VENT={tank_config.P_VENT/1e5:.0f} bar, P_MIN={tank_config.P_MIN/1e5:.0f} bar")

        # Setup each tank
        for i, (tank_geom, tank_config) in enumerate(zip(self.tank_geometries, self.config.tanks)):
            print(f"Setting up Tank {i+1} ({tank_config.name})...")

            tank_properties = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}")
            thermal_model = self._create_thermal_model(tank_properties)
            dynamic_model = IsochoricModelSwitcher(
                scenario=tank_config.scenario,
                p_min=tank_config.P_MIN,
                p_vent=tank_config.P_VENT,
                tank_volume=tank_properties['volume']
            )

            self.tanks.append(tank_geom)
            self.thermal_models.append(thermal_model)
            self.dynamic_models.append(dynamic_model)

            print(f"   ✅ Tank {i+1}: V={tank_properties['volume']:.4f} m³, A_in={tank_properties['inner_surface_area']:.3f} m²")

    def _setup_coupling_rules(self):
        """Setup inter-tank coupling based on rules."""
        print(f"\n🔗 COUPLING SETUP:")

        if not self.coupling_rules:
            print("   No coupling rules specified - tanks operate independently")
            # Still need to populate the tank properties cache
            for i, tank_geom in enumerate(self.tank_geometries):
                self._cached_tank_properties[i] = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}", tank_index=i)
            return

        for rule in self.coupling_rules:
            valve = PressureTriggeredValve(
                source_tank=rule.get('source_tank', 0),
                target_tank=rule.get('target_tank', 1),
                opening_pressure=rule.get('opening_pressure', 17e5),  # 17 bar default
                closing_pressure=rule.get('closing_pressure', 18e5),  # 18 bar default
                max_flow_rate=rule.get('max_flow_rate', 0.15),        # 150 g/s default
                orifice_diameter=rule.get('orifice_diameter', 0.01)   # 10 mm default
            )
            self.coupling_valves.append(valve)

            print(f"   🔗 Valve: Tank{rule.get('source_tank', 0)} → Tank{rule.get('target_tank', 1)}")
            print(f"      Opens at {rule.get('opening_pressure', 17e5)/1e5:.0f} bar, closes at {rule.get('closing_pressure', 18e5)/1e5:.0f} bar")
            print(f"      Max flow rate: {rule.get('max_flow_rate', 0.15)*1000:.1f} g/s")
            print(f"      Orifice diameter: {rule.get('orifice_diameter', 0.01)*1000:.1f} mm")

        print(f"   ✅ {len(self.coupling_valves)} coupling rules configured")

        # Cache tank properties to avoid repeated calculations during simulation
        for i, tank_geom in enumerate(self.tank_geometries):
            self._cached_tank_properties[i] = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}", tank_index=i)

    def _get_tank_properties(self, tank: SphericalTank, tank_id: str = "Unknown", tank_index: int = -1):
        """Calculate tank properties from SphericalTank geometry"""
        if tank is None:
            print(f"   ⚠️  No {tank_id} tank provided, using minimal defaults")
            return {
                'volume': 0.1,
                'inner_surface_area': 1.0,
                'outer_surface_area': 1.05,
                'inner_diameter': 0.6,
                'outer_diameter': 0.61,
                'liner_mass': 1.0,
                'wall_mass': 10.0
            }

        inner_radius = tank.radius  # Tank internal radius

        # Layer thicknesses (typical values for cryocompressed hydrogen tanks)
        thickness_liner = 0.005      # 5mm aluminum liner
        thickness_wall = 0.020       # 20mm composite wall
        thickness_insulation = 0.050 # 50mm insulation

        # Calculate radii at each layer
        liner_outer_radius = inner_radius + thickness_liner
        wall_outer_radius = liner_outer_radius + thickness_wall
        external_radius = wall_outer_radius + thickness_insulation

        # Calculate areas and volume
        volume = (4/3) * math.pi * inner_radius**3
        inner_surface_area = 4 * math.pi * inner_radius**2
        outer_surface_area = 4 * math.pi * external_radius**2

        # Calculate masses - use reasonable defaults if not specified
        # For a 0.5 m³ tank, typical structural mass ratios are:
        # Liner: ~2-3 kg/m³ of tank volume (aluminum liner)
        # Wall: ~5-8 kg/m³ of tank volume (composite pressure vessel)
        liner_mass = getattr(tank.sections[0], 'liner_mass', volume * 2.5) if hasattr(tank, 'sections') and tank.sections else volume * 2.5
        wall_mass = getattr(tank.sections[0], 'wall_mass', volume * 6.5) if hasattr(tank, 'sections') and tank.sections else volume * 6.5

        # Only print properties during initialization, not during simulation
        if hasattr(self, '_properties_printed') and tank_index not in self._properties_printed:
            print(f"   🔧 {tank_id} properties calculated:")
            print(f"      Volume: {volume:.6f} m³")
            print(f"      Inner surface area: {inner_surface_area:.4f} m²")
            print(f"      Outer surface area: {outer_surface_area:.4f} m²")
            print(f"      Inner radius: {inner_radius:.3f} m")
            print(f"      External radius: {external_radius:.3f} m")
            print(f"      Liner mass: {liner_mass:.2f} kg")
            print(f"      Wall mass: {wall_mass:.2f} kg")
            self._properties_printed.add(tank_index)

        return {
            'volume': volume,
            'inner_surface_area': inner_surface_area,
            'outer_surface_area': outer_surface_area,
            'inner_diameter': 2 * inner_radius,
            'outer_diameter': 2 * external_radius,
            'liner_mass': liner_mass,
            'wall_mass': wall_mass,
            'radius': inner_radius,
            'inner_radius': inner_radius
        }

    def _create_thermal_model(self, tank_properties: Dict[str, float]):
        """Create thermal model for tank"""
        return StopsModelThermalModel(
            tank_volume=tank_properties['volume'],
            inner_surface_area=tank_properties['inner_surface_area'],
            outer_surface_area=tank_properties['outer_surface_area'],
            inner_diameter=tank_properties['inner_diameter'],
            liner_mass=tank_properties['liner_mass'],
            wall_mass=tank_properties['wall_mass'],
            ambient_temperature=self.config.AMBIENT_TEMPERATURE
        )

    def create_initial_state(self) -> np.ndarray:
        """Create initial state vector for all tanks."""
        print("Creating initial tank system state...")

        state = []

        for i, tank_config in enumerate(self.config.tanks):
            print(f"Tank {i+1}: ", end="")

            if tank_config.MASS_INIT is not None:
                # Use specified mass
                m_init = tank_config.MASS_INIT
                tank_volume = self._cached_tank_properties[i]['volume']
                density_init = m_init / tank_volume
                print(f"Using specified mass {m_init:.2f} kg in {tank_volume:.4f} m³ volume")
                print(f"        Resulting density: {density_init:.2f} kg/m³")
            else:
                # Calculate mass from P, T conditions
                try:
                    density_init = PropsSI("D", "P", tank_config.P_INIT, "T", tank_config.T_INIT, "hydrogen")
                    tank_volume = self._cached_tank_properties[i]['volume']
                    m_init = density_init * tank_volume
                    print(f"Calculated from P={tank_config.P_INIT/1e5:.0f} bar, T={tank_config.T_INIT:.1f} K")
                    print(f"        Density: {density_init:.2f} kg/m³, Volume: {tank_volume:.4f} m³")
                    print(f"        Resulting mass: {m_init:.2f} kg")
                except Exception as e:
                    print(f"⚠️  Error calculating initial state for Tank {i+1}: {e}")
                    # Use default values
                    m_init = 10.0  # kg
                    print(f"        Using default mass: {m_init:.2f} kg")

            # Add tank state: [mass, temperature, solid_temperature]
            state.extend([m_init, tank_config.T_INIT, tank_config.T_INIT + 0.1])

        initial_state = np.array(state)
        print(f"Initial conditions summary:")
        for i in range(len(self.config.tanks)):
            idx = i * 3
            print(f"  Tank {i+1}: m={initial_state[idx]:.2f}kg, T={initial_state[idx+1]:.1f}K, Ts={initial_state[idx+2]:.1f}K")

        return initial_state

    def ode_system(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for the multi-tank system.

        State vector y = [m1, T1, Ts1, m2, T2, Ts2, ..., mN, TN, TsN]
        Each tank has 3 state variables: mass, temperature, solid temperature
        """
        try:
            # Check if we've already flagged stopping
            if hasattr(self, '_stop_requested') and self._stop_requested:
                return np.zeros(len(y))

            # Create multi-tank state from current state vector
            multi_state = MultiTankState.from_state_vector(y, self.tanks, t)

            # Initialize derivatives for all tanks
            n_tanks = len(self.tanks)
            dydt = np.zeros(3 * n_tanks)

            # Calculate coupling mass flows
            coupling_flows = self._calculate_coupling_flows(multi_state, t)

            # Calculate derivatives for each tank
            for i in range(n_tanks):
                tank_state = multi_state.tank_states[i]

                # Check stopping criteria: minimum density
                tank_volume = self._cached_tank_properties[i]['volume']
                current_density = tank_state.fuel_mass / tank_volume

                if current_density <= self.config.minimum_density:
                    if not hasattr(self, '_density_stop_printed'):
                        print(f"   ⏹️  Tank {i+1} reached minimum density: {current_density:.2f} ≤ {self.config.minimum_density:.2f} kg/m³")
                        self._density_stop_printed = True
                    # Set very small derivatives instead of zero to allow graceful termination
                    dydt = np.ones(len(y)) * 1e-12
                    return dydt

                # Prevent numerical instability near empty tank
                if tank_state.fuel_mass <= 1.0:  # 1 kg minimum
                    if not hasattr(self, '_empty_warn_printed'):
                        print(f"   ⚠️  Tank {i+1} approaching empty: {tank_state.fuel_mass:.2f} kg")
                        self._empty_warn_printed = True
                    tank_state.fuel_mass = max(tank_state.fuel_mass, 1.0)

                # Create mission-based flow functions
                def fuel_flow_func(time): return 0.0  # No fuel inflow for discharge scenario
                def discharge_flow_func(time): return self._get_discharge_flow_rate(time, i)  # Mission-based discharge

                # Add coupling contributions
                net_coupling_flow = sum(coupling_flows.get(f"to_{i}", [])) - sum(coupling_flows.get(f"from_{i}", []))

                # Get thermal derivatives from thermal model using correct interface
                Q_solid = self.thermal_models[i].compute_heat_flux(t, tank_state)
                dTs_dt = self.thermal_models[i].compute_solid_temperature_derivative(t, tank_state)

                # Get state derivatives from dynamic model
                state_derivatives = self.dynamic_models[i].compute_state_derivatives(
                    time=t,
                    state=tank_state,
                    fuel_flow_func=fuel_flow_func,
                    discharge_flow_func=discharge_flow_func,
                    Q_solid=Q_solid,
                    dTs_dt=dTs_dt,
                    Q_discharge=0.0
                )

                # Pack derivatives: [dm/dt, dT/dt, dTs/dt]
                idx = i * 3
                dydt[idx] = state_derivatives.fuel_mass_derivative + net_coupling_flow  # Mass derivative with coupling
                dydt[idx + 1] = state_derivatives.temperature_derivative  # Temperature derivative
                dydt[idx + 2] = state_derivatives.solid_temperature_derivative  # Solid temperature derivative

            return dydt

        except Exception as e:
            print(f"❌ Failed to create tank system state at t={t:.1f}s: {e}")
            # Return zero derivatives to prevent integration failure
            return np.zeros(len(y))

    def _calculate_coupling_flows(self, multi_state: MultiTankState, t: float) -> Dict[str, List[float]]:
        """Calculate mass flows between tanks due to coupling valves."""
        flows = {}

        for valve in self.coupling_valves:
            source_state = multi_state.tank_states[valve.source_tank]
            target_state = multi_state.tank_states[valve.target_tank]

            # Calculate valve flow
            flow_rate = valve.calculate_flow(source_state, target_state, t)

            # Record flows
            if f"from_{valve.source_tank}" not in flows:
                flows[f"from_{valve.source_tank}"] = []
            if f"to_{valve.target_tank}" not in flows:
                flows[f"to_{valve.target_tank}"] = []

            flows[f"from_{valve.source_tank}"].append(flow_rate)
            flows[f"to_{valve.target_tank}"].append(flow_rate)

        return flows

    def _get_discharge_flow_rate(self, time: float, tank_index: int) -> float:
        """
        Get discharge flow rate for specific tank at given time based on mission profile.

        Args:
            time: Current time [s]
            tank_index: Tank index (0-based)

        Returns:
            Discharge flow rate [kg/s] (positive = outflow)
        """
        if self.config.mission_profile is None:
            return 0.0

        # For single tank scenarios, tank 0 gets the mission discharge flow
        if tank_index != 0:
            return 0.0

        try:
            # Find which mission section we're in
            current_time = 0.0

            for section in self.config.mission_profile.sections:
                section_end_time = current_time + section.duration

                if time <= section_end_time:
                    # We're in this section - calculate flow rate
                    section_time = time - current_time

                    # Extract outflows from this section
                    for flow in section.fuel_flows:
                        if hasattr(flow, 'mass_flow'):
                            if isinstance(flow.mass_flow, list):
                                # Time-varying flow: linear interpolation
                                start_rate = abs(flow.mass_flow[0])
                                end_rate = abs(flow.mass_flow[-1])
                                progress = section_time / section.duration if section.duration > 0 else 0
                                return start_rate + (end_rate - start_rate) * progress
                            else:
                                # Constant flow rate
                                return abs(flow.mass_flow)

                    # No flows in this section
                    return 0.0

                current_time = section_end_time

            # Past mission end
            return 0.0

        except Exception as e:
            print(f"⚠️  Error calculating discharge flow at t={time:.1f}s: {e}")
            return 0.0

    def run_analysis(self, solver_method: str = "RK45") -> MultiTankResults:
        """
        Run complete tank system analysis.

        Args:
            solver_method: ODE solver to use ("RK45", "LSODA", "Radau", etc.)

        Returns:
            MultiTankResults with time series data
        """
        print(f"\n🚀 Starting TankSystem-based simulation...")
        print(f"   Analysis: {getattr(self.config, 'analysis_name', 'Tank System Analysis')}")
        print(f"   Tanks: {len(self.tanks)}")
        print(f"   Solver: {solver_method}")

        # Setup solver with default timestep
        default_timestep = 1.0  # seconds

        if solver_method == "LSODA":
            self.solver = LSODASolver(timestep=default_timestep)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "RK45":
            self.solver = RK45Solver(timestep=default_timestep)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "Radau":
            self.solver = RadauSolver(timestep=default_timestep)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "DOP853":
            self.solver = DOP853Solver(timestep=default_timestep)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "BDF":
            self.solver = BDFSolver(timestep=default_timestep)
            self.solver.set_ode_function(self.ode_system)
        else:
            raise ValueError(f"Unknown solver method: {solver_method}")

        # Create initial conditions
        y0 = self.create_initial_state()

        # Integration setup - use mission duration from configuration
        duration_seconds = self.config.MISSION_DURATION
        duration_hours = duration_seconds / 3600.0
        t_span = (0.0, duration_seconds)

        print(f"Integration setup:")
        print(f"   • Duration: {duration_hours:.3f} hours ({duration_seconds:.0f} seconds)")
        print(f"   • Time step: 1.0s")
        print(f"   • Expected points: {int(duration_seconds) + 1}")

        # Run integration
        print("Starting ODE integration...")
        start_time = time.time()

        # Reset any stopping flags
        if hasattr(self, '_stop_requested'):
            delattr(self, '_stop_requested')
        if hasattr(self, '_density_stop_printed'):
            delattr(self, '_density_stop_printed')
        if hasattr(self, '_empty_warn_printed'):
            delattr(self, '_empty_warn_printed')

        solution = self.solver.integrate_full(
            t_span=t_span,
            y0=y0,
            rtol=1e-6,
            atol=1e-9,
            max_step=10.0
        )

        elapsed_time = time.time() - start_time
        print(f"✅ Integration completed in {elapsed_time:.1f}s")
        print(f"   Final time: {solution.t[-1]/3600:.2f} hours")
        print(f"   Data points: {len(solution.t)}")

        # Convert solution to MultiTankState objects
        multi_tank_states = []
        for i in range(len(solution.t)):
            current_time = solution.t[i]

            # Create MultiTankState for this time step
            tank_states = []
            flow_data = []  # Store flow information for this time step

            for tank_idx in range(len(self.tanks)):
                state_idx = tank_idx * 3  # Each tank has 3 state variables: m, T, Ts

                # Create tank state from solution
                tank_state = IsochoricTankState(
                    tank=self.tanks[tank_idx],
                    fuel_mass=solution.y[state_idx, i],
                    temperature=solution.y[state_idx + 1, i],
                    solid_temperature=solution.y[state_idx + 2, i]
                )

                # Recompute derived properties
                tank_state.compute_pressure()
                tank_state.get_hydrogen_properties()

                # Calculate flow rates for this time step
                discharge_flow = self._get_discharge_flow_rate(current_time, tank_idx)
                inflow_rate = 0.0  # No inflow for discharge scenario
                outflow_rate = discharge_flow  # Discharge is outflow
                vent_rate = 0.0  # TODO: Implement venting logic if needed
                coupling_inflow_rate = 0.0  # TODO: Implement coupling flows
                coupling_outflow_rate = 0.0

                # Store flow data for this tank at this time step
                flow_data.append({
                    'inflow_rate': inflow_rate,
                    'outflow_rate': outflow_rate,
                    'vent_rate': vent_rate,
                    'coupling_inflow_rate': coupling_inflow_rate,
                    'coupling_outflow_rate': coupling_outflow_rate
                })

                tank_states.append(tank_state)

            # Create MultiTankState and manually set flow data
            multi_tank_state = MultiTankState(tank_states=tank_states)

            # Set flow rates on individual tank states
            for tank_idx, tank_state in enumerate(tank_states):
                flow_info = flow_data[tank_idx]
                tank_state.inflow_rate = flow_info['inflow_rate']
                tank_state.outflow_rate = flow_info['outflow_rate']
                tank_state.vent_rate = flow_info['vent_rate']
                tank_state.coupling_inflow_rate = flow_info['coupling_inflow_rate']
                tank_state.coupling_outflow_rate = flow_info['coupling_outflow_rate']
            multi_tank_states.append(multi_tank_state)

        # Return results
        return MultiTankResults(
            times=solution.t,
            multi_tank_states=multi_tank_states,
            tank_metadata=[
                {
                    'name': config.name,
                    'tank_id': i,
                    'scenario': config.scenario
                }
                for i, config in enumerate(self.config.tanks)
            ]
        )