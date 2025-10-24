"""
General tank system core engine for hydrogen storage analysis.

This module provides the main TankSystem class that can manage any number of tanks
(from 1 to N) with unified integration and inter-tank coupling capabilities.
"""

import math
import time
import numpy as np
# Matplotlib is not required for core simulation; avoid hard dependency at import time
from CoolProp.CoolProp import PropsSI
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from src.tank_design.tank_shapes import SphericalTank
from src.thermodynamics.isochoric_thermal_model import StopsModelThermalModel
from ..solver import (
    LSODASolver, RK45Solver, RadauSolver, DOP853Solver, BDFSolver
)
from src.thermodynamics.tank_states import IsochoricTankState
from src.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher

from .state_management import MultiTankState, MultiTankResults
from src.multi_tank.coupling.inter_tank_coupling import PressureTriggeredValve, OHEXExtractionCoupling
from src.fluids.flow_physics import FlowPhysics


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
    target_density: float = None         # Target density for refuel missions [kg/m³]

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
                 coupling_rules: List[Dict] = None,
                 scenario_config=None):
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
        self.scenario_config = scenario_config  # Store full scenario config for materials
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

        # Store coupling flows for post-processing (time -> {tank_idx: flow_rate})
        self.coupling_flow_history = {}

        # Initialize flow physics if configuration is available
        self.flow_physics = None
        if scenario_config and hasattr(scenario_config, 'config_dict'):
            # Check for 'physics' section (new schema) or 'flow_physics' (legacy)
            physics_config = scenario_config.config_dict.get('physics', scenario_config.config_dict.get('flow_physics'))
            if physics_config:
                try:
                    self.flow_physics = FlowPhysics(physics_config)
                    print(f"✓ Flow physics initialized from configuration")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to initialize flow physics from configuration. "
                        f"Check your 'physics:' section in the YAML file. Error: {e}"
                    ) from e
            elif self.coupling_rules:
                # Only raise error if we have coupling rules that need flow physics
                raise ValueError(
                    "No 'physics' section found in configuration, but coupling rules are defined. "
                    "Flow physics configuration is required for multi-tank systems with coupling. "
                    "Add a 'physics:' section to your YAML configuration file with orifice_flow, "
                    "safety_limits, and other flow physics parameters."
                )
            else:
                # No coupling rules, no physics needed
                print("   No coupling rules - flow physics not required")

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

            tank_properties = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}", tank_index=i)
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

    def _extract_mission_profile_data(self) -> dict:
        """Extract mission profile data from system configuration."""
        if not self.config.mission_profile:
            return {}

        try:
            # Import flow types
            from src.mission.mission_sections import OutFlow

            # Extract time points and flow rates from mission sections
            times = [0.0]
            flow_rates = [0.0]
            current_time = 0.0

            for section in self.config.mission_profile.sections:
                current_time += section.duration
                times.append(current_time)

                # Find OutFlow rate for this section
                section_flow_rate = 0.0
                for flow in section.fuel_flows:
                    if isinstance(flow, OutFlow):
                        # Get flow rate from mass_flow attribute (use absolute value for positive rate)
                        if isinstance(flow.mass_flow, list):
                            section_flow_rate = abs(flow.mass_flow[0])  # First value for start of section
                        else:
                            section_flow_rate = abs(flow.mass_flow)
                        break

                flow_rates.append(section_flow_rate)

            return {
                'time_s': times,
                'flow_rate_kg_s': flow_rates
            }

        except Exception as e:
            print(f"   ⚠️ Failed to extract mission profile: {e}")
            return {}

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
            rule_type = rule.get('type', 'pressure_triggered_valve')

            if rule_type == 'pressure_triggered_valve':
                # Create pressure-triggered valve
                source_idx = rule.get('source_tank', 0)
                target_idx = rule.get('target_tank', 1)

                valve = PressureTriggeredValve(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    p_open=rule.get('opening_pressure', 17e5),  # 17 bar default
                    p_close=rule.get('closing_pressure', 18e5),  # 18 bar default
                    max_flow_rate=rule.get('max_flow_rate', 0.005),       # 5 g/s default (realistic for pressurization)
                    orifice_diameter=rule.get('orifice_diameter', 0.001),  # 1 mm default (realistic for pressurization)
                    flow_physics=self.flow_physics,  # Pass configuration-driven flow physics
                    valve_time_constant_s=rule.get('valve_time_constant_s', 0.5)  # First-order dynamics time constant
                )

                # Add tank name attributes expected by _calculate_coupling_flows
                valve.source_tank = source_idx
                valve.target_tank = target_idx

                self.coupling_valves.append(valve)

                print(f"   🔗 Valve: Tank{source_idx+1} → Tank{target_idx+1}")
                print(f"      Opens at {rule.get('opening_pressure', 17e5)/1e5:.0f} bar, closes at {rule.get('closing_pressure', 18e5)/1e5:.0f} bar")
                print(f"      Max flow rate: {rule.get('max_flow_rate', 0.005)*1000:.1f} g/s")
                print(f"      Orifice diameter: {rule.get('orifice_diameter', 0.001)*1000:.1f} mm")

            elif rule_type == 'mission_adaptive_pressure_valve':
                # Create mission-adaptive pressure valve with dynamic thresholds
                from src.multi_tank.coupling.inter_tank_coupling import MissionAdaptivePressureValve

                source_idx = rule.get('source_tank', 0)
                target_idx = rule.get('target_tank', 1)

                # Get target tank configuration for minimum pressure
                target_tank_config = {}
                if target_idx < len(self.config.tanks):
                    target_tank_config = {
                        'minimum_pressure': self.config.tanks[target_idx].P_MIN
                    }

                valve = MissionAdaptivePressureValve(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    mission_profile=rule.get('mission_profile', {}),
                    discharge_piping=rule.get('piping', rule.get('discharge_piping', {})),
                    control_params=rule.get('control_parameters', rule.get('control_params', {})),
                    max_flow_rate=rule.get('flow_physics', {}).get('safety_limits', {}).get('max_flow_rate_kg_s', rule.get('max_flow_rate', 0.005)),
                    orifice_diameter=rule.get('flow_physics', {}).get('orifice_flow', {}).get('orifice_diameter_m', rule.get('orifice_diameter', 0.001)),
                    coupling_id=rule.get('coupling_id', 'mission_adaptive_pressure_valve'),
                    flow_physics=self.flow_physics,  # Pass configuration-driven flow physics
                    target_tank_config=target_tank_config  # Pass target tank configuration
                )

                # Add tank name attributes expected by _calculate_coupling_flows
                valve.source_tank = source_idx
                valve.target_tank = target_idx

                # If no hardcoded mission profile in coupling rule, set it from system config
                if not rule.get('mission_profile', {}) and self.config.mission_profile:
                    mission_data = self._extract_mission_profile_data()
                    if mission_data:
                        valve.set_mission_profile(mission_data)

                self.coupling_valves.append(valve)

                piping_params = rule.get('piping', rule.get('discharge_piping', {}))
                pipe_d = piping_params.get('diameter_m', 0.01)
                pipe_l = piping_params.get('length_m', 2.0)
                control_params = rule.get('control_parameters', rule.get('control_params', {}))
                margin = control_params.get('pressure_margin_bar', 1.0)
                control_interval = control_params.get('control_interval_s', 10.0)

                # Require explicit minimum pressure in config - no silent defaults
                if 'minimum_pressure' not in target_tank_config:
                    raise KeyError(
                        f"Coupling rule {rule.get('coupling_id', 'unknown')} requires "
                        f"'minimum_pressure' in target tank configuration. "
                        f"No default value provided - must be explicitly configured."
                    )
                target_min_pressure = target_tank_config['minimum_pressure'] / 1e5

                print(f"   🔗 Adaptive Valve: Tank{source_idx+1} → Tank{target_idx+1} (Mission-Adaptive)")
                print(f"      Dynamic thresholds based on real-time mission flow")
                print(f"      Discharge piping: {pipe_d*1000:.0f}mm × {pipe_l:.1f}m")
                print(f"      Pressure margin: {margin:.1f} bar")
                max_flow = rule.get('flow_physics', {}).get('safety_limits', {}).get('max_flow_rate_kg_s', rule.get('max_flow_rate', 0.05))
                orifice_d = rule.get('flow_physics', {}).get('orifice_flow', {}).get('orifice_diameter_m', rule.get('orifice_diameter', 0.001))
                print(f"      Max flow rate: {max_flow*1000:.0f} g/s")
                print(f"      Orifice diameter: {orifice_d*1000:.1f} mm")
                print(f"   Control system: Time-based ({control_interval:.1f}s intervals)")
                print(f"   Target tank minimum pressure: {target_min_pressure:.1f} bar")

            elif rule_type == 'pressure_governor':
                # New margin-free governor mode
                from src.multi_tank.coupling.inter_tank_coupling import PressureGovernorValve

                source_idx = rule.get('source_tank', 0)
                target_idx = rule.get('target_tank', 1)

                # Target tank config for minimum pressure (diagnostics)
                target_tank_config = {}
                if target_idx < len(self.config.tanks):
                    target_tank_config = {
                        'minimum_pressure': self.config.tanks[target_idx].P_MIN
                    }

                valve = PressureGovernorValve(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    mission_profile=rule.get('mission_profile', {}),
                    discharge_piping=rule.get('piping', rule.get('discharge_piping', {})),
                    control_params=rule.get('control_parameters', rule.get('control_params', {})),
                    max_flow_rate=rule.get('flow_physics', {}).get('safety_limits', {}).get('max_flow_rate_kg_s', rule.get('max_flow_rate', 0.005)),
                    orifice_diameter=rule.get('flow_physics', {}).get('orifice_flow', {}).get('orifice_diameter_m', rule.get('orifice_diameter', 0.001)),
                    coupling_id=rule.get('coupling_id', 'pressure_governor'),
                    flow_physics=self.flow_physics,
                    target_tank_config=target_tank_config
                )

                valve.source_tank = source_idx
                valve.target_tank = target_idx

                # Inject mission from system if not hardcoded in rule
                if not rule.get('mission_profile', {}) and self.config.mission_profile:
                    mission_data = self._extract_mission_profile_data()
                    if mission_data:
                        valve.set_mission_profile(mission_data)

                self.coupling_valves.append(valve)

                piping_params = rule.get('piping', rule.get('discharge_piping', {}))
                pipe_d = piping_params.get('diameter_m', 0.01)
                pipe_l = piping_params.get('length_m', 2.0)
                control_params = rule.get('control_parameters', rule.get('control_params', {}))
                control_interval = control_params.get('control_interval_s', 1.0)
                gain = control_params.get('pressure_gain_kg_s_per_bar', 0.05)
                print(f"   🔗 Pressure Governor: Tank{source_idx+1} → Tank{target_idx+1} (margin-free)")
                print(f"      Discharge piping: {pipe_d*1000:.0f}mm × {pipe_l:.1f}m")
                print(f"      Control cadence: {control_interval:.1f}s, gain: {gain:.3f} kg/s/bar")

            elif rule_type == 'feedforward_pressure_enforcer':
                # Create feedforward pressure enforcer (per-derivative-eval algebraic control)
                from src.multi_tank.coupling.inter_tank_coupling import FeedforwardPressureEnforcer

                source_idx = rule.get('source_tank', 0)
                target_idx = rule.get('target_tank', 1)

                # Get source tank configuration for minimum pressure (if available)
                source_tank_config = {}
                if source_idx < len(self.config.tanks):
                    source_tank_config = {
                        'minimum_pressure': self.config.tanks[source_idx].P_MIN
                    }

                # Get target tank configuration for minimum pressure (if available)
                target_tank_config = {}
                if target_idx < len(self.config.tanks):
                    target_tank_config = {
                        'minimum_pressure': self.config.tanks[target_idx].P_MIN
                    }

                valve = FeedforwardPressureEnforcer(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    mission_profile=rule.get('mission_profile', {}),
                    discharge_piping=rule.get('discharge_piping', {}),
                    control_params=rule.get('control_parameters', rule.get('control_params', {})),
                    max_flow_rate=rule.get('flow_physics', {}).get('safety_limits', {}).get('max_flow_rate_kg_s', rule.get('max_flow_rate', 0.005)),
                    orifice_diameter=rule.get('flow_physics', {}).get('orifice_flow', {}).get('orifice_diameter_m', rule.get('orifice_diameter', 0.001)),
                    coupling_id=rule.get('coupling_id', 'feedforward_pressure_enforcer'),
                    flow_physics=self.flow_physics,
                    source_tank_config=source_tank_config,
                    target_tank_config=target_tank_config
                )

                valve.source_tank = source_idx
                valve.target_tank = target_idx

                # Inject mission from system if not hardcoded in rule
                if not rule.get('mission_profile', {}) and self.config.mission_profile:
                    mission_data = self._extract_mission_profile_data()
                    if mission_data:
                        valve.set_mission_profile(mission_data)

                self.coupling_valves.append(valve)

                piping_params = rule.get('piping', rule.get('discharge_piping', {}))
                pipe_d = piping_params.get('diameter_m', 0.01)
                pipe_l = piping_params.get('length_m', 2.0)
                control_params = rule.get('control_parameters', rule.get('control_params', {}))
                bracket_margin = control_params.get('bracket_margin', 1.2)
                tol_bar = control_params.get('tol_pressure_bar', 0.02)
                max_iters = control_params.get('max_bisection_iters', 10)
                print(f"   🔗 Feedforward Enforcer: Tank{source_idx+1} → Tank{target_idx+1} (algebraic per-step)")
                print(f"      Discharge piping: {pipe_d*1000:.0f}mm × {pipe_l:.1f}m")
                print(f"      Bisection: margin×{bracket_margin:.2f}, tol {tol_bar:.3f} bar, iters {max_iters}")

            elif rule_type == 'mass_flow_pid_valve':
                # Create mass flow PID controlled valve with direct flow-to-flow control
                from src.multi_tank.coupling.inter_tank_coupling import MassFlowPIDControlledValve

                source_idx = rule.get('source_tank', 0)
                target_idx = rule.get('target_tank', 1)

                # Get target tank configuration for minimum pressure
                target_tank_config = {}
                if target_idx < len(self.config.tanks):
                    target_tank_config = {
                        'minimum_pressure': self.config.tanks[target_idx].P_MIN
                    }

                valve = MassFlowPIDControlledValve(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    mission_profile=rule.get('mission_profile', {}),
                    control_params=rule.get('control_parameters', rule.get('control_params', {})),
                    max_flow_rate=rule.get('flow_physics', {}).get('safety_limits', {}).get('max_flow_rate_kg_s', rule.get('max_flow_rate', 0.005)),
                    orifice_diameter=rule.get('flow_physics', {}).get('orifice_flow', {}).get('orifice_diameter_m', rule.get('orifice_diameter', 0.001)),
                    coupling_id=rule.get('coupling_id', 'mass_flow_pid_valve'),
                    flow_physics=self.flow_physics  # Pass configuration-driven flow physics
                )

                # Add tank name attributes expected by _calculate_coupling_flows
                valve.source_tank = source_idx
                valve.target_tank = target_idx

                # If no hardcoded mission profile in coupling rule, set it from system config
                if not rule.get('mission_profile', {}) and self.config.mission_profile:
                    mission_data = self._extract_mission_profile_data()
                    if mission_data:
                        valve.set_mission_profile(mission_data)

                self.coupling_valves.append(valve)

                piping_params = rule.get('piping', rule.get('discharge_piping', {}))
                pipe_d = piping_params.get('diameter_m', 0.01)
                pipe_l = piping_params.get('length_m', 2.0)
                control_params = rule.get('control_parameters', rule.get('control_params', {}))
                control_interval = control_params.get('control_interval_s', 10.0)

                kp = control_params.get('pid_kp', 1.0)
                ki = control_params.get('pid_ki', 0.1)
                kd = control_params.get('pid_kd', 0.01)

                print(f"   🔗 Flow-Matching Valve: Tank{source_idx+1} → Tank{target_idx+1} (Direct Flow Control)")
                print(f"      Flow-to-flow PID control (bypass pressure dynamics)")
                print(f"      PID gains: kp={kp}, ki={ki}, kd={kd}")
                print(f"      Discharge piping: {pipe_d*1000:.0f}mm × {pipe_l:.1f}m")
                max_flow = rule.get('flow_physics', {}).get('safety_limits', {}).get('max_flow_rate_kg_s', rule.get('max_flow_rate', 0.05))
                orifice_d = rule.get('flow_physics', {}).get('orifice_flow', {}).get('orifice_diameter_m', rule.get('orifice_diameter', 0.001))
                print(f"      Max flow rate: {max_flow*1000:.0f} g/s")
                print(f"      Orifice diameter: {orifice_d*1000:.1f} mm")

            elif rule_type == 'ohex_extraction':
                # Create OHEX extraction coupling
                from src.multi_tank.coupling.inter_tank_coupling import OHEXExtractionCoupling

                source_idx = rule.get('source_tank', 1)  # Default to tank 2 (LH2)

                # Require explicit min_extraction_pressure - no silent defaults
                if 'min_extraction_pressure' not in rule:
                    raise KeyError(
                        f"OHEX extraction coupling {rule.get('coupling_id', 'unknown')} requires "
                        f"'min_extraction_pressure' parameter. "
                        f"No default value provided - must be explicitly configured."
                    )

                ohex_coupling = OHEXExtractionCoupling(
                    source_idx=source_idx,
                    mission_profile=rule.get('mission_profile', {}),
                    min_extraction_pressure=rule['min_extraction_pressure'],
                    coupling_id=rule.get('coupling_id', 'ohex_extraction')
                )

                # Add tank name attributes expected by _calculate_coupling_flows
                ohex_coupling.source_tank = source_idx
                ohex_coupling.target_tank = -1  # No target tank for extraction

                self.coupling_valves.append(ohex_coupling)

                print(f"   🔗 OHEX Extraction: Tank{source_idx+1} → OHEX")
                print(f"      Min extraction pressure: {rule.get('min_extraction_pressure', 3.0e5)/1e5:.1f} bar")
                print(f"      Mission profile: {len(rule.get('mission_profile', {}).get('time_s', []))} time points")

            else:
                print(f"   ⚠️  Unsupported coupling rule type '{rule_type}' - skipping")
                continue

        print(f"   ✅ {len(self.coupling_valves)} coupling rules configured")

        # Cache tank properties to avoid repeated calculations during simulation
        for i, tank_geom in enumerate(self.tank_geometries):
            self._cached_tank_properties[i] = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}", tank_index=i)

    def _get_tank_properties(self, tank: SphericalTank, tank_id: str = "Unknown", tank_index: int = -1):
        """Calculate tank properties from SphericalTank geometry.

        Raises:
            ValueError: If tank is None (indicates configuration error)
        """
        if tank is None:
            raise ValueError(
                f"No tank object provided for {tank_id} (index {tank_index}). "
                f"This indicates a configuration error - cannot calculate geometry from None. "
                f"Check that tank was properly initialized before calling this method."
            )

        inner_radius = tank.radius  # Tank internal radius

        # Calculate thicknesses using proper netting analysis and NIST materials FROM CONFIG
        from src.tank_design.structural_models import CompositeCylinder, CompositeSphericalEndCap
        import math

        # Get materials from configuration - NO HARDCODED VALUES
        if not self.scenario_config:
            raise RuntimeError("No scenario configuration available - cannot determine materials")

        # Per-tank materials are now mandatory - the get_tank_materials method will handle the error

        # Get materials from config - support per-tank materials
        # Convert 0-based tank_index to 1-based tank_id for YAML configuration
        tank_id_yaml = tank_index + 1
        tank_materials = self.scenario_config.get_tank_materials(tank_id_yaml)
        liner_material = tank_materials.get('liner')
        composite_material = tank_materials.get('composite')

        if not liner_material or not composite_material:
            raise RuntimeError(f"Liner or composite material not found in configuration for tank {tank_id_yaml}")

        # Get thicknesses from configuration - support per-tank configuration
        # Use per-tank material config (convert 0-based index to 1-based tank ID)
        materials_config = self.scenario_config.get_tank_material_config(tank_id_yaml)
        thickness_liner = materials_config.get('liner', {}).get('thickness', None)
        if thickness_liner is None:
            raise RuntimeError("Liner thickness not specified in configuration")

        thickness_insulation = materials_config.get('insulation', {}).get('thickness', None)
        if thickness_insulation is None:
            raise RuntimeError("Insulation thickness not specified in configuration")

        # Get design parameters from configuration - NO HARDCODED VALUES
        safety_factor = materials_config.get('safety_margin', None)
        if safety_factor is None:
            raise RuntimeError("Safety margin not specified in configuration")

        # Get design pressure from tank configuration (use passed tank parameter)
        design_pressure = None
        working_pressure = None

        # First try to get from tank object if it has these attributes
        if hasattr(tank, 'venting_pressure'):
            design_pressure = float(tank.venting_pressure)
        if hasattr(tank, 'initial_pressure'):
            working_pressure = float(tank.initial_pressure)

        # If not found in tank object, try getting from scenario config tank geometries
        if design_pressure is None or working_pressure is None:
            tank_geometries = self.scenario_config.tank_geometries
            # Find the tank geometry by index (tank_index should match)
            if tank_index >= 0:
                tank_ids = list(tank_geometries.keys())
                if tank_index < len(tank_ids):
                    tank_id_key = tank_ids[tank_index]
                    tank_config = tank_geometries[tank_id_key]

                    if design_pressure is None and 'venting_pressure' in tank_config:
                        design_pressure = float(tank_config['venting_pressure'])
                    if working_pressure is None and 'initial_pressure' in tank_config:
                        working_pressure = float(tank_config['initial_pressure'])

        if design_pressure is None:
            raise RuntimeError("Design pressure (venting_pressure) not found in tank configuration")
        if working_pressure is None:
            raise RuntimeError("Working pressure (initial_pressure) not found in configuration")

        # Calculate composite wall thickness using netting analysis for cylindrical+spherical geometry
        # Tank section interface for structural model
        class TankSectionInterface:
            def __init__(self, radius, material):
                self.radius = radius
                self.material = material

        tank_section = TankSectionInterface(inner_radius, composite_material)
        cylinder_model = CompositeCylinder()
        endcap_model = CompositeSphericalEndCap()

        # Calculate thickness for both sections
        cylinder_thickness = cylinder_model.compute_thickness(tank_section, working_pressure)
        endcap_thickness = endcap_model.compute_thickness(tank_section, working_pressure)
        thickness_wall = max(cylinder_thickness, endcap_thickness)  # Governing thickness

        print(f"   🔧 Netting Analysis Results for {tank_id}:")
        print(f"      Radius: {inner_radius:.3f} m")
        print(f"      Design pressure: {design_pressure/1e5:.0f} bar, Working: {working_pressure/1e5:.0f} bar")
        print(f"      Safety factor: {safety_factor:.2f} (from config)")
        print(f"      Liner: {type(liner_material).__name__}, thickness: {thickness_liner*1000:.1f}mm")
        print(f"      Composite: {type(composite_material).__name__}, σ_failure = {composite_material.failure_stress/1e6:.0f} MPa")
        print(f"      Winding angle: {math.degrees(composite_material.winding_angle):.1f}°")
        print(f"      Cylinder thickness: {cylinder_thickness*1000:.1f} mm")
        print(f"      Endcap thickness: {endcap_thickness*1000:.1f} mm")
        print(f"      Governing wall thickness: {thickness_wall*1000:.1f} mm")

        # Calculate radii at each layer
        liner_outer_radius = inner_radius + thickness_liner
        wall_outer_radius = liner_outer_radius + thickness_wall
        external_radius = wall_outer_radius + thickness_insulation

        # Calculate areas and volume
        volume = (4/3) * math.pi * inner_radius**3
        inner_surface_area = 4 * math.pi * inner_radius**2
        outer_surface_area = 4 * math.pi * external_radius**2

        # Calculate masses using proper cylindrical+spherical geometry and NIST densities
        # Geometry: Cylindrical section (L = 3R) + 2 spherical endcaps (R)
        cylinder_length = 3 * inner_radius  # L/R = 3.0

        # Liner mass (aluminum, inner shell)
        liner_inner_radius = inner_radius
        liner_outer_radius = inner_radius + thickness_liner
        # Cylindrical section + spherical endcaps
        liner_cyl_volume = math.pi * (liner_outer_radius**2 - liner_inner_radius**2) * cylinder_length
        liner_sphere_volume = 2 * (4/3) * math.pi * (liner_outer_radius**3 - liner_inner_radius**3)
        liner_total_volume = liner_cyl_volume + liner_sphere_volume
        liner_mass = liner_material.density * liner_total_volume

        # Wall mass (composite, outer shell)
        wall_inner_radius = liner_outer_radius
        wall_outer_radius = liner_outer_radius + thickness_wall
        # Cylindrical section + spherical endcaps
        wall_cyl_volume = math.pi * (wall_outer_radius**2 - wall_inner_radius**2) * cylinder_length
        wall_sphere_volume = 2 * (4/3) * math.pi * (wall_outer_radius**3 - wall_inner_radius**3)
        wall_total_volume = wall_cyl_volume + wall_sphere_volume
        wall_mass = composite_material.density * wall_total_volume

        print(f"      Geometry: Cylinder (L={cylinder_length:.2f}m) + 2 spherical endcaps")
        print(f"      Liner mass: {liner_mass:.1f} kg (ρ={liner_material.density} kg/m³)")
        print(f"      Wall mass: {wall_mass:.1f} kg (ρ={composite_material.density} kg/m³)")

        # Update the properties to include calculated thickness for orchestrator
        thickness_info = {
            'wall_thickness': thickness_wall,
            'cylinder_thickness': cylinder_thickness,
            'endcap_thickness': endcap_thickness,
            'composite_density': composite_material.density,
            'cylindrical_section_length': cylinder_length
        }

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
            'inner_radius': inner_radius,
            # Add netting analysis results
            **thickness_info
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
        # --- Debug heartbeat and stuck-step detection ---
        import os
        debug_enabled = os.environ.get("H2_DEBUG", "0") == "1"

        # Detect potential stuck integration (solver calling RHS without advancing time)
        if not hasattr(self, '_last_ode_t'):
            self._last_ode_t = t
            self._ode_stuck_count = 0
            self._ode_heartbeat_last_t = -1e9
        dt_sim = t - getattr(self, '_last_ode_t', t)
        if dt_sim <= 1e-12:
            self._ode_stuck_count = getattr(self, '_ode_stuck_count', 0) + 1
            if debug_enabled and self._ode_stuck_count in (100, 1000, 5000):
                print(f"[ODE DEBUG] Potential stuck integration near t={t:.6f}s (no progress for {self._ode_stuck_count} calls)")
        else:
            self._ode_stuck_count = 0
        self._last_ode_t = t

        # Heartbeat: print state summary at start, then every 60s of simulated time or when debug enabled
        if debug_enabled or (t - getattr(self, '_ode_heartbeat_last_t', -1e9) >= 60.0) or (t <= 1.0):
            tank_summaries = []
            for i in range(len(self.tanks)):
                idx = i * 3
                m_i = float(y[idx])
                T_i = float(y[idx + 1])
                vol = getattr(self.tanks[i], 'volume', 1.0)
                density = m_i / max(vol, 1e-9)
                # Quick pressure estimate using saturation-aware helper to avoid warnings
                try:
                    from src.fluids.coolprop_safe import safe_pressure_from_T_rho
                    P_i = safe_pressure_from_T_rho(max(T_i, 1.0), max(density, 1e-9), "hydrogen")
                except Exception:
                    from CoolProp.CoolProp import PropsSI
                    P_i = float(PropsSI("P", "T", max(T_i, 1.0), "Dmass", max(density, 1e-9), "hydrogen"))
                tank_summaries.append((P_i / 1e5, m_i))
            press_str = ", ".join([f"P{i+1}={p:.2f}bar" for i, (p, _) in enumerate(tank_summaries)])
            mass_str = ", ".join([f"m{i+1}={m:.2f}kg" for i, (_, m) in enumerate(tank_summaries)])
            print(f"[ODE HB] t={t:.1f}s | {press_str} | {mass_str}")
            self._ode_heartbeat_last_t = t
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

                # Check stopping criteria: density-based stopping
                tank_volume = self._cached_tank_properties[i]['volume']
                current_density = tank_state.fuel_mass / tank_volume

                # Check for minimum density (discharge/dormancy missions)
                if current_density <= self.config.minimum_density:
                    if not hasattr(self, '_min_density_stop_printed'):
                        print(f"   ⏹️  Tank {i+1} reached minimum density: {current_density:.2f} ≤ {self.config.minimum_density:.2f} kg/m³")
                        self._min_density_stop_printed = True
                    # Set very small derivatives instead of zero to allow graceful termination
                    dydt = np.ones(len(y)) * 1e-12
                    return dydt

                # Check for target density (refuel missions) - just log, don't terminate here
                if self.config.target_density is not None and current_density >= self.config.target_density:
                    if not hasattr(self, '_target_density_stop_printed'):
                        print(f"   ⏹️  Tank {i+1} reached target density: {current_density:.2f} ≥ {self.config.target_density:.2f} kg/m³")
                        self._target_density_stop_printed = True
                        # Set a flag to indicate stopping should occur
                        self._stop_requested = True

                # Prevent numerical instability near empty tank
                if tank_state.fuel_mass <= 1.0:  # 1 kg minimum
                    if not hasattr(self, '_empty_warn_printed'):
                        print(f"   ⚠️  Tank {i+1} approaching empty: {tank_state.fuel_mass:.2f} kg")
                        self._empty_warn_printed = True
                    tank_state.fuel_mass = max(tank_state.fuel_mass, 1.0)

                # Get coupling contribution for this tank (simplified pattern)
                net_coupling_flow = coupling_flows[i]

                # Calculate coupling enthalpy (hydrogen coming from other tanks)
                coupling_enthalpy = 0.0
                if net_coupling_flow > 0:  # Receiving hydrogen from other tanks
                    # Find source tank with highest pressure (likely Tank 1 supplying Tank 2)
                    source_tank_idx = 0 if i == 1 else 1  # Simple 2-tank case
                    if source_tank_idx < len(multi_state.tank_states):
                        source_state = multi_state.tank_states[source_tank_idx]
                        from CoolProp.CoolProp import PropsSI
                        coupling_enthalpy = PropsSI("Hmass", "T", source_state.temperature,
                                                    "Dmass", source_state.density, "hydrogen")
                        # print(f"coupling enthalpy for Tank {i+1} from Tank {source_tank_idx+1}: {coupling_enthalpy:.1f} J/kg")


                # Helper to ensure scalar float from any flow source
                def _as_float(val: Any) -> float:
                    try:
                        # Handle list/tuple/ndarray by taking the first element
                        if isinstance(val, (list, tuple)):
                            return float(val[0])
                        import numpy as _np
                        if isinstance(val, _np.ndarray):
                            return float(val.flat[0])
                        return float(val)
                    except Exception:
                        return 0.0

                # Create flow functions that include both mission and coupling flows
                def fuel_flow_func(time):
                    mission_inflow = _as_float(self._get_inflow_rate(time, i))  # Mission-based inflow (refuel)
                    coupling_inflow = max(0.0, _as_float(net_coupling_flow))  # Positive coupling = inflow
                    return mission_inflow + coupling_inflow

                def discharge_flow_func(time):
                    mission_outflow = _as_float(self._get_outflow_rate(time, i))  # Mission-based outflow (discharge)
                    coupling_outflow = max(0.0, _as_float(-net_coupling_flow))  # Negative coupling = outflow
                    return mission_outflow + coupling_outflow

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
                    Q_discharge=0.0,
                    coupling_enthalpy=coupling_enthalpy  # Pass coupling enthalpy for proper energy balance
                )

                # Pack derivatives: [dm/dt, dT/dt, dTs/dt]
                # Note: coupling flows are now included in the flow functions, so temperature derivatives
                # automatically account for coupling enthalpy effects
                idx = i * 3
                dydt[idx] = state_derivatives.fuel_mass_derivative  # Mass derivative already includes coupling
                dydt[idx + 1] = state_derivatives.temperature_derivative  # Temperature derivative includes coupling enthalpy
                dydt[idx + 2] = state_derivatives.solid_temperature_derivative  # Solid temperature derivative

            # print some debug info (throttled to avoid performance issues)
            if hasattr(self, '_last_debug_time'):
                if t - self._last_debug_time > 100:  # Print every 100 seconds
                    if len(self.tanks) >= 2:
                        print(f"t={t:.1f}s, Tank1_mass={y[0]:.2f}kg, Tank2_mass={y[3]:.2f}kg")
                    else:
                        print(f"t={t:.1f}s, Tank1_mass={y[0]:.2f}kg")
                    self._last_debug_time = t
            else:
                self._last_debug_time = t

            return dydt

        except Exception as e:
            # Print richer diagnostics to locate type/units issues without crashing the solver
            import traceback
            print(f"❌ Failed to create tank system state at t={t:.6f}s: {e}")
            traceback.print_exc()
            # Return zero derivatives to prevent integration failure
            return np.zeros(len(y))

    def _calculate_coupling_flows(self, multi_state: MultiTankState, t: float) -> Dict[int, float]:
        """
        Calculate net mass flow rate for each tank due to coupling.

        Clean simple implementation - ALL coupling valves use same interface:
        flow_rate = valve.calculate_flow(source_state, target_state, t)
        """
        # Initialize coupling flows for all tanks (positive = inflow, negative = outflow)
        coupling_flows = {i: 0.0 for i in range(len(self.tanks))}

        for valve in self.coupling_valves:
            source_state = multi_state.tank_states[valve.source_tank]

            # Handle OHEX extraction (no target tank)
            if valve.target_tank == -1:
                flow_rate = valve.calculate_flow(source_state, None, t)
                if flow_rate > 0:
                    coupling_flows[valve.source_tank] -= flow_rate
            else:
                # Standard inter-tank coupling
                target_state = multi_state.tank_states[valve.target_tank]
                flow_rate = valve.calculate_flow(source_state, target_state, t)

                if flow_rate > 0:
                    coupling_flows[valve.source_tank] -= flow_rate
                    coupling_flows[valve.target_tank] += flow_rate

        # Store coupling flows for post-processing
        if any(abs(flow) > 1e-9 for flow in coupling_flows.values()):
            self.coupling_flow_history[t] = coupling_flows.copy()

        return coupling_flows

    def _estimate_coupling_flows_for_postprocessing(self, multi_state: MultiTankState, t: float) -> Dict[int, float]:
        """
        Estimate coupling flows for post-processing without relying on valve state machines.

        This method approximates coupling flows based on current pressure conditions,
        since valve state is not preserved between post-processing calls.
        """
        import math
        coupling_flows = {i: 0.0 for i in range(len(self.tanks))}

        for valve in self.coupling_valves:
            if hasattr(valve, 'activation_threshold') and hasattr(valve, 'deactivation_threshold'):
                # PressureTriggeredValve - estimate based on pressure conditions
                source_state = multi_state.tank_states[valve.source_tank]
                target_state = multi_state.tank_states[valve.target_tank]

                # Ensure pressures are computed
                if hasattr(source_state, 'compute_pressure'):
                    source_state.compute_pressure()
                if hasattr(target_state, 'compute_pressure'):
                    target_state.compute_pressure()

                target_pressure = target_state.pressure

                # Estimate if valve would be active based on pressure conditions
                # Use a more permissive threshold for post-processing estimation
                # (avoids strict hysteresis that requires state history)
                mid_threshold = (valve.activation_threshold + valve.deactivation_threshold) / 2

                if target_pressure <= mid_threshold and source_state.pressure > target_state.pressure:
                    # Estimate flow rate using simplified physics
                    P1, P2 = source_state.pressure, target_state.pressure
                    T1 = source_state.temperature
                    rho1 = source_state.fuel_mass / source_state.tank.volume

                    # Use valve's effective area and max flow rate
                    effective_area = valve.effective_area
                    max_flow_rate = valve.max_flow_rate

                    # Simplified choked flow calculation
                    gamma = 1.4  # Heat capacity ratio for hydrogen
                    R_specific = 4124  # J/(kg⋅K) specific gas constant for hydrogen
                    P_crit_ratio = (2/(gamma+1))**(gamma/(gamma-1))
                    discharge_coeff = 0.6

                    if P2/P1 < P_crit_ratio:
                        # Choked flow
                        sonic_velocity = math.sqrt(gamma * R_specific * T1)
                        flow_rate = discharge_coeff * effective_area * rho1 * sonic_velocity
                    else:
                        # Subsonic flow
                        velocity = math.sqrt(2 * (P1 - P2) / rho1)
                        flow_rate = discharge_coeff * effective_area * rho1 * velocity

                    # Apply limits
                    flow_rate = min(flow_rate, max_flow_rate)
                    max_safe_flow = 0.1 * source_state.fuel_mass
                    flow_rate = min(flow_rate, max_safe_flow)

                    # Apply to coupling flows
                    coupling_flows[valve.source_tank] -= flow_rate
                    coupling_flows[valve.target_tank] += flow_rate

            elif hasattr(valve, 'target_tank') and valve.target_tank == -1:
                # OHEX extraction coupling
                flow_rate = valve.calculate_flow(multi_state.tank_states[valve.source_tank], None, t)
                if flow_rate > 0:
                    coupling_flows[valve.source_tank] -= flow_rate

            elif hasattr(valve, 'update_time_based_control'):
                # MissionAdaptivePressureValve - special handling for time-based control
                from src.multi_tank.coupling.inter_tank_coupling import MissionAdaptivePressureValve
                if isinstance(valve, MissionAdaptivePressureValve):
                    source_state = multi_state.tank_states[valve.source_tank]
                    target_state = multi_state.tank_states[valve.target_tank]

                    # Ensure pressures are computed
                    if hasattr(source_state, 'compute_pressure'):
                        source_state.compute_pressure()
                    if hasattr(target_state, 'compute_pressure'):
                        target_state.compute_pressure()

                    # Get LH2 density
                    if hasattr(target_state, 'density'):
                        lh2_density = target_state.density
                    else:
                        lh2_density = target_state.fuel_mass / target_state.tank.volume

                    # Force update time-based control for post-processing
                    valve.last_control_update = -1.0  # Force update
                    valve.update_time_based_control(t, target_state.pressure, lh2_density)

                    # Now calculate flow rate
                    tank_states = [source_state if i == valve.source_tank else
                                 target_state if i == valve.target_tank else None
                                 for i in range(len(multi_state.tank_states))]

                    flow_rate = valve.calculate_flow_rate(t, multi_state.tank_states)

                    if flow_rate > 0:
                        coupling_flows[valve.source_tank] -= flow_rate
                        coupling_flows[valve.target_tank] += flow_rate

                # Check for MassFlowPIDControlledValve
                from src.multi_tank.coupling.inter_tank_coupling import MassFlowPIDControlledValve
                if isinstance(valve, MassFlowPIDControlledValve):
                    source_state = multi_state.tank_states[valve.source_tank]
                    target_state = multi_state.tank_states[valve.target_tank]

                    # Ensure pressures are computed
                    if hasattr(source_state, 'compute_pressure'):
                        source_state.compute_pressure()
                    if hasattr(target_state, 'compute_pressure'):
                        target_state.compute_pressure()

                    # Calculate flow rate using the flow-matching control
                    flow_rate = valve.calculate_flow_rate(t, multi_state.tank_states)

                    if flow_rate > 0:
                        coupling_flows[valve.source_tank] -= flow_rate
                        coupling_flows[valve.target_tank] += flow_rate
                else:
                    # Other coupling types with time-based control
                    source_state = multi_state.tank_states[valve.source_tank]
                    target_state = multi_state.tank_states[valve.target_tank] if valve.target_tank >= 0 else None
                    flow_rate = valve.calculate_flow(source_state, target_state, t)

                    if flow_rate > 0:
                        coupling_flows[valve.source_tank] -= flow_rate
                        if valve.target_tank >= 0:
                            coupling_flows[valve.target_tank] += flow_rate

            else:
                # Other coupling types - try to use their calculate_flow method
                source_state = multi_state.tank_states[valve.source_tank]
                target_state = multi_state.tank_states[valve.target_tank] if valve.target_tank >= 0 else None
                flow_rate = valve.calculate_flow(source_state, target_state, t)

                if flow_rate > 0:
                    coupling_flows[valve.source_tank] -= flow_rate
                    if valve.target_tank >= 0:
                        coupling_flows[valve.target_tank] += flow_rate

        return coupling_flows

    def _calculate_coupling_flows_stateless(self, multi_state: MultiTankState, t: float) -> Dict[int, float]:
        """Calculate coupling flows using stateless pressure evaluation for post-processing."""
        coupling_flows = {i: 0.0 for i in range(len(self.tanks))}

        for valve in self.coupling_valves:
            # Only handle PressureTriggeredValve for now
            if not hasattr(valve, 'activation_threshold'):
                continue

            source_state = multi_state.tank_states[valve.source_tank]
            target_state = multi_state.tank_states[valve.target_tank]

            # Ensure pressures are computed
            if hasattr(source_state, 'compute_pressure'):
                source_state.compute_pressure()
            if hasattr(target_state, 'compute_pressure'):
                target_state.compute_pressure()

            # Check if valve should be active based on pressure conditions
            # For post-processing, we use a simplified logic without hysteresis state
            target_pressure = target_state.pressure
            source_pressure = source_state.pressure

            # Debug pressure conditions
            if t < 100:  # Debug early times
                print(f"    Valve T{valve.source_tank}→T{valve.target_tank}: P_target={target_pressure/1e5:.1f}bar, P_source={source_pressure/1e5:.1f}bar")
                print(f"    Activation_threshold={valve.activation_threshold/1e5:.1f}bar, Pressure_diff={source_pressure-target_pressure:.0f}Pa")

            # Valve should be active if target pressure is below activation threshold
            # and there's sufficient pressure difference for flow
            should_be_active = (
                target_pressure <= valve.activation_threshold and
                source_pressure > target_pressure + 1e5  # At least 1 bar pressure difference
            )

            if should_be_active:
                # Calculate flow using the valve's flow physics
                try:
                    # Create mock tank_states list for compatibility
                    tank_states = [None] * max(valve.source_tank + 1, valve.target_tank + 1)
                    tank_states[valve.source_tank] = source_state
                    tank_states[valve.target_tank] = target_state

                    # Temporarily set valve active for flow calculation
                    original_state = valve.is_active
                    valve.is_active = True

                    flow_rate = valve.calculate_flow_rate(t, tank_states)

                    # Restore original state
                    valve.is_active = original_state

                    if flow_rate > 0:
                        coupling_flows[valve.source_tank] -= flow_rate
                        coupling_flows[valve.target_tank] += flow_rate

                except Exception as e:
                    # Fall back to simple orifice flow calculation
                    if source_pressure > target_pressure:
                        # Simple choked flow approximation
                        P_ratio = target_pressure / source_pressure
                        if P_ratio < 0.528:  # Choked flow
                            flow_rate = 0.6 * valve.effective_area * source_pressure * math.sqrt(0.7 / (287 * source_state.temperature))
                        else:  # Subsonic flow
                            flow_rate = 0.6 * valve.effective_area * math.sqrt(2 * source_state.density * (source_pressure - target_pressure))

                        # Apply reasonable limits
                        flow_rate = min(flow_rate, valve.max_flow_rate)
                        flow_rate = min(flow_rate, 0.1 * source_state.fuel_mass)  # Don't drain tank too fast

                        if flow_rate > 1e-6:  # Only apply significant flows
                            coupling_flows[valve.source_tank] -= flow_rate
                            coupling_flows[valve.target_tank] += flow_rate

        return coupling_flows

    def _get_outflow_rate(self, time: float, tank_index: int) -> float:
        """
        Get outflow (discharge) rate for specific tank at given time based on mission profile.

        Args:
            time: Current time [s]
            tank_index: Tank index (0-based)

        Returns:
            Outflow rate [kg/s] (positive = outflow)
        """
        return self._get_flow_rate(time, tank_index, 'OutFlow')

    def _get_inflow_rate(self, time: float, tank_index: int) -> float:
        """
        Get inflow (refuel) rate for specific tank at given time based on mission profile.

        Args:
            time: Current time [s]
            tank_index: Tank index (0-based)

        Returns:
            Inflow rate [kg/s] (positive = inflow)
        """
        return self._get_flow_rate(time, tank_index, 'InFlow')

    def _get_flow_rate(self, time: float, tank_index: int, flow_type: str) -> float:
        """
        Get flow rate for specific tank at given time and flow type.

        Args:
            time: Current time [s]
            tank_index: Tank index (0-based)
            flow_type: 'InFlow' or 'OutFlow'

        Returns:
            Flow rate [kg/s] (positive)
        """
        if self.config.mission_profile is None:
            return 0.0

        # For single tank scenarios, the only tank gets the mission flow
        # For multi-tank scenarios, check mission assignment
        if len(self.tanks) == 1:
            # Single tank case: the only tank gets all mission flows
            pass  # Continue to flow calculation
        else:
            # Multi-tank case: orchestrator handles mission assignment via method override
            # This code path should not be reached when orchestrator is used
            # Default fallback: only first tank gets mission flows
            if tank_index != 0:
                return 0.0

        try:
            # Import flow types
            from src.mission.mission_sections import InFlow, OutFlow

            # Find which mission section we're in
            current_time = 0.0

            for section in self.config.mission_profile.sections:
                section_end_time = current_time + section.duration

                if time <= section_end_time:
                    # We're in this section - calculate flow rate
                    section_time = time - current_time

                    # Extract flows of the specified type from this section
                    for flow in section.fuel_flows:
                        # Check if this is the right flow type
                        if flow_type == 'InFlow' and isinstance(flow, InFlow):
                            target_flow = flow
                        elif flow_type == 'OutFlow' and isinstance(flow, OutFlow):
                            target_flow = flow
                        else:
                            continue

                        # Extract flow rate
                        if hasattr(target_flow, 'mass_flow'):
                            if isinstance(target_flow.mass_flow, list):
                                # Time-varying flow: proper handling of multi-point profiles
                                if len(target_flow.mass_flow) == 1:
                                    return abs(target_flow.mass_flow[0])
                                elif len(target_flow.mass_flow) == 2:
                                    # Two-point linear interpolation (original behavior)
                                    start_rate = abs(target_flow.mass_flow[0])
                                    end_rate = abs(target_flow.mass_flow[-1])
                                    progress = section_time / section.duration if section.duration > 0 else 0
                                    return start_rate + (end_rate - start_rate) * progress
                                else:
                                    # Multi-point profile: interpolate across the full array
                                    flow_values = [abs(rate) for rate in target_flow.mass_flow]
                                    if section.duration <= 0:
                                        return flow_values[0]

                                    # Calculate which segment we're in
                                    progress = section_time / section.duration
                                    progress = max(0.0, min(1.0, progress))  # Clamp to [0,1]

                                    # Map progress to array index
                                    array_pos = progress * (len(flow_values) - 1)
                                    idx = int(array_pos)

                                    if idx >= len(flow_values) - 1:
                                        return flow_values[-1]

                                    # Linear interpolation between adjacent points
                                    frac = array_pos - idx
                                    return flow_values[idx] + (flow_values[idx + 1] - flow_values[idx]) * frac
                            else:
                                # Constant flow rate
                                return abs(target_flow.mass_flow)

                    # No flows of this type in this section
                    return 0.0

                current_time = section_end_time

            # Past mission end
            return 0.0

        except Exception as e:
            print(f"⚠️  Error calculating discharge flow at t={time:.1f}s: {e}")
            return 0.0

    def _create_density_event(self):
        """Create density stopping event for refuel missions."""
        def density_event(t, y):
            """Event function to detect when target density is reached."""
            # Check all tanks - stop if ANY tank reaches target density
            states_per_tank = 3  # [mass, temp, temp_solid]

            for i in range(len(self.tanks)):
                mass_idx = i * states_per_tank
                mass = y[mass_idx]

                if mass <= 0:
                    continue  # Skip empty tanks

                # Calculate density for this tank
                tank_volume = self._cached_tank_properties[i]['volume']
                current_density = mass / tank_volume

                # Check if this tank has reached target density
                if current_density >= self.config.target_density:
                    return current_density - self.config.target_density

            # No tank has reached target yet - return negative value
            return -1.0

        # Configure event properties
        density_event.terminal = True    # Stop integration when event occurs
        density_event.direction = 1      # Trigger when density increases (refuel)

        return density_event

    def run_analysis(self, solver_method: str = "RK45", solver_config: dict = None) -> MultiTankResults:
        """
        Run complete tank system analysis.

        Args:
            solver_method: ODE solver to use ("RK45", "LSODA", "Radau", etc.)
            solver_config: Optional solver configuration parameters

        Returns:
            MultiTankResults with time series data
        """
        print(f"\n🚀 Starting TankSystem-based simulation...")
        print(f"   Analysis: {getattr(self.config, 'analysis_name', 'Tank System Analysis')}")
        print(f"   Tanks: {len(self.tanks)}")
        print(f"   Solver: {solver_method}")

        # Setup solver with configuration parameters
        solver_config = solver_config or {}

        # Extract solver parameters with defaults and ensure proper types
        timestep = float(solver_config.get('time_step', 1.0))  # Default timestep
        rtol = float(solver_config.get('rtol', 1e-6))
        atol = float(solver_config.get('atol', 1e-9))
        max_step = solver_config.get('max_step', None)
        if max_step is not None:
            max_step = float(max_step)

        # Create solver configuration
        solver_params = {
            'timestep': timestep,
            'rtol': rtol,
            'atol': atol,
        }
        if max_step is not None:
            solver_params['max_step'] = max_step

        print(f"   🔧 Solver parameters: timestep={timestep}s, rtol={rtol:.0e}, atol={atol:.0e}, max_step={max_step}")

        if solver_method == "LSODA":
            self.solver = LSODASolver(**solver_params)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "RK45":
            self.solver = RK45Solver(**solver_params)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "Radau":
            self.solver = RadauSolver(**solver_params)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "DOP853":
            self.solver = DOP853Solver(**solver_params)
            self.solver.set_ode_function(self.ode_system)
        elif solver_method == "BDF":
            self.solver = BDFSolver(**solver_params)
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

        # Use solver config parameters instead of hardcoded values
        integration_params = {
            't_span': t_span,
            'y0': y0,
            'rtol': rtol,
            'atol': atol
        }

        # Add max_step only if provided
        if max_step is not None:
            integration_params['max_step'] = max_step

        # Add density stopping events if target density is specified
        if self.config.target_density is not None:
            integration_params['events'] = self._create_density_event()
            print(f"   🔧 Added target density event: {self.config.target_density:.1f} kg/m³")

        print(f"   🔧 Solver parameters: timestep={timestep}s, rtol={rtol}, atol={atol}, max_step={max_step}")

        solution = self.solver.integrate_full(**integration_params)

        elapsed_time = time.time() - start_time
        print(f"✅ Integration completed in {elapsed_time:.1f}s")
        print(f"   Final time: {solution.t[-1]/3600:.2f} hours")
        print(f"   Data points: {len(solution.t)}")

        # Check if stopped due to event
        if hasattr(solution, 't_events') and solution.t_events and len(solution.t_events[0]) > 0:
            event_time = solution.t_events[0][0]
            print(f"   🎯 Stopped by density event at t={event_time:.1f}s ({event_time/3600:.3f}h)")

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

                # Calculate basic flow rates for this time step
                discharge_flow = self._get_outflow_rate(current_time, tank_idx)
                inflow_rate = 0.0  # No inflow for discharge scenario
                outflow_rate = discharge_flow  # Discharge is outflow
                vent_rate = 0.0  # TODO: Implement venting logic if needed

                # Store basic flow data for this tank at this time step (coupling flows calculated later)
                flow_data.append({
                    'inflow_rate': inflow_rate,
                    'outflow_rate': outflow_rate,
                    'vent_rate': vent_rate,
                    'coupling_inflow_rate': 0.0,  # Will be calculated after all tank states are created
                    'coupling_outflow_rate': 0.0
                })

                # Set basic flow attributes on tank state (kg/s)
                tank_state.inflow_rate = inflow_rate
                tank_state.outflow_rate = outflow_rate
                tank_state.vent_rate = vent_rate
                tank_state.coupling_inflow_rate = 0.0
                tank_state.coupling_outflow_rate = 0.0

                tank_states.append(tank_state)

            # Calculate coupling flows now that all tank states are available
            if len(tank_states) > 1:  # Multi-tank system
                # Use stored coupling flows from simulation instead of recalculating
                coupling_flows = {i: 0.0 for i in range(len(tank_states))}

                # Find the closest stored coupling flows from simulation
                if self.coupling_flow_history:
                    # Find the closest time in history
                    closest_time = min(self.coupling_flow_history.keys(),
                                     key=lambda t: abs(t - current_time))

                    # Only use if within reasonable tolerance (1 second)
                    if abs(closest_time - current_time) <= 1.0:
                        coupling_flows = self.coupling_flow_history[closest_time].copy()

                # Debug: Print coupling flows at this timestep
                if any(abs(flow) > 1e-6 for flow in coupling_flows.values()):
                    flows_str = ", ".join([f"T{i}:{flow*1000:.1f}g/s" for i, flow in coupling_flows.items() if abs(flow) > 1e-6])
                    print(f"  Post-processing t={current_time:.1f}s: {flows_str} (from history)")
                else:
                    # Only print occasionally to avoid spam
                    if i < 5 or i % 200 == 0:
                        print(f"  Post-processing t={current_time:.1f}s: All coupling flows are zero")

                # Update flow data with coupling flows AND set tank state attributes
                for tank_idx in range(len(tank_states)):
                    if tank_idx in coupling_flows:
                        net_coupling_flow = coupling_flows[tank_idx]
                        coupling_inflow = max(0.0, net_coupling_flow)   # Positive = receiving
                        coupling_outflow = max(0.0, -net_coupling_flow) # Negative = sending

                        # Update flow_data
                        flow_data[tank_idx]['coupling_inflow_rate'] = coupling_inflow
                        flow_data[tank_idx]['coupling_outflow_rate'] = coupling_outflow

                        # Update tank state attributes (convert to kg/s for consistency)
                        tank_states[tank_idx].coupling_inflow_rate = coupling_inflow
                        tank_states[tank_idx].coupling_outflow_rate = coupling_outflow

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