"""
General tank system core engine for hydrogen storage analysis.

This module provides the main TankSystem class that can manage any number of tanks
(from 1 to N) with unified integration and inter-tank coupling capabilities.

Author: Dante Raso
"""

import math
import time
import numpy as np
# Matplotlib is not required for core simulation; avoid hard dependency at import time
from CoolProp.CoolProp import PropsSI
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from toplab.demand import Demand
from toplab.tank_design.tank_shapes import CapsuleTank
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel
from toplab.materials.insulation_models import insulation_model_from_config
from ..solver import (
    LSODASolver, RK45Solver, RadauSolver, DOP853Solver, BDFSolver, RK4FixedSolver
)
from toplab.thermodynamics.tank_states import IsochoricTankState
from toplab.utilities.output_control import silent_when_quiet
from toplab.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher
from toplab.dynamics.edge_flow import EdgeFlow

from .state_management import MultiTankState, MultiTankResults
from toplab.coupling import EdgeContext, build_coupling
from toplab.network.network import SINK, CouplingEdge
from toplab.fluids.fluid import DEFAULT_COOLPROP_FLUID
from toplab.fluids.flow_physics import FlowPhysics
from toplab.peripheral_components.factory import build_peripheral_component_chain


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
    MISSION_DURATION: float = NotImplementedError  # complain
    tanks: List[TankConfig] = None       # Tank configurations
    mission_profile: Any = None          # Mission profile for flow calculations
    minimum_density: float = 5.8         # Stopping density [kg/m³]
    target_density: float = None         # Target density for refuel missions [kg/m³]
    outflow_demands: Dict[int, Demand] = field(default_factory=dict)  # tank_index (0-based) → discharge demand
    inflow_demands: Dict[int, Demand] = field(default_factory=dict)   # tank_index (0-based) → refuel demand

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

    @silent_when_quiet
    def __init__(self,
                 tank_geometries: List[CapsuleTank],
                 config: TankSystemConfig,
                 couplings: List[CouplingEdge] = None,
                 analysis_config=None,
                 verbosity: str = "summary"):
        """
        Initialize tank system.

        Args:
            tank_geometries: List of SphericalTank objects (geometry and materials)
            config: TankSystemConfig with tank parameters and scenarios
            couplings: Coupling edges from the Network (optional)
            analysis_config: AnalysisConfig supplying physics, materials and per-tank limits
        """
        self.verbosity = self._normalize_verbosity(verbosity)
        self._ode_progress_interval_s = 60.0
        self._ode_progress_bucket = -1
        print("Initializing TankSystem...")

        self.tank_geometries = tank_geometries
        self.config = config
        self.analysis_config = analysis_config
        self.couplings = couplings or []

        # Validate configuration
        if len(tank_geometries) != len(config.tanks):
            raise ValueError(f"Number of tank geometries ({len(tank_geometries)}) must match "
                           f"number of tank configs ({len(config.tanks)})")

        # Initialize system components
        self.tanks = []
        self.thermal_models = []
        self.dynamic_models = []
        self.coupling_valves = []

        # Per-edge flows [kg/s] at the accepted solution points (n_edges x n_points), set by run_analysis
        self.edge_flow_history = np.zeros((0, 0))
        self._edge_slices = []   # slice of the state vector owned by each edge

        # Initialize flow physics if configuration is available
        self.flow_physics = None
        if analysis_config is not None:
            physics_config = analysis_config.physics
            if physics_config:
                try:
                    self.flow_physics = FlowPhysics(physics_config)
                    print(f"✓ Flow physics initialized from configuration")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to initialize flow physics from configuration. "
                        f"Check your 'physics:' section in the YAML file. Error: {e}"
                    ) from e

        # Initialize caching system
        self._cached_tank_properties = {}
        self._properties_printed = set()  # Track which tanks have printed properties
        self._min_densities = {}   # Per-tank minimum densities (non-CH2 tanks)
        self._min_pressures = {}   # Per-tank minimum pressures (CH2 tanks)
        self._tank_fluids = {}     # Per-tank fluid type string (upper-cased)
        self._coolprop_fluids = {}  # Per-tank CoolProp fluid name

        # Setup tanks and coupling
        self._setup_tanks()
        self._setup_coupling_rules()

        print(f"Tank system initialized with {len(self.tanks)} tanks and {len(self.coupling_valves)} coupling rules")

    @staticmethod
    def _normalize_verbosity(verbosity: str) -> str:
        level = (verbosity or "summary").strip().lower()
        if level not in {"quiet", "summary", "debug"}:
            return "summary"
        return level

    def _is_quiet(self) -> bool:
        return self.verbosity == "quiet"

    def _is_debug(self) -> bool:
        return self.verbosity == "debug"

    def _log_summary(self, message: str) -> None:
        if not self._is_quiet():
            print(message)

    def _log_debug(self, message: str) -> None:
        if self._is_debug():
            print(message)

    def _log_warning(self, message: str) -> None:
        # Keep warnings visible even in quiet mode.
        print(message)

    def _setup_tanks(self):
        """Setup tanks from provided geometries and configurations."""
        print(f"\nTANK SETUP:")

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

            # Determine fluid type and choose the appropriate stopping criterion.
            tank_spec = self.analysis_config.tanks[i] if self.analysis_config else None
            tank_data = tank_spec.geometry_data() if tank_spec else {}
            fluid_type = tank_spec.fluid if tank_spec else ''
            self._tank_fluids[i] = fluid_type
            self._coolprop_fluids[i] = tank_spec.coolprop if tank_spec else DEFAULT_COOLPROP_FLUID

            if fluid_type == 'CH2':
                # CH2: stop when pressure drops to minimum_pressure
                raw_p_min = tank_data.get('minimum_pressure')
                if raw_p_min is None:
                    raise ValueError(
                        f"Tank {i + 1} (CH2): 'minimum_pressure' must be set in operating_limits."
                    )
                self._min_pressures[i] = float(raw_p_min)
                self._min_densities[i] = 0.0  # unused for CH2; keep dict consistent
                tank_properties['minimum_density'] = 0.0
                print(f"   Tank {i+1} (CH2): pressure-based stopping at {float(raw_p_min)/1e5:.1f} bar")
            else:
                # All other fluid types: stop on minimum density
                per_tank_min = None
                raw = tank_data.get('minimum_density')
                if raw is not None:
                    try:
                        per_tank_min = float(raw)
                    except (TypeError, ValueError):
                        pass
                if per_tank_min is None:
                    raise ValueError(
                        f"Tank {i + 1}: 'minimum_density' must be set in the node's stopping_criteria. "
                        "Add 'minimum_density: <value>' under the node's stopping_criteria in your YAML config."
                    )
                self._min_densities[i] = per_tank_min
                tank_properties['minimum_density'] = per_tank_min
                print(f"   Tank {i+1}: V={tank_properties['volume']:.4f} m³, A_in={tank_properties['inner_surface_area']:.3f} m², rho_min={per_tank_min:.3f} kg/m³")

            self.tanks.append(tank_geom)
            self.thermal_models.append(thermal_model)
            self.dynamic_models.append(dynamic_model)

    def _extract_mission_profile_data(self) -> dict:
        """Extract mission profile data from system configuration."""
        if not self.config.mission_profile:
            return {}

        try:
            # Import flow types
            from toplab.missions.mission_sections import OutFlow

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
            print(f"   WARNING: Failed to extract mission profile: {e}")
            return {}

    def _setup_coupling_rules(self):
        """Build the coupling edges through the edge registry."""
        print(f"\nCOUPLING SETUP:")

        if not self.couplings:
            print("   No couplings specified - tanks operate independently")
        else:
            ctx = EdgeContext(
                minimum_pressures={i: t.P_MIN for i, t in enumerate(self.config.tanks)},
                fluids=self._coolprop_fluids,
                flow_physics=self.flow_physics,
                mission_data=self._extract_mission_profile_data(),
            )
            for edge in self.couplings:
                coupling = build_coupling(edge, ctx)
                self.coupling_valves.append(coupling)
                target = "sink" if edge.target == SINK else f"Tank{edge.target+1}"
                print(f"   🔗 {edge.coupling_type} '{edge.edge_id}': Tank{edge.source+1} → {target}")
            print(f"   {len(self.coupling_valves)} coupling rules configured")

        start = 5 * len(self.config.tanks)
        for edge in self.coupling_valves:
            self._edge_slices.append(slice(start, start + edge.n_states))
            start += edge.n_states

        # Cache tank properties to avoid repeated calculations during simulation
        for i, tank_geom in enumerate(self.tank_geometries):
            self._cached_tank_properties[i] = self._get_tank_properties(tank_geom, tank_id=f"Tank{i+1}", tank_index=i)

    def _calculate_coupling_enthalpy(self, tank_idx: int, multi_state: MultiTankState, edge_flows: List[float]) -> float:
        inflow_edges = [
            {'valve': edge, 'source_tank': edge.source_tank, 'flow_rate': flow}
            for edge, flow in zip(self.coupling_valves, edge_flows)
            if edge.target_tank == tank_idx and flow > 0.0
        ]
        if not inflow_edges:
            return 0.0

        total_inflow = 0.0
        enthalpy_flow = 0.0
        target_state = multi_state.tank_states[tank_idx]

        for inflow_edge in inflow_edges:
            flow_rate = inflow_edge['flow_rate']
            if flow_rate <= 0.0:
                continue

            source_state = multi_state.tank_states[inflow_edge['source_tank']]
            delivered_enthalpy = inflow_edge['valve'].get_delivered_enthalpy(
                source_state,
                target_state,
                flow_rate,
            )
            total_inflow += flow_rate
            enthalpy_flow += flow_rate * delivered_enthalpy

        if total_inflow <= 0.0:
            return 0.0

        return enthalpy_flow / total_inflow

    def _get_tank_properties(self, tank: CapsuleTank, tank_id: str = "Unknown", tank_index: int = -1):
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
        from toplab.tank_design.structural_models import CompositeCylinder, CompositeSphericalEndCap
        import math

        # Get materials from configuration - NO HARDCODED VALUES
        if not self.analysis_config:
            raise RuntimeError("No analysis configuration available - cannot determine materials")

        tank_spec = self.analysis_config.tanks[tank_index]
        tank_id_yaml = tank_spec.node_id
        tank_materials = tank_spec.nist_materials()
        liner_material = tank_materials.get('liner')
        composite_material = tank_materials.get('composite')

        if not liner_material or not composite_material:
            raise RuntimeError(f"Liner or composite material not found in configuration for tank {tank_id_yaml}")

        materials_config = tank_spec.materials
        thickness_liner = materials_config.get('liner', {}).get('thickness', None)
        if thickness_liner is None:
            raise RuntimeError("Liner thickness not specified in configuration")

        insulation_config = materials_config.get('insulation', {})
        thickness_insulation = insulation_config.get('thickness')
        if thickness_insulation is None:
            raise RuntimeError(f"'insulation.thickness' not specified in configuration for tank {tank_id_yaml}")
        thickness_insulation = float(thickness_insulation)

        shell_thickness = insulation_config.get('shell_thickness')
        if shell_thickness is None:
            raise RuntimeError(f"'insulation.shell_thickness' not specified in configuration for tank {tank_id_yaml}")
        shell_thickness = float(shell_thickness)

        alpha_amb = insulation_config.get('alpha_amb')
        if alpha_amb is None:
            raise RuntimeError(f"'insulation.alpha_amb' not specified in configuration for tank {tank_id_yaml}")
        alpha_amb = float(alpha_amb)

        emissivity_shell = insulation_config.get('emissivity')
        if emissivity_shell is None:
            raise RuntimeError(f"'insulation.emissivity' not specified in configuration for tank {tank_id_yaml}")
        emissivity_shell = float(emissivity_shell)

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

        # If not found in tank object, fall back to the tank spec
        if design_pressure is None or working_pressure is None:
            if tank_index >= 0:
                tank_config = tank_spec.geometry_data()
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

        # Burst pressure FS scales the design pressure for wall thickness
        design_pressure_structural = working_pressure * safety_factor

        # Calculate thickness for both sections
        cylinder_thickness = cylinder_model.compute_thickness(tank_section, design_pressure_structural)
        endcap_thickness = endcap_model.compute_thickness(tank_section, design_pressure_structural)
        thickness_wall = max(cylinder_thickness, endcap_thickness)  # Governing thickness

        print(f"   Netting Analysis Results for {tank_id}:")
        print(f"      Radius: {inner_radius:.3f} m")
        print(f"      Working pressure: {working_pressure/1e5:.0f} bar, Safety factor: {safety_factor:.2f}")
        print(f"      Structural design pressure: {design_pressure_structural/1e5:.0f} bar")
        # Radii for each layer (inside → outside)
        liner_outer_radius = inner_radius + thickness_liner
        wall_outer_radius = liner_outer_radius + thickness_wall   # = r_structure
        r_structure = wall_outer_radius
        r_shell = r_structure + thickness_insulation              # outer radius of insulation
        r_shell_outer = r_shell + shell_thickness                 # outer radius of outer Al shell

        # Surface areas and volumes
        volume = tank.volume
        inner_surface_area = tank.surface_area
        cylinder_length = tank.cylindrical_section_length
        # A_shell = 2π r_shell L + 4π r_shell² (per governing equations, at foam-shell interface)
        outer_surface_area = 2 * math.pi * r_shell * cylinder_length + 4 * math.pi * r_shell ** 2
        outer_volume = (4 / 3) * math.pi * r_shell_outer ** 3 + math.pi * r_shell_outer ** 2 * cylinder_length

        # Layer masses using capsule (cylinder + 2 hemispherical endcaps) geometry
        def _layer_mass(density, r_inner, r_outer):
            cyl = math.pi * (r_outer ** 2 - r_inner ** 2) * cylinder_length
            sph = (4 / 3) * math.pi * (r_outer ** 3 - r_inner ** 3)
            return density * (cyl + sph)

        liner_mass = _layer_mass(liner_material.density, inner_radius, liner_outer_radius)
        wall_mass  = _layer_mass(composite_material.density, liner_outer_radius, wall_outer_radius)
        insulation_model = insulation_model_from_config(insulation_config)
        foam_mass  = _layer_mass(insulation_model.density, r_structure, r_shell)
        shell_mass = _layer_mass(liner_material.density, r_shell, r_shell_outer)  # same Al as liner

        print(f"\n[{tank_id} Geometry]")
        print(f"  Inner radius:     {inner_radius:.4f} m")
        print(f"  Structure radius: {r_structure:.4f} m  "
              f"(liner {thickness_liner*1000:.1f}mm + wall {thickness_wall*1000:.1f}mm)")
        print(f"  Shell radius:     {r_shell:.4f} m  (insulation {thickness_insulation*1000:.0f}mm)")
        print(f"  Shell outer:      {r_shell_outer:.4f} m  (shell {shell_thickness*1000:.1f}mm)")
        print(f"  Cylinder length:  {cylinder_length:.4f} m  (φ={tank.phi:.2f})")
        print(f"\n[{tank_id} Structural – netting analysis]")
        print(f"  Working pressure:  {working_pressure/1e5:.0f} bar  →  "
              f"design: {design_pressure_structural/1e5:.0f} bar  (SF={safety_factor:.1f})")
        print(f"  Liner:   {liner_material.name}, {thickness_liner*1000:.1f} mm, {liner_mass:.1f} kg")
        print(f"  Wall:    {composite_material.name}, {thickness_wall*1000:.1f} mm "
              f"({math.degrees(composite_material.winding_angle):.1f}°), {wall_mass:.1f} kg")
        print(f"  Foam:    {insulation_model.name}, {thickness_insulation*1000:.0f} mm, {foam_mass:.1f} kg")
        print(f"  Shell:   {liner_material.name}, {shell_thickness*1000:.1f} mm, {shell_mass:.1f} kg")
        dry_mass = liner_mass + wall_mass + foam_mass + shell_mass
        print(f"  Total dry mass: {dry_mass:.1f} kg")
        print(f"\n[{tank_id} Thermal areas]")
        print(f"  Inner surface (A_in):  {inner_surface_area:.4f} m²")
        print(f"  Shell surface (A_sh):  {outer_surface_area:.4f} m²")

        # Netting analysis detail for reference
        print(f"\n[{tank_id} Netting analysis detail]")
        print(f"  Cylinder wall: {cylinder_thickness*1000:.1f} mm  |  Endcap: {endcap_thickness*1000:.1f} mm  "
              f"(governing: {thickness_wall*1000:.1f} mm)")

        thickness_info = {
            'wall_thickness': thickness_wall,
            'cylinder_thickness': cylinder_thickness,
            'endcap_thickness': endcap_thickness,
            'composite_density': composite_material.density,
            'cylindrical_section_length': cylinder_length,
            'phi': tank.phi,
        }

        if hasattr(self, '_properties_printed') and tank_index not in self._properties_printed:
            self._properties_printed.add(tank_index)

        return {
            'volume': volume,
            'outer_volume': outer_volume,
            'inner_surface_area': inner_surface_area,
            'outer_surface_area': outer_surface_area,
            'inner_diameter': 2 * inner_radius,
            'outer_diameter': 2 * r_shell_outer,  # true outer envelope (includes shell)
            'inner_radius': inner_radius,
            'r_structure': r_structure,
            'r_shell': r_shell,
            'liner_mass': liner_mass,
            'wall_mass': wall_mass,
            'foam_mass': foam_mass,
            'insulation_model': insulation_model,
            'shell_mass': shell_mass,
            'liner_material': liner_material,
            'wall_material': composite_material,
            'shell_material': liner_material,  # outer shell is same Al as liner
            'alpha_amb': alpha_amb,
            'emissivity_shell': emissivity_shell,
            'thickness_wall': thickness_wall,
            **thickness_info,
        }

    def _create_thermal_model(self, tank_properties: Dict[str, float]):
        """Create InsulatedTankThermalModel for a tank from its geometry property dict."""
        return InsulatedTankThermalModel(
            tank_volume=tank_properties['volume'],
            inner_surface_area=tank_properties['inner_surface_area'],
            inner_diameter=tank_properties['inner_diameter'],
            r_structure=tank_properties['r_structure'],
            r_shell=tank_properties['r_shell'],
            cylinder_length=tank_properties['cylindrical_section_length'],
            liner_mass=tank_properties['liner_mass'],
            wall_mass=tank_properties['wall_mass'],
            foam_mass=tank_properties['foam_mass'],
            shell_mass=tank_properties['shell_mass'],
            ambient_temperature=self.config.AMBIENT_TEMPERATURE,
            alpha_amb=tank_properties['alpha_amb'],
            emissivity_shell=tank_properties['emissivity_shell'],
            liner_material=tank_properties['liner_material'],
            wall_material=tank_properties['wall_material'],
            shell_material=tank_properties['shell_material'],
            insulation_model=tank_properties['insulation_model'],
        )

    def create_initial_state(self) -> np.ndarray:
        """Create initial state vector for all tanks."""
        print("Creating initial tank system state...")

        state = []

        for i, tank_config in enumerate(self.config.tanks):
            print(f"Tank {i+1}: ", end="")

            if tank_config.MASS_INIT is not None:
                m_init = tank_config.MASS_INIT
                tank_volume = self._cached_tank_properties[i]['volume']
                density_init = m_init / tank_volume
                print(f"  Tank {i+1}: specified mass {m_init:.2f} kg, "
                      f"density {density_init:.2f} kg/m³")
            else:
                density_init = PropsSI("D", "P", tank_config.P_INIT, "T", tank_config.T_INIT, self._coolprop_fluids[i])
                tank_volume = self._cached_tank_properties[i]['volume']
                m_init = density_init * tank_volume
                print(f"  Tank {i+1}: P={tank_config.P_INIT/1e5:.0f} bar, "
                      f"T={tank_config.T_INIT:.1f} K, "
                      f"ρ={density_init:.2f} kg/m³, "
                      f"m={m_init:.2f} kg")

            # State: [mass, T_H2, T_structure, T_insulation, T_shell]
            # Structure starts slightly above H2 and shell starts at ambient.
            # Initialize insulation at the temperature that balances its two
            # half-layer heat flows, avoiding a nonphysical initial storage pulse.
            T_h2_init = tank_config.T_INIT
            T_structure_init = T_h2_init + 0.1
            T_shell_init = self.config.AMBIENT_TEMPERATURE
            T_insulation_init = self.thermal_models[i].determine_initial_insulation_temperature(
                T_structure_init, T_shell_init
            )
            state.extend([
                m_init,
                T_h2_init,
                T_structure_init,
                T_insulation_init,
                T_shell_init,
            ])

        initial_state = np.array(state)
        print("Initial state summary:")
        for i in range(len(self.config.tanks)):
            idx = 5 * i
            print(f"  {self.config.tanks[i].name}: "
                  f"m={initial_state[idx]:.2f} kg  "
                f"T_H2={initial_state[idx+1]:.1f} K  "
                f"T_struct={initial_state[idx+2]:.1f} K  "
                f"T_ins={initial_state[idx+3]:.1f} K  "
                f"T_shell={initial_state[idx+4]:.1f} K")

        return initial_state

    def ode_system(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for the multi-tank system.

        State vector y = [m1, T_H2, T_struct, T_ins, T_shell, ..., mN, T_H2,N, T_struct,N, T_ins,N, T_shell,N].
        Each tank has five state variables.
        """
        # Periodic heartbeat: one print per progress bucket (default every 60 s sim-time)
        progress_bucket = int(max(0.0, t) // self._ode_progress_interval_s)
        if progress_bucket > getattr(self, '_ode_progress_bucket', -1):
            self._ode_progress_bucket = progress_bucket
            lines = [f"t = {t:.1f} s ({t/3600:.3f} h)"]
            for i in range(len(self.tanks)):
                idx_hb = 5 * i
                m_i   = max(float(y[idx_hb]),     1e-9)
                T_i   = max(float(y[idx_hb + 1]), 14.0)
                T_s   = float(y[idx_hb + 2])
                T_ins = float(y[idx_hb + 3])
                T_sh  = float(y[idx_hb + 4])
                rho_i = m_i / max(self.tanks[i].volume, 1e-9)
                try:
                    from toplab.fluids.coolprop_safe import safe_pressure_from_T_rho
                    P_i = safe_pressure_from_T_rho(T_i, rho_i, self._coolprop_fluids[i])
                except Exception:
                    P_i = PropsSI("P", "T", T_i, "Dmass", rho_i, self._coolprop_fluids[i])
                thermal = self.thermal_models[i]
                try:
                    Q_amb = thermal.compute_ambient_to_shell_heat_flux(T_sh)
                    Q_sh_ins = thermal.compute_shell_to_insulation_heat_flux(T_sh, T_ins)
                    Q_ins_str = thermal.compute_insulation_to_structure_heat_flux(T_ins, T_s)
                    alpha_s = thermal.get_alpha_s(T_i, T_s, P_i)
                    Q_str = alpha_s * thermal.A_in * (T_s - T_i)
                    heat_str = (f"Q_amb={Q_amb:+.1f}W  Q_sh-ins={Q_sh_ins:+.1f}W  "
                                f"Q_ins-str={Q_ins_str:+.1f}W  Q_str-H2={Q_str:+.1f}W")
                except Exception as e:
                    heat_str = f"heat flows unavailable ({e})"
                lines.append(
                    f"  {self.config.tanks[i].name}: "
                    f"T_H2={T_i:.1f}K  T_struct={T_s:.1f}K  T_ins={T_ins:.1f}K  T_shell={T_sh:.1f}K  |  "
                    f"{heat_str}  |  "
                    f"P={P_i/1e5:.1f}bar  ρ={rho_i:.2f}kg/m³"
                )
            self._log_summary("\n".join(lines))

        # Build multi-tank state and compute derivatives
        multi_state = MultiTankState.from_state_vector(
            y[:5 * len(self.tanks)], self.tanks, t, fluids=[self._coolprop_fluids[i] for i in range(len(self.tanks))]
        )

        n_tanks = len(self.tanks)
        dydt = np.zeros(len(y))

        edge_rates = self._edge_flow_rates(multi_state, t, y)
        coupling_flows = self._net_coupling_flows(edge_rates)

        for i in range(n_tanks):
            tank_state = multi_state.tank_states[i]

            if tank_state.fuel_mass <= 1.0:
                if not hasattr(self, '_empty_warn_printed'):
                    self._log_warning(f"  WARNING: {self.config.tanks[i].name} approaching empty: "
                                      f"{tank_state.fuel_mass:.2f} kg")
                    self._empty_warn_printed = True
                tank_state.fuel_mass = max(tank_state.fuel_mass, 1.0)

            net_coupling_flow = coupling_flows[i]
            coupling_enthalpy = 0.0
            if net_coupling_flow > 0:
                coupling_enthalpy = self._calculate_coupling_enthalpy(i, multi_state, edge_rates)

            def _as_float(val: Any) -> float:
                if isinstance(val, (list, tuple)):
                    return float(val[0])
                import numpy as _np
                if isinstance(val, _np.ndarray):
                    return float(val.flat[0])
                return float(val)

            edge_flows = []

            mission_inflow = _as_float(self._get_inflow_rate(t, i))
            if mission_inflow > 0.0:
                edge_flows.append(EdgeFlow(
                    mdot=mission_inflow, h=0.0,
                    edge_type='refuel', from_node=-1, to_node=i,
                ))

            mission_outflow = _as_float(self._get_outflow_rate(t, i))
            if mission_outflow > 0.0:
                edge_flows.append(EdgeFlow(
                    mdot=mission_outflow, h=0.0,
                    edge_type='discharge', from_node=i, to_node=-1,
                ))

            coupling_inflow = max(0.0, _as_float(net_coupling_flow))
            if coupling_inflow > 0.0:
                edge_flows.append(EdgeFlow(
                    mdot=coupling_inflow, h=coupling_enthalpy,
                    edge_type='coupling', from_node=-2, to_node=i,
                ))

            coupling_outflow = max(0.0, _as_float(-net_coupling_flow))
            if coupling_outflow > 0.0:
                edge_flows.append(EdgeFlow(
                    mdot=coupling_outflow, h=0.0,
                    edge_type='coupling', from_node=i, to_node=-2,
                ))

            # Thermal derivatives
            Q_structure_to_h2 = self.thermal_models[i].compute_structure_to_h2_heat_flux(t, tank_state)
            dT_structure_dt = self.thermal_models[i].compute_structure_temperature_derivative(t, tank_state)
            dT_insulation_dt = self.thermal_models[i].compute_insulation_temperature_derivative(t, tank_state)
            dT_shell_dt = self.thermal_models[i].compute_shell_temperature_derivative(t, tank_state)

            state_derivatives = self.dynamic_models[i].compute_state_derivatives(
                time=t,
                state=tank_state,
                edge_flows=edge_flows,
                Q_structure_to_h2=Q_structure_to_h2,
                dT_structure_dt=dT_structure_dt,
                dT_insulation_dt=dT_insulation_dt,
                dT_shell_dt=dT_shell_dt,
                Q_discharge=0.0,
                tank_index=i,
            )

            idx = i * 5
            dydt[idx]     = state_derivatives.fuel_mass_derivative
            dydt[idx + 1] = state_derivatives.h2_temperature_derivative
            dydt[idx + 2] = state_derivatives.structure_temperature_derivative
            dydt[idx + 3] = state_derivatives.insulation_temperature_derivative
            dydt[idx + 4] = state_derivatives.shell_temperature_derivative

        for edge, sl in zip(self.coupling_valves, self._edge_slices):
            if edge.n_states:
                dydt[sl] = edge.state_derivatives(t, multi_state.tank_states, y[sl])

        return dydt

    def _edge_flow_rates(self, multi_state: MultiTankState, t: float, y: np.ndarray) -> List[float]:
        """Mass flow [kg/s, >= 0] through each coupling edge; a pure function of (t, y) and the edge modes."""
        return [max(0.0, float(edge.flow(t, multi_state.tank_states, y[sl])))
                for edge, sl in zip(self.coupling_valves, self._edge_slices)]

    def _net_coupling_flows(self, edge_rates: List[float]) -> Dict[int, float]:
        """Net coupling mass flow per tank (positive = inflow)."""
        net = {i: 0.0 for i in range(len(self.tanks))}
        for edge, flow in zip(self.coupling_valves, edge_rates):
            net[edge.source_tank] -= flow
            if edge.target_tank >= 0:
                net[edge.target_tank] += flow
        return net

    def _gross_coupling_flows(self, edge_rates) -> Dict[int, Dict[str, float]]:
        gross = {i: {'inflow': 0.0, 'outflow': 0.0} for i in range(len(self.tanks))}
        for edge, flow in zip(self.coupling_valves, edge_rates):
            gross[edge.source_tank]['outflow'] += flow
            if edge.target_tank >= 0:
                gross[edge.target_tank]['inflow'] += flow
        return gross

    def _edge_flows_at(self, t: float, y: np.ndarray) -> List[float]:
        multi_state = MultiTankState.from_state_vector(
            y[:5 * len(self.tanks)], self.tanks, t, fluids=[self._coolprop_fluids[i] for i in range(len(self.tanks))]
        )
        return self._edge_flow_rates(multi_state, t, y)

    def _mode_switch_event(self, switch):
        """Terminal event for an edge mode switch (a tank pressure crossing a threshold)."""
        from toplab.fluids.coolprop_safe import safe_pressure_from_T_rho
        idx, volume = switch.tank_idx, self._cached_tank_properties[switch.tank_idx]['volume']
        fluid = self._coolprop_fluids[idx]

        def event(t, y):
            return safe_pressure_from_T_rho(float(y[5 * idx + 1]), float(y[5 * idx]) / volume, fluid) - switch.pressure_pa

        event.terminal = True
        event.direction = switch.direction
        return event

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
        """Mission flow [kg/s, positive] into ('InFlow') or out of ('OutFlow') a tank."""
        demands = self.config.inflow_demands if flow_type == 'InFlow' else self.config.outflow_demands
        demand = demands.get(tank_index)
        return demand.mass_flow(time) if demand is not None else 0.0

    def _create_min_density_events(self) -> list:
        """Create one terminal event per tank: pressure-based for CH2, density-based for all others."""
        events = []
        for tank_idx in range(len(self.tanks)):
            vol = self._cached_tank_properties[tank_idx]['volume']

            if self._tank_fluids.get(tank_idx, '') == 'CH2':
                p_min = self._min_pressures[tank_idx]

                def make_pressure_event(idx, p_min_pa, volume, fluid):
                    from CoolProp.CoolProp import PropsSI as _PropsSI

                    def event(t, y):
                        m = float(y[idx * 5])
                        T = float(y[idx * 5 + 1])
                        rho = m / max(float(volume), 1e-9)
                        try:
                            P = _PropsSI("P", "T", max(T, 14.0), "Dmass", max(rho, 1e-9), fluid)
                            return P - p_min_pa
                        except Exception:
                            return 1.0  # stay positive so a CoolProp error does not falsely trigger

                    event.terminal = True
                    event.direction = -1  # fire when pressure falls through p_min
                    return event

                events.append(make_pressure_event(tank_idx, p_min, vol, self._coolprop_fluids[tank_idx]))
            else:
                min_rho = self._min_densities[tank_idx]

                def make_event(idx, minimum, volume):
                    def event(t, y):
                        return y[idx * 5] / volume - minimum
                    event.terminal = True
                    event.direction = -1  # density decreasing
                    return event

                events.append(make_event(tank_idx, min_rho, vol))
        return events

    def _create_density_event(self):
        """Create density stopping event for refuel missions."""
        def density_event(t, y):
            """Event function to detect when target density is reached."""
            # Check all tanks - stop if ANY tank reaches target density
            states_per_tank = 5  # [mass, T_H2, T_structure, T_insulation, T_shell]

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

    def compute_coupling_stream_history(self, results: MultiTankResults) -> list:
        """
        Post-processing helper: re-run each coupling valve's peripheral component chain
        at every output time step and return the stream state (T, P, mdot) before and
        after each component.

        Only valves that have a non-empty component_chain (for split_chain entries) or
        non-empty main_conditioning_chain / discharge_conditioning_chain are included.

        Returns
        -------
        list of dicts, one per qualifying entry::

            {
                'stream_type': 'split_chain' | 'main_discharge' | 'fuel_cell_discharge' | 'mixed_fuel_cell',
                'coupling_id': str,
                'source_tank': int,   # 0-based
                'target_tank': int,   # 0-based, or -1 for sink
                'times_h':    ndarray,   # output times [h]
                'components': [          # for split_chain / main_discharge / fuel_cell_discharge
                    {
                        'name':   str,
                        'inlet':  {'mdot_gs': ndarray, 'temperature_K': ndarray, 'pressure_bar': ndarray},
                        'outlet': {'mdot_gs': ndarray, 'temperature_K': ndarray, 'pressure_bar': ndarray},
                    },
                    ...
                ],
                # Additional keys for 'mixed_fuel_cell' only:
                'mdot_gs':        ndarray,   # total mixed mass flow [g/s]
                'temperature_K':  ndarray,   # mixed temperature [K]
                'pressure_bar':   ndarray,   # mixing pressure [bar]
                'mdot_main_gs':   ndarray,   # 95% conditioned stream [g/s]
                'mdot_discharge_gs': ndarray, # Tank 2 discharge stream [g/s]
            }
        """
        import numpy as np
        from CoolProp.CoolProp import PropsSI
        from toplab.peripheral_components.base import PeripheralFlowState
        from toplab.coupling import (
            ProportionalSplitCoupling,
            PressureTriggeredDischarge,
        )

        histories = []
        n_timesteps = len(results.times)
        times_h = results.times / 3600.0

        def _process_chain(chain, source_state, mdot, target_pressure):
            """Walk *chain* starting from *source_state* at *mdot* kg/s.
            Returns list of dicts {name, inlet_mdot, inlet_T, inlet_P, outlet_mdot, outlet_T, outlet_P}.
            """
            comp_data = [
                {
                    'name':        type(comp).__name__,
                    'inlet_mdot':  np.full(n_timesteps, np.nan),
                    'inlet_T':     np.full(n_timesteps, np.nan),
                    'inlet_P':     np.full(n_timesteps, np.nan),
                    'outlet_mdot': np.full(n_timesteps, np.nan),
                    'outlet_T':    np.full(n_timesteps, np.nan),
                    'outlet_P':    np.full(n_timesteps, np.nan),
                }
                for comp in chain
            ]
            return comp_data  # pre-allocated; caller fills per timestep

        def _fill_chain_at_ts(comp_data, chain, source_state, mdot, target_pressure, ts_idx):
            """Fill one time-step's worth of inlet/outlet data for a component chain."""
            stream = PeripheralFlowState.from_tank_state(source_state, mdot)
            for ci, component in enumerate(chain):
                cd = comp_data[ci]
                cd['inlet_mdot'][ts_idx] = stream.mass_flow_rate * 1000.0
                cd['inlet_T'][ts_idx]    = stream.temperature
                cd['inlet_P'][ts_idx]    = stream.pressure / 1e5

                stream = component.process_stream(stream, target_pressure=target_pressure)

                cd['outlet_mdot'][ts_idx] = stream.mass_flow_rate * 1000.0
                cd['outlet_T'][ts_idx]    = stream.temperature
                cd['outlet_P'][ts_idx]    = stream.pressure / 1e5
            return stream  # final outlet stream after full chain

        def _comp_data_to_list(comp_data):
            return [
                {
                    'name': cd['name'],
                    'inlet':  {'mdot_gs': cd['inlet_mdot'], 'temperature_K': cd['inlet_T'],  'pressure_bar': cd['inlet_P']},
                    'outlet': {'mdot_gs': cd['outlet_mdot'], 'temperature_K': cd['outlet_T'], 'pressure_bar': cd['outlet_P']},
                }
                for cd in comp_data
            ]

        # ----------------------------------------------------------------
        # Pass 1: split_chain and main_discharge entries
        # ----------------------------------------------------------------
        main_discharge_entry = None   # will be used to compute mixed stream

        for valve in self.coupling_valves:
            source_tank_idx = valve.source_tank

            # --- split_chain (full ODE chain: main conditioning + compressor) ---
            if valve.component_chain:
                n_components = len(valve.component_chain)
                comp_data = _process_chain(valve.component_chain, None, 0, None)

                for ts_idx in range(n_timesteps):
                    t = results.times[ts_idx]
                    source_state = results.multi_tank_states[ts_idx].tank_states[source_tank_idx]
                    if source_state.pressure is None and hasattr(source_state, 'compute_pressure'):
                        source_state.compute_pressure()
                    if source_state.pressure is None:
                        continue

                    target_pressure = None
                    if valve.target_tank >= 0:
                        tgt = results.multi_tank_states[ts_idx].tank_states[valve.target_tank]
                        if tgt.pressure is None and hasattr(tgt, 'compute_pressure'):
                            tgt.compute_pressure()
                        target_pressure = tgt.pressure

                    try:
                        mdot = float(self.edge_flow_history[self.coupling_valves.index(valve), ts_idx])
                    except Exception:
                        continue
                    if mdot <= 0.0:
                        continue

                    try:
                        _fill_chain_at_ts(comp_data, valve.component_chain, source_state, mdot, target_pressure, ts_idx)
                    except Exception as e:
                        print(f"   WARNING: split_chain re-run failed at t={t:.1f}s: {e}")

                histories.append({
                    'stream_type': 'split_chain',
                    'coupling_id': valve.coupling_id,
                    'source_tank': valve.source_tank,
                    'target_tank': valve.target_tank,
                    'times_h':    times_h,
                    'components': _comp_data_to_list(comp_data),
                })

            # --- main_discharge (95% of Tank 1 outflow, through HEX + PressureReg only) ---
            if (
                isinstance(valve, ProportionalSplitCoupling)
                and valve.main_conditioning_chain
            ):
                chain = valve.main_conditioning_chain
                comp_data_main = _process_chain(chain, None, 0, None)

                # Arrays for the outlet state (used later for mixing)
                main_outlet_T   = np.full(n_timesteps, np.nan)
                main_outlet_P   = np.full(n_timesteps, np.nan)
                main_outlet_h   = np.full(n_timesteps, np.nan)
                main_mdot_gs    = np.full(n_timesteps, np.nan)

                for ts_idx in range(n_timesteps):
                    t = results.times[ts_idx]
                    source_state = results.multi_tank_states[ts_idx].tank_states[source_tank_idx]
                    if source_state.pressure is None and hasattr(source_state, 'compute_pressure'):
                        source_state.compute_pressure()
                    if source_state.pressure is None:
                        continue

                    # Main discharge: (1 - split_fraction) * full mission outflow
                    try:
                        mdot_split = float(self.edge_flow_history[self.coupling_valves.index(valve), ts_idx])
                    except Exception:
                        continue
                    if mdot_split <= 0.0:
                        continue
                    mdot_main = mdot_split * (1.0 - valve.split_fraction) / valve.split_fraction

                    try:
                        outlet_stream = _fill_chain_at_ts(
                            comp_data_main, chain, source_state, mdot_main, None, ts_idx
                        )
                        main_outlet_T[ts_idx]  = outlet_stream.temperature
                        main_outlet_P[ts_idx]  = outlet_stream.pressure
                        main_outlet_h[ts_idx]  = outlet_stream.enthalpy
                        main_mdot_gs[ts_idx]   = mdot_main * 1000.0
                    except Exception as e:
                        print(f"   WARNING: main_discharge re-run failed at t={t:.1f}s: {e}")

                main_discharge_entry = {
                    'stream_type': 'main_discharge',
                    'coupling_id': valve.coupling_id + '_main',
                    'source_tank': valve.source_tank,
                    'target_tank': -1,
                    'times_h':    times_h,
                    'components': _comp_data_to_list(comp_data_main),
                    # Extra arrays for mixing
                    '_outlet_T_K':  main_outlet_T,
                    '_outlet_P_Pa': main_outlet_P,
                    '_outlet_h':    main_outlet_h,
                    '_mdot_gs':     main_mdot_gs,
                }
                histories.append(main_discharge_entry)

        # ----------------------------------------------------------------
        # Pass 2: fuel_cell_discharge (Tank 2 discharge through PressureReg)
        # ----------------------------------------------------------------
        fuel_cell_entry = None

        for valve in self.coupling_valves:
            if not (isinstance(valve, PressureTriggeredDischarge) and valve.discharge_conditioning_chain):
                continue

            source_tank_idx = valve.source_tank
            chain = valve.discharge_conditioning_chain
            comp_data_fc = _process_chain(chain, None, 0, None)

            # Get Tank 2 gross coupling outflow from post-processed results [g/s]
            tank_data = results._extract_tank_arrays(source_tank_idx)
            tank2_outflow_gs = tank_data.get('coupling_outflow_rates', np.zeros(n_timesteps))

            # Arrays for the outlet state (used for mixing)
            fc_outlet_T   = np.full(n_timesteps, np.nan)
            fc_outlet_P   = np.full(n_timesteps, np.nan)
            fc_outlet_h   = np.full(n_timesteps, np.nan)
            fc_mdot_gs    = np.full(n_timesteps, np.nan)

            for ts_idx in range(n_timesteps):
                t = results.times[ts_idx]
                mdot = tank2_outflow_gs[ts_idx] / 1000.0  # g/s → kg/s
                if mdot <= 0.0 or np.isnan(mdot):
                    continue

                source_state = results.multi_tank_states[ts_idx].tank_states[source_tank_idx]
                if source_state.pressure is None and hasattr(source_state, 'compute_pressure'):
                    source_state.compute_pressure()
                if source_state.pressure is None:
                    continue

                try:
                    outlet_stream = _fill_chain_at_ts(
                        comp_data_fc, chain, source_state, mdot, None, ts_idx
                    )
                    fc_outlet_T[ts_idx]  = outlet_stream.temperature
                    fc_outlet_P[ts_idx]  = outlet_stream.pressure
                    fc_outlet_h[ts_idx]  = outlet_stream.enthalpy
                    fc_mdot_gs[ts_idx]   = mdot * 1000.0
                except Exception as e:
                    print(f"   WARNING: fuel_cell_discharge re-run failed at t={t:.1f}s: {e}")

            fuel_cell_entry = {
                'stream_type': 'fuel_cell_discharge',
                'coupling_id': valve.coupling_id + '_discharge',
                'source_tank': source_tank_idx,
                'target_tank': -1,
                'times_h':    times_h,
                'components': _comp_data_to_list(comp_data_fc),
                # Extra arrays for mixing
                '_outlet_T_K':  fc_outlet_T,
                '_outlet_P_Pa': fc_outlet_P,
                '_outlet_h':    fc_outlet_h,
                '_mdot_gs':     fc_mdot_gs,
            }
            histories.append(fuel_cell_entry)

        # ----------------------------------------------------------------
        # Pass 3: mixed_fuel_cell  (enthalpy-weighted mix at mixer junction)
        # ----------------------------------------------------------------
        if main_discharge_entry is not None and fuel_cell_entry is not None:
            A_T   = main_discharge_entry['_outlet_T_K']
            A_P   = main_discharge_entry['_outlet_P_Pa']
            A_h   = main_discharge_entry['_outlet_h']
            A_mdot = main_discharge_entry['_mdot_gs']

            B_T   = fuel_cell_entry['_outlet_T_K']
            B_P   = fuel_cell_entry['_outlet_P_Pa']
            B_h   = fuel_cell_entry['_outlet_h']
            B_mdot = fuel_cell_entry['_mdot_gs']

            mix_fluid = self._coolprop_fluids[main_discharge_entry['source_tank']]
            mix_mdot_gs = np.full(n_timesteps, np.nan)
            mix_T_K     = np.full(n_timesteps, np.nan)
            mix_P_bar   = np.full(n_timesteps, np.nan)

            for ts_idx in range(n_timesteps):
                mA = A_mdot[ts_idx]
                mB = B_mdot[ts_idx]
                has_A = (not np.isnan(mA)) and mA > 0.0
                has_B = (not np.isnan(mB)) and mB > 0.0

                if not has_A and not has_B:
                    continue

                if has_A and has_B:
                    # Enthalpy-weighted mix
                    m_total = mA + mB
                    h_mix = (mA * A_h[ts_idx] + mB * B_h[ts_idx]) / m_total
                    # Use average outlet pressure (both should be ~3 bar)
                    P_mix = (A_P[ts_idx] + B_P[ts_idx]) / 2.0
                    try:
                        T_mix = PropsSI("T", "P", P_mix, "Hmass", h_mix, mix_fluid)
                    except Exception:
                        T_mix = np.nan
                    mix_mdot_gs[ts_idx] = m_total
                    mix_T_K[ts_idx]     = T_mix
                    mix_P_bar[ts_idx]   = P_mix / 1e5
                elif has_A:
                    mix_mdot_gs[ts_idx] = mA
                    mix_T_K[ts_idx]     = A_T[ts_idx]
                    mix_P_bar[ts_idx]   = A_P[ts_idx] / 1e5
                else:
                    mix_mdot_gs[ts_idx] = mB
                    mix_T_K[ts_idx]     = B_T[ts_idx]
                    mix_P_bar[ts_idx]   = B_P[ts_idx] / 1e5

            histories.append({
                'stream_type':      'mixed_fuel_cell',
                'coupling_id':      'fuel_cell_mixed',
                'source_tank':      -1,
                'target_tank':      -1,
                'times_h':          times_h,
                'components':       [],
                'mdot_gs':          mix_mdot_gs,
                'temperature_K':    mix_T_K,
                'pressure_bar':     mix_P_bar,
                'mdot_main_gs':     A_mdot,
                'mdot_discharge_gs': B_mdot,
            })

        return histories

    @silent_when_quiet
    def run_analysis(self, solver_method: str = "RK45", solver_config: dict = None) -> MultiTankResults:
        """
        Run complete tank system analysis.

        Args:
            solver_method: ODE solver to use ("RK45", "LSODA", "Radau", etc.)
            solver_config: Optional solver configuration parameters

        Returns:
            MultiTankResults with time series data
        """
        self._log_summary(f"\nStarting TankSystem-based simulation...")
        self._log_summary(f"   Analysis: {getattr(self.config, 'analysis_name', 'Tank System Analysis')}")
        self._log_summary(f"   Tanks: {len(self.tanks)}")
        self._log_summary(f"   Solver: {solver_method}")

        # Reset per-run history dicts so re-runs on the same object start clean

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

        self._log_summary(f"   Solver parameters: timestep={timestep}s, rtol={rtol:.0e}, atol={atol:.0e}, max_step={max_step}")

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
        elif solver_method == "RK4_FIXED":
            self.solver = RK4FixedSolver(**solver_params)
            self.solver.set_ode_function(self.ode_system)
        else:
            raise ValueError(f"Unknown solver method: {solver_method}")

        # Create initial conditions
        y0 = self.create_initial_state()
        initial_tanks = MultiTankState.from_state_vector(
            y0, self.tanks, 0.0, fluids=[self._coolprop_fluids[i] for i in range(len(self.tanks))]
        )
        edge_states = []
        for edge in self.coupling_valves:
            edge_states.extend(edge.initialize(initial_tanks.tank_states))
        y0 = np.concatenate([y0, edge_states])

        # Integration setup - use mission duration from configuration
        duration_seconds = float(self.config.MISSION_DURATION)

        # Optional override (primarily for tests/smoke runs)
        max_simulation_time = solver_config.get('max_simulation_time', None)
        if max_simulation_time is not None:
            duration_seconds = min(duration_seconds, float(max_simulation_time))

        duration_hours = duration_seconds / 3600.0
        t_span = (0.0, duration_seconds)

        self._log_summary("Integration setup:")
        self._log_summary(f"   • Duration: {duration_hours:.3f} hours ({duration_seconds:.0f} seconds)")
        self._log_summary(f"   • Time step: {timestep:.3f}s")
        self._log_summary(f"   • Expected points: {int(duration_seconds / timestep) + 1 if timestep > 0 else 'unknown'}")

        # Run integration
        self._log_summary("Starting ODE integration...")
        start_time = time.time()

        # Reset any stopping flags
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

        # Build event list: one terminal event per tank (min density) + optional target density
        all_events = self._create_min_density_events()
        if self.config.target_density is not None:
            all_events.append(self._create_density_event())
            self._log_summary(f"   Added target density event: {self.config.target_density:.1f} kg/m³")
        integration_params['events'] = all_events
        self._log_summary(f"   Added {len(self.tanks)} minimum-density stopping event(s)")

        self._log_debug(f"   Solver parameters: timestep={timestep}s, rtol={rtol}, atol={atol}, max_step={max_step}")

        from toplab.solver.hybrid import integrate_with_mode_switches
        t_span_ = integration_params.pop('t_span')
        y0_ = integration_params.pop('y0')
        stop_events = integration_params.pop('events')
        solution = integrate_with_mode_switches(
            self.solver, self.coupling_valves, stop_events, t_span_, y0_,
            switch_event=self._mode_switch_event, edge_flows=self._edge_flows_at,
            log=self._log_summary, **integration_params,
        )

        # Fallback stopping check: if events were missed (e.g. due to ODE exceptions
        # corrupting LSODA's internal step), truncate at the first minimum-density
        # crossing so results are never past the intended stop point.
        _n = len(self.tanks)
        events_already_fired = (
            hasattr(solution, 't_events')
            and solution.t_events
            and any(len(te) > 0 for te in solution.t_events[:_n])
        )
        if not events_already_fired:
            truncate_step = None
            truncate_tank = None
            for step_i in range(len(solution.t)):
                for tank_i in range(_n):
                    if self._tank_fluids.get(tank_i, '') == 'CH2':
                        # Pressure-based check for CH2 tanks
                        m_val = float(solution.y[tank_i * 5, step_i])
                        T_val = float(solution.y[tank_i * 5 + 1, step_i])
                        vol_val = self._cached_tank_properties[tank_i]['volume']
                        rho_val = m_val / max(vol_val, 1e-9)
                        try:
                            from CoolProp.CoolProp import PropsSI as _P
                            P_val = _P("P", "T", max(T_val, 14.0), "Dmass", max(rho_val, 1e-9), self._coolprop_fluids[tank_i])
                        except Exception:
                            continue
                        if P_val < self._min_pressures[tank_i]:
                            truncate_step = step_i
                            truncate_tank = tank_i
                            break
                    else:
                        mass_val = float(solution.y[tank_i * 5, step_i])
                        vol_val = self._cached_tank_properties[tank_i]['volume']
                        if mass_val / max(vol_val, 1e-9) < self._min_densities[tank_i]:
                            truncate_step = step_i
                            truncate_tank = tank_i
                            break
                if truncate_step is not None:
                    break
            if truncate_step is not None:
                t_stop = solution.t[truncate_step]
                self._log_summary(
                    f"   WARNING: stopping event was not caught by solver; "
                    f"truncating solution at Tank {truncate_tank + 1} stopping criterion "
                    f"(t={t_stop:.1f}s, {t_stop / 3600:.3f}h)."
                )
                solution.t = solution.t[:truncate_step + 1]
                solution.y = solution.y[:, :truncate_step + 1]
                solution.edge_flows = solution.edge_flows[:, :truncate_step + 1]

        elapsed_time = time.time() - start_time
        self._log_summary(f"Integration completed in {elapsed_time:.1f}s")
        self._log_summary(f"   Final time: {solution.t[-1]/3600:.2f} hours")
        self._log_summary(f"   Data points: {len(solution.t)}")

        # Report stop reason
        n_tanks = len(self.tanks)
        stopped_early = False
        if hasattr(solution, 't_events') and solution.t_events:
            for tank_idx in range(n_tanks):
                if tank_idx < len(solution.t_events) and len(solution.t_events[tank_idx]) > 0:
                    event_time = solution.t_events[tank_idx][0]
                    self._log_summary(
                        f"   Tank {tank_idx + 1} reached its minimum density before completing the mission "
                        f"(t={event_time:.1f}s, {event_time / 3600:.3f}h)."
                    )
                    stopped_early = True
            # Check target density event (appended after per-tank events)
            target_idx = n_tanks
            if (self.config.target_density is not None
                    and target_idx < len(solution.t_events)
                    and len(solution.t_events[target_idx]) > 0):
                event_time = solution.t_events[target_idx][0]
                self._log_summary(
                    f"   Target density reached at t={event_time:.1f}s ({event_time / 3600:.3f}h)."
                )
                stopped_early = True
        if not stopped_early:
            self._log_summary("   Mission completed: all tanks ran to the end of the mission profile.")

        self.edge_flow_history = solution.edge_flows

        # Convert solution to MultiTankState objects
        multi_tank_states = []
        for i in range(len(solution.t)):
            current_time = solution.t[i]

            # Create MultiTankState for this time step
            tank_states = []
            flow_data = []  # Store flow information for this time step

            for tank_idx in range(len(self.tanks)):
                state_idx = tank_idx * 5

                # Create tank state from solution
                tank_state = IsochoricTankState(
                    tank=self.tanks[tank_idx],
                    fuel_mass=solution.y[state_idx, i],
                    h2_temperature=solution.y[state_idx + 1, i],
                    structure_temperature=solution.y[state_idx + 2, i],
                    insulation_temperature=solution.y[state_idx + 3, i],
                    shell_temperature=solution.y[state_idx + 4, i],
                    fluid=self._coolprop_fluids[tank_idx],
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
                # Use gross coupling flows from simulation (tracks inflow/outflow per tank separately)
                gross_flows = self._gross_coupling_flows(solution.edge_flows[:, i])

                # Debug: Print coupling flows at this timestep
                any_nonzero = any(
                    gross_flows[i]['inflow'] > 1e-6 or gross_flows[i]['outflow'] > 1e-6
                    for i in gross_flows
                )
                # if any_nonzero:
                #     parts = []
                #     for i in gross_flows:
                #         inf = gross_flows[i]['inflow']
                #         out = gross_flows[i]['outflow']
                #         if inf > 1e-6 or out > 1e-6:
                #             parts.append(f"T{i}: +{inf*1000:.1f}/-{out*1000:.1f}g/s")
                #     # print(f"  Post-processing t={current_time:.1f}s: {', '.join(parts)} (from history)")
                # else:
                    # if i < 5 or i % 200 == 0:
                        # print(f"  Post-processing t={current_time:.1f}s: All coupling flows are zero")

                # Update flow data with gross coupling flows and set tank state attributes
                for tank_idx in range(len(tank_states)):
                    coupling_inflow = gross_flows[tank_idx]['inflow']
                    coupling_outflow = gross_flows[tank_idx]['outflow']

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