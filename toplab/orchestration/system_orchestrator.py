"""
System orchestrator for multi-tank hydrogen analysis.

This module provides the main orchestration class that integrates:
- Typed AnalysisConfig + Network
- MultiTankSystem DAE physics engine
- Mission sequence execution with stopping criteria
- Results validation and output generation

Key Features:
- Load YAML configurations via toplab.config.load_config
- Execute mission sequences (discharge -> refuel -> dormancy)
- Multi-tank coupling with pressure-triggered valves
- Enhanced stopping criteria (density + time-based)
- Production-ready orchestration framework

Author: Dante Raso
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from toplab.config import AnalysisConfig, load_config
from toplab.missions.factory import build_mission
from toplab.missions.mission import Mission
from toplab.network import SINK, Network
from toplab.network.coupling_rules import to_tank_system_rules

# Multi-tank DAE physics engine
from toplab.system.tank_system import TankSystem, TankSystemConfig, TankConfig
from toplab.utilities.tank_geometry import create_tank_from_fuel_mass
from toplab.utilities.output_control import silent_when_quiet

# Mission framework
from toplab.missions.isochoric_missions import DischargeMission

# Heat flow data collection for iHEX calculation
from toplab.dynamics.isochoric_dynamic_models import set_heat_flow_data_collector

# Utilities
from CoolProp.CoolProp import PropsSI


@dataclass(frozen=True)
class Stream:
    """One mission, or a ``fraction`` of it, served by a tank."""
    label: str
    mission: Mission
    fraction: float = 1.0


class SystemOrchestrator:
    """
    System orchestrator for running multi-tank hydrogen storage analysis.

    The SystemOrchestrator provides a high-level interface for configuring
    and running tank system analyses. It handles:
    - Mission profile configuration and parsing
    - Tank geometry sizing and configuration
    - Analysis execution and results collection
    """

    @silent_when_quiet
    def __init__(self, config: AnalysisConfig = None, config_path: str = None, verbosity: str = "summary"):
        self.verbosity = self._normalize_verbosity(verbosity)

        if config is None:
            if config_path is None:
                raise ValueError("Must provide either config or config_path")
            config = load_config(config_path)
        self.config = config
        self.network = Network(config)

        # One Mission per configured mission entry; the first is the primary (timing/ambient reference).
        self.missions = [build_mission(spec) for spec in config.missions]
        self.mission_profile = self.missions[0]
        print(f"   Loaded {len(self.missions)} mission(s); primary profile: {config.missions[0].profile}")

        self._streams = self._resolve_streams()

        # Flat per-tank dicts. Sizing stores its calculated initial state back into these.
        self._tank_data = [tank.geometry_data() for tank in config.tanks]

        self.tank_geometries = self._create_tank_geometries()
        print(f"   Created {len(self.tank_geometries)} tank geometries")

        self.tank_system_config = self._create_tank_system_config()

        self.couplings = self.network.coupling_edges()
        for coupling in self.couplings:
            print(f"   Coupling {coupling.edge_id}: {coupling.coupling_type}")

        self.tank_system = TankSystem(
            tank_geometries=self.tank_geometries,
            config=self.tank_system_config,
            coupling_rules=to_tank_system_rules(self.couplings),
            analysis_config=self.config,
            verbosity=self.verbosity,
        )
        print(f"   TankSystem DAE engine initialised")

        self._setup_mission_sequences()

        self.results = None

        # Heat flow data collection for iHEX calculations
        self.heat_flow_data = {
            't': [],           # Time [s]
            'qdot_disch': [],  # Discharge heat rate [W] (iHEX requirements from DAE)
            'qdot_ohex': [],   # oHEX heat rate [W] (calculated in post-processing)
            'mdot_disch': [],  # Discharge mass flow rate [kg/s]
            'T': [],           # Temperature [K]
            'rho': []          # Density [kg/m³]
        }
        self.validation_results = None

        print("System Orchestrator ready for mission execution!")
        print(f"   Tanks: {len(self.tank_geometries)}")
        print(f"   Missions: {len(self.config.missions)}")
        print(f"   Analysis type: {self.config.name}")

    def _resolve_streams(self) -> Dict[int, List[Stream]]:
        """Route every mission to the tank(s) that serve it, via the network."""
        streams: Dict[int, List[Stream]] = {}
        for spec, mission in zip(self.config.missions, self.missions):
            for tank_index, fraction in self.network.demand_targets(spec.assigned_to_node).items():
                tank_streams = streams.setdefault(tank_index, [])
                label = spec.parameters.get('column_name', f"Stream {len(tank_streams) + 1}")
                tank_streams.append(Stream(label, mission, fraction))

        assigned = {spec.assigned_to_node for spec in self.config.missions}
        for mixer_id, mixer in self.network.mixers.items():
            if mixer_id not in assigned:
                raise ValueError(
                    f"Mixer node {mixer_id} has no mission: set 'assigned_to_node: {mixer_id}' "
                    f"on the mission(s) feeding it."
                )
            print(f"   Mixer {mixer_id}: tank shares " +
                  ", ".join(f"Tank {tid}={f:.3f}" for tid, f in mixer.fractions.items()))
        return streams

    @staticmethod
    def _normalize_verbosity(verbosity: str) -> str:
        level = (verbosity or "summary").strip().lower()
        if level not in {"quiet", "summary", "debug"}:
            return "summary"
        return level

    def _required_fuel(self, tank_index: int) -> float:
        """Fuel [kg] the tank must carry: its share of every mission routed to it."""
        streams = self._streams.get(tank_index)
        if not streams:
            # Tanks with no demand of their own are sized against the primary mission.
            return self.mission_profile.required_fuel
        return sum(s.fraction * s.mission.required_fuel for s in streams)

    def _create_tank_geometries(self) -> List[Any]:
        """Create tank geometries from the tank specs."""
        tank_geometries = []

        for tank_index, (tank, geometry_data) in enumerate(zip(self.config.tanks, self._tank_data)):
            tank_id = tank.node_id
            print(f"   Creating Tank {tank_id} geometry...")

            phi = geometry_data.get('phi', 0.0)
            initial_pressure = float(geometry_data['initial_pressure'])

            if 'fuel_mass' in geometry_data:
                fuel_mass = float(geometry_data['fuel_mass'])
                initial_temperature = geometry_data.get('initial_temperature', 53.25)

                tank_geom = create_tank_from_fuel_mass(
                    fuel_mass=fuel_mass,
                    initial_pressure=initial_pressure,
                    initial_temperature=initial_temperature,
                    operating_pressure=initial_pressure * 1.125,  # 12.5% margin
                    phi=phi,
                    safety_margin=1.1,
                    liner_thickness=0.005,
                    insulation_thickness=0.05
                )
                print(f"     From fuel mass: {fuel_mass}kg → V={tank_geom.volume:.4f}m³")

            elif 'radius' in geometry_data:
                from toplab.tank_design.tank_shapes import CapsuleTank
                from toplab.materials.nist_materials import NISTMaterial

                radius = float(geometry_data['radius'])
                material = NISTMaterial.aluminum_6061T6_nist()
                operating_pressure = geometry_data.get('venting_pressure', geometry_data['initial_pressure'])
                tank_geom = CapsuleTank(radius=radius, material=material, operating_pressure=operating_pressure, phi=phi)
                print(f"     From radius: {radius}m → V={tank_geom.volume:.4f}m³")

            else:
                from toplab.materials.nist_materials import NISTMaterial

                material = NISTMaterial.aluminum_6061T6_nist()
                operating_pressure = geometry_data.get('venting_pressure', 450e5)  # Default 450 bar
                if 'venting_pressure' not in geometry_data:
                    print(f"WARNING: Using fallback: venting_pressure = {operating_pressure/1e5:.0f} bar for Tank {tank_id} (consider adding to YAML config)")

                tank_geom = self._create_mission_sized_tank(
                    geometry_data, material, operating_pressure, self._required_fuel(tank_index),
                    fluid=tank.coolprop)
                print(f"     Mission-sized geometry: V={tank_geom.volume:.4f}m³")

            tank_geometries.append(tank_geom)

        return tank_geometries

    def _create_mission_sized_tank(self, geometry_data, material, operating_pressure, required_fuel_mass, fluid):
        """
        Create a capsule tank sized to hold ``required_fuel_mass`` at the tank's initial state.

        Stores the calculated initial temperature/density back into ``geometry_data``.
        """
        from CoolProp.CoolProp import PropsSI
        import math

        print(f"   Sizing for fuel requirement: {required_fuel_mass:.2f} kg")

        # Tanks are always sized at discharge conditions, even for refuel/dormancy missions.
        if self.config.missions[0].type != 'discharge':
            geometry_data['initial_pressure'] = 400e5  # 400 bar - discharge pressure
            geometry_data['initial_density'] = 78.0    # kg/m³ - discharge density

        initial_pressure = float(geometry_data['initial_pressure'])  # Pa

        if 'initial_density' in geometry_data:
            target_density = float(geometry_data['initial_density'])  # kg/m³
            try:
                initial_temp = PropsSI("T", "P", initial_pressure, "D", target_density, fluid)
                print(f"     Solved initial temperature: {initial_temp:.2f} K (from P={initial_pressure/1e5:.0f} bar, ρ={target_density:.1f} kg/m³)")
            except Exception as e:
                print(f"     Warning: Could not solve for temperature from P,ρ: {e}")
                initial_temp = geometry_data.get('initial_temperature', 53.25)
                target_density = PropsSI("D", "P", initial_pressure, "T", initial_temp, fluid)
                print(f"     Using fallback: T={initial_temp:.2f} K, ρ={target_density:.1f} kg/m³")
        else:
            initial_temp = geometry_data.get('initial_temperature', 53.25)  # K
            target_density = PropsSI("D", "P", initial_pressure, "T", initial_temp, fluid)
            print(f"     Calculated density: {target_density:.2f} kg/m³ (from P={initial_pressure/1e5:.0f} bar, T={initial_temp:.1f} K)")

        fill_fraction = geometry_data.get('fill_fraction', 0.90)  # 90% fill default
        required_volume = required_fuel_mass / (target_density * fill_fraction)

        phi = float(geometry_data.get('phi', 0.0))

        # V = πr³(φ + 4/3), where φ = L/r and L is cylindrical section length.
        volume_factor = math.pi * (max(phi, 0.0) + 4.0 / 3.0)
        tank_radius = (required_volume / volume_factor) ** (1/3)

        print(f"     Fill fraction: {fill_fraction:.0%}")
        print(f"     Required volume: {required_volume:.4f} m³")
        print(f"     Tank radius: {tank_radius:.3f} m")
        print(f"     Tank phi (L/R): {phi:.3f}")

        geometry_data['calculated_initial_temperature'] = initial_temp
        geometry_data['calculated_initial_density'] = target_density

        from toplab.tank_design.tank_shapes import CapsuleTank
        return CapsuleTank(radius=tank_radius, material=material, operating_pressure=operating_pressure, phi=phi)

    def _create_tank_system_config(self) -> TankSystemConfig:
        """Create TankSystemConfig from the typed config."""
        mission = self.config.missions[0]

        tank_configs = []
        for tank, geometry_data in zip(self.config.tanks, self._tank_data):
            tank_id = tank.node_id
            initial_pressure = float(geometry_data['initial_pressure'])
            venting_pressure = float(geometry_data.get('venting_pressure', initial_pressure * 1.125))
            minimum_pressure = float(geometry_data.get('minimum_pressure', min(15e5, initial_pressure * 0.1)))
            if 'minimum_pressure' not in geometry_data:
                print(f"WARNING: Using fallback: minimum_pressure = {minimum_pressure/1e5:.1f} bar for Tank {tank_id} (calculated from initial_pressure)")

            if 'calculated_initial_temperature' in geometry_data:
                initial_temp = geometry_data['calculated_initial_temperature']
            elif 'initial_temperature' in geometry_data:
                initial_temp = geometry_data['initial_temperature']
            elif 'initial_density' in geometry_data:
                try:
                    density = float(geometry_data['initial_density'])
                    initial_temp = PropsSI("T", "P", initial_pressure, "D", density, tank.coolprop)
                except Exception:
                    initial_temp = 53.25  # Default cryogenic temperature
            else:
                initial_temp = 53.25  # Default

            tank_configs.append(TankConfig(
                P_INIT=initial_pressure,
                T_INIT=initial_temp,
                P_VENT=venting_pressure,
                P_MIN=minimum_pressure,
                MASS_INIT=geometry_data.get('initial_mass'),  # None if not specified
                scenario=mission.type.upper(),  # DISCHARGE, REFUEL, DORMANCY
                name=f"Tank{tank_id}"
            ))

            if tank.fluid == 'CH2':
                p_stop_bar = float(geometry_data.get('minimum_pressure', 0)) / 1e5
                print(f"   Tank {tank_id} config: P_init={initial_pressure/1e5:.0f}bar, T_init={initial_temp:.1f}K, P_stop={p_stop_bar:.1f}bar (CH2 pressure-based)")
            else:
                print(f"   Tank {tank_id} config: P_init={initial_pressure/1e5:.0f}bar, T_init={initial_temp:.1f}K, ρ_stop={geometry_data.get('minimum_density', 5.8):.1f}kg/m³")

        stopping_criteria = self.config.stopping_criteria

        # CH2 tanks use pressure-based stopping and have no minimum_density; skip them.
        if 'minimum_density' in stopping_criteria:
            minimum_density = float(stopping_criteria['minimum_density'])
        else:
            non_ch2 = [d for t, d in zip(self.config.tanks, self._tank_data) if t.fluid != 'CH2']
            ref_tank = non_ch2[0] if non_ch2 else self._tank_data[0]
            minimum_density = float(ref_tank.get('minimum_density', 5.8))
            if 'minimum_density' not in ref_tank and non_ch2:
                print(f"WARNING: Using fallback: minimum_density = {minimum_density} kg/m³ for stopping criteria (consider adding to YAML config)")

        target_density = None
        if 'target_density' in stopping_criteria:
            target_density = float(stopping_criteria['target_density'])
        elif 'maximum_density' in stopping_criteria:
            target_density = float(stopping_criteria['maximum_density'])

        outflow_demands, inflow_demands = self._build_demands()

        return TankSystemConfig(
            AMBIENT_TEMPERATURE=mission.ambient_temperature,
            MISSION_DURATION=self._calculate_mission_duration(),
            tanks=tank_configs,
            mission_profile=self.mission_profile,
            minimum_density=minimum_density,
            target_density=target_density,
            outflow_demands=outflow_demands,
            inflow_demands=inflow_demands,
        )

    def _build_demands(self):
        """Per-tank (0-based index) outflow and inflow Demands from the routed missions."""
        from toplab.demand import SumDemand, inflow_demand, outflow_demand

        outflows, inflows = {}, {}
        for tank_index, streams in self._streams.items():
            outflows[tank_index] = SumDemand(outflow_demand(s.mission).scaled(s.fraction) for s in streams)
            inflows[tank_index] = SumDemand(inflow_demand(s.mission).scaled(s.fraction) for s in streams)
        return outflows, inflows

    def _calculate_mission_duration(self) -> float:
        """Longest mission duration [s]."""
        return max(sum(section.duration for section in m.sections) for m in self.missions)

    def _setup_mission_sequences(self):
        """Setup mission-specific flow rates and sequences."""
        print("   Setting up mission sequences...")

        mission_count = len(self.config.missions)
        if mission_count > 1:
            print(f"     Multi-mission sequences detected ({mission_count} missions)")

        for tank_idx, demand in sorted(self.tank_system.config.outflow_demands.items()):
            print(f"     Tank {tank_idx + 1}: discharge demand {demand.total_mass():.2f} kg over {demand.duration:.0f} s")

        print(f"   ✓ Mission flow profile configured for DAE system")

    @silent_when_quiet
    def run_simulation(self, solver_method: str = None, solver_config: dict = None) -> Any:
        """
        Execute the complete multi-tank simulation.

        Args:
            solver_method: Override solver method (RK45, LSODA, etc.). If None, reads from config.solver.method
            solver_config: Optional solver configuration parameters (timestep, rtol, atol, max_step). If None, reads from config.solver

        Returns:
            MultiTankResults: Analysis results from DAE integration
        """
        # Read solver configuration from scenario config if not provided
        if solver_method is None or solver_config is None:
            solver_config_dict = self.config.solver

            if solver_method is None:
                solver_method = solver_config_dict.get('method', 'RK45')

            if solver_config is None:
                # Convert string values to float for numerical parameters
                rtol = solver_config_dict.get('rtol', 1e-6)
                atol = solver_config_dict.get('atol', 1e-9)
                max_step = solver_config_dict.get('max_step', None)

                # Handle string values from YAML
                if isinstance(rtol, str):
                    rtol = float(rtol)
                if isinstance(atol, str):
                    atol = float(atol)
                if isinstance(max_step, str):
                    max_step = float(max_step)

                solver_config = {
                    'rtol': rtol,
                    'atol': atol,
                    'max_step': max_step
                }

        print("\nStarting simulation...")
        print(f"   Analysis: {self.config.name}")
        print(f"   Tanks: {len(self.tank_geometries)}")
        print(f"   Mission: {self.config.missions[0].type}")
        print(f"   Solver: {solver_method} (from {'config' if solver_method != 'RK45' else 'default'})")
        if solver_config:
            print(f"   Tolerances: rtol={solver_config.get('rtol')}, atol={solver_config.get('atol')}")

        start_time = time.time()

        try:
            # Clear previous heat flow data and set up collector
            for key in self.heat_flow_data:
                self.heat_flow_data[key].clear()
            set_heat_flow_data_collector(self.heat_flow_data)
            print("   Heat flow data collector configured for iHEX extraction")

            # Run the MultiTankSystem DAE simulation
            self.results = self.tank_system.run_analysis(solver_method, solver_config)

            end_time = time.time()
            wall_time = end_time - start_time

            print("Simulation completed successfully!")
            print(f"   Wall time: {wall_time:.2f} seconds")
            print(f"   Final time: {self.results.times[-1]/3600:.2f} hours")
            print(f"   Data points: {len(self.results.times)}")

            # Debug heat flow data collection
            qdot_disch_count = len(self.heat_flow_data['qdot_disch'])
            qdot_disch_nonzero = sum(1 for q in self.heat_flow_data['qdot_disch'] if abs(q) > 1e-3)
            max_qdot_disch = max(abs(q) for q in self.heat_flow_data['qdot_disch']) if self.heat_flow_data['qdot_disch'] else 0
            print(f"   Heat flow data: {qdot_disch_count} points, {qdot_disch_nonzero} non-zero, max = {max_qdot_disch:.1f}W")

            # Display final states
            final_multi_state = self.results.multi_tank_states[-1]
            for i in range(len(self.tank_geometries)):
                tank_state = final_multi_state.get_tank_state(i)
                print(f"   Tank {i+1}: m={tank_state.fuel_mass:.2f}kg, "
                        f"T_H2={tank_state.h2_temperature:.1f}K, "
                      f"ρ={tank_state.density:.1f}kg/m³")

            return self.results

        except Exception as e:
            print(f"ERROR: Simulation failed: {e}")
            raise

    @silent_when_quiet
    def validate_results(self) -> Dict[str, Any]:
        """
        Validate multi-tank simulation results.

        Returns:
            Dictionary with validation results
        """
        if not self.results:
            raise ValueError("No results available. Run simulation first.")

        print("\nValidating simulation results...")
        from toplab.post.mass_balance import check_mass_balance

        rtol = 1e-3
        balances = check_mass_balance(self.results)
        for b in balances:
            status = "ok" if b.passed(rtol) else "FAILED"
            print(f"   Tank {b.tank_index + 1} mass balance: residual {b.residual:.4g} kg "
                  f"({b.relative_residual:.2e} of initial mass) {status}")
        self.validation_results = {
            "overall": all(b.passed(rtol) for b in balances),
            "mass_balance": balances,
        }

        return self.validation_results

    def _calculate_ohex_requirements(self, tank_index: int = 0) -> List[float]:
        """
        Calculate OHEX (Outboard Heat Exchanger) heat requirements.

        Uses enthalpy difference method: Q_oHEX = mdot * (h_target - h_current)
        where h_target is at standard fuel cell inlet conditions.

        Args:
            tank_index: Index of tank to calculate for

        Returns:
            List of OHEX heat requirements [W] for each time point
        """
        if not self.results:
            return []

        # Get OHEX target conditions from config
        hex_config = self.config.output_plots.get('heat_exchanger_requirements', {})
        target_temp = float(hex_config.get('ohex_target_temperature', 200.0))  # K
        target_press = float(hex_config.get('ohex_target_pressure', 20e5))    # Pa

        # Log fallback usage for heat exchanger parameters
        if 'ohex_target_temperature' not in hex_config:
            print(f"WARNING: Using fallback: ohex_target_temperature = {target_temp} K (now available in YAML config)")
        if 'ohex_target_pressure' not in hex_config:
            print(f"WARNING: Using fallback: ohex_target_pressure = {target_press/1e5:.0f} bar (now available in YAML config)")

        fluid = self.config.tanks[tank_index].coolprop
        try:
            # Calculate target enthalpy (constant for all time points)
            h_target = PropsSI("Hmass", "T", target_temp, "P", target_press, fluid)
        except Exception as e:
            print(f"   WARNING: Could not calculate OHEX target enthalpy: {e}")
            return [0.0] * len(self.results.times)

        qdot_ohex = []
        tank_series = self.results.get_tank_series(tank_index)

        # Add statistics tracking
        max_q_ohex = 0.0
        valid_points = 0
        print(f"   Calculating OHEX for {len(tank_series.states)} time points...")

        for i, state in enumerate(tank_series.states):
            try:
                # Get current state conditions
                T_current = state.h2_temperature
                m_current = state.fuel_mass

                # Get tank volume for density calculation
                tank_volume = self.tank_geometries[tank_index].volume
                rho_current = m_current / tank_volume  # kg/m³

                # Get actual mass flow rate used during simulation (stored in tank state)
                mass_rate = abs(state.outflow_rate)

                if T_current > 0 and rho_current > 0 and mass_rate > 0:
                    # Calculate current pressure and enthalpy with saturation-aware helpers
                    try:
                        from toplab.fluids.coolprop_safe import safe_pressure_from_T_rho, safe_enthalpy
                        p_current = safe_pressure_from_T_rho(T_current, rho_current, fluid)
                        # Assume gas enthalpy for outlet stream when in two-phase
                        h_current = safe_enthalpy(T_current, p_current, assume_gas_when_twophase=True, fluid=fluid)
                    except Exception:
                        # Fallback to direct CoolProp calls
                        p_current = PropsSI("P", "T", T_current, "Dmass", rho_current, fluid)
                        h_current = PropsSI("Hmass", "T", T_current, "P", p_current, fluid)

                    # Calculate OHEX heat requirement
                    q_ohex = mass_rate * (h_target - h_current)  # [W]
                    q_ohex_final = max(0.0, q_ohex)  # Ensure non-negative
                    qdot_ohex.append(q_ohex_final)

                    # Track statistics
                    if q_ohex_final > max_q_ohex:
                        max_q_ohex = q_ohex_final
                    if q_ohex_final > 0:
                        valid_points += 1
                else:
                    qdot_ohex.append(0.0)

            except Exception:
                # Silently handle CoolProp errors (common at extreme conditions)
                qdot_ohex.append(0.0)

        print(f"   OHEX calculated: {valid_points}/{len(qdot_ohex)} points, max = {max_q_ohex:.0f}W")
        return qdot_ohex

    def _calculate_ihex_requirements(self, tank_index: int = 0) -> List[float]:
        """
        Extract iHEX requirements from DAE simulation data.

        iHEX requirements come directly from the qdot_disch values calculated
        during the DAE integration in the isochoric dynamic models.
        This represents the Configuration B discharge heat requirement that
        is computed as part of the energy balance to maintain pressure constraints.

        Args:
            tank_index: Index of tank to calculate for (currently unused - uses global heat flow data)

        Returns:
            List of iHEX heat requirements [W] for each time point from DAE simulation
        """
        if not self.results:
            return []

        print(f"   Extracting iHEX requirements from DAE simulation data...")

        # Check if heat flow data was collected during simulation
        if not hasattr(self, 'heat_flow_data') or not self.heat_flow_data['qdot_disch']:
            print("   WARNING: No heat flow data collected, returning zero iHEX requirements")
            return [0.0] * len(self.results.times)

        # Get raw data from DAE integration
        heat_times = self.heat_flow_data['t']
        heat_qdot_disch = self.heat_flow_data['qdot_disch']

        if len(heat_times) != len(heat_qdot_disch):
            print(f"   WARNING: Heat flow time/data length mismatch: {len(heat_times)} vs {len(heat_qdot_disch)}")
            return [0.0] * len(self.results.times)

        if len(heat_times) == 0:
            print("   WARNING: No heat flow data collected during integration")
            return [0.0] * len(self.results.times)

        # Interpolate heat flow data to match results timeline
        import numpy as np
        results_times = np.array(self.results.times)
        heat_times_array = np.array(heat_times)
        heat_qdot_array = np.array(heat_qdot_disch)

        # Handle length mismatch by interpolation
        if len(heat_times) != len(self.results.times):
            print(f"   Interpolating heat flow data: {len(heat_times)} → {len(self.results.times)} points")
            qdot_ihex = np.interp(results_times, heat_times_array, heat_qdot_array).tolist()
        else:
            qdot_ihex = heat_qdot_disch.copy()

        # Statistics
        valid_points = sum(1 for q in qdot_ihex if abs(q) > 1e-3)
        max_q_ihex = max(abs(q) for q in qdot_ihex) if qdot_ihex else 0.0

        print(f"   iHEX extracted from DAE: {valid_points}/{len(qdot_ihex)} points, max = {max_q_ihex:.0f}W")
        return qdot_ihex

    def _compute_per_stream_outflows(self, tank_index: int) -> Optional[Dict[str, np.ndarray]]:
        """
        For tanks with multiple discharge streams (e.g. FC + GT dual-flow), return a dict
        mapping each stream label → outflow array [g/s, negative convention] at simulation times.
        Returns None when only a single stream is present.
        """
        if not self.results:
            return None
        streams = self._streams.get(tank_index, [])
        if len(streams) < 2:
            return None

        from toplab.demand import outflow_demand
        times = self.results.times
        result = {}
        for stream in streams:
            demand = outflow_demand(stream.mission).scaled(stream.fraction)
            flows_gs = np.array([-demand.mass_flow(t) * 1000.0 for t in times])
            result[stream.label] = flows_gs
        return result

    @silent_when_quiet
    def generate_plots(self, save_path: str = None) -> Any:
        """
        Generate plots from simulation results.

        Args:
            save_path: Optional path to save plot file

        Returns:
            Figure object from plotting
        """
        if not self.results:
            raise ValueError("No results available. Run simulation first.")

        print("Generating multi-tank analysis plots...")

        try:
            # Import the new plotting module
            from toplab.plotting.multi_tank_plotting import DelftColourPlotter

            # Get plotting configuration from config
            plot_config = self.config.output_plots
            style_config = plot_config.get('style', {})

            # Get output formats from config (defaults to ['png'] if not specified)
            output_config = self.config.output
            output_formats = output_config.get('plot_formats', ['png'])
            enable_pgf = 'pgf' in output_formats

            if enable_pgf:
                print(f"   PGF output enabled for LaTeX compatibility")

            # Load font configuration and update global font settings
            font_config = style_config.get('font', {})
            if font_config:
                from toplab.plotting.plot_style_sb import update_font_settings

                master_size = font_config.get('master_size')
                legend_size = font_config.get('legend_size')
                font_name = font_config.get('font_name')

                # Update global font settings if specified in config
                update_font_settings(
                    master_size=master_size,
                    legend_size=legend_size,
                    font_name=font_name
                )

                print(f"   Font configuration loaded: master={master_size}, legend={legend_size}, font='{font_name}'")

            # Create plotter with analysis name and styling options from config
            plotter = DelftColourPlotter(
                analysis_name=self.config.name,
                use_greyscale=style_config.get('use_greyscale', False),
                enable_multi_tank_overlay=style_config.get('enable_multi_tank_overlay', False),
                output_formats=output_formats,
                enable_pgf=enable_pgf
            )

            # Get figure_prefix from config (if specified)
            figure_prefix = plot_config.get('figure_prefix', None)
            if figure_prefix:
                print(f"   Using figure prefix from config: '{figure_prefix}'")

            return self._generate_single_mission_plots(plotter, save_path, figure_prefix)

        except Exception as e:
            print(f"   WARNING: Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_single_mission_plots(self, plotter, save_path: str = None, figure_prefix: str = None):
        """Generate plots for single mission analysis."""
        figures = []

        # Check if multi-tank overlay is enabled and we have multiple tanks
        num_tanks = len(self.tank_geometries)
        use_overlay = plotter.enable_multi_tank_overlay and num_tanks > 1

        if use_overlay:
            print(f"   Plotting multi-tank evolution (overlay mode, {num_tanks} tanks)...")

            # Create reference lines from first tank configuration (for overlay mode)
            tank_config_data = self._tank_data[0]
            ref_line_config = self.config.output_plots.get('reference_lines', {})
            reference_lines = plotter.create_reference_lines_from_config(tank_config_data, ref_line_config)

            # Add mission ambient temperature if available
            mission = self.config.missions[0]
            reference_lines['T_ambient'] = mission.ambient_temperature

            # Generate single overlay plot for all tanks
            overlay_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem
                overlay_save_path = save_dir / f"{base_name}_evolution_all_tanks{save_ext}"

            # Get event lines configuration from tank_states config
            tank_states_config = self.config.output_plots.get('tank_states', {})
            event_lines = tank_states_config.get('event_lines', [])

            # Get axis limits and legend locations from config
            xlim = tank_states_config.get('xlim', None)
            ylim_pressure = tank_states_config.get('ylim_pressure', None)
            ylim_temperature = tank_states_config.get('ylim_temperature', None)
            ylim_density = tank_states_config.get('ylim_density', None)
            legend_locations = tank_states_config.get('legend_locations', None)

            fig = plotter.plot_tank_evolution(
                results=self.results,
                tank_index=0,  # Not used in overlay mode
                reference_lines=reference_lines,
                reference_lines_config=ref_line_config,
                event_lines=event_lines,
                save_path=str(overlay_save_path) if overlay_save_path else None,
                overlay_all_tanks=True,
                xlim=xlim,
                ylim_pressure=ylim_pressure,
                ylim_temperature=ylim_temperature,
                ylim_density=ylim_density,
                legend_locations=legend_locations
            )
            figures.append(fig)

        # Generate per-tank plots (individual evolution plots if not overlaying, plus other plot types)
        for tank_idx in range(num_tanks):
            # Create reference lines from tank configuration
            tank_config_data = self._tank_data[tank_idx]
            ref_line_config = self.config.output_plots.get('reference_lines', {})
            reference_lines = plotter.create_reference_lines_from_config(tank_config_data, ref_line_config)

            # Add mission ambient temperature if available
            mission = self.config.missions[0]
            reference_lines['T_ambient'] = mission.ambient_temperature

            # Generate individual tank evolution plot (only if not using overlay mode)
            if not use_overlay:
                print(f"   Plotting Tank {tank_idx + 1} evolution...")

                # Read tank state plotting configuration
                tank_states_config = self.config.output_plots.get('tank_states', {})
                event_lines = tank_states_config.get('event_lines', [])

                # Axis limits and legend locations
                xlim = tank_states_config.get('xlim', None)
                ylim_pressure = tank_states_config.get('ylim_pressure', None)
                ylim_temperature = tank_states_config.get('ylim_temperature', None)
                ylim_density = tank_states_config.get('ylim_density', None)
                legend_locations = tank_states_config.get('legend_locations', None)

                # Separate figures config and dpi
                separate_figures = tank_states_config.get('separate_figures', False)
                dpi = tank_states_config.get('dpi', 900)

                # File/dir naming
                tank_save_path = None
                save_dir = None
                filename_prefix = None
                if save_path:
                    from pathlib import Path
                    save_dir = Path(save_path).parent
                    save_ext = Path(save_path).suffix or '.png'
                    base_name = figure_prefix if figure_prefix else Path(save_path).stem
                    ts_filename = tank_states_config.get('filename', 'evolution')

                    if separate_figures:
                        # Provide a prefix; function will append _pressure/_temperature/_density and keep extension from save_path
                        filename_prefix = f"{base_name}_{ts_filename}_tank{tank_idx + 1}"
                    else:
                        tank_save_path = save_dir / f"{base_name}_{ts_filename}_tank{tank_idx + 1}{save_ext}"
                result_figs = plotter.plot_tank_evolution(
                    results=self.results,
                    tank_index=tank_idx,
                    reference_lines=reference_lines,
                    reference_lines_config=ref_line_config,
                    event_lines=event_lines,
                    save_path=str(tank_save_path) if (tank_save_path and not separate_figures) else None,
                    xlim=xlim,
                    ylim_pressure=ylim_pressure,
                    ylim_temperature=ylim_temperature,
                    ylim_density=ylim_density,
                    legend_locations=legend_locations,
                    separate_figures=separate_figures,
                    save_dir=str(save_dir) if (save_dir and separate_figures) else None,
                    filename_prefix=filename_prefix,
                    dpi=dpi
                )
                # Handle return type (dict of figures when separate_figures=True)
                if isinstance(result_figs, dict):
                    figures.extend(result_figs.values())
                else:
                    figures.append(result_figs)

            # Generate density-temperature plot for this tank
            dt_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem
                dt_save_path = save_dir / f"{base_name}_density_temperature_tank{tank_idx + 1}{save_ext}"

            # Get density-temperature plot configuration
            dt_config = self.config.output_plots.get('density_temperature', {})

            # Extract plot parameters from config with defaults
            isobar_pressures = dt_config.get('isobar_pressures', [450, 400, 100, 15, 5])
            show_reference_pressures = dt_config.get('show_reference_pressures', True)
            include_saturation_line = dt_config.get('include_saturation_line', True)
            include_isobars = dt_config.get('include_isobars', True)

            # Only pass reference pressures if config allows it
            ref_pressures = reference_lines if show_reference_pressures else None

            # Respect the configuration settings for saturation line and isobars
            # (The old logic was unnecessarily disabling these for multi-tank systems)
            show_saturation = include_saturation_line
            show_isobars = include_isobars

            # Extract valve events for multi-tank systems
            valve_events = None
            if len(self.tank_geometries) > 1:
                try:
                    valve_events = plotter._extract_valve_events(self.results)
                except Exception as e:
                    print(f"   WARNING: Valve event extraction failed for tank {tank_idx}: {e}")

            # Get arrow configuration from density_temperature config
            dt_config = self.config.output_plots.get('density_temperature', {})
            arrow_position = dt_config.get('arrow_position', 0.25)
            arrow_size = dt_config.get('arrow_size', 1.0)
            valve_marker_size = dt_config.get('valve_marker_size', 8.0)
            legend_location = dt_config.get('legend_location', 'upper right')
            xlim = dt_config.get('xlim', None)
            ylim = dt_config.get('ylim', None)

            dt_fig = plotter.plot_density_temperature(
                results=self.results,
                tank_index=tank_idx,
                include_saturation_line=show_saturation,
                include_isobars=show_isobars,
                isobar_pressures=isobar_pressures,
                reference_pressures=ref_pressures,
                valve_events=valve_events,
                arrow_position=arrow_position,
                arrow_size=arrow_size,
                valve_marker_size=valve_marker_size,
                xlim=xlim,
                ylim=ylim,
                save_path=str(dt_save_path) if dt_save_path else None,
                legend_location=legend_location
            )
            figures.append(dt_fig)

            # Get mass flow plot configuration
            mf_config = self.config.output_plots.get('mass_flows', {})

            # Generate mass flow plot for this tank
            mf_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem

                # Use filename from config if available
                mf_filename = mf_config.get('filename', 'mass_flows')
                mf_save_path = save_dir / f"{base_name}_{mf_filename}_tank{tank_idx + 1}{save_ext}"

            # Check if mass flow plots are enabled
            if mf_config.get('enabled', True):
                # Extract plot parameters from config with defaults
                include_venting_flow = mf_config.get('include_venting_flow', True)
                include_coupling_flows = mf_config.get('include_coupling_flows', True)
                xlim = mf_config.get('xlim', None)
                ylim = mf_config.get('ylim', None)
                legend_location = mf_config.get('legend_location', 'best')

                per_stream = self._compute_per_stream_outflows(tank_idx)

                mf_fig = plotter.plot_mass_flows(
                    results=self.results,
                    tank_index=tank_idx,
                    include_venting_flow=include_venting_flow,
                    include_coupling_flows=include_coupling_flows,
                    save_path=str(mf_save_path) if mf_save_path else None,
                    xlim=xlim,
                    ylim=ylim,
                    legend_location=legend_location,
                    per_stream_outflows=per_stream,
                )
                figures.append(mf_fig)

            # Generate thermal-network heat-flow plot for this tank
            thermal_flow_config = self.config.output_plots.get('thermal_heat_flows', {})
            if thermal_flow_config.get('enabled', True):
                thermal_flow_save_path = None
                if save_path:
                    from pathlib import Path
                    save_dir = Path(save_path).parent
                    save_ext = Path(save_path).suffix or '.png'
                    base_name = figure_prefix if figure_prefix else Path(save_path).stem
                    thermal_flow_filename = thermal_flow_config.get('filename', 'thermal_heat_flows')
                    thermal_flow_save_path = save_dir / f"{base_name}_{thermal_flow_filename}_tank{tank_idx + 1}{save_ext}"

                thermal_flow_fig = plotter.plot_thermal_heat_flows(
                    results=self.results,
                    thermal_model=self.tank_system.thermal_models[tank_idx],
                    tank_index=tank_idx,
                    save_path=str(thermal_flow_save_path) if thermal_flow_save_path else None,
                    xlim=thermal_flow_config.get('xlim'),
                    ylim=thermal_flow_config.get('ylim'),
                    legend_location=thermal_flow_config.get('legend_location', 'best'),
                )
                figures.append(thermal_flow_fig)

            # Get heat exchanger plot configuration
            hex_config = self.config.output_plots.get('heat_exchanger_requirements', {})

            # Generate heat exchanger plot for this tank (if enabled)
            if hex_config.get('enabled', False):
                hex_save_path = None
                if save_path:
                    from pathlib import Path
                    save_dir = Path(save_path).parent
                    save_ext = Path(save_path).suffix or '.png'
                    base_name = figure_prefix if figure_prefix else Path(save_path).stem

                    # Use filename from config if available
                    hex_filename = hex_config.get('filename', 'hex')
                    hex_save_path = save_dir / f"{base_name}_{hex_filename}_tank{tank_idx + 1}{save_ext}"

                # Calculate OHEX data if needed
                qdot_ohex = self._calculate_ohex_requirements(tank_idx) if hex_config.get('include_ohex', True) else []

                # Calculate iHEX data if needed
                qdot_ihex = self._calculate_ihex_requirements(tank_idx) if hex_config.get('include_ihex', True) else [0.0] * len(self.results.times)

                # Prepare heat exchanger data
                heat_exchanger_data = {
                    'times': self.results.times / 3600.0,  # Convert to hours
                    'ihex_requirements': qdot_ihex,
                    'ohex_requirements': qdot_ohex
                }

                xlim = hex_config.get('xlim', None)
                ylim = hex_config.get('ylim', None)
                legend_location = hex_config.get('legend_location', 'best')

                hex_fig = plotter.plot_heat_exchanger_requirements(
                    heat_exchanger_data=heat_exchanger_data,
                    tank_index=tank_idx,
                    include_ohex=hex_config.get('include_ohex', True),
                    include_total=hex_config.get('include_total', True),
                    save_path=str(hex_save_path) if hex_save_path else None,
                    xlim=xlim,
                    ylim=ylim,
                    legend_location=legend_location
                )
                figures.append(hex_fig)

        # Generate pressure requirements plot for mission-adaptive systems (not per-tank)
        pressure_req_config = self.config.output_plots.get('pressure_requirements', {})
        pressure_req_enabled = pressure_req_config.get('enabled', False)

        if pressure_req_enabled:
            pressure_req_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem

                # Use filename from config if available
                pressure_req_filename = pressure_req_config.get('filename', 'pressure_requirements')
                pressure_req_save_path = save_dir / f"{base_name}_{pressure_req_filename}{save_ext}"

            # Get tank index from config (default to 1 for LH2 tank)
            tank_index = pressure_req_config.get('tank_index', 1)

            try:
                pressure_req_fig = plotter.plot_pressure_requirements(
                    orchestrator=self,
                    tank_index=tank_index,
                    save_path=str(pressure_req_save_path) if pressure_req_save_path else None
                )
                figures.append(pressure_req_fig)
            except Exception as e:
                print(f"   WARNING: Pressure requirements plot failed: {e}")

        # Count plots generated
        mass_flows_enabled = self.config.output_plots.get('mass_flows', {}).get('enabled', True)
        thermal_heat_flows_enabled = self.config.output_plots.get('thermal_heat_flows', {}).get('enabled', True)
        heat_exchanger_enabled = self.config.output_plots.get('heat_exchanger_requirements', {}).get('enabled', False)

        # -------------------------------------------------------------------
        # Peripheral component chain plots (HEX, compressor, pump…)
        # -------------------------------------------------------------------
        from toplab.coupling.inter_tank_coupling import ProportionalSplitCoupling, PressureTriggeredDischarge

        # Only attempt if the tank_system has coupling valves
        if hasattr(self, 'tank_system') and self.tank_system is not None:
            # --- Component chain plots (one per coupling with components) ---
            try:
                stream_histories = self.tank_system.compute_coupling_stream_history(self.results)
                for sh in stream_histories:
                    # Only plot peripheral-component charts for the 5% split chain
                    if sh.get('stream_type', 'split_chain') != 'split_chain':
                        continue
                    if not sh['components']:
                        continue
                    pc_save_path = None
                    if save_path:
                        from pathlib import Path
                        save_dir = Path(save_path).parent
                        save_ext = Path(save_path).suffix or '.png'
                        base_name = figure_prefix if figure_prefix else Path(save_path).stem
                        safe_id = sh['coupling_id'].replace(' ', '_').replace('/', '_').replace('→', '_to_')
                        pc_save_path = save_dir / f"{base_name}_components_{safe_id}{save_ext}"

                    pc_fig = plotter.plot_peripheral_components(
                        stream_history=sh,
                        save_path=str(pc_save_path) if pc_save_path else None,
                    )
                    figures.append(pc_fig)

                    # --- HEX / compressor power & energy plot ---
                    try:
                        _pwr_cfg = self.config.output_plots.get('hex_compressor_power', {})
                        if _pwr_cfg.get('enabled', True):
                            pwr_save_path = None
                            if save_path:
                                from pathlib import Path as _Path
                                _save_dir = _Path(save_path).parent
                                _save_ext = _Path(save_path).suffix or '.png'
                                _base_name = figure_prefix if figure_prefix else _Path(save_path).stem
                                _safe_id = sh['coupling_id'].replace(' ', '_').replace('/', '_').replace('→', '_to_')
                                _pwr_fname = _pwr_cfg.get('filename', 'power')
                                pwr_save_path = _save_dir / f"{_base_name}_{_pwr_fname}_{_safe_id}{_save_ext}"

                            pwr_fig = plotter.plot_hex_compressor_power(
                                stream_history=sh,
                                save_path=str(pwr_save_path) if pwr_save_path else None,
                                xlim=_pwr_cfg.get('xlim', None),
                                ylim_power=_pwr_cfg.get('ylim_power', None),
                                ylim_energy=_pwr_cfg.get('ylim_energy', None),
                                legend_location=_pwr_cfg.get('legend_location_power', 'best'),
                                fluid=self.config.tanks[sh['source_tank']].coolprop,
                            )
                            figures.append(pwr_fig)
                    except Exception as _pwr_e:
                        print(f"   WARNING: HEX/compressor power plot failed: {_pwr_e}")

            except Exception as e:
                print(f"   WARNING: Peripheral component plots failed: {e}")
                import traceback
                traceback.print_exc()

            # --- Fuel cell inlet plot: find the PressureTriggeredDischarge sink ---
            try:
                discharge_tank_idx = None
                for valve in self.tank_system.coupling_valves:
                    if isinstance(valve, PressureTriggeredDischarge):
                        discharge_tank_idx = valve.source_tank
                        break

                if discharge_tank_idx is not None:
                    _fc_cfg = self.config.output_plots.get('fuel_cell_inlet', {})
                    if _fc_cfg.get('enabled', True):
                        fc_save_path = None
                        if save_path:
                            from pathlib import Path
                            save_dir = Path(save_path).parent
                            save_ext = Path(save_path).suffix or '.png'
                            base_name = figure_prefix if figure_prefix else Path(save_path).stem
                            _fc_fname = _fc_cfg.get('filename', 'fuel_cell_inlet')
                            fc_save_path = save_dir / f"{base_name}_{_fc_fname}{save_ext}"

                        fc_fig = plotter.plot_fuel_cell_inlet(
                            results=self.results,
                            source_tank_idx=discharge_tank_idx,
                            stream_histories=stream_histories,
                            save_path=str(fc_save_path) if fc_save_path else None,
                            xlim=_fc_cfg.get('xlim', None),
                            ylim_mdot=_fc_cfg.get('ylim_mdot', None),
                            ylim_temp=_fc_cfg.get('ylim_temp', None),
                            ylim_pres=_fc_cfg.get('ylim_pres', None),
                            legend_location=_fc_cfg.get('legend_location', 'best'),
                        )
                        figures.append(fc_fig)
            except Exception as e:
                print(f"   WARNING: Fuel cell inlet plot failed: {e}")
                import traceback
                traceback.print_exc()
        # -------------------------------------------------------------------

        plots_per_tank = 2  # Always: evolution + density-temperature
        if mass_flows_enabled:
            plots_per_tank += 1
        if thermal_heat_flows_enabled:
            plots_per_tank += 1
        if heat_exchanger_enabled:
            plots_per_tank += 1

        total_plots = len(self.results.tank_metadata) * plots_per_tank
        if pressure_req_enabled:
            total_plots += 1  # Add pressure requirements plot

        plot_types = ['evolution', 'density-temperature']
        if mass_flows_enabled:
            plot_types.append('mass flows')
        if thermal_heat_flows_enabled:
            plot_types.append('thermal heat flows')
        if heat_exchanger_enabled:
            plot_types.append('heat exchanger requirements')
        if pressure_req_enabled:
            plot_types.append('pressure requirements')

        print(f"   Generated {len(figures)} tank plots ({plots_per_tank} plots per tank: {' + '.join(plot_types)})")

        # Return the first figure (or all figures if multiple tanks)
        return figures[0] if len(figures) == 1 else figures

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the scenario configuration and results."""
        summary = {
            'scenario': {
                'name': self.config.name,
                'description': self.config.description,
                'version': self.config.version
            },
            'tanks': {
                'count': len(self.tank_geometries),
                'geometries': [
                    {
                        'volume': tank.volume,
                        'radius': getattr(tank, 'radius', 'N/A')
                    }
                    for tank in self.tank_geometries
                ]
            },
            'missions': {
                'count': len(self.config.missions),
                'sequences': [
                    {
                        'type': mission.type,
                        'profile': mission.profile,
                        'ambient_temperature': mission.ambient_temperature
                    }
                    for mission in self.config.missions
                ]
            },
            'tank_materials': {
                str(tank.node_id): sorted(tank.materials) for tank in self.config.tanks
            }
        }

        # Add results summary if available
        if self.results:
            final_multi_state = self.results.multi_tank_states[-1]
            summary['results'] = {
                'final_time_hours': self.results.times[-1] / 3600,
                'data_points': len(self.results.times),
                'final_tank_states': [
                    {
                        'fuel_mass': final_multi_state.get_tank_state(i).fuel_mass,
                        'h2_temperature': final_multi_state.get_tank_state(i).h2_temperature,
                        'density': final_multi_state.get_tank_state(i).density
                    }
                    for i in range(len(self.tank_geometries))
                ]
            }

        if self.validation_results:
            summary['validation'] = self.validation_results

        return summary

    def get_comprehensive_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary with all input/output parameters for benchmarking."""
        import numpy as np

        if not self.results or not self.results.multi_tank_states:
            raise RuntimeError("No simulation results available. Run simulation first.")

        summary = {
            'analysis_name': self.config.name,
            'tank_count': len(self.tank_geometries),
            'tanks': {}
        }

        for tank_idx in range(len(self.tank_geometries)):
            tank_name = f"Tank_{tank_idx + 1}"
            # Geometry and mass properties already computed during TankSystem initialisation
            tank_props = self.tank_system._cached_tank_properties[tank_idx]

            # TankSystemConfig holds the values actually used in the simulation
            tank_sys_cfg = self.tank_system.config.tanks[tank_idx]

            # Per-tank material thicknesses via the same path used by _get_tank_properties
            mat_cfg = self.config.tanks[tank_idx].materials
            liner_thickness_m     = float(mat_cfg['liner']['thickness'])
            insulation_thickness_m = float(mat_cfg['insulation']['thickness'])

            initial_state = self.results.multi_tank_states[0].get_tank_state(tank_idx)

            # === DIRECT INPUTS ===
            initial_pressure_bar  = float(tank_sys_cfg.P_INIT) / 1e5
            initial_temperature_K = float(tank_sys_cfg.T_INIT)
            tank_volume_m3        = float(tank_props['volume'])
            initial_density_kg_m3 = float(initial_state.fuel_mass) / tank_volume_m3
            vent_pressure_bar     = float(tank_sys_cfg.P_VENT) / 1e5
            min_pressure_bar      = float(tank_sys_cfg.P_MIN) / 1e5

            if self.tank_system._tank_fluids.get(tank_idx, '') == 'CH2':
                min_density_kg_m3 = 0.0  # CH2 uses pressure-based stopping
            else:
                min_density_kg_m3 = float(self.tank_system._min_densities[tank_idx])

            direct_inputs = {
                'initial_pressure_bar':   initial_pressure_bar,
                'initial_temperature_K':  initial_temperature_K,
                'initial_density_kg_m3':  initial_density_kg_m3,
                'vent_pressure_bar':      vent_pressure_bar,
                'min_pressure_bar':       min_pressure_bar,
                'min_density_kg_m3':      min_density_kg_m3,
                'liner_thickness_m':      liner_thickness_m,
                'insulation_thickness_m': insulation_thickness_m,
            }

            # === PREPROCESSED INPUTS ===
            fluid_name = self.config.tanks[tank_idx].coolprop
            P_crit_bar = PropsSI("Pcrit", fluid_name) / 1e5
            if initial_pressure_bar <= P_crit_bar:
                T_sat = float(PropsSI("T", "P", initial_pressure_bar * 1e5, "Q", 0, fluid_name))
            else:
                T_sat = float(PropsSI("Tcrit", fluid_name))
            delta_T = initial_temperature_K - T_sat

            preprocessed_inputs = {
                'T_sat_K':   T_sat,
                'delta_T_K': delta_T,
            }

            # === DIRECT OUTPUTS ===
            tank_radius_m    = float(tank_props['radius'])
            tank_diameter_m  = float(tank_props['inner_diameter'])
            surface_area_m2  = float(tank_props['inner_surface_area'])
            wall_thickness_m = float(tank_props['thickness_wall'])
            liner_mass_kg    = float(tank_props['liner_mass'])
            wall_mass_kg     = float(tank_props['wall_mass'])
            total_tank_mass_kg = liner_mass_kg + wall_mass_kg
            fuel_mass_kg     = float(initial_state.fuel_mass)

            direct_outputs = {
                'tank_volume_m3':      tank_volume_m3,
                'tank_radius_m':       tank_radius_m,
                'tank_diameter_m':     tank_diameter_m,
                'surface_area_m2':     surface_area_m2,
                'wall_thickness_m':    wall_thickness_m,
                'fuel_mass_kg':        fuel_mass_kg,
                'liner_mass_kg':       liner_mass_kg,
                'wall_mass_kg':        wall_mass_kg,
                'total_tank_mass_kg':  total_tank_mass_kg,
            }

            # === POSTPROCESSED OUTPUTS ===
            times_array = np.asarray(self.results.times, dtype=float)
            mission_duration_hours = float(times_array[-1]) / 3600.0

            vent_pressure_pa = vent_pressure_bar * 1e5
            tank_series = self.results.get_tank_series(tank_idx)
            time_to_vent_hours = mission_duration_hours  # default: never vented
            for i, state in enumerate(tank_series.states):
                if float(state.pressure) >= vent_pressure_pa:
                    time_to_vent_hours = float(times_array[i]) / 3600.0
                    break

            # Gravimetric efficiency = fuel / (fuel + structure)
            structure_mass = liner_mass_kg + wall_mass_kg
            mass_efficiency = fuel_mass_kg / (fuel_mass_kg + structure_mass)

            # Volumetric efficiency = inner volume / outer volume (capsule-correct, from tank_props)
            outer_volume_m3 = float(tank_props['outer_volume'])
            volumetric_efficiency = tank_volume_m3 / outer_volume_m3 if outer_volume_m3 > 0.0 else 0.0

            fuel_energy_mj = fuel_mass_kg * 120.0  # H2 LHV ~120 MJ/kg

            qdot_ohex = self._calculate_ohex_requirements(tank_idx)
            if len(qdot_ohex) == len(times_array) and len(times_array) > 1:
                dt = np.diff(times_array)
                ohex_energy_mj = float(np.sum(np.asarray(qdot_ohex[:-1]) * dt)) / 1e6
            else:
                ohex_energy_mj = 0.0

            qdot_ihex = self._calculate_ihex_requirements(tank_idx)
            if len(qdot_ihex) == len(times_array) and len(times_array) > 1:
                dt = np.diff(times_array)
                ihex_energy_mj = float(np.sum(np.asarray(qdot_ihex[:-1]) * dt)) / 1e6
            else:
                ihex_energy_mj = 0.0

            total_energy_mj = ohex_energy_mj + ihex_energy_mj

            postprocessed_outputs = {
                'mission_duration_hours': mission_duration_hours,
                'time_to_vent_hours':     time_to_vent_hours,
                'mass_efficiency':        mass_efficiency,
                'volumetric_efficiency':  volumetric_efficiency,
                'fuel_energy_MJ':         fuel_energy_mj,
                'ohex_energy_MJ':         ohex_energy_mj,
                'ihex_energy_MJ':         ihex_energy_mj,
                'total_energy_MJ':        total_energy_mj,
            }

            summary['tanks'][tank_name] = {
                'direct_inputs':         direct_inputs,
                'preprocessed_inputs':   preprocessed_inputs,
                'direct_outputs':        direct_outputs,
                'postprocessed_outputs': postprocessed_outputs,
            }

        return summary

    def print_scenario_summary(self):
        """Print comprehensive scenario summary."""
        summary = self.get_scenario_summary()

        print("\n🔍 SCENARIO CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Analysis: {summary['scenario']['name']}")
        print(f"Description: {summary['scenario']['description']}")
        print(f"Version: {summary['scenario']['version']}")
        print(f"Tanks: {summary['tanks']['count']}")
        print(f"Missions: {summary['missions']['count']}")
        print(f"Materials: {summary['materials']['count']} ({', '.join(summary['materials']['types'])})")

        if 'results' in summary:
            print(f"\nResults:")
            print(f"  Final time: {summary['results']['final_time_hours']:.2f} hours")
            print(f"  Data points: {summary['results']['data_points']}")
            for i, tank_state in enumerate(summary['results']['final_tank_states']):
                print(f"  Tank {i+1}: m={tank_state['fuel_mass']:.2f}kg, "
                      f"T={tank_state['temperature']:.1f}K, "
                      f"ρ={tank_state['density']:.1f}kg/m³")

        if 'validation' in summary:
            overall = summary['validation']['overall']
            print(f"  Validation: {'PASSED' if overall else 'FAILED'}")

        print("=" * 60)

    def print_comprehensive_analysis_summary(self):
        """Print comprehensive analysis parameter summary table."""
        summary = self.get_comprehensive_analysis_summary()

        print(f"\nCOMPREHENSIVE ANALYSIS SUMMARY: {summary['analysis_name']}")
        print("=" * 100)

        # Header
        header = f"{'Parameter':<35} {'Unit':<12}"
        for tank_name in summary['tanks'].keys():
            header += f"{tank_name:<15}"
        print(header)
        print("-" * 100)

        # Define parameter groups and their display information
        param_groups = [
            ("DIRECT INPUTS", [
                ('initial_pressure_bar', 'bar', 'Initial Pressure'),
                ('initial_temperature_K', 'K', 'Initial Temperature'),
                ('initial_density_kg_m3', 'kg/m³', 'Initial Density'),
                ('vent_pressure_bar', 'bar', 'Vent Pressure'),
                ('min_pressure_bar', 'bar', 'Minimum Pressure'),
                ('min_density_kg_m3', 'kg/m³', 'Minimum Density'),
                ('liner_thickness_m', 'm', 'Liner Thickness'),
                ('insulation_thickness_m', 'm', 'Insulation Thickness')
            ]),
            ("PREPROCESSED INPUTS", [
                ('T_sat_K', 'K', 'Saturation Temperature'),
                ('delta_T_K', 'K', 'ΔT from Saturation')
            ]),
            ("DIRECT OUTPUTS", [
                ('tank_volume_m3', 'm³', 'Tank Volume'),
                ('tank_radius_m', 'm', 'Tank Radius'),
                ('tank_diameter_m', 'm', 'Tank Diameter'),
                ('surface_area_m2', 'm²', 'Surface Area'),
                ('wall_thickness_m', 'm', 'Wall Thickness'),
                ('fuel_mass_kg', 'kg', 'Fuel Mass'),
                ('liner_mass_kg', 'kg', 'Liner Mass'),
                ('wall_mass_kg', 'kg', 'Wall Mass'),
                ('total_tank_mass_kg', 'kg', 'Total Tank Mass')
            ]),
            ("POSTPROCESSED OUTPUTS", [
                ('mission_duration_hours', 'hours', 'Mission Duration'),
                ('time_to_vent_hours', 'hours', 'Time to Vent'),
                ('mass_efficiency', '-', 'Mass Efficiency'),
                ('volumetric_efficiency', '-', 'Volumetric Efficiency'),
                ('fuel_energy_MJ', 'MJ', 'Fuel Energy'),
                ('ohex_energy_MJ', 'MJ', 'OHEX Energy'),
                ('ihex_energy_MJ', 'MJ', 'iHEX Energy'),
                ('total_energy_MJ', 'MJ', 'Total Energy')
            ])
        ]

        # Print each parameter group
        for group_name, params in param_groups:
            print(f"\n{group_name}")
            print("-" * 50)

            for param_key, unit, display_name in params:
                row = f"{display_name:<35} {unit:<12}"

                for tank_name in summary['tanks'].keys():
                    tank_data = summary['tanks'][tank_name]

                    # Find the parameter in the appropriate group
                    value = None
                    for group_key in ['direct_inputs', 'preprocessed_inputs', 'direct_outputs', 'postprocessed_outputs']:
                        if param_key in tank_data[group_key]:
                            value = tank_data[group_key][param_key]
                            break

                    if value is not None:
                        # Format based on parameter type
                        if 'efficiency' in param_key:
                            formatted_value = f"{value:.3f}"
                        elif 'energy' in param_key or 'MJ' in unit:
                            formatted_value = f"{value:.1f}"
                        elif 'hours' in unit:
                            formatted_value = f"{value:.2f}"
                        elif 'bar' in unit:
                            formatted_value = f"{value:.0f}"
                        elif param_key in ['tank_volume_m3', 'surface_area_m2']:
                            formatted_value = f"{value:.3f}"
                        elif param_key in ['tank_radius_m', 'tank_diameter_m']:
                            formatted_value = f"{value:.2f}"
                        elif 'thickness' in param_key:
                            formatted_value = f"{value:.3f}"
                        elif 'mass' in param_key:
                            formatted_value = f"{value:.1f}"
                        elif 'density' in param_key:
                            formatted_value = f"{value:.1f}"
                        elif 'temperature' in param_key.lower() or unit == 'K':
                            formatted_value = f"{value:.1f}"
                        else:
                            formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = "N/A"

                    row += f"{formatted_value:<15}"

                print(row)

        print("\n" + "=" * 100)
        print("Comprehensive analysis summary complete")
        print("   📝 All input parameters, tank geometry, and performance metrics displayed")



    @silent_when_quiet
    def save_comprehensive_results(self) -> str:
        """
        Generate and save comprehensive results report to text file.

        This method runs after simulation completion and generates a complete
        analysis report using the exact data sources specified in the requirements table.

        Returns:
            str: Path to the saved results file
        """
        from datetime import datetime
        from pathlib import Path

        if not self.results:
            raise RuntimeError("No simulation results available. Run simulation first.")

        output_config = self.config.output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        mission_sections = []
        if getattr(self, 'mission_profile', None) and getattr(self.mission_profile, 'sections', None):
            mission_sections = [section.duration for section in self.mission_profile.sections]

        times_array = np.asarray(self.results.times, dtype=float)
        total_duration_s = float(times_array[-1]) if times_array.size > 0 else 0.0
        data_points = int(times_array.size)

        def _fmt(value: Any, decimals: int = 2, default: str = "N/A") -> str:
            if value is None:
                return default
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return default
                value = value.flat[0]
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return default
                value = value[0]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return str(value)
            if np.isnan(value):
                return default
            return f"{value:.{decimals}f}"

        def _line(label: str, value: Any) -> str:
            return f"{label:<32} {value}\n"

        def _get_tank_props(tank_geometry: Any, tank_index: int) -> Dict[str, Any]:
            if hasattr(self, 'tank_system') and self.tank_system:
                try:
                    return self.tank_system._get_tank_properties(tank_geometry, f"Tank{tank_index + 1}", tank_index)
                except Exception:
                    return {}
            return {}

        total_initial_mass = 0.0
        total_final_mass = 0.0
        total_fuel_consumed = 0.0
        total_dry_mass = 0.0
        total_inner_volume = 0.0
        total_outer_volume = 0.0
        total_ihex_kwh = 0.0
        total_ohex_kwh = 0.0
        total_hex_kwh = 0.0

        per_tank_blocks = []
        for tank_index, tank_geometry in enumerate(self.tank_geometries):
            tank_name = f"Tank_{tank_index + 1}"
            tank_config = self._tank_data[tank_index]
            tank_material_config = self.config.tanks[tank_index].materials
            insulation_config = tank_material_config.get('insulation', {})
            initial_state = self.results.multi_tank_states[0].get_tank_state(tank_index)
            final_state = self.results.multi_tank_states[-1].get_tank_state(tank_index)
            tank_props = _get_tank_props(tank_geometry, tank_index)

            initial_mass = float(initial_state.fuel_mass)
            final_mass = float(final_state.fuel_mass)
            fuel_consumed = initial_mass - final_mass

            try:
                qdot_ihex = self._calculate_ihex_requirements(tank_index)
                ihex_kwh = np.trapz(np.abs(np.asarray(qdot_ihex, dtype=float)), times_array) / 3.6e6 if len(qdot_ihex) == data_points else 0.0
            except Exception:
                ihex_kwh = 0.0

            try:
                qdot_ohex = self._calculate_ohex_requirements(tank_index)
                ohex_kwh = np.trapz(np.asarray(qdot_ohex, dtype=float), times_array) / 3.6e6 if len(qdot_ohex) == data_points else 0.0
            except Exception:
                ohex_kwh = 0.0

            hex_kwh = ihex_kwh + ohex_kwh
            liner_mass = float(tank_props.get('liner_mass', 0.0))
            composite_wall_mass = float(tank_props.get('wall_mass', 0.0))
            insulation_mass = float(tank_props.get('foam_mass', 0.0))
            outer_shell_mass = float(tank_props.get('shell_mass', 0.0))
            dry_mass = liner_mass + composite_wall_mass + insulation_mass + outer_shell_mass

            inner_volume = float(getattr(tank_geometry, 'volume', tank_props.get('volume', 0.0)))
            outer_volume = float(tank_props.get('outer_volume', 0.0))

            total_initial_mass += initial_mass
            total_final_mass += final_mass
            total_fuel_consumed += fuel_consumed
            total_dry_mass += dry_mass
            total_inner_volume += inner_volume
            total_outer_volume += outer_volume
            total_ihex_kwh += ihex_kwh
            total_ohex_kwh += ohex_kwh
            total_hex_kwh += hex_kwh

            block = []
            block.append(f"{tank_name.upper()}\n")
            block.append("-" * 80 + "\n")
            block.append(_line("Configured pressure [bar]:", _fmt(tank_config.get('initial_pressure', 0.0) / 1e5, 1)))
            block.append(_line("Configured temperature [K]:", _fmt(tank_config.get('initial_temperature', tank_config.get('calculated_initial_temperature')), 2)))
            block.append(_line("Configured density [kg/m^3]:", _fmt(tank_config.get('initial_density', tank_config.get('calculated_initial_density')), 2)))
            block.append(_line("Minimum pressure [bar]:", _fmt(tank_config.get('minimum_pressure', 0.0) / 1e5, 1)))
            block.append(_line("Venting pressure [bar]:", _fmt(tank_config.get('venting_pressure', 0.0) / 1e5, 1)))
            block.append(_line("Inner volume [m^3]:", _fmt(inner_volume, 4)))
            block.append(_line("Outer volume [m^3]:", _fmt(outer_volume, 4)))
            block.append(_line("Tank radius [m]:", _fmt(getattr(tank_geometry, 'radius', 0.0), 3)))
            block.append(_line("Tank phi (L/R) [-]:", _fmt(getattr(tank_geometry, 'phi', 0.0), 3)))
            block.append(_line("Cylindrical length [m]:", _fmt(getattr(tank_geometry, 'cylindrical_section_length', 0.0), 3)))
            block.append(_line("Total length [m]:", _fmt(getattr(tank_geometry, 'total_height', 0.0), 3)))
            block.append(_line("Inner area [m^2]:", _fmt(tank_props.get('inner_surface_area', 0.0), 3)))
            block.append(_line("Outer area [m^2]:", _fmt(tank_props.get('outer_surface_area', 0.0), 3)))
            block.append(_line("Liner thickness [m]:", _fmt(tank_material_config.get('liner', {}).get('thickness'), 4)))
            block.append(_line("Composite wall thickness [m]:", _fmt(tank_props.get('wall_thickness', 0.0), 4)))
            block.append(_line("Insulation thickness [m]:", _fmt(insulation_config.get('thickness'), 4)))
            block.append(_line("Outer shell thickness [m]:", _fmt(insulation_config.get('shell_thickness'), 4)))
            block.append(_line("Liner mass [kg]:", _fmt(liner_mass, 2)))
            block.append(_line("Composite wall mass [kg]:", _fmt(composite_wall_mass, 2)))
            block.append(_line("Insulation mass [kg]:", _fmt(insulation_mass, 2)))
            block.append(_line("Outer shell mass [kg]:", _fmt(outer_shell_mass, 2)))
            block.append(_line("Total dry mass [kg]:", _fmt(dry_mass, 2)))
            block.append(_line("Initial fuel mass [kg]:", _fmt(initial_mass, 2)))
            block.append(_line("Final fuel mass [kg]:", _fmt(final_mass, 2)))
            block.append(_line("Fuel consumed [kg]:", _fmt(fuel_consumed, 2)))
            block.append(_line("Initial pressure actual [bar]:", _fmt(initial_state.pressure / 1e5, 2)))
            block.append(_line("Initial H2 temperature actual [K]:", _fmt(initial_state.h2_temperature, 2)))
            block.append(_line("Initial density actual [kg/m^3]:", _fmt(initial_state.density, 2)))
            block.append(_line("Final H2 temperature [K]:", _fmt(final_state.h2_temperature, 2)))
            block.append(_line("Final pressure [bar]:", _fmt(final_state.pressure / 1e5, 2)))
            block.append(_line("Final density [kg/m^3]:", _fmt(final_state.density, 2)))
            block.append(_line("Mission duration [h]:", _fmt(total_duration_s / 3600.0, 3)))
            block.append(_line("iHEX energy [kWh]:", _fmt(ihex_kwh, 3)))
            block.append(_line("OHEX energy [kWh]:", _fmt(ohex_kwh, 3)))
            block.append(_line("Total HEX energy [kWh]:", _fmt(hex_kwh, 3)))
            per_tank_blocks.append("".join(block))

        report_lines = []
        report_lines.append(f"{self.config.name} - Results Report\n")
        report_lines.append(f"Generated: {timestamp}\n")
        report_lines.append(f"Description: {self.config.description}\n\n")

        report_lines.append("SYSTEM OVERVIEW\n")
        report_lines.append("=" * 80 + "\n")
        report_lines.append(_line("Tank count:", len(self.tank_geometries)))
        report_lines.append(_line("Coupling rule count:", len(self.couplings)))
        report_lines.append(_line("Mission count:", len(self.config.missions)))
        report_lines.append(_line("Mission sections:", len(mission_sections)))
        report_lines.append(_line("Mission duration [h]:", _fmt(total_duration_s / 3600.0, 3)))
        report_lines.append(_line("Data points:", data_points))
        if mission_sections:
            mission_duration_str = ", ".join(_fmt(duration, 1) for duration in mission_sections)
            report_lines.append(_line("Mission section durations [s]:", mission_duration_str))
        report_lines.append("\n")

        report_lines.append("SYSTEM TOTALS\n")
        report_lines.append("=" * 80 + "\n")
        report_lines.append(_line("Initial fuel mass [kg]:", _fmt(total_initial_mass, 2)))
        report_lines.append(_line("Final fuel mass [kg]:", _fmt(total_final_mass, 2)))
        report_lines.append(_line("Fuel consumed [kg]:", _fmt(total_fuel_consumed, 2)))
        report_lines.append(_line("Total dry mass [kg]:", _fmt(total_dry_mass, 2)))
        report_lines.append(_line("Total inner volume [m^3]:", _fmt(total_inner_volume, 4)))
        report_lines.append(_line("Total outer volume [m^3]:", _fmt(total_outer_volume, 4)))
        total_system_mass = total_initial_mass + total_dry_mass
        gravimetric_efficiency = total_initial_mass / total_system_mass if total_system_mass > 0 else 0.0
        volumetric_efficiency = total_inner_volume / total_outer_volume if total_outer_volume > 0 else 0.0
        report_lines.append(_line("Gravimetric efficiency [-]:", _fmt(gravimetric_efficiency, 4)))
        report_lines.append(_line("Volumetric efficiency [-]:", _fmt(volumetric_efficiency, 4)))
        report_lines.append(_line("Total iHEX energy [kWh]:", _fmt(total_ihex_kwh, 3)))
        report_lines.append(_line("Total OHEX energy [kWh]:", _fmt(total_ohex_kwh, 3)))
        report_lines.append(_line("Total HEX energy [kWh]:", _fmt(total_hex_kwh, 3)))
        if total_duration_s > 0:
            report_lines.append(_line("Average net fuel use [kg/s]:", _fmt(total_fuel_consumed / total_duration_s, 6)))
        report_lines.append("\n")

        report_lines.append("COUPLING SUMMARY\n")
        report_lines.append("=" * 80 + "\n")
        if self.couplings:
            for idx, coupling in enumerate(self.couplings, start=1):
                source = self.network.tank_ids[coupling.source]
                target = 'sink' if coupling.target == SINK else self.network.tank_ids[coupling.target]
                report_lines.append(
                    f"{idx}. {coupling.edge_id}: {coupling.coupling_type} (source={source}, target={target})\n"
                )
        else:
            report_lines.append("No coupling rules configured.\n")
        report_lines.append("\n")

        report_lines.append("PER-TANK DETAILS\n")
        report_lines.append("=" * 80 + "\n\n")
        report_lines.append("\n".join(per_tank_blocks))
        report_lines.append("\n")
        report_lines.append("=" * 80 + "\n")
        report_lines.append("Report generated from SystemOrchestrator summary data and simulation results.\n")

        report_text = "".join(report_lines)

        # Create output directory and save file
        output_dir = Path(output_config.get('results_directory', 'output/results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"comprehensive_report.txt"

        with open(output_file, 'w') as f:
            f.write(report_text)

        print(f"COMPREHENSIVE RESULTS SUMMARY:")
        print(f"   Tanks: {len(self.tank_geometries)}")
        print(f"   Fuel consumed: {total_fuel_consumed:.2f} kg")
        print(f"   Total iHEX Energy: {total_ihex_kwh:.3f} kWh")
        print(f"   Total OHEX Energy: {total_ohex_kwh:.3f} kWh")
        print(f"   Total HEX Energy: {total_hex_kwh:.3f} kWh")

        return str(output_file)