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
from toplab.sizing import TankDesign, size_tank

# Multi-tank DAE physics engine
from toplab.system.tank_system import TankSystem, TankSystemConfig, TankConfig
from toplab.orchestration.run_result import RunResult
from toplab.utilities.output_control import silent_when_quiet

# Mission framework
from toplab.missions.isochoric_missions import DischargeMission

# Heat flow data collection for iHEX calculation

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
    def __init__(self, config: AnalysisConfig = None, config_path: str = None, verbosity: str = "summary",
                 sizing_config: AnalysisConfig = None):
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

        self._streams = self._resolve_streams(config, self.missions)

        # Tanks are sized from ``sizing_config`` when given (e.g. a dormancy check re-using
        # the discharge design); otherwise from the analysis itself.
        if sizing_config is None:
            self._sizing_streams = self._streams
            self._sizing_primary = self.mission_profile
        else:
            sizing_missions = [build_mission(spec) for spec in sizing_config.missions]
            self._sizing_streams = self._resolve_streams(sizing_config, sizing_missions, announce=False)
            self._sizing_primary = sizing_missions[0]

        self.designs = self._size_tanks()
        self.tank_geometries = [d.geometry for d in self.designs]
        print(f"   Created {len(self.tank_geometries)} tank geometries")

        self.tank_system_config = self._create_tank_system_config()

        self.couplings = self.network.coupling_edges()
        for coupling in self.couplings:
            print(f"   Coupling {coupling.edge_id}: {coupling.coupling_type}")

        self.tank_system = TankSystem(
            tank_geometries=self.tank_geometries,
            config=self.tank_system_config,
            couplings=self.couplings,
            analysis_config=self.config,
            verbosity=self.verbosity,
        )
        print(f"   TankSystem DAE engine initialised")

        self._setup_mission_sequences()

        self.results = None

        self.validation_results = None

        print("System Orchestrator ready for mission execution!")
        print(f"   Tanks: {len(self.tank_geometries)}")
        print(f"   Missions: {len(self.config.missions)}")
        print(f"   Analysis type: {self.config.name}")

    def _resolve_streams(self, config: AnalysisConfig, missions: List[Any],
                         announce: bool = True) -> Dict[int, List[Stream]]:
        """Route every mission to the tank(s) that serve it, via the network."""
        streams: Dict[int, List[Stream]] = {}
        for spec, mission in zip(config.missions, missions):
            for tank_index, fraction in self.network.demand_targets(spec.assigned_to_node).items():
                tank_streams = streams.setdefault(tank_index, [])
                label = spec.parameters.get('column_name', f"Stream {len(tank_streams) + 1}")
                tank_streams.append(Stream(label, mission, fraction))

        assigned = {spec.assigned_to_node for spec in config.missions}
        for mixer_id, mixer in self.network.mixers.items():
            if mixer_id not in assigned:
                raise ValueError(
                    f"Mixer node {mixer_id} has no mission: set 'assigned_to_node: {mixer_id}' "
                    f"on the mission(s) feeding it."
                )
            if announce:
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
        streams = self._sizing_streams.get(tank_index)
        if not streams:
            # Tanks with no demand of their own are sized against the primary mission.
            return self._sizing_primary.required_fuel
        return sum(s.fraction * s.mission.required_fuel for s in streams)

    def _size_tanks(self) -> List[TankDesign]:
        """Size every tank (pure); report what the sizer decided."""
        designs = []
        for tank_index, tank in enumerate(self.config.tanks):
            print(f"   Sizing Tank {tank.node_id}...")
            design = size_tank(tank, self._required_fuel(tank_index))
            for note in design.notes:
                print(f"     {note}")
            print(f"     Geometry: V={design.geometry.volume:.4f} m³ ({design.method})")
            designs.append(design)
        return designs

    def _create_tank_system_config(self) -> TankSystemConfig:
        """Create TankSystemConfig from the typed config."""
        mission = self.config.missions[0]

        tank_configs = []
        for tank, design in zip(self.config.tanks, self.designs):
            tank_configs.append(TankConfig(
                P_INIT=design.initial_pressure,
                T_INIT=design.initial_temperature,
                P_VENT=design.venting_pressure,
                P_MIN=design.minimum_pressure,
                MASS_INIT=design.initial_mass,
                scenario=mission.type.upper(),  # DISCHARGE, REFUEL, DORMANCY
                name=f"Tank{tank.node_id}"
            ))

            stop = (f"P_stop={design.minimum_pressure / 1e5:.1f}bar (CH2 pressure-based)" if tank.fluid == 'CH2'
                    else f"ρ_stop={tank.geometry_data().get('minimum_density', 5.8):.1f}kg/m³")
            print(f"   Tank {tank.node_id} config: P_init={design.initial_pressure / 1e5:.0f}bar, "
                  f"T_init={design.initial_temperature:.1f}K, {stop}")

        stopping_criteria = self.config.stopping_criteria

        # CH2 tanks use pressure-based stopping and have no minimum_density; skip them.
        if 'minimum_density' in stopping_criteria:
            minimum_density = float(stopping_criteria['minimum_density'])
        else:
            non_ch2 = [t.geometry_data() for t in self.config.tanks if t.fluid != 'CH2']
            ref_tank = non_ch2[0] if non_ch2 else self.config.tanks[0].geometry_data()
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
            # Run the MultiTankSystem DAE simulation
            self.results = self.tank_system.run_analysis(solver_method, solver_config)

            end_time = time.time()
            wall_time = end_time - start_time

            print("Simulation completed successfully!")
            print(f"   Wall time: {wall_time:.2f} seconds")
            print(f"   Final time: {self.results.times[-1]/3600:.2f} hours")
            print(f"   Data points: {len(self.results.times)}")

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

    def run_result(self) -> RunResult:
        """Snapshot of the finished run for post-processing and reporting."""
        return RunResult(
            config=self.config, network=self.network, results=self.results, designs=self.designs,
            tank_geometries=self.tank_geometries, couplings=self.couplings, streams=self._streams,
            mission_profile=self.mission_profile, tank_system=self.tank_system,
            validation_results=self.validation_results)

    # --- Delegates: derived quantities, plots and reports live outside the orchestrator ---

    def generate_plots(self, save_path: str = None) -> Any:
        from toplab.report.plots import generate_plots
        return generate_plots(self.run_result(), save_path)

    def get_scenario_summary(self) -> Dict[str, Any]:
        from toplab.report.summary import get_scenario_summary
        return get_scenario_summary(self.run_result())

    def get_comprehensive_analysis_summary(self) -> Dict[str, Any]:
        from toplab.report.summary import get_comprehensive_analysis_summary
        return get_comprehensive_analysis_summary(self.run_result())

    def print_scenario_summary(self):
        from toplab.report.summary import print_scenario_summary
        print_scenario_summary(self.run_result())

    def print_comprehensive_analysis_summary(self):
        from toplab.report.summary import print_comprehensive_analysis_summary
        print_comprehensive_analysis_summary(self.run_result())

    def save_comprehensive_results(self) -> str:
        from toplab.report.summary import save_comprehensive_results
        return save_comprehensive_results(self.run_result())
