from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence
from contextlib import nullcontext, redirect_stderr, redirect_stdout

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configuration.scenario_configuration import ScenarioConfig
from src.optimization.sweep_runner import BaseSweepStudy, SweepResult, SweepRuntimeConfig
from src.orchestration.system_orchestrator import SystemOrchestrator


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "analysis" / "coupled_ch2_cch2" / "coupled_ch2_cch2_config.yaml"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "optimization" / "pressure_buffer_opt" / "output" / "pressure_buffer_sweep.txt"
DEFAULT_SWEEP_CONFIG_PATH = PROJECT_ROOT / "optimization" / "pressure_buffer_opt" / "pressure_buffer_sweep.yaml"


@dataclass(frozen=True)
class DesignVector:
    tank_1_radius: float
    tank_1_phi: float
    tank_2_radius: float
    tank_2_phi: float


class PressureBufferSweep(BaseSweepStudy[DesignVector]):
    def __init__(
        self,
        base_config_path: str | Path = DEFAULT_CONFIG_PATH,
        verbosity: str = "quiet",
        report_path: str | Path = DEFAULT_REPORT_PATH,
        silent: bool = True,
        save_plots: bool = False,
        save_data: bool = False,
        ranking: Sequence[str] | None = None,
        require_mission_completion: bool = False,
        case_timeout_s: float | None = None,
        sweep_config_path: str | Path = DEFAULT_SWEEP_CONFIG_PATH,
    ) -> None:
        runtime_config = SweepRuntimeConfig(
            sweep_config_path=Path(sweep_config_path).resolve(),
            base_config_path=Path(base_config_path).resolve(),
            report_path=Path(report_path).resolve(),
            verbosity=verbosity,
            silent=silent,
            save_plots=save_plots,
            save_data=save_data,
            ranking=tuple(ranking or (
                "mission_completed",
                "gravimetric_efficiency",
                "volumetric_efficiency",
                "mission_completion_ratio",
            )),
            require_mission_completion=require_mission_completion,
            case_timeout_s=None if case_timeout_s is None else float(case_timeout_s),
        )
        super().__init__(runtime_config)
        self.base_scenario = ScenarioConfig.from_yaml(self.base_config_path)
        self.baseline_design = self._resolve_baseline_design()

    @classmethod
    def from_sweep_config(cls, sweep_config_path: str | Path = DEFAULT_SWEEP_CONFIG_PATH) -> tuple["PressureBufferSweep", dict]:
        config_path, sweep_config = cls.load_sweep_config(sweep_config_path)
        runtime_config = cls.create_runtime_config(
            sweep_config_path=config_path,
            sweep_config=sweep_config,
            project_root=PROJECT_ROOT,
            default_base_config=DEFAULT_CONFIG_PATH,
            default_report_path=DEFAULT_REPORT_PATH,
        )
        sweep = cls(
            base_config_path=runtime_config.base_config_path,
            verbosity=runtime_config.verbosity,
            report_path=runtime_config.report_path,
            silent=runtime_config.silent,
            save_plots=runtime_config.save_plots,
            save_data=runtime_config.save_data,
            ranking=runtime_config.ranking,
            require_mission_completion=runtime_config.require_mission_completion,
            case_timeout_s=runtime_config.case_timeout_s,
            sweep_config_path=runtime_config.sweep_config_path,
        )
        return sweep, sweep_config

    def _clone_base_config(self) -> dict:
        return copy.deepcopy(self.base_scenario.config_dict)

    def _resolve_baseline_design(self) -> DesignVector:
        scenario = ScenarioConfig(
            config_dict=self._clone_base_config(),
            config_format=self.base_scenario.config_format,
            config_path=str(self.base_config_path),
        )
        with self._stdout_context() as stdout_target:
            stderr_context = redirect_stderr(stdout_target) if self.silent else nullcontext()
            stdout_redirect = redirect_stdout(stdout_target) if self.silent else nullcontext()
            with stdout_redirect, stderr_context:
                orchestrator = SystemOrchestrator(scenario_config=scenario, verbosity="quiet")
        tank_1 = orchestrator.tank_geometries[0]
        tank_2 = orchestrator.tank_geometries[1]
        return DesignVector(
            tank_1_radius=float(tank_1.radius),
            tank_1_phi=float(getattr(tank_1, "phi", 0.0)),
            tank_2_radius=float(tank_2.radius),
            tank_2_phi=float(getattr(tank_2, "phi", 0.0)),
        )

    def build_cartesian_designs(
        self,
        tank_1_radii: Sequence[float] | None = None,
        tank_1_phis: Sequence[float] | None = None,
        tank_2_radii: Sequence[float] | None = None,
        tank_2_phis: Sequence[float] | None = None,
        max_points: int | None = 10,
    ) -> list[DesignVector]:
        baseline = self.baseline_design
        tank_1_radii = tank_1_radii or (baseline.tank_1_radius * 0.95, baseline.tank_1_radius * 1.05)
        tank_1_phis = tank_1_phis or (max(0.0, baseline.tank_1_phi - 0.5), baseline.tank_1_phi + 0.5)
        tank_2_radii = tank_2_radii or (baseline.tank_2_radius * 0.95, baseline.tank_2_radius * 1.05)
        tank_2_phis = tank_2_phis or (max(0.0, baseline.tank_2_phi - 0.5), baseline.tank_2_phi + 0.5)

        product_iter = itertools.product(tank_1_radii, tank_1_phis, tank_2_radii, tank_2_phis)
        if max_points is not None:
            product_iter = itertools.islice(product_iter, max_points)

        return [
            DesignVector(
                tank_1_radius=float(tank_1_radius),
                tank_1_phi=float(tank_1_phi),
                tank_2_radius=float(tank_2_radius),
                tank_2_phi=float(tank_2_phi),
            )
            for tank_1_radius, tank_1_phi, tank_2_radius, tank_2_phi in product_iter
        ]

    def build_design_points_from_config(self, sweep_config: dict) -> list[DesignVector]:
        sweep_section = sweep_config.get("sweep", {})
        mode = str(sweep_section.get("mode", "cartesian")).strip().lower()

        if mode == "explicit":
            return self._build_explicit_design_points(sweep_config)
        if mode != "cartesian":
            raise ValueError(f"Unsupported sweep mode '{mode}'. Expected 'cartesian' or 'explicit'.")

        variables = sweep_config.get("design_variables", {})
        return self.build_cartesian_designs(
            tank_1_radii=_coerce_float_sequence(variables.get("tank_1_radius")),
            tank_1_phis=_coerce_float_sequence(variables.get("tank_1_phi")),
            tank_2_radii=_coerce_float_sequence(variables.get("tank_2_radius")),
            tank_2_phis=_coerce_float_sequence(variables.get("tank_2_phi")),
            max_points=sweep_section.get("max_points", 10),
        )

    def evaluate_design(self, design: DesignVector) -> SweepResult:
        scenario = ScenarioConfig(
            config_dict=self._build_config_for_design(design),
            config_format=self.base_scenario.config_format,
            config_path=str(self.base_config_path),
        )

        try:
            with self._stdout_context() as stdout_target:
                stderr_context = redirect_stderr(stdout_target) if self.silent else nullcontext()
                stdout_redirect = redirect_stdout(stdout_target) if self.silent else nullcontext()
                with stdout_redirect, stderr_context:
                    orchestrator = SystemOrchestrator(scenario_config=scenario, verbosity=self.verbosity)
                    results = orchestrator.run_simulation()
                    return self._extract_metrics(design, orchestrator, results)
        except Exception as exc:
            return SweepResult(
                design=design,
                mission_completed=False,
                mission_completion_ratio=0.0,
                mission_duration_s=0.0,
                target_duration_s=0.0,
                gravimetric_efficiency=0.0,
                volumetric_efficiency=0.0,
                total_fuel_mass_kg=0.0,
                structure_mass_kg=0.0,
                total_inner_volume_m3=0.0,
                total_outer_volume_m3=0.0,
                error=str(exc),
            )

    def write_report(
        self,
        results: Sequence[SweepResult],
        report_path: str | Path | None = None,
    ) -> Path:
        report_path = self.report_path if report_path is None else Path(report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)

        successful = [result for result in results if result.is_successful]
        feasible = [result for result in successful if result.mission_completed]
        ranked_pool = feasible if self.require_mission_completion else successful
        ranked = sorted(
            ranked_pool,
            key=self._ranking_key,
            reverse=True,
        )

        lines = [
            "Pressure Buffer Sweep",
            "====================",
            f"Base config: {self.base_config_path}",
            f"Evaluated points: {len(results)}",
            f"Successful runs: {len(successful)}",
            f"Mission-complete runs: {len(feasible)}",
            f"Case wall-time constraint: {self._format_case_timeout_constraint()}",
            "",
            "Baseline design",
            f"  Tank 1: radius={self.baseline_design.tank_1_radius:.4f} m, phi={self.baseline_design.tank_1_phi:.4f}",
            f"  Tank 2: radius={self.baseline_design.tank_2_radius:.4f} m, phi={self.baseline_design.tank_2_phi:.4f}",
            "",
            "Ranked results",
        ]

        for index, result in enumerate(ranked, start=1):
            status = "PASS" if result.mission_completed else "SHORT"
            lines.extend(
                [
                    f"{index:02d}. {status}",
                    (
                        "    design: "
                        f"r1={result.design.tank_1_radius:.4f} m, "
                        f"phi1={result.design.tank_1_phi:.4f}, "
                        f"r2={result.design.tank_2_radius:.4f} m, "
                        f"phi2={result.design.tank_2_phi:.4f}"
                    ),
                    (
                        "    metrics: "
                        f"mission={result.mission_duration_s/3600.0:.3f}/{result.target_duration_s/3600.0:.3f} h "
                        f"({result.mission_completion_ratio:.3%}), "
                        f"gravimetric={result.gravimetric_efficiency:.4f}, "
                        f"volumetric={result.volumetric_efficiency:.4f}"
                    ),
                    (
                        "    totals: "
                        f"fuel={result.total_fuel_mass_kg:.3f} kg, "
                        f"structure={result.structure_mass_kg:.3f} kg, "
                        f"inner_volume={result.total_inner_volume_m3:.4f} m^3, "
                        f"outer_volume={result.total_outer_volume_m3:.4f} m^3"
                    ),
                ]
            )

        failed = [result for result in results if not result.is_successful]
        if failed:
            lines.extend(["", "Failed runs"])
            for result in failed:
                lines.append(
                    (
                        "  "
                        f"r1={result.design.tank_1_radius:.4f}, phi1={result.design.tank_1_phi:.4f}, "
                        f"r2={result.design.tank_2_radius:.4f}, phi2={result.design.tank_2_phi:.4f}: "
                        f"{result.error}"
                    )
                )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def _ranking_key(self, result: SweepResult) -> tuple[float | bool, ...]:
        values: list[float | bool] = []
        for field_name in self.ranking:
            value = getattr(result, field_name)
            values.append(value)
        return tuple(values)

    def format_design_vector(self, design: DesignVector) -> str:
        return (
            f"(r1={design.tank_1_radius:.4f}, phi1={design.tank_1_phi:.4f}, "
            f"r2={design.tank_2_radius:.4f}, phi2={design.tank_2_phi:.4f})"
        )

    def _build_config_for_design(self, design: DesignVector) -> dict:
        config_dict = self._clone_base_config()
        output_config = config_dict.setdefault("output", {})
        output_config["silent"] = self.silent
        output_config["save_plots"] = self.save_plots
        output_config["save_data"] = self.save_data

        nodes = config_dict.get("network", {}).get("nodes", [])
        node_map = {node.get("node_id"): node for node in nodes}

        self._apply_geometry(node_map[1], design.tank_1_radius, design.tank_1_phi)
        self._apply_geometry(node_map[2], design.tank_2_radius, design.tank_2_phi)

        # Keep optimization assumptions consistent across all runs.
        self._apply_pressure_constraints(config_dict)

        if "network" in config_dict:
            config_dict["coupling_rules"] = ScenarioConfig._compile_network_coupling_rules(config_dict)

        return config_dict

    @staticmethod
    def _apply_pressure_constraints(config_dict: dict) -> None:
        network = config_dict.get("network", {})

        for node in network.get("nodes", []):
            if node.get("type") != "tank":
                continue
            initial_pressure = float(node.get("initial_conditions", {}).get("pressure", 0.0))
            if initial_pressure <= 0.0:
                continue
            operating_limits = node.setdefault("operating_limits", {})
            operating_limits["venting_pressure"] = 1.5 * initial_pressure

        for edge in network.get("edges", []):
            if edge.get("connection_type") != "pressure_compensation":
                continue
            activation = edge.setdefault("activation_conditions", {})
            activation["pressure_open_bar"] = 16.0
            activation["pressure_close_bar"] = 30.0

    @staticmethod
    def _apply_geometry(node: dict, radius: float, phi: float) -> None:
        geometry = node.setdefault("geometry", {})
        geometry["radius"] = float(radius)
        geometry["phi"] = float(phi)
        geometry.pop("mission_based_sizing", None)

    @staticmethod
    def _extract_metrics(design: DesignVector, orchestrator: SystemOrchestrator, results) -> SweepResult:
        times = np.asarray(results.times, dtype=float)
        mission_duration_s = float(times[-1]) if times.size > 0 else 0.0
        target_duration_s = float(orchestrator.tank_system.config.MISSION_DURATION)
        mission_completion_ratio = mission_duration_s / target_duration_s if target_duration_s > 0.0 else 0.0
        mission_completed = mission_duration_s >= (0.999 * target_duration_s) if target_duration_s > 0.0 else False

        initial_state = results.multi_tank_states[0]
        total_fuel_mass_kg = 0.0
        structure_mass_kg = 0.0
        total_inner_volume_m3 = 0.0
        total_outer_volume_m3 = 0.0

        for tank_index, tank in enumerate(orchestrator.tank_geometries):
            tank_state = initial_state.get_tank_state(tank_index)
            properties = orchestrator.tank_system._cached_tank_properties[tank_index]

            total_fuel_mass_kg += float(tank_state.fuel_mass)
            structure_mass_kg += float(properties["liner_mass"]) + float(properties["wall_mass"])
            total_inner_volume_m3 += float(properties["volume"])

            outer_radius = 0.5 * float(properties["outer_diameter"])
            cylinder_length = float(properties.get("cylindrical_section_length", 0.0))
            sphere_volume = (4.0 / 3.0) * math.pi * outer_radius ** 3
            cylinder_volume = math.pi * outer_radius ** 2 * cylinder_length
            total_outer_volume_m3 += sphere_volume + cylinder_volume

        total_mass_kg = total_fuel_mass_kg + structure_mass_kg
        gravimetric_efficiency = total_fuel_mass_kg / total_mass_kg if total_mass_kg > 0.0 else 0.0
        volumetric_efficiency = total_inner_volume_m3 / total_outer_volume_m3 if total_outer_volume_m3 > 0.0 else 0.0

        return SweepResult(
            design=design,
            mission_completed=mission_completed,
            mission_completion_ratio=mission_completion_ratio,
            mission_duration_s=mission_duration_s,
            target_duration_s=target_duration_s,
            gravimetric_efficiency=gravimetric_efficiency,
            volumetric_efficiency=volumetric_efficiency,
            total_fuel_mass_kg=total_fuel_mass_kg,
            structure_mass_kg=structure_mass_kg,
            total_inner_volume_m3=total_inner_volume_m3,
            total_outer_volume_m3=total_outer_volume_m3,
        )

    def _build_explicit_design_points(self, sweep_config: dict) -> list[DesignVector]:
        raw_points = sweep_config.get("design_points", [])
        if not raw_points:
            raise ValueError("Sweep config uses mode 'explicit' but defines no design_points.")

        design_points: list[DesignVector] = []
        for point in raw_points:
            design_points.append(
                DesignVector(
                    tank_1_radius=float(point["tank_1_radius"]),
                    tank_1_phi=float(point["tank_1_phi"]),
                    tank_2_radius=float(point["tank_2_radius"]),
                    tank_2_phi=float(point["tank_2_phi"]),
                )
            )
        return design_points


def _coerce_float_sequence(values: Sequence[float] | None) -> Sequence[float] | None:
    if values is None:
        return None
    return [float(value) for value in values]


def main() -> None:
    sweep_config_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SWEEP_CONFIG_PATH

    if sweep_config_path.exists():
        sweep, sweep_config = PressureBufferSweep.from_sweep_config(sweep_config_path)
        design_points = sweep.build_design_points_from_config(sweep_config)
    else:
        sweep = PressureBufferSweep()
        design_points = sweep.build_cartesian_designs()

    results = sweep.run_sweep(design_points)
    report_path = sweep.write_report(results)
    print(f"Sweep completed. Report written to {report_path}")


if __name__ == "__main__":
    main()