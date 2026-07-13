from __future__ import annotations

import copy
import csv
import math
import tempfile
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configuration.scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator
from src.optimization.sweep_runner import SweepResult
from optimization.pressure_buffer_sweep.pressure_buffer_sweep import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_SWEEP_CONFIG_PATH,
    DesignVector,
    PressureBufferSweep,
)


DEFAULT_SENSITIVITY_CONFIG_PATH = PROJECT_ROOT / "optimization" / "presure_buffer_opt" / "pressure_buffer_sensitivity.yaml"


OBJECTIVE_DEFS: dict[str, dict[str, str]] = {
    "gravimetric_efficiency": {
        "sense": "max",
        "label": "Maximize gravimetric efficiency",
    },
    "volumetric_efficiency": {
        "sense": "max",
        "label": "Maximize volumetric efficiency",
    },
    "vent_time_after_mission_s": {
        "sense": "max",
        "label": "Maximize time-to-vent after mission",
    },
}

# Active design variables per objective — fully decoupled so no wasted simulations.
OBJECTIVE_ACTIVE_VARIABLES: dict[str, list[str]] = {
    "gravimetric_efficiency": ["radius_scale", "length_scale"],
    "volumetric_efficiency": ["radius_scale", "length_scale"],
    "vent_time_after_mission_s": ["insulation_scale"],
}


@dataclass(frozen=True)
class SharedScaleDesign:
    radius_scale: float
    length_scale: float
    insulation_scale: float = 1.0


@dataclass
class ObjectiveEvaluation:
    objective_name: str
    objective_value: float
    objective_score: float
    design: SharedScaleDesign
    design_vector: DesignVector
    discharge: SweepResult
    vent_time_after_mission_s: float | None


class PressureBufferSensitivityStudy:
    def __init__(
        self,
        sensitivity_config_path: str | Path = DEFAULT_SENSITIVITY_CONFIG_PATH,
    ) -> None:
        self.config_path = Path(sensitivity_config_path).resolve()
        with open(self.config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        study_cfg = config.get("study", {})
        output_cfg = config.get("output", {})

        base_config = study_cfg.get("base_config", config.get("base_config", DEFAULT_CONFIG_PATH))
        if not Path(base_config).is_absolute():
            base_config = (PROJECT_ROOT / base_config).resolve()

        report_directory = study_cfg.get("report_directory", "optimization/presure_buffer_opt/output")
        if not Path(report_directory).is_absolute():
            report_directory = (PROJECT_ROOT / report_directory).resolve()
        self.report_directory = Path(report_directory)

        sweep_config_path = study_cfg.get("sweep_config", DEFAULT_SWEEP_CONFIG_PATH)
        if not Path(sweep_config_path).is_absolute():
            sweep_config_path = (PROJECT_ROOT / sweep_config_path).resolve()

        self.objectives = tuple(study_cfg.get("objective_sequence", [
            "gravimetric_efficiency",
            "vent_time_after_mission_s",
            "volumetric_efficiency",
        ]))

        steps_cfg = study_cfg.get("steps_per_objective", 10)
        if isinstance(steps_cfg, dict):
            # Per-objective mapping; missing keys fall back to "default" then 10.
            _default = int(steps_cfg.get("default", 10))
            self.steps_per_objective: dict[str, int] = {
                obj: int(steps_cfg.get(obj, _default))
                for obj in OBJECTIVE_DEFS
            }
        else:
            _n = int(steps_cfg)
            self.steps_per_objective = {obj: _n for obj in OBJECTIVE_DEFS}

        bt_cfg = study_cfg.get("backtracking", {})
        initial_step_cfg = bt_cfg.get("initial_step", {})
        min_step_cfg = bt_cfg.get("min_step", {})
        self.initial_step: dict[str, float] = {
            "radius_scale":    float(initial_step_cfg.get("radius_scale",    0.20)),
            "length_scale":    float(initial_step_cfg.get("length_scale",    0.20)),
            "insulation_scale": float(initial_step_cfg.get("insulation_scale", 0.30)),
        }
        self.min_step: dict[str, float] = {
            "radius_scale":    float(min_step_cfg.get("radius_scale",    0.01)),
            "length_scale":    float(min_step_cfg.get("length_scale",    0.01)),
            "insulation_scale": float(min_step_cfg.get("insulation_scale", 0.02)),
        }
        self.max_backtracks = int(bt_cfg.get("max_backtracks", 5))

        bounds_cfg = study_cfg.get("bounds", {})
        radius_bounds    = bounds_cfg.get("radius_scale",    [0.5, 2.0])
        length_bounds    = bounds_cfg.get("length_scale",    [0.5, 2.5])
        insulation_bounds = bounds_cfg.get("insulation_scale", [0.1, 4.0])
        self.radius_bounds    = (float(radius_bounds[0]),    float(radius_bounds[1]))
        self.length_bounds    = (float(length_bounds[0]),    float(length_bounds[1]))
        self.insulation_bounds = (float(insulation_bounds[0]), float(insulation_bounds[1]))

        self.phi_max = float(study_cfg.get("phi_max", 8.0))

        dormancy_cfg = study_cfg.get("dormancy", {})
        self.dormancy_duration_s = float(dormancy_cfg.get("duration_s", 30.0 * 24.0 * 3600.0))

        # Cached baseline discharge for vent-time evaluations (geometry never changes there)
        self._vent_time_cached_discharge: SweepResult | None = None

        verbosity = output_cfg.get("verbosity", "quiet")
        silent = bool(output_cfg.get("silent", True))
        save_plots = bool(output_cfg.get("save_plots", False))
        save_data = bool(output_cfg.get("save_data", False))

        # We reuse the sweep class runtime for robust design evaluation.
        self.sweep = PressureBufferSweep(
            base_config_path=base_config,
            verbosity=verbosity,
            report_path=DEFAULT_REPORT_PATH,
            silent=silent,
            save_plots=save_plots,
            save_data=save_data,
            sweep_config_path=sweep_config_path,
        )

        self.baseline = self.sweep.baseline_design
        self.base_length_1 = self.baseline.tank_1_radius * self.baseline.tank_1_phi
        self.base_length_2 = self.baseline.tank_2_radius * self.baseline.tank_2_phi
        self.base_insulation_thickness_m = self._read_baseline_insulation_thickness()

    def run_all(self) -> dict[str, list[ObjectiveEvaluation]]:
        self.report_directory.mkdir(parents=True, exist_ok=True)
        all_histories: dict[str, list[ObjectiveEvaluation]] = {}

        for objective_name in self.objectives:
            history = self._run_objective(objective_name)
            all_histories[objective_name] = history
            self._write_objective_report(objective_name, history)

        self._write_summary_csv(all_histories)
        return all_histories

    def _run_objective(self, objective_name: str) -> list[ObjectiveEvaluation]:
        if objective_name not in OBJECTIVE_DEFS:
            raise ValueError(f"Unsupported objective '{objective_name}'")

        # For vent-time: run the baseline discharge ONCE and cache it.
        # Geometry stays at (1.0, 1.0) throughout — only insulation is active.
        if objective_name == "vent_time_after_mission_s":
            baseline_dv = self._to_design_vector(SharedScaleDesign(1.0, 1.0, 1.0))
            baseline_cfg = self.sweep._build_config_for_design(baseline_dv)
            self._vent_time_cached_discharge, _ = self._run_discharge(baseline_cfg, baseline_dv)

        design = SharedScaleDesign(radius_scale=1.0, length_scale=1.0, insulation_scale=1.0)
        history: list[ObjectiveEvaluation] = [self._evaluate(design, objective_name)]

        for _ in range(self.steps_per_objective.get(objective_name, 10)):
            current_eval = history[-1]
            sensitivities = self._compute_local_sensitivities(design, current_eval, objective_name)

            # Backtracking line search: start with initial_step, halve on failure.
            step_sizes = {var: self.initial_step[var] for var in self._active_vars(objective_name)}
            found = False
            for _ in range(self.max_backtracks):
                candidate = self._apply_step(design, sensitivities, step_sizes)
                candidate_eval = self._evaluate(candidate, objective_name)
                if candidate_eval.objective_score > current_eval.objective_score:
                    design = candidate
                    history.append(candidate_eval)
                    found = True
                    break
                step_sizes = {v: s * 0.5 for v, s in step_sizes.items()}
                if all(step_sizes[v] < self.min_step[v] for v in step_sizes):
                    break

            if not found:
                break  # Converged — no improvement at any step size.

        return history

    def _compute_local_sensitivities(
        self,
        design: SharedScaleDesign,
        current_eval: ObjectiveEvaluation,
        objective_name: str,
    ) -> dict[str, float]:
        # Only probe the variables that are active for this objective.
        sensitivities: dict[str, float] = {}
        for var_name in self._active_vars(objective_name):
            delta = self.initial_step[var_name]
            perturbed = self._offset_design(design, var_name, delta)
            eval_plus = self._evaluate(perturbed, objective_name)
            sensitivities[var_name] = (eval_plus.objective_score - current_eval.objective_score) / delta
        return sensitivities

    def _active_vars(self, objective_name: str) -> list[str]:
        return list(OBJECTIVE_ACTIVE_VARIABLES.get(objective_name, ["radius_scale", "length_scale", "insulation_scale"]))

    def _get_bounds(self, var_name: str) -> tuple[float, float]:
        return {"radius_scale": self.radius_bounds, "length_scale": self.length_bounds, "insulation_scale": self.insulation_bounds}[var_name]

    def _apply_step(
        self,
        design: SharedScaleDesign,
        sensitivities: dict[str, float],
        step_sizes: dict[str, float],
    ) -> SharedScaleDesign:
        r, l, ins = design.radius_scale, design.length_scale, design.insulation_scale
        for var_name, step in step_sizes.items():
            direction = _sign(sensitivities.get(var_name, 0.0))
            bounds = self._get_bounds(var_name)
            if var_name == "radius_scale":
                r = self._clamp(r + direction * step, bounds)
            elif var_name == "length_scale":
                l = self._clamp(l + direction * step, bounds)
            elif var_name == "insulation_scale":
                ins = self._clamp(ins + direction * step, bounds)
        return SharedScaleDesign(radius_scale=r, length_scale=l, insulation_scale=ins)

    def _evaluate(self, design: SharedScaleDesign, objective_name: str) -> ObjectiveEvaluation:
        design_vector = self._to_design_vector(design)
        config_dict = self.sweep._build_config_for_design(design_vector)
        self._apply_insulation_scaling(config_dict, design.insulation_scale)

        if objective_name == "vent_time_after_mission_s":
            # No discharge needed: dormancy starts from the fully-loaded base-config state.
            vent_time = self._run_dormancy_from_base(config_dict)
            discharge = self._vent_time_cached_discharge or SweepResult(
                design=design_vector,
                mission_completed=False, mission_completion_ratio=0.0,
                mission_duration_s=0.0, target_duration_s=0.0,
                gravimetric_efficiency=0.0, volumetric_efficiency=0.0,
                total_fuel_mass_kg=0.0, structure_mass_kg=0.0,
                total_inner_volume_m3=0.0, total_outer_volume_m3=0.0,
            )
            objective_value = float(vent_time)
            return ObjectiveEvaluation(
                objective_name=objective_name,
                objective_value=objective_value,
                objective_score=self._to_objective_score(objective_name, objective_value),
                design=design,
                design_vector=design_vector,
                discharge=discharge,
                vent_time_after_mission_s=vent_time,
            )

        discharge, _ = self._run_discharge(config_dict, design_vector)
        if not discharge.is_successful:
            return ObjectiveEvaluation(
                objective_name=objective_name,
                objective_value=self._fallback_objective_value(objective_name),
                objective_score=float("-inf"),
                design=design,
                design_vector=design_vector,
                discharge=discharge,
                vent_time_after_mission_s=None,
            )
        objective_value = self._extract_objective_value(objective_name, discharge, None)
        return ObjectiveEvaluation(
            objective_name=objective_name,
            objective_value=objective_value,
            objective_score=self._to_objective_score(objective_name, objective_value),
            design=design,
            design_vector=design_vector,
            discharge=discharge,
            vent_time_after_mission_s=None,
        )

    def _run_discharge(self, config_dict: dict, design_vector: DesignVector) -> tuple[SweepResult, Any]:
        scenario = ScenarioConfig(
            config_dict=config_dict,
            config_format=self.sweep.base_scenario.config_format,
            config_path=str(self.sweep.base_config_path),
        )

        try:
            with self.sweep._stdout_context() as stdout_target:
                stderr_context = redirect_stderr(stdout_target) if self.sweep.silent else nullcontext()
                stdout_redirect = redirect_stdout(stdout_target) if self.sweep.silent else nullcontext()
                with stdout_redirect, stderr_context:
                    orchestrator = SystemOrchestrator(scenario_config=scenario, verbosity=self.sweep.verbosity)
                    results = orchestrator.run_simulation()
                    metrics = self.sweep._extract_metrics(design_vector, orchestrator, results)
            return metrics, results
        except Exception as exc:
            failed = SweepResult(
                design=design_vector,
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
            return failed, None

    def _run_dormancy_from_base(self, base_config_dict: dict) -> float:
        """Run a dormancy simulation from the fully-loaded base config and return time to first vent [s]."""
        dormancy_config = self._build_dormancy_config_from_base(base_config_dict)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as tmp_file:
            yaml.dump(dormancy_config, tmp_file, default_flow_style=False, allow_unicode=True)
            tmp_path = tmp_file.name
        try:
            scenario = ScenarioConfig(
                config_dict=dormancy_config,
                config_format=self.sweep.base_scenario.config_format,
                config_path=tmp_path,
            )
            with self.sweep._stdout_context() as stdout_target:
                stderr_context = redirect_stderr(stdout_target) if self.sweep.silent else nullcontext()
                stdout_redirect = redirect_stdout(stdout_target) if self.sweep.silent else nullcontext()
                with stdout_redirect, stderr_context:
                    orchestrator = SystemOrchestrator(scenario_config=scenario, verbosity=self.sweep.verbosity)
                    dormancy_results = orchestrator.run_simulation()
        except Exception:
            return self.dormancy_duration_s
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return self._first_vent_time_s(dormancy_config, dormancy_results)

    def _build_dormancy_config_from_base(self, config_dict: dict) -> dict:
        """Build a dormancy config from a fully-loaded (pre-mission) config dict."""
        dormancy_config = copy.deepcopy(config_dict)

        mission = dormancy_config.setdefault("mission", {})
        mission["type"] = "dormancy"
        mission["profile"] = "constant_flow"
        mission["flow_rate"] = 0.0
        mission["duration"] = float(self.dormancy_duration_s)
        mission["parameters"] = {}

        for node in dormancy_config.get("network", {}).get("nodes", []):
            if node.get("type") != "tank":
                continue
            # Density stopping is irrelevant for a full stationary tank.
            sc = node.setdefault("stopping_criteria", {})
            sc["minimum_density"] = 0.01
            sc["use_density_stopping_events"] = False
            sc["use_duration_stopping"] = True

        # Large integration steps are appropriate for slow thermal dormancy.
        solver = dormancy_config.setdefault("solver", {})
        solver["time_step"] = 300.0
        solver["max_step"] = 3600.0

        output_config = dormancy_config.setdefault("output", {})
        output_config["silent"] = self.sweep.silent
        output_config["save_plots"] = False
        output_config["save_data"] = False
        return dormancy_config

    def _first_vent_time_s(self, config_dict: dict, results: Any) -> float:
        tank_nodes = [node for node in config_dict.get("network", {}).get("nodes", []) if node.get("type") == "tank"]
        tank_nodes.sort(key=lambda node: int(node.get("node_id", 0)))
        vent_pressures = [
            float(node.get("operating_limits", {}).get("venting_pressure", math.inf))
            for node in tank_nodes
        ]

        for time_s, multi_state in zip(results.times, results.multi_tank_states):
            for tank_index, vent_pressure in enumerate(vent_pressures):
                tank_state = multi_state.get_tank_state(tank_index)
                pressure = float(tank_state.pressure)
                if pressure >= vent_pressure:
                    return float(time_s)

        # No vent event detected within the dormancy window.
        return self.dormancy_duration_s

    def _to_design_vector(self, design: SharedScaleDesign) -> DesignVector:
        r1 = self.baseline.tank_1_radius * design.radius_scale
        r2 = self.baseline.tank_2_radius * design.radius_scale

        l1 = self.base_length_1 * design.length_scale
        l2 = self.base_length_2 * design.length_scale

        # Cap phi so tanks don't become unrealistically elongated.
        phi1 = min(l1 / r1, self.phi_max) if r1 > 0.0 else self.phi_max
        phi2 = min(l2 / r2, self.phi_max) if r2 > 0.0 else self.phi_max

        return DesignVector(
            tank_1_radius=float(r1),
            tank_1_phi=float(max(0.0, phi1)),
            tank_2_radius=float(r2),
            tank_2_phi=float(max(0.0, phi2)),
        )

    def _read_baseline_insulation_thickness(self) -> float:
        nodes = self.sweep.base_scenario.config_dict.get("network", {}).get("nodes", [])
        for node in nodes:
            if node.get("type") == "tank":
                insulation = node.get("materials", {}).get("insulation", {})
                if "thickness" in insulation:
                    return float(insulation["thickness"])
        return 0.05

    def _apply_insulation_scaling(self, config_dict: dict, insulation_scale: float) -> None:
        """Scale insulation thickness and HTC on all tank nodes in-place.

        Effective heat-transfer coefficient U = k/d, so U scales inversely with
        thickness.  Both fields are updated so the thermal model sees the correct
        conductance change.
        """
        if insulation_scale == 1.0:
            return
        for node in config_dict.get("network", {}).get("nodes", []):
            if node.get("type") != "tank":
                continue
            insulation = node.get("materials", {}).get("insulation", {})
            if "thickness" in insulation:
                insulation["thickness"] = float(insulation["thickness"]) * insulation_scale
            if "heat_transfer_coefficient" in insulation:
                # Thicker insulation → lower effective HTC (U ∝ 1/d)
                insulation["heat_transfer_coefficient"] = (
                    float(insulation["heat_transfer_coefficient"]) / insulation_scale
                )

    def _extract_objective_value(self, objective_name: str, discharge: SweepResult, vent_time: float | None) -> float:
        if objective_name == "gravimetric_efficiency":
            return float(discharge.gravimetric_efficiency)
        if objective_name == "volumetric_efficiency":
            return float(discharge.volumetric_efficiency)
        if objective_name == "vent_time_after_mission_s":
            if vent_time is None:
                return self.dormancy_duration_s
            return float(vent_time)
        raise ValueError(f"Unsupported objective '{objective_name}'")

    @staticmethod
    def _to_objective_score(objective_name: str, objective_value: float) -> float:
        sense = OBJECTIVE_DEFS[objective_name]["sense"]
        return objective_value if sense == "max" else -objective_value

    def _fallback_objective_value(self, objective_name: str) -> float:
        if OBJECTIVE_DEFS[objective_name]["sense"] == "max":
            return 0.0
        return self.dormancy_duration_s

    @staticmethod
    def _clamp(value: float, bounds: tuple[float, float]) -> float:
        lower, upper = bounds
        return max(lower, min(upper, value))

    def _offset_design(self, design: SharedScaleDesign, var_name: str, delta: float) -> SharedScaleDesign:
        if var_name == "radius_scale":
            return SharedScaleDesign(
                radius_scale=self._clamp(design.radius_scale + delta, self.radius_bounds),
                length_scale=design.length_scale,
                insulation_scale=design.insulation_scale,
            )
        if var_name == "length_scale":
            return SharedScaleDesign(
                radius_scale=design.radius_scale,
                length_scale=self._clamp(design.length_scale + delta, self.length_bounds),
                insulation_scale=design.insulation_scale,
            )
        if var_name == "insulation_scale":
            return SharedScaleDesign(
                radius_scale=design.radius_scale,
                length_scale=design.length_scale,
                insulation_scale=self._clamp(design.insulation_scale + delta, self.insulation_bounds),
            )
        raise ValueError(f"Unknown design variable '{var_name}'")

    def _write_objective_report(self, objective_name: str, history: list[ObjectiveEvaluation]) -> Path:
        report_path = self.report_directory / f"pressure_buffer_sensitivity_{objective_name}.txt"
        objective_meta = OBJECTIVE_DEFS[objective_name]

        lines = [
            "Pressure Buffer Sensitivity Study",
            "===============================",
            f"Objective: {objective_meta['label']}",
            f"Sense: {objective_meta['sense']}",
            f"Starting point: attached base YAML geometry (scale = 1.0)",
            "",
            "Iterative history",
        ]

        for index, evaluation in enumerate(history):
            dv = evaluation.design_vector
            tank_1_length = dv.tank_1_radius * dv.tank_1_phi
            tank_2_length = dv.tank_2_radius * dv.tank_2_phi
            lines.extend(
                [
                    f"{index:02d}. objective_value={evaluation.objective_value:.6f}, objective_score={evaluation.objective_score:.6f}",
                    (
                        "    shared scales: "
                        f"radius={evaluation.design.radius_scale:.5f}, "
                        f"length={evaluation.design.length_scale:.5f}, "
                        f"insulation={evaluation.design.insulation_scale:.5f} "
                        f"(t={evaluation.design.insulation_scale * self.base_insulation_thickness_m * 1000:.1f} mm)"
                    ),
                    (
                        "    tank geometry: "
                        f"tank1(r={dv.tank_1_radius:.4f}m, L={tank_1_length:.4f}m, phi={dv.tank_1_phi:.4f}), "
                        f"tank2(r={dv.tank_2_radius:.4f}m, L={tank_2_length:.4f}m, phi={dv.tank_2_phi:.4f})"
                    ),
                    (
                        "    discharge metrics: "
                        f"mission_completion={evaluation.discharge.mission_completion_ratio:.3%}, "
                        f"gravimetric={evaluation.discharge.gravimetric_efficiency:.5f}, "
                        f"volumetric={evaluation.discharge.volumetric_efficiency:.5f}"
                    ),
                ]
            )
            if evaluation.vent_time_after_mission_s is not None:
                lines.append(
                    f"    dormancy metric: first_vent_time={evaluation.vent_time_after_mission_s/3600.0:.4f} h"
                )
            if evaluation.discharge.error:
                lines.append(f"    error: {evaluation.discharge.error}")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def _write_summary_csv(self, histories: dict[str, list[ObjectiveEvaluation]]) -> Path:
        summary_path = self.report_directory / "pressure_buffer_sensitivity_summary.csv"
        fieldnames = [
            "objective",
            "iteration",
            "objective_value",
            "objective_score",
            "radius_scale",
            "length_scale",
            "insulation_scale",
            "insulation_thickness_mm",
            "tank_1_radius_m",
            "tank_1_length_m",
            "tank_1_phi",
            "tank_2_radius_m",
            "tank_2_length_m",
            "tank_2_phi",
            "mission_completion_ratio",
            "gravimetric_efficiency",
            "volumetric_efficiency",
            "vent_time_after_mission_s",
            "error",
        ]

        with open(summary_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()

            for objective_name, history in histories.items():
                for index, evaluation in enumerate(history):
                    dv = evaluation.design_vector
                    writer.writerow(
                        {
                            "objective": objective_name,
                            "iteration": index,
                            "objective_value": evaluation.objective_value,
                            "objective_score": evaluation.objective_score,
                            "radius_scale": evaluation.design.radius_scale,
                            "length_scale": evaluation.design.length_scale,
                            "insulation_scale": evaluation.design.insulation_scale,
                            "insulation_thickness_mm": evaluation.design.insulation_scale * self.base_insulation_thickness_m * 1000.0,
                            "tank_1_radius_m": dv.tank_1_radius,
                            "tank_1_length_m": dv.tank_1_radius * dv.tank_1_phi,
                            "tank_1_phi": dv.tank_1_phi,
                            "tank_2_radius_m": dv.tank_2_radius,
                            "tank_2_length_m": dv.tank_2_radius * dv.tank_2_phi,
                            "tank_2_phi": dv.tank_2_phi,
                            "mission_completion_ratio": evaluation.discharge.mission_completion_ratio,
                            "gravimetric_efficiency": evaluation.discharge.gravimetric_efficiency,
                            "volumetric_efficiency": evaluation.discharge.volumetric_efficiency,
                            "vent_time_after_mission_s": (
                                "" if evaluation.vent_time_after_mission_s is None else evaluation.vent_time_after_mission_s
                            ),
                            "error": "" if evaluation.discharge.error is None else evaluation.discharge.error,
                        }
                    )

        return summary_path


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def main() -> None:
    config_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SENSITIVITY_CONFIG_PATH
    study = PressureBufferSensitivityStudy(config_path)
    histories = study.run_all()

    print("Sensitivity study completed.")
    for objective_name, history in histories.items():
        if not history:
            continue
        best = max(history, key=lambda item: item.objective_score)
        print(
            f"  - {objective_name}: best_value={best.objective_value:.6f}, "
            f"radius_scale={best.design.radius_scale:.5f}, length_scale={best.design.length_scale:.5f}, "
            f"insulation_scale={best.design.insulation_scale:.5f}"
        )
    print(f"Reports written to {study.report_directory}")


if __name__ == "__main__":
    main()
