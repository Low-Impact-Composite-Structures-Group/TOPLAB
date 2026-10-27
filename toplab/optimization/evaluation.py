"""Typed candidate evaluation shared by the optimisers.

A candidate is a list of :class:`DesignVariable` values applied to the typed
``AnalysisConfig`` with ``dataclasses.replace``; no YAML is written. Evaluation
runs the discharge mission, the dormancy hold and aft-fuselage packaging.
"""

from __future__ import annotations

import copy
import csv
import contextlib
import os
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from toplab.config import AnalysisConfig, load_config
from toplab.packaging.aft_placement import AftFuselageDimensions, maximize_packaging_compaction

PARAMETERS = ("radius", "phi", "insulation_thickness")
OBJECTIVES = ("eta_g", "eta_v", "eta_vent", "zeta_p")
RESIDUAL_NAMES = ("g_dis", "g_dorm", "g_packaging")
_ALIASES = {"r": "radius", "t_ins": "insulation_thickness"}
# Residual assigned to every constraint of a candidate that failed to evaluate.
FAILED_VIOLATION = 1.0e3
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class TankDesign:
    radius: float
    phi: float
    insulation_thickness: float


@dataclass(frozen=True)
class DesignVariable:
    """One scalar design variable on one tank (``tank`` is the 1-based tank index)."""
    tank: int
    parameter: str
    lower: float
    upper: float
    baseline: float

    @property
    def name(self) -> str:
        return f"{self.parameter}_{self.tank}"


def parse_design_variables(raw: Mapping[str, Mapping[str, float]]) -> list[DesignVariable]:
    """Parse ``{radius_1: {min, max, baseline}, ...}`` into an ordered list."""
    variables = []
    for name, spec in raw.items():
        match = re.fullmatch(r"(radius|phi|insulation_thickness|r|t_ins)(?:_?(\d+))?", name)
        if match is None:
            raise ValueError(
                f"Design variable '{name}' must be '<parameter>' or '<parameter>_<tank>' "
                f"with parameter in {PARAMETERS}.")
        parameter = _ALIASES.get(match.group(1), match.group(1))
        # A bare name (e.g. 'radius') targets tank 1.
        tank = int(match.group(2)) if match.group(2) else 1
        variables.append(DesignVariable(tank, parameter, float(spec["min"]),
                                        float(spec["max"]), float(spec["baseline"])))
    if not variables:
        raise ValueError("design_variables must define at least one variable.")
    names = [v.name for v in variables]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate design variables after alias resolution: {names}.")
    for variable in variables:
        if not variable.lower < variable.upper:
            raise ValueError(f"Design variable '{variable.name}' needs min < max.")
        if not variable.lower <= variable.baseline <= variable.upper:
            raise ValueError(f"Design variable '{variable.name}' baseline lies outside [min, max].")
    return variables


def apply_design(config: AnalysisConfig, variables: Sequence[DesignVariable],
                 values: Sequence[float]) -> AnalysisConfig:
    """Return a copy of ``config`` with the design values applied and mission sizing disabled."""
    if len(values) != len(variables):
        raise ValueError(f"Expected {len(variables)} design values, got {len(values)}.")
    geometries = []
    materials = []
    for tank in config.tanks:
        geometry = dict(tank.geometry)
        # A fixed radius replaces mission sizing for every tank, including untouched ones.
        geometry.pop("mission_based_sizing", None)
        geometries.append(geometry)
        materials.append(copy.deepcopy(tank.materials))
    for variable, value in zip(variables, values):
        if not 1 <= variable.tank <= len(geometries):
            raise ValueError(f"Design variable {variable.name} refers to a tank that does not exist.")
        index = variable.tank - 1
        if variable.parameter == "insulation_thickness":
            materials[index].setdefault("insulation", {})["thickness"] = float(value)
        else:
            geometries[index][variable.parameter] = float(value)
    tanks = [replace(tank, geometry=g, materials=m)
             for tank, g, m in zip(config.tanks, geometries, materials)]
    return replace(config, tanks=tanks, dormancy_check=None,
                   output={**config.output, "silent": True, "save_plots": False, "save_data": False})


def packaging_dimensions(config: AnalysisConfig) -> AftFuselageDimensions:
    values = (config.packaging or {}).get("aft_fuselage_dimensions")
    if not values:
        raise ValueError("Base configuration must define packaging.aft_fuselage_dimensions.")
    return AftFuselageDimensions(**{key: float(value) for key, value in values.items()})


@dataclass
class CandidateResult:
    designs: list[TankDesign] = field(default_factory=list)
    eta_g: float = 0.0
    eta_v: float = 0.0
    eta_vent: float = 0.0
    zeta_p: float = 0.0
    f_vent: float = 0.0
    vented_mass_kg: float = 0.0
    per_tank_vented_fraction: list[float] = field(default_factory=list)
    discharge_residuals: list[float] = field(default_factory=list)
    discharge_residual: float = 0.0
    dormancy_residuals: list[float] = field(default_factory=list)
    dormancy_residual: float = 0.0
    packaging_feasible: bool = False
    packages: bool = False
    z_shift_star: float = 0.0
    mission_completed: bool = False
    initial_hydrogen_masses_kg: list[float] = field(default_factory=list)
    final_hydrogen_masses_kg: list[float] = field(default_factory=list)
    structural_masses_kg: list[float] = field(default_factory=list)
    internal_volumes_m3: list[float] = field(default_factory=list)
    external_volumes_m3: list[float] = field(default_factory=list)
    placements: list[Any] = field(default_factory=list)
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return (self.error is None and self.mission_completed and
                self.discharge_residual <= 0.0 and self.dormancy_residual <= 0.0 and
                self.packaging_feasible)

    @property
    def objectives(self) -> list[float]:
        return [self.eta_g, self.eta_v, self.eta_vent, self.zeta_p]


def physical_residuals(result: CandidateResult) -> dict[str, float]:
    """Return the physical constraints in ``g <= 0`` form.

    ``g_dis`` is the discharge shortfall in hours, ``g_dorm`` the largest per-tank
    vented fraction relative to its limit minus one, and ``g_packaging`` the negated
    packaging compaction margin ``-zeta_p``. A candidate that failed to evaluate
    violates every constraint by :data:`FAILED_VIOLATION`.
    """
    if result.error is not None:
        return {name: FAILED_VIOLATION for name in RESIDUAL_NAMES}
    return {"g_dis": float(result.discharge_residual) / SECONDS_PER_HOUR,
            "g_dorm": float(result.dormancy_residual),
            "g_packaging": -float(result.zeta_p)}


def dormancy_limits(base: AnalysisConfig, cfg: Mapping[str, Any]) -> tuple[float, float]:
    """Return ``(hold_period_s, maximum_vented_fraction)``.

    The optimisation YAML ``dormancy`` section wins; otherwise the base config's
    ``dormancy_check`` supplies ``duration_h`` and ``maximum_vented_percentage``,
    ``maximum_loss_percent`` or ``maximum_loss_percent_per_hour`` (scaled by the hold).
    """
    settings = cfg.get("dormancy", {}) or {}
    check = base.dormancy_check or {}
    hours = settings.get("hold_period_h", check.get("duration_h"))
    if hours is None:
        raise ValueError("Define dormancy.hold_period_h or dormancy_check.duration_h.")
    hours = float(hours)
    percent = settings.get("maximum_vented_percentage", check.get("maximum_vented_percentage"))
    if percent is None:
        percent = check.get("maximum_loss_percent")
    if percent is None and "maximum_loss_percent_per_hour" in check:
        percent = float(check["maximum_loss_percent_per_hour"]) * hours
    if percent is None or float(percent) <= 0.0:
        raise ValueError("Define a positive dormancy.maximum_vented_percentage "
                         "(or dormancy_check.maximum_loss_percent) for the venting limit.")
    return hours * SECONDS_PER_HOUR, float(percent) / 100.0


def venting_performance(vented_fraction: float, maximum_vented_fraction: float) -> float:
    """Return the signed venting performance; values below zero are retained."""
    if maximum_vented_fraction <= 0.0:
        raise ValueError("maximum_vented_fraction must be positive.")
    return 1.0 - float(vented_fraction) / float(maximum_vented_fraction)


class CandidateEvaluator:
    """Evaluate discharge, dormancy, system metrics and aft packaging for one design."""

    def __init__(self, base_config: AnalysisConfig, variables: Sequence[DesignVariable],
                 maximum_vented_fraction: float, hold_period_s: float, progress: bool = True) -> None:
        self.base_config = base_config
        self.variables = list(variables)
        self.packaging_dimensions = packaging_dimensions(base_config)
        self.maximum_vented_fraction = float(maximum_vented_fraction)
        self.hold_period_s = float(hold_period_s)
        self.progress = bool(progress)
        self.evaluation_count = 0
        self._stdout = sys.stdout

    def _progress(self, message: str) -> None:
        if self.progress:
            print(f"[evaluation {self.evaluation_count}] {message}", file=self._stdout, flush=True)

    def evaluate(self, values: Sequence[float]) -> CandidateResult:
        """Evaluate one design; failures are captured in ``CandidateResult.error``.

        The simulation's own console output is suppressed; only progress lines are shown.
        """
        self._stdout = sys.stdout
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            return self._evaluate(values)

    def _evaluate(self, values: Sequence[float]) -> CandidateResult:
        from toplab.orchestration.run_analysis import run_dormancy_check
        from toplab.orchestration.system_orchestrator import SystemOrchestrator

        result = CandidateResult()
        self.evaluation_count += 1
        started = time.perf_counter()
        self._progress(f"start x={[float(v) for v in values]}")
        try:
            candidate = apply_design(self.base_config, self.variables, values)
            orchestrator = SystemOrchestrator(candidate, verbosity="quiet")
            discharge = orchestrator.run_simulation()
            tank_count = len(orchestrator.tank_geometries)

            target_s = float(orchestrator.tank_system.config.MISSION_DURATION)
            actual_s = float(discharge.times[-1])
            result.mission_completed = actual_s >= target_s
            result.discharge_residuals = [target_s - actual_s] * tank_count
            result.discharge_residual = max(result.discharge_residuals, default=0.0)
            result.initial_hydrogen_masses_kg = [
                float(discharge.multi_tank_states[0].get_tank_state(i).fuel_mass) for i in range(tank_count)]
            result.final_hydrogen_masses_kg = [
                float(discharge.multi_tank_states[-1].get_tank_state(i).fuel_mass) for i in range(tank_count)]
            properties = orchestrator.tank_system._cached_tank_properties
            result.structural_masses_kg = [
                float(properties[i]["liner_mass"]) + float(properties[i]["wall_mass"])
                for i in range(tank_count)]
            result.internal_volumes_m3 = [float(properties[i]["volume"]) for i in range(tank_count)]
            result.external_volumes_m3 = [float(properties[i]["outer_volume"]) for i in range(tank_count)]
            hydrogen = sum(result.initial_hydrogen_masses_kg)
            total_mass = hydrogen + sum(result.structural_masses_kg)
            result.eta_g = hydrogen / total_mass if total_mass > 0.0 else 0.0
            external = sum(result.external_volumes_m3)
            result.eta_v = sum(result.internal_volumes_m3) / external if external > 0.0 else 0.0
            self._progress(f"discharge actual_s={actual_s:.1f} target_s={target_s:.1f} eta_g={result.eta_g:.4f}")

            dormancy_case = replace(candidate, dormancy_check={
                "enabled": True, "duration_h": self.hold_period_s / 3600.0,
                "maximum_loss_percent": 100.0})
            dormancy = run_dormancy_check(dormancy_case, silent=True)
            initial = [t["initial_mass_kg"] for t in dormancy["tanks"]]
            vented = [t["vented_mass_kg"] for t in dormancy["tanks"]]
            result.vented_mass_kg = sum(vented)
            result.per_tank_vented_fraction = [v / m if m > 0.0 else 0.0 for v, m in zip(vented, initial)]
            result.f_vent = result.vented_mass_kg / sum(initial) if sum(initial) > 0.0 else 0.0
            limit = self.maximum_vented_fraction
            result.eta_vent = venting_performance(result.f_vent, limit)
            result.dormancy_residuals = [f / limit - 1.0 for f in result.per_tank_vented_fraction]
            result.dormancy_residual = max(result.dormancy_residuals, default=0.0)
            self._progress(f"dormancy f_vent={result.f_vent:.6g} g_dorm={result.dormancy_residual:.6g}")

            radii = [float(properties[i]["outer_diameter"]) / 2.0 for i in range(tank_count)]
            half_lengths = [float(properties[i].get("cylindrical_section_length", 0.0)) / 2.0
                            for i in range(tank_count)]
            packaging = maximize_packaging_compaction(radii, half_lengths, self.packaging_dimensions)
            result.packaging_feasible = result.packages = bool(packaging.feasible)
            result.z_shift_star = float(packaging.z_shift)
            result.zeta_p = float(packaging.zeta_p)
            result.placements = packaging.placements
            result.designs = [
                TankDesign(float(g.radius), float(g.phi),
                           float(candidate.tanks[i].materials
                                 .get("insulation", {}).get("thickness", 0.0)))
                for i, g in enumerate(orchestrator.tank_geometries)]
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            self._progress(f"error={result.error}")
        self._progress(f"done elapsed={time.perf_counter() - started:.1f}s "
                       f"objectives={result.objectives} feasible={result.feasible}")
        return result


@dataclass
class OptimizationProblem:
    """Everything an optimiser needs, built from one optimisation YAML."""
    path: Path
    cfg: dict[str, Any]
    base_config: AnalysisConfig
    variables: list[DesignVariable]
    evaluator: Any

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(v.lower, v.upper) for v in self.variables]

    @property
    def baseline(self) -> np.ndarray:
        return np.asarray([v.baseline for v in self.variables], dtype=float)

    @property
    def results_dir(self) -> Path:
        output = self.cfg.get("output", {}) or {}
        directory = Path(output.get("results_dir", output.get("results_directory", "output")))
        return directory if directory.is_absolute() else (self.path.parent / directory).resolve()


def load_problem(path: Path | str, evaluator: Any = None) -> OptimizationProblem:
    """Load an optimisation YAML, its base config, design variables and evaluator.

    ``evaluator`` replaces the real :class:`CandidateEvaluator` (used by tests).
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    base_rel = (cfg.get("optimization", {}) or {}).get("base_config", cfg.get("base_config"))
    if base_rel is None:
        raise ValueError("Optimization configuration must define base_config.")
    base_config = load_config((path.parent / base_rel).resolve())
    variables = parse_design_variables(cfg.get("design_variables", {}) or {})
    if evaluator is None:
        hold_s, limit = dormancy_limits(base_config, cfg)
        progress = next((bool(cfg[k]["evaluation_progress"]) for k in ("solver", "nsga2")
                         if "evaluation_progress" in (cfg.get(k) or {})), True)
        evaluator = CandidateEvaluator(base_config, variables, limit, hold_s, progress=progress)
    return OptimizationProblem(path, cfg, base_config, variables, evaluator)


def candidate_row(variables: Sequence[DesignVariable], x: Sequence[float],
                  candidate: CandidateResult) -> dict[str, Any]:
    """One flat CSV row: design variables by name, metrics, residuals and status."""
    row: dict[str, Any] = {v.name: float(value) for v, value in zip(variables, x)}
    row.update({name: getattr(candidate, name) for name in OBJECTIVES})
    row.update(physical_residuals(candidate))
    row.update({"f_vent": candidate.f_vent, "packaging_feasible": candidate.packaging_feasible,
                "feasible": candidate.feasible, "error": candidate.error or ""})
    return row


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


class RowLogger:
    """Append rows to a CSV as they are produced so a crash loses nothing.

    The file is created (truncating any old one) on the first row, flushed and fsynced after
    every row. If a later row introduces a new column the file is rewritten with the wider header.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._rows: list[Mapping[str, Any]] = []
        self._fieldnames: list[str] = []

    def append(self, row: Mapping[str, Any]) -> None:
        import os
        new = [key for key in row if key not in self._fieldnames]
        self._rows.append(row)
        if new or not self._fieldnames:
            self._fieldnames += new
            write_rows(self.path, self._rows)
            return
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self._fieldnames).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
