"""Typed candidate evaluation shared by the optimisers.

A candidate is a list of :class:`DesignVariable` values applied to the typed
``AnalysisConfig`` with ``dataclasses.replace``; no YAML is written. Evaluation
runs the discharge mission, the dormancy hold and aft-fuselage packaging.
"""

from __future__ import annotations

import copy
import csv
import contextlib
import math
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
from toplab.optimization.insulation_sizing import (
    DormancyPoint, InsulationGrid, size_insulation)
from toplab.packaging.aft_placement import AftFuselageDimensions, maximize_packaging_compaction

PARAMETERS = ("radius", "phi", "insulation_thickness")
# Every metric reported per candidate (SQP may target any of them).
OBJECTIVES = ("eta_g", "eta_v", "eta_vent", "zeta_p")
# Metrics NSGA-II maximises. Venting is a dormancy constraint (g_dorm), not an objective.
NSGA2_OBJECTIVES = ("eta_g", "eta_v", "zeta_p")
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


def with_insulation(config: AnalysisConfig, thickness: float) -> AnalysisConfig:
    """Return a copy of ``config`` with every tank's insulation set to ``thickness`` [m]."""
    tanks = []
    for tank in config.tanks:
        materials = copy.deepcopy(tank.materials)
        materials.setdefault("insulation", {})["thickness"] = float(thickness)
        tanks.append(replace(tank, materials=materials))
    return replace(config, tanks=tanks)


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
    structural_masses_kg: list[float] = field(default_factory=list)  # liner + composite wall
    insulation_masses_kg: list[float] = field(default_factory=list)
    shell_masses_kg: list[float] = field(default_factory=list)
    dry_masses_kg: list[float] = field(default_factory=list)  # structure + insulation + outer shell
    internal_volumes_m3: list[float] = field(default_factory=list)
    external_volumes_m3: list[float] = field(default_factory=list)
    placements: list[Any] = field(default_factory=list)
    error: str | None = None
    # Provenance: ``eval_id`` is assigned once by the optimiser driver (rank 0) per executed
    # candidate; ``worker_rank``/``pid`` record where the analysis actually ran.
    # Set when discharge proves the mission cannot be completed and the dormancy and packaging
    # analyses were therefore not run: their fields below are placeholders, not results.
    discharge_failed: bool = False
    # Internal insulation sizing (NSGA-II): the thickness used for the whole design, how it was
    # found and how many dormancy simulations that took. ``None`` when thickness is not sized.
    t_ins_star: float | None = None
    sizing_status: str = ""
    n_dormancy_evals: int = 0
    eval_id: int = 0
    worker_rank: int = 0
    pid: int = 0

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
    if result.discharge_failed:
        # Rejected through g_dis alone; the unevaluated constraints must not add violation.
        return {"g_dis": float(result.discharge_residual) / SECONDS_PER_HOUR,
                "g_dorm": 0.0, "g_packaging": 0.0}
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


def _mpi_rank() -> int:
    try:
        from mpi4py import MPI
        return int(MPI.COMM_WORLD.Get_rank())
    except ImportError:
        return 0


class CandidateEvaluator:
    """Evaluate discharge, dormancy, system metrics and aft packaging for one design."""

    def __init__(self, base_config: AnalysisConfig, variables: Sequence[DesignVariable],
                 maximum_vented_fraction: float, hold_period_s: float, progress: bool = True,
                 skip_after_discharge_failure: bool = False,
                 insulation_grid: InsulationGrid | None = None) -> None:
        self.base_config = base_config
        self.variables = list(variables)
        self.packaging_dimensions = packaging_dimensions(base_config)
        self.maximum_vented_fraction = float(maximum_vented_fraction)
        self.hold_period_s = float(hold_period_s)
        self.progress = bool(progress)
        # Skip dormancy and packaging when the mission is not completed (NSGA-II only).
        self.skip_after_discharge_failure = bool(skip_after_discharge_failure)
        # When set, insulation is sized internally on this grid instead of being a design variable.
        self.insulation_grid = insulation_grid
        self.evaluation_count = 0
        self._stdout = sys.stdout

    insulation_grid: InsulationGrid | None = None

    def _progress(self, message: str) -> None:
        if self.progress:
            print(f"[evaluation {self.evaluation_count}] {message}", file=self._stdout, flush=True)

    def evaluate(self, values: Sequence[float]) -> CandidateResult:
        """Evaluate one design; failures are captured in ``CandidateResult.error``.

        The simulation's own console output is suppressed; only progress lines are shown.
        """
        self._stdout = sys.stdout
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = self._evaluate(values)
        result.pid = os.getpid()
        result.worker_rank = _mpi_rank()
        return result

    @staticmethod
    def _designs(candidate: AnalysisConfig, orchestrator: Any) -> list[TankDesign]:
        return [TankDesign(float(g.radius), float(g.phi),
                           float(candidate.tanks[i].materials.get("insulation", {}).get("thickness", 0.0)))
                for i, g in enumerate(orchestrator.tank_geometries)]

    def _run_discharge(self, candidate: AnalysisConfig, result: CandidateResult) -> Any:
        """Run the mission; fill the discharge fields, ``eta_g`` and ``eta_v``. Returns the orchestrator."""
        from toplab.orchestration.system_orchestrator import SystemOrchestrator

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
        result.insulation_masses_kg = [float(properties[i]["foam_mass"]) for i in range(tank_count)]
        result.shell_masses_kg = [float(properties[i]["shell_mass"]) for i in range(tank_count)]
        result.dry_masses_kg = [s + f + o for s, f, o in zip(
            result.structural_masses_kg, result.insulation_masses_kg, result.shell_masses_kg)]
        result.internal_volumes_m3 = [float(properties[i]["volume"]) for i in range(tank_count)]
        result.external_volumes_m3 = [float(properties[i]["outer_volume"]) for i in range(tank_count)]
        hydrogen = sum(result.initial_hydrogen_masses_kg)
        # Same system definition as the report: fuel / (fuel + total dry mass incl. insulation and shell).
        total_mass = hydrogen + sum(result.dry_masses_kg)
        result.eta_g = hydrogen / total_mass if total_mass > 0.0 else 0.0
        external = sum(result.external_volumes_m3)
        result.eta_v = sum(result.internal_volumes_m3) / external if external > 0.0 else 0.0
        outcome = "mission completed" if result.mission_completed else "mission NOT completed"
        self._progress(f"  discharge done: {outcome} (actual {actual_s:.1f} s, target {target_s:.1f} s), "
                       f"eta_g={result.eta_g:.4f}, eta_v={result.eta_v:.4f}")
        return orchestrator

    def _run_dormancy(self, candidate: AnalysisConfig) -> DormancyPoint:
        """Run the dormancy hold; ``g_dorm`` is the worst per-tank vented fraction over its limit, minus one."""
        from toplab.orchestration.run_analysis import run_dormancy_check

        dormancy_case = replace(candidate, dormancy_check={
            "enabled": True, "duration_h": self.hold_period_s / 3600.0, "maximum_loss_percent": 100.0})
        dormancy = run_dormancy_check(dormancy_case, silent=True)
        initial = [t["initial_mass_kg"] for t in dormancy["tanks"]]
        vented = [t["vented_mass_kg"] for t in dormancy["tanks"]]
        fractions = [v / m if m > 0.0 else 0.0 for v, m in zip(vented, initial)]
        limit = self.maximum_vented_fraction
        residuals = [f / limit - 1.0 for f in fractions]
        f_vent = sum(vented) / sum(initial) if sum(initial) > 0.0 else 0.0
        return DormancyPoint(max(residuals, default=0.0), f_vent,
                             {"vented_mass_kg": sum(vented), "per_tank_vented_fraction": fractions,
                              "residuals": residuals})

    @staticmethod
    def _record_dormancy(result: CandidateResult, point: DormancyPoint, limit: float) -> None:
        data = point.data
        result.vented_mass_kg = data["vented_mass_kg"]
        result.per_tank_vented_fraction = data["per_tank_vented_fraction"]
        result.f_vent = point.f_vent
        result.eta_vent = venting_performance(point.f_vent, limit)
        result.dormancy_residuals = data["residuals"]
        result.dormancy_residual = point.g_dorm

    def _run_packaging(self, orchestrator: Any, result: CandidateResult) -> None:
        properties = orchestrator.tank_system._cached_tank_properties
        tank_count = len(orchestrator.tank_geometries)
        radii = [float(properties[i]["outer_diameter"]) / 2.0 for i in range(tank_count)]
        half_lengths = [float(properties[i].get("cylindrical_section_length", 0.0)) / 2.0
                        for i in range(tank_count)]
        packaging = maximize_packaging_compaction(radii, half_lengths, self.packaging_dimensions)
        result.packaging_feasible = result.packages = bool(packaging.feasible)
        result.z_shift_star = float(packaging.z_shift)
        result.zeta_p = float(packaging.zeta_p)
        result.placements = packaging.placements

    def _evaluate(self, values: Sequence[float]) -> CandidateResult:
        result = CandidateResult()
        self.evaluation_count += 1
        started = time.perf_counter()
        self._progress(f"start, design vector x={[float(v) for v in values]}")
        try:
            candidate = apply_design(self.base_config, self.variables, values)
            if self.insulation_grid is not None:
                self._evaluate_sized(candidate, result)
            else:
                self._evaluate_fixed(candidate, result)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            self._progress(f"error={result.error}")
        self._progress(f"done elapsed={time.perf_counter() - started:.1f}s "
                       f"objectives={result.objectives} feasible={result.feasible}")
        return result

    def _evaluate_fixed(self, candidate: AnalysisConfig, result: CandidateResult) -> None:
        """Insulation comes from the candidate itself (design variable or base config)."""
        self._progress("step 1 of 3: discharge")
        orchestrator = self._run_discharge(candidate, result)
        if self.skip_after_discharge_failure and not result.mission_completed:
            result.discharge_failed = True
            result.designs = self._designs(candidate, orchestrator)
            self._progress("step 2 of 3: dormancy skipped, discharge failed")
            self._progress("step 3 of 3: packaging skipped, discharge failed")
            return
        self._progress("step 2 of 3: dormancy")
        point = self._run_dormancy(candidate)
        self._record_dormancy(result, point, self.maximum_vented_fraction)
        result.n_dormancy_evals = 1
        self._progress(f"  dormancy done: f_vent={result.f_vent:.6g}, g_dorm={result.dormancy_residual:.6g}")
        self._progress("step 3 of 3: packaging")
        self._run_packaging(orchestrator, result)
        self._progress(f"  packaging done: feasible={result.packaging_feasible}, zeta_p={result.zeta_p:.4f}")
        result.designs = self._designs(candidate, orchestrator)

    def _evaluate_sized(self, candidate: AnalysisConfig, result: CandidateResult) -> None:
        """Discharge check, discrete insulation sizing, then the full design at ``t_ins*``.

        Order: discharge (screen) -> dormancy (sizing) -> discharge again only if ``t*`` differs
        from the screening thickness -> packaging, so packaging never precedes the dynamic analyses.
        """
        grid = self.insulation_grid
        # The thinnest insulation the model can simulate screens the mission; t = 0 cannot be built.
        screen = next(t for t in grid.values if t > 0.0)
        self._progress(f"step 1 of 3: discharge (screening at {screen * 1e3:.2f} mm insulation)")
        screening = self._run_discharge(with_insulation(candidate, screen), result)
        if self.skip_after_discharge_failure and not result.mission_completed:
            result.discharge_failed = True
            result.t_ins_star = screen
            result.sizing_status = "discharge_failed"
            result.designs = self._designs(with_insulation(candidate, screen), screening)
            self._progress("step 2 of 3: dormancy skipped, discharge failed")
            self._progress("step 3 of 3: packaging skipped, discharge failed")
            return

        self._progress("step 2 of 3: dormancy (insulation sizing by bisection)")
        simulations = [0]

        def dormancy_at(thickness: float) -> DormancyPoint:
            if thickness <= 0.0:
                # A bare tank cannot be modelled (shell radius equals structure radius): it counts as violating.
                return DormancyPoint(math.inf, math.nan)
            simulations[0] += 1
            point = self._run_dormancy(with_insulation(candidate, thickness))
            self._progress(f"  bisection {simulations[0]}: t_ins={thickness * 1e3:.2f} mm, "
                           f"f_vent={point.f_vent:.6g}, g_dorm={point.g_dorm:.6g}")
            return point

        sizing = size_insulation(dormancy_at, grid)
        final = with_insulation(candidate, sizing.t_ins)
        if sizing.t_ins == screen:
            orchestrator = screening
        else:
            self._progress(f"  re-running discharge at t_ins*={sizing.t_ins * 1e3:.2f} mm")
            orchestrator = self._run_discharge(final, result)
        result.t_ins_star = sizing.t_ins
        result.sizing_status = sizing.status
        result.n_dormancy_evals = simulations[0]
        self._record_dormancy(result, sizing.point, self.maximum_vented_fraction)
        self._progress(f"  dormancy complete after {simulations[0]} bisections: "
                       f"t_ins*={sizing.t_ins * 1e3:.2f} mm ({sizing.status})")
        self._progress("step 3 of 3: packaging")
        self._run_packaging(orchestrator, result)
        self._progress(f"  packaging done: feasible={result.packaging_feasible}, zeta_p={result.zeta_p:.4f}")
        result.designs = self._designs(final, orchestrator)


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
        """``$TOPLAB_RESULTS_DIR`` if set (e.g. a cluster scratch folder), else ``output.results_dir``."""
        override = os.environ.get("TOPLAB_RESULTS_DIR")
        if override:
            return Path(override).resolve()
        output = self.cfg.get("output", {}) or {}
        directory = Path(output.get("results_dir", output.get("results_directory", "output")))
        return directory if directory.is_absolute() else (self.path.parent / directory).resolve()


def load_problem(path: Path | str, evaluator: Any = None,
                 skip_after_discharge_failure: bool = False,
                 size_insulation_internally: bool = False) -> OptimizationProblem:
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
    grid = None
    if size_insulation_internally:
        thickness = [v.name for v in variables if v.parameter == "insulation_thickness"]
        if thickness:
            raise ValueError(
                f"Insulation thickness is sized internally and cannot be a design variable "
                f"(remove {thickness}); configure the 'insulation_sizing' section instead.")
        grid = InsulationGrid.from_config(cfg.get("insulation_sizing"))
    if evaluator is None:
        hold_s, limit = dormancy_limits(base_config, cfg)
        progress = next((bool(cfg[k]["evaluation_progress"]) for k in ("solver", "nsga2")
                         if "evaluation_progress" in (cfg.get(k) or {})), True)
        evaluator = CandidateEvaluator(base_config, variables, limit, hold_s, progress=progress,
                                      skip_after_discharge_failure=skip_after_discharge_failure,
                                      insulation_grid=grid)
    return OptimizationProblem(path, cfg, base_config, variables, evaluator)


def candidate_row(variables: Sequence[DesignVariable], x: Sequence[float],
                  candidate: CandidateResult) -> dict[str, Any]:
    """One flat CSV row: design variables by name, metrics, residuals and status."""
    row: dict[str, Any] = {v.name: float(value) for v, value in zip(variables, x)}
    row.update({name: getattr(candidate, name) for name in OBJECTIVES})
    row.update(physical_residuals(candidate))
    if candidate.t_ins_star is not None:
        row.update({"t_ins_star": candidate.t_ins_star, "sizing_status": candidate.sizing_status,
                    "n_dormancy_evals": candidate.n_dormancy_evals})
    row.update({"f_vent": candidate.f_vent, "packaging_feasible": candidate.packaging_feasible,
                "feasible": candidate.feasible, "discharge_failed": candidate.discharge_failed,
                "error": candidate.error or ""})
    if candidate.discharge_failed:
        # Blank cells mark quantities that were never computed (not zero results).
        for name in ("eta_vent", "zeta_p", "g_dorm", "g_packaging", "f_vent", "packaging_feasible"):
            row[name] = ""
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
