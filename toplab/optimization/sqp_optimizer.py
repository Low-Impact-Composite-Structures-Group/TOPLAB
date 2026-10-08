"""Multi-tank SLSQP optimization with epsilon-constraint Pareto sweeps."""

from __future__ import annotations

import copy
import csv
import itertools
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize

from toplab.packaging.aft_placement import AftFuselageDimensions, maximize_packaging_compaction


@dataclass(frozen=True)
class TankDesign:
    radius: float
    phi: float
    insulation_thickness: float


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


EvalResult = CandidateResult


@dataclass(frozen=True)
class EpsilonTriplet:
    eta_v: float
    eta_vent: float
    zeta_p: float


@dataclass
class SQPSubproblemResult:
    epsilon: EpsilonTriplet
    x0: list[float]
    x: list[float]
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    candidate: CandidateResult
    feasible: bool
    error: str | None = None


@dataclass
class ParetoSweepResult:
    epsilons: list[EpsilonTriplet]
    subproblems: list[SQPSubproblemResult]
    pareto: list[SQPSubproblemResult]
    candidate_evaluations: int


@dataclass
class SingleObjectiveResult:
    objective: str
    x0: list[float]
    x: list[float]
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    initial_candidate: CandidateResult
    candidate: CandidateResult
    physical_residuals: tuple[float, float, float]
    unique_evaluations: int
    feasible: bool
    error: str | None = None


def decode_design_vector(values: Sequence[float]) -> list[TankDesign]:
    if len(values) == 0 or len(values) % 3:
        raise ValueError("A design vector must contain three values per tank.")
    return [TankDesign(float(values[i]), float(values[i + 1]), float(values[i + 2]))
            for i in range(0, len(values), 3)]


def venting_performance(vented_fraction: float, maximum_vented_fraction: float) -> float:
    """Return the signed venting performance; values below zero are retained."""
    if maximum_vented_fraction <= 0.0:
        raise ValueError("maximum_vented_fraction must be positive.")
    return 1.0 - float(vented_fraction) / float(maximum_vented_fraction)


def epsilon_residuals(epsilon: EpsilonTriplet, result: CandidateResult) -> tuple[float, float, float]:
    """Return canonical g <= 0 residuals for the three epsilon constraints."""
    return (epsilon.eta_v - result.eta_v,
            epsilon.eta_vent - result.eta_vent,
            epsilon.zeta_p - result.zeta_p)


def physical_constraint_residuals(result: CandidateResult) -> dict[str, float]:
    """Return named physical residuals; every value <= 0 is feasible.

    ``g_dis`` is a time shortfall in seconds. ``g_dorm`` is the largest
    per-tank vented fraction divided by its allowed fraction, minus one.
    ``g_packaging`` is the signed compaction-margin constraint. Since
    ``zeta_p`` is positive for feasible forward compaction and negative for an
    unshifted placement violation, ``-zeta_p <= 0`` is the smooth packaging
    proxy used by SLSQP. The Boolean placement result remains a final validity
    guard because the placement search itself is sampled and numerical.
    """
    return {
        "g_dis": float(result.discharge_residual),
        "g_dorm": float(result.dormancy_residual),
        "g_packaging": -float(result.zeta_p),
    }


def constraint_residuals(result: CandidateResult,
                          epsilon: EpsilonTriplet | None = None) -> dict[str, float]:
    """Return all active constraints using the canonical ``g <= 0`` sign."""
    residuals = physical_constraint_residuals(result)
    if epsilon is not None:
        epsilon_values = epsilon_residuals(epsilon, result)
        residuals.update({
            "g_epsilon_v": epsilon_values[0],
            "g_epsilon_vent": epsilon_values[1],
            "g_epsilon_p": epsilon_values[2],
        })
    return residuals


class SQPCandidateEvaluator:
    """Evaluate discharge, dormancy, system metrics, and aft packaging once."""

    def __init__(self, base_config_path: Path, packaging_dimensions: AftFuselageDimensions,
                 maximum_vented_fraction: float, hold_period_s: float,
                 progress: bool = True) -> None:
        self.base_config_path = Path(base_config_path)
        self.packaging_dimensions = packaging_dimensions
        self.maximum_vented_fraction = float(maximum_vented_fraction)
        self.hold_period_s = float(hold_period_s)
        self.progress = bool(progress)
        with self.base_config_path.open("r", encoding="utf-8") as stream:
            self.base_config = yaml.safe_load(stream) or {}
        self.evaluation_count = 0

    def _progress(self, message: str) -> None:
        if self.progress:
            print(f"[SQP evaluation {self.evaluation_count}] {message}", flush=True)

    def evaluate(self, values: Sequence[float]) -> CandidateResult:
        designs = decode_design_vector(values)
        result = CandidateResult(designs=designs)
        self.evaluation_count += 1
        vector = [float(value) for value in values]
        started = time.perf_counter()
        self._progress(f"start x={vector}")
        try:
            stage_started = time.perf_counter()
            self._progress("discharge start")
            discharge_config = self._build_candidate_config(designs)
            with _TemporaryConfig(discharge_config, self.base_config_path.parent) as path:
                from toplab.configuration.scenario_configuration import ScenarioConfig
                from toplab.orchestration.system_orchestrator import SystemOrchestrator
                scenario = ScenarioConfig.from_yaml(path)
                orchestrator = SystemOrchestrator(scenario, verbosity="quiet")
                discharge = orchestrator.run_simulation()
            self._progress(
                f"discharge done elapsed={time.perf_counter() - stage_started:.2f}s")

            tank_count = len(orchestrator.tank_geometries)
            target_s = float(orchestrator.tank_system.config.MISSION_DURATION)
            actual_s = float(discharge.times[-1])
            result.mission_completed = actual_s >= target_s
            result.discharge_residuals = [target_s - actual_s] * tank_count
            result.discharge_residual = max(result.discharge_residuals, default=0.0)
            self._progress(
                f"discharge metrics actual_s={actual_s:.3f} target_s={target_s:.3f} "
                f"g_dis={result.discharge_residual:.3f}")
            result.initial_hydrogen_masses_kg = [
                float(discharge.multi_tank_states[0].get_tank_state(i).fuel_mass)
                for i in range(tank_count)]
            result.final_hydrogen_masses_kg = [
                float(discharge.multi_tank_states[-1].get_tank_state(i).fuel_mass)
                for i in range(tank_count)]
            properties = orchestrator.tank_system._cached_tank_properties
            result.structural_masses_kg = [
                float(properties[i]["liner_mass"]) + float(properties[i]["wall_mass"])
                for i in range(tank_count)]
            result.internal_volumes_m3 = [float(properties[i]["volume"]) for i in range(tank_count)]
            result.external_volumes_m3 = [float(properties[i]["outer_volume"]) for i in range(tank_count)]
            self._calculate_system_metrics(result)

            stage_started = time.perf_counter()
            self._progress("dormancy start")
            dormancy_config = self._build_dormancy_config(discharge_config)
            with _TemporaryConfig(dormancy_config, self.base_config_path.parent) as path:
                dormancy_scenario = ScenarioConfig.from_yaml(path)
                dormancy_orchestrator = SystemOrchestrator(dormancy_scenario, verbosity="quiet")
                dormancy = dormancy_orchestrator.run_simulation()
            self._progress(
                f"dormancy done elapsed={time.perf_counter() - stage_started:.2f}s")
            dormancy_initial = [
                float(dormancy.multi_tank_states[0].get_tank_state(i).fuel_mass)
                for i in range(tank_count)]
            dormancy_final = [
                float(dormancy.multi_tank_states[-1].get_tank_state(i).fuel_mass)
                for i in range(tank_count)]
            vented = [max(0.0, a - b) for a, b in zip(dormancy_initial, dormancy_final)]
            result.vented_mass_kg = sum(vented)
            total_initial = sum(dormancy_initial)
            result.per_tank_vented_fraction = [
                loss / initial if initial > 0.0 else 0.0
                for loss, initial in zip(vented, dormancy_initial)]
            result.f_vent = result.vented_mass_kg / total_initial if total_initial > 0.0 else 0.0
            limit = self.maximum_vented_fraction
            result.eta_vent = venting_performance(result.f_vent, limit) if limit > 0.0 else 0.0
            result.dormancy_residuals = [
                fraction / limit - 1.0 if limit > 0.0 else float("inf")
                for fraction in result.per_tank_vented_fraction]
            result.dormancy_residual = max(result.dormancy_residuals, default=0.0)
            self._progress(
                f"dormancy metrics f_vent={result.f_vent:.6g} "
                f"eta_vent={result.eta_vent:.6g} g_dorm={result.dormancy_residual:.6g}")

            stage_started = time.perf_counter()
            self._progress("packaging start")
            radii = [float(properties[i]["outer_diameter"]) / 2.0 for i in range(tank_count)]
            half_lengths = [
                float(properties[i].get("cylindrical_section_length", 0.0)) / 2.0
                for i in range(tank_count)]
            packaging = maximize_packaging_compaction(radii, half_lengths, self.packaging_dimensions)
            result.packaging_feasible = bool(packaging.feasible)
            result.packages = result.packaging_feasible
            result.z_shift_star = float(packaging.z_shift)
            result.zeta_p = float(packaging.zeta_p)
            result.placements = packaging.placements
            self._progress(
                f"packaging done elapsed={time.perf_counter() - stage_started:.2f}s "
                f"feasible={result.packaging_feasible} zeta_p={result.zeta_p:.6g}")
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            self._progress(f"error={result.error}")
        self._progress(
            f"done elapsed={time.perf_counter() - started:.2f}s "
            f"objectives={result.objectives} feasible={result.feasible}")
        return result

    @staticmethod
    def _calculate_system_metrics(result: CandidateResult) -> None:
        hydrogen = sum(result.initial_hydrogen_masses_kg)
        total_mass = hydrogen + sum(result.structural_masses_kg)
        result.eta_g = hydrogen / total_mass if total_mass > 0.0 else 0.0
        external = sum(result.external_volumes_m3)
        result.eta_v = sum(result.internal_volumes_m3) / external if external > 0.0 else 0.0

    def _build_candidate_config(self, designs: Sequence[TankDesign]) -> dict[str, Any]:
        config = copy.deepcopy(self.base_config)
        tanks = [node for node in config.get("network", {}).get("nodes", []) if node.get("type") == "tank"]
        if len(tanks) != len(designs):
            raise ValueError(f"Expected {len(designs)} tank nodes, found {len(tanks)}.")
        for tank, design in zip(tanks, designs):
            geometry = tank.setdefault("geometry", {})
            geometry.pop("mission_based_sizing", None)
            geometry["radius"] = design.radius
            geometry["phi"] = design.phi
            tank.setdefault("materials", {}).setdefault("insulation", {})["thickness"] = design.insulation_thickness
        config.setdefault("output", {}).update({"silent": True, "save_plots": False, "save_data": False})
        config.pop("dormancy_check", None)
        return config

    def _build_dormancy_config(self, discharge_config: dict[str, Any]) -> dict[str, Any]:
        config = copy.deepcopy(discharge_config)
        config.pop("missions", None)
        config["mission"] = {
            "type": "dormancy", "profile": "constant_flow", "ambient_temperature": 288.15,
            "assigned_to_node": 1, "flow_rate": 0.0, "duration": self.hold_period_s}
        return config


class _TemporaryConfig:
    def __init__(self, config: dict[str, Any], directory: Path) -> None:
        self.config, self.directory, self.path = config, directory, None

    def __enter__(self) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=self.directory, delete=False) as stream:
            yaml.safe_dump(self.config, stream)
            self.path = Path(stream.name)
        return str(self.path)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def _packaging_dimensions(raw: dict[str, Any]) -> AftFuselageDimensions:
    values = raw.get("packaging", {}).get("aft_fuselage_dimensions")
    if not values:
        raise ValueError("Base configuration must define packaging.aft_fuselage_dimensions.")
    return AftFuselageDimensions(**{key: float(value) for key, value in values.items()})


def _epsilon_values(raw: Any, name: str) -> list[float]:
    spec = raw.get(name, {}) if isinstance(raw, dict) else {}
    if isinstance(spec, dict) and "values" in spec:
        values = [float(value) for value in spec["values"]]
    elif isinstance(spec, dict) and {"min", "max", "count"} <= set(spec):
        values = np.linspace(float(spec["min"]), float(spec["max"]), int(spec["count"])).tolist()
    else:
        values = []
    if not values:
        raise ValueError(f"epsilon_constraints.{name} must define values or min/max/count.")
    return values


def _non_dominated(results: Sequence[SQPSubproblemResult], tolerance: float = 1e-8) -> list[SQPSubproblemResult]:
    kept = []
    for candidate in results:
        metrics = np.asarray(candidate.candidate.objectives)
        dominated = any(
            other is not candidate
            and np.all(np.asarray(other.candidate.objectives) >= metrics - tolerance)
            and np.any(np.asarray(other.candidate.objectives) > metrics + tolerance)
            for other in results)
        if not dominated:
            kept.append(candidate)
    return kept


def _unique_results(results: Sequence[SQPSubproblemResult], tolerance: float = 1e-8) -> list[SQPSubproblemResult]:
    unique: list[SQPSubproblemResult] = []
    for candidate in results:
        metrics = np.asarray(candidate.candidate.objectives)
        if any(np.all(np.abs(metrics - np.asarray(other.candidate.objectives)) <= tolerance)
               and np.all(np.abs(np.asarray(candidate.x) - np.asarray(other.x)) <= tolerance)
               for other in unique):
            continue
        unique.append(candidate)
    return unique


class SQPOptimizer:
    """Run one epsilon-constrained SLSQP problem or a configured sweep."""

    def __init__(self, opt_config_path: Path):
        self.opt_config_path = Path(opt_config_path)
        with self.opt_config_path.open("r", encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream) or {}
        base_rel = self.cfg.get("base_config", self.cfg.get("optimization", {}).get("base_config"))
        if base_rel is None:
            raise ValueError("Optimization configuration must define base_config.")
        self.base_config_path = (self.opt_config_path.parent / base_rel).resolve()
        with self.base_config_path.open("r", encoding="utf-8") as stream:
            base_raw = yaml.safe_load(stream) or {}
        self.base_raw = base_raw
        self.evaluator = SQPCandidateEvaluator(
            self.base_config_path, _packaging_dimensions(base_raw),
            self._maximum_vented_fraction(base_raw, self.cfg), self._hold_period_s(base_raw, self.cfg),
            progress=bool(self.cfg.get("solver", {}).get("evaluation_progress", True)))
        self._cache: dict[tuple[float, ...], CandidateResult] = {}

    @staticmethod
    def _hold_period_s(base_raw: dict[str, Any], optimizer_raw: dict[str, Any] | None = None) -> float:
        settings = (optimizer_raw or {}).get("dormancy", {})
        hours = settings.get("hold_period_h", base_raw.get("dormancy_check", {}).get("duration_h"))
        return float(hours) * 3600.0

    @staticmethod
    def _maximum_vented_fraction(base_raw: dict[str, Any], optimizer_raw: dict[str, Any] | None = None) -> float:
        settings = (optimizer_raw or {}).get("dormancy", {})
        if "maximum_vented_percentage" in settings:
            return float(settings["maximum_vented_percentage"]) / 100.0
        dormancy = base_raw.get("dormancy_check", {})
        if "maximum_vented_percentage" in dormancy:
            return float(dormancy["maximum_vented_percentage"]) / 100.0
        return float(dormancy.get("maximum_loss_percent_per_hour", 0.0)) * float(dormancy.get("duration_h", 12.0)) / 100.0

    def _variable_specs(self) -> list[dict[str, float]]:
        variables = self.cfg.get("design_variables", {})
        if all(name in variables for name in ("radius", "phi", "insulation_thickness")):
            tanks = [node for node in self.base_raw.get("network", {}).get("nodes", []) if node.get("type") == "tank"]
            return [variables[name] for _ in tanks for name in ("radius", "phi", "insulation_thickness")]
        fields = ("radius", "phi", "insulation_thickness")
        indexed: dict[int, dict[str, dict[str, float]]] = {}
        for name, spec in variables.items():
            match = re.fullmatch(r"(radius|r|phi|insulation_thickness|t_ins)_?(\d+)", name)
            if match is None:
                continue
            field_name = {"r": "radius", "t_ins": "insulation_thickness"}.get(match.group(1), match.group(1))
            indexed.setdefault(int(match.group(2)), {})[field_name] = spec
        if indexed and all(all(field in indexed[tank] for field in fields) for tank in indexed):
            return [indexed[tank][field] for tank in sorted(indexed) for field in fields]
        return [variables[name] for name in variables]

    def _bounds_and_x0(self) -> tuple[list[tuple[float, float]], np.ndarray]:
        specs = self._variable_specs()
        return ([(float(spec["min"]), float(spec["max"])) for spec in specs],
                np.asarray([float(spec["baseline"]) for spec in specs], dtype=float))

    def _get(self, x: Sequence[float]) -> CandidateResult:
        key = tuple(float(value) for value in x)
        if key not in self._cache:
            self._cache[key] = self.evaluator.evaluate(key)
        return self._cache[key]

    def solve_epsilon_subproblem(self, epsilon: EpsilonTriplet, x0: Sequence[float]) -> SQPSubproblemResult:
        bounds, _ = self._bounds_and_x0()
        clipped = np.clip(np.asarray(x0, dtype=float), [b[0] for b in bounds], [b[1] for b in bounds])

        def residuals(x: Sequence[float]) -> tuple[float, ...]:
            result = self._get(x)
            values = constraint_residuals(result, epsilon)
            return tuple(values[name] for name in (
                "g_dis", "g_dorm", "g_epsilon_v", "g_epsilon_vent",
                "g_epsilon_p", "g_packaging"))

        constraints = [{"type": "ineq", "fun": lambda x, i=i: -residuals(x)[i]} for i in range(6)]
        solver = self.cfg.get("solver", {})
        options = {"maxiter": int(solver.get("max_iter", solver.get("maxiter", 100))),
                   "ftol": float(solver.get("tol", solver.get("ftol", 1e-8))),
                   "disp": bool(solver.get("disp", False))}
        if "finite_diff_step" in solver:
            options["eps"] = solver["finite_diff_step"]
        try:
            opt = minimize(lambda x: -self._get(x).eta_g, clipped, method="SLSQP",
                           bounds=bounds, constraints=constraints, options=options)
            final = self._get(opt.x)
            # SLSQP status is an optimizer diagnostic, not the physical
            # feasibility test. A useful endpoint can satisfy every TOPLAB
            # constraint even when SLSQP stops at its iteration limit.
            feasibility_tolerance = float(solver.get("feasibility_tolerance", options["ftol"]))
            final_residuals = residuals(opt.x)
            feasible = bool(final.feasible and all(
                value <= feasibility_tolerance for value in final_residuals))
            return SQPSubproblemResult(epsilon, clipped.tolist(), np.asarray(opt.x).tolist(),
                                       bool(opt.success), int(opt.status), str(opt.message),
                                       int(getattr(opt, "nit", 0)), int(getattr(opt, "nfev", 0)),
                                       final, feasible)
        except Exception as exc:
            final = self._get(clipped)
            return SQPSubproblemResult(epsilon, clipped.tolist(), clipped.tolist(), False, -1,
                                       str(exc), 0, 0, final, False, str(exc))

    def solve_single_objective(self, objective: str, x0: Sequence[float] | None = None) -> SingleObjectiveResult:
        """Maximize one performance metric subject to physical constraints."""
        objective_names = {"eta_g", "eta_v", "eta_vent", "zeta_p"}
        if objective not in objective_names:
            raise ValueError(f"objective must be one of {sorted(objective_names)}")

        bounds, baseline = self._bounds_and_x0()
        initial = baseline if x0 is None else np.asarray(x0, dtype=float)
        clipped = np.clip(initial, [b[0] for b in bounds], [b[1] for b in bounds])

        def physical_residuals(x: Sequence[float]) -> tuple[float, ...]:
            values = physical_constraint_residuals(self._get(x))
            return tuple(values[name] for name in ("g_dis", "g_dorm", "g_packaging"))

        constraints = [{"type": "ineq", "fun": lambda x, i=i: -physical_residuals(x)[i]} for i in range(3)]
        solver = self.cfg.get("solver", {})
        options = {"maxiter": int(solver.get("max_iter", solver.get("maxiter", 100))),
                   "ftol": float(solver.get("tol", solver.get("ftol", 1e-8))),
                   "disp": bool(solver.get("disp", False))}
        if "finite_diff_step" in solver:
            options["eps"] = solver["finite_diff_step"]

        try:
            initial_candidate = self._get(clipped)
            opt = minimize(lambda x: -float(getattr(self._get(x), objective)), clipped,
                           method="SLSQP", bounds=bounds, constraints=constraints,
                           options=options)
            final = self._get(opt.x)
            tolerance = float(solver.get("feasibility_tolerance", options["ftol"]))
            feasible = bool(final.feasible and all(value <= tolerance for value in physical_residuals(opt.x)))
            return SingleObjectiveResult(
                objective, clipped.tolist(), np.asarray(opt.x).tolist(), bool(opt.success),
                int(opt.status), str(opt.message), int(getattr(opt, "nit", 0)),
                int(getattr(opt, "nfev", 0)), initial_candidate, final,
                tuple(float(value) for value in physical_residuals(opt.x)),
                len(self._cache), feasible)
        except Exception as exc:
            final = self._get(clipped)
            initial_candidate = self._get(clipped)
            return SingleObjectiveResult(objective, clipped.tolist(), clipped.tolist(), False,
                                         -1, str(exc), 0, 0, initial_candidate, final,
                                         tuple(float(value) for value in physical_residuals(clipped)),
                                         len(self._cache), False, str(exc))

    def run_epsilon_sweep(self) -> ParetoSweepResult:
        epsilon_raw = self.cfg.get("epsilon_constraints", {})
        epsilons = [EpsilonTriplet(*values) for values in itertools.product(
            _epsilon_values(epsilon_raw, "eta_v"),
            _epsilon_values(epsilon_raw, "eta_vent"),
            _epsilon_values(epsilon_raw, "zeta_p"))]
        bounds, baseline = self._bounds_and_x0()
        previous = baseline
        successful, results = [], []
        for epsilon in epsilons:
            run = self.solve_epsilon_subproblem(epsilon, previous)
            results.append(run)
            if run.feasible:
                previous = np.clip(run.x, [b[0] for b in bounds], [b[1] for b in bounds])
                successful.append(run)
        unique = _unique_results(successful)
        return ParetoSweepResult(epsilons, results, _non_dominated(unique), self.evaluator.evaluation_count)

    def run(self) -> ParetoSweepResult:
        return self.run_epsilon_sweep()


def evaluate_design(r1: float, r2: float, phi1: float, phi2: float,
                    base_raw: dict, base_config_path: Path) -> CandidateResult:
    """Compatibility helper for callers of the former two-tank wrapper."""
    evaluator = SQPCandidateEvaluator(
        base_config_path, _packaging_dimensions(base_raw),
        SQPOptimizer._maximum_vented_fraction(base_raw), SQPOptimizer._hold_period_s(base_raw))
    return evaluator.evaluate([r1, phi1, 0.026, r2, phi2, 0.026])


def serialize_single_objective(result: SingleObjectiveResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "objective": result.objective, "x0": result.x0, "x": result.x,
        "success": result.success, "feasible": result.feasible,
        "status": result.status, "message": result.message,
        "nit": result.nit, "nfev": result.nfev,
        "unique_evaluations": result.unique_evaluations,
        "g_dis": result.physical_residuals[0],
        "g_dorm": result.physical_residuals[1],
        "g_packaging": result.physical_residuals[2],
        "error": result.error or result.candidate.error or "",
        "initial_eta_g": result.initial_candidate.eta_g,
        "initial_eta_v": result.initial_candidate.eta_v,
        "initial_eta_vent": result.initial_candidate.eta_vent,
        "initial_zeta_p": result.initial_candidate.zeta_p,
        "eta_g": result.candidate.eta_g, "eta_v": result.candidate.eta_v,
        "eta_vent": result.candidate.eta_vent, "zeta_p": result.candidate.zeta_p,
        "z_shift_star": result.candidate.z_shift_star,
        "packaging_feasible": result.candidate.packaging_feasible,
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def serialize_evaluation_history(optimizer: SQPOptimizer, path: Path) -> None:
    """Write every unique candidate evaluated by the optimizer callbacks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for x, candidate in optimizer._cache.items():
        rows.append({
            "x": list(x), "eta_g": candidate.eta_g, "eta_v": candidate.eta_v,
            "eta_vent": candidate.eta_vent, "zeta_p": candidate.zeta_p,
            "g_dis": candidate.discharge_residual, "g_dorm": candidate.dormancy_residual,
            "g_packaging": physical_constraint_residuals(candidate)["g_packaging"],
            "packaging_feasible": candidate.packaging_feasible,
            "feasible": candidate.feasible, "error": candidate.error or "",
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0]) if rows else ["x"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_sweep(result: ParetoSweepResult, path: Path, runs: Sequence[SQPSubproblemResult] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = result.subproblems if runs is None else runs
    rows = [{
        "epsilon_v": run.epsilon.eta_v, "epsilon_vent": run.epsilon.eta_vent,
        "epsilon_p": run.epsilon.zeta_p, "x": run.x, "success": run.success,
        "feasible": run.feasible, "status": run.status, "message": run.message,
        "eta_g": run.candidate.eta_g, "eta_v": run.candidate.eta_v,
        "eta_vent": run.candidate.eta_vent, "zeta_p": run.candidate.zeta_p,
        "z_shift_star": run.candidate.z_shift_star,
        "g_dis": run.candidate.discharge_residual,
        "g_dorm": run.candidate.dormancy_residual,
        "g_packaging": physical_constraint_residuals(run.candidate)["g_packaging"],
        "g_epsilon_v": run.epsilon.eta_v - run.candidate.eta_v,
        "g_epsilon_vent": run.epsilon.eta_vent - run.candidate.eta_vent,
        "g_epsilon_p": run.epsilon.zeta_p - run.candidate.zeta_p,
        "packaging_feasible": run.candidate.packaging_feasible,
    } for run in selected]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["epsilon_v"])
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["CandidateResult", "EpsilonTriplet", "EvalResult", "ParetoSweepResult", "SingleObjectiveResult",
           "SQPCandidateEvaluator", "SQPOptimizer", "SQPSubproblemResult", "TankDesign",
           "constraint_residuals", "decode_design_vector", "epsilon_residuals", "evaluate_design",
           "physical_constraint_residuals", "serialize_evaluation_history",
           "serialize_single_objective", "serialize_sweep", "venting_performance"]
