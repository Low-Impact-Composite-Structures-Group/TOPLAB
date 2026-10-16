"""Multi-tank SLSQP optimization with epsilon-constraint Pareto sweeps."""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize

from toplab.config import load_config
from toplab.optimization.evaluation import (
    CandidateEvaluator, CandidateResult, DesignVariable, TankDesign,
    parse_design_variables, venting_performance)

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
        self.base_config = load_config(self.base_config_path)
        self.variables: list[DesignVariable] = parse_design_variables(self.cfg.get("design_variables", {}))
        self.evaluator = CandidateEvaluator(
            self.base_config, self.variables,
            self._maximum_vented_fraction(self.base_config, self.cfg),
            self._hold_period_s(self.base_config, self.cfg),
            progress=bool(self.cfg.get("solver", {}).get("evaluation_progress", True)))
        self._cache: dict[tuple[float, ...], CandidateResult] = {}
        self.history: list[dict[str, Any]] = []
        self._active_epsilon: EpsilonTriplet | None = None

    @staticmethod
    def _hold_period_s(base: Any, optimizer_raw: dict[str, Any] | None = None) -> float:
        settings = (optimizer_raw or {}).get("dormancy", {})
        hours = settings.get("hold_period_h", (base.dormancy_check or {}).get("duration_h"))
        if hours is None:
            raise ValueError("Define dormancy.hold_period_h or dormancy_check.duration_h.")
        return float(hours) * 3600.0

    @staticmethod
    def _maximum_vented_fraction(base: Any, optimizer_raw: dict[str, Any] | None = None) -> float:
        settings = (optimizer_raw or {}).get("dormancy", {})
        if "maximum_vented_percentage" in settings:
            return float(settings["maximum_vented_percentage"]) / 100.0
        dormancy = base.dormancy_check or {}
        if "maximum_vented_percentage" in dormancy:
            return float(dormancy["maximum_vented_percentage"]) / 100.0
        return float(dormancy.get("maximum_loss_percent_per_hour", 0.0)) * float(dormancy.get("duration_h", 12.0)) / 100.0

    def _bounds_and_x0(self) -> tuple[list[tuple[float, float]], np.ndarray]:
        return ([(v.lower, v.upper) for v in self.variables],
                np.asarray([v.baseline for v in self.variables], dtype=float))

    def _get(self, x: Sequence[float]) -> CandidateResult:
        key = tuple(float(value) for value in x)
        if key not in self._cache:
            self._cache[key] = self.evaluator.evaluate(key)
            self.history.append({
                "evaluation": len(self.history) + 1,
                "epsilon": self._active_epsilon,
                "x": key,
                "candidate": self._cache[key],
            })
        return self._cache[key]

    def _best_feasible(self, objective: str, tolerance: float) -> tuple[tuple[float, ...], CandidateResult] | None:
        feasible = [(entry["x"], entry["candidate"]) for entry in self.history
                    if entry["candidate"].feasible and all(
                        value <= tolerance
                        for value in physical_constraint_residuals(entry["candidate"]).values())]
        return max(feasible, key=lambda item: getattr(item[1], objective), default=None)

    def solve_epsilon_subproblem(self, epsilon: EpsilonTriplet, x0: Sequence[float]) -> SQPSubproblemResult:
        bounds, _ = self._bounds_and_x0()
        clipped = np.clip(np.asarray(x0, dtype=float), [b[0] for b in bounds], [b[1] for b in bounds])
        self._active_epsilon = epsilon

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
            final, x_final = self._get(opt.x), np.asarray(opt.x, dtype=float)
            tolerance = float(solver.get("feasibility_tolerance", options["ftol"]))
            feasible = bool(final.feasible and all(value <= tolerance for value in physical_residuals(x_final)))
            message = str(opt.message)
            if not feasible:
                # The packaging margin is non-smooth across its feasibility boundary, so
                # SLSQP can stall on the infeasible side; fall back to the best feasible point seen.
                best = self._best_feasible(objective, tolerance)
                if best is not None:
                    x_final, final = np.asarray(best[0], dtype=float), best[1]
                    feasible = True
                    message = f"best feasible evaluated point (SLSQP: {message})"
            return SingleObjectiveResult(
                objective, clipped.tolist(), x_final.tolist(), bool(opt.success),
                int(opt.status), message, int(getattr(opt, "nit", 0)),
                int(getattr(opt, "nfev", 0)), initial_candidate, final,
                tuple(float(value) for value in physical_residuals(x_final)),
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
    """Write every unique candidate evaluated, in order, tagged with the active epsilon triplet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in optimizer.history:
        candidate, epsilon = entry["candidate"], entry["epsilon"]
        rows.append({
            "evaluation": entry["evaluation"],
            "epsilon_v": "" if epsilon is None else epsilon.eta_v,
            "epsilon_vent": "" if epsilon is None else epsilon.eta_vent,
            "epsilon_p": "" if epsilon is None else epsilon.zeta_p,
            "x": list(entry["x"]), "eta_g": candidate.eta_g, "eta_v": candidate.eta_v,
            "eta_vent": candidate.eta_vent, "zeta_p": candidate.zeta_p,
            "g_dis": candidate.discharge_residual, "g_dorm": candidate.dormancy_residual,
            "g_packaging": physical_constraint_residuals(candidate)["g_packaging"],
            "packaging_feasible": candidate.packaging_feasible,
            "feasible": candidate.feasible, "error": candidate.error or "",
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0]) if rows else ["evaluation"]
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
           "SQPOptimizer", "SQPSubproblemResult", "TankDesign",
           "constraint_residuals", "epsilon_residuals", "physical_constraint_residuals", "serialize_evaluation_history",
           "serialize_single_objective", "serialize_sweep", "venting_performance"]
