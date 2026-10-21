"""SLSQP optimisation: one objective, optional epsilon constraints on the others.

A problem maximises one of ``eta_g``, ``eta_v``, ``eta_vent`` or ``zeta_p`` over any
non-empty subset of the design variables (radius, phi, insulation thickness, per tank)
subject to the physical constraints (discharge, dormancy venting, aft packaging).

* No ``epsilon_constraints`` -> a single-objective solve.
* ``epsilon_constraints`` on any of the other metrics -> the epsilon-constraint method:
  one solve per point of the Cartesian grid of epsilon values (``metric >= epsilon``),
  warm-started from the previous feasible solution, plus the non-dominated set.
* ``objective`` may be a list; each objective is then optimised in turn.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from toplab.optimization.evaluation import (
    FAILED_VIOLATION, OBJECTIVES, RESIDUAL_NAMES, CandidateResult, OptimizationProblem,
    candidate_row, load_problem, physical_residuals)


@dataclass
class SQPRun:
    """Outcome of one SLSQP solve (one objective, one epsilon point)."""
    objective: str
    epsilon: dict[str, float]
    x0: list[float]
    x: list[float]
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    initial: CandidateResult
    candidate: CandidateResult
    residuals: dict[str, float]
    feasible: bool
    error: str | None = None

    @property
    def value(self) -> float:
        return float(getattr(self.candidate, self.objective))


@dataclass
class SQPResult:
    objective: str
    runs: list[SQPRun]
    pareto: list[SQPRun]

    @property
    def best(self) -> SQPRun | None:
        """Feasible run with the highest objective value."""
        return max((run for run in self.runs if run.feasible), key=lambda run: run.value, default=None)


def epsilon_grid(raw: Mapping[str, Any] | None, objective: str) -> list[dict[str, float]]:
    """Cartesian product of the configured epsilon values, excluding the objective itself.

    Each entry is ``{metric: {values: [...]}}``, ``{metric: {min, max, count}}`` or a bare
    number/list. An empty grid yields ``[{}]`` (a plain single-objective solve).
    """
    axes: dict[str, list[float]] = {}
    for name, spec in (raw or {}).items():
        if name not in OBJECTIVES:
            raise ValueError(f"epsilon_constraints.{name} is not one of {OBJECTIVES}.")
        if name == objective:
            continue
        if isinstance(spec, Mapping) and "values" in spec:
            values = [float(v) for v in spec["values"]]
        elif isinstance(spec, Mapping) and {"min", "max", "count"} <= set(spec):
            values = np.linspace(float(spec["min"]), float(spec["max"]), int(spec["count"])).tolist()
        elif isinstance(spec, (int, float)):
            values = [float(spec)]
        elif isinstance(spec, (list, tuple)):
            values = [float(v) for v in spec]
        else:
            values = []
        if not values:
            raise ValueError(f"epsilon_constraints.{name} must define values or min/max/count.")
        axes[name] = values
    return [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]


def non_dominated(runs: Sequence[SQPRun], metrics: Sequence[str], tolerance: float = 1e-8) -> list[SQPRun]:
    """Unique runs not dominated in ``metrics`` (all maximised)."""
    points = [np.asarray([getattr(run.candidate, m) for m in metrics]) for run in runs]
    kept: list[int] = []
    for i, p in enumerate(points):
        dominated = any(np.all(q >= p - tolerance) and np.any(q > p + tolerance)
                        for j, q in enumerate(points) if j != i)
        duplicate = any(np.all(np.abs(points[j] - p) <= tolerance) for j in kept)
        if not dominated and not duplicate:
            kept.append(i)
    return [runs[i] for i in kept]


class SQPOptimizer:
    """Run SLSQP for the objective(s) and epsilon grid defined in an optimisation YAML."""

    def __init__(self, opt_config_path: Path | str, evaluator: Any = None) -> None:
        self.problem: OptimizationProblem = load_problem(opt_config_path, evaluator)
        self.cfg = self.problem.cfg
        self.variables = self.problem.variables
        self.evaluator = self.problem.evaluator
        self.solver: dict[str, Any] = self.cfg.get("solver", {}) or {}
        self.objectives = self._parse_objectives(self.cfg.get("objective"))
        method = str(self.solver.get("method", "SLSQP")).upper()
        if method != "SLSQP":
            raise ValueError(f"solver.method must be SLSQP, got '{method}'.")
        self._step = self._finite_difference_step()
        self._cache: dict[tuple[float, ...], CandidateResult] = {}
        self._solve_keys: set[tuple[float, ...]] = set()
        self.history: list[dict[str, Any]] = []
        self._active: tuple[str, dict[str, float]] = ("", {})

    @staticmethod
    def _parse_objectives(raw: Any) -> list[str]:
        names = [raw] if isinstance(raw, str) else list(raw or [])
        if not names:
            raise ValueError(f"Define 'objective' as one of {OBJECTIVES} (or a list of them).")
        unknown = [n for n in names if n not in OBJECTIVES]
        if unknown:
            raise ValueError(f"Unknown objective(s) {unknown}; choose from {OBJECTIVES}.")
        return names

    def _finite_difference_step(self) -> Any:
        """Scalar, list in variable order, or dict keyed by variable name or parameter."""
        raw = self.solver.get("finite_diff_step")
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            steps = []
            for v in self.variables:
                key = v.name if v.name in raw else v.parameter
                if key not in raw:
                    raise ValueError(f"solver.finite_diff_step has no entry for '{v.name}' or '{v.parameter}'.")
                steps.append(float(raw[key]))
            return steps
        if isinstance(raw, (list, tuple)):
            if len(raw) != len(self.variables):
                raise ValueError(f"solver.finite_diff_step has {len(raw)} entries for "
                                 f"{len(self.variables)} design variables {[v.name for v in self.variables]}.")
            return [float(s) for s in raw]
        return float(raw)

    def _get(self, x: Sequence[float]) -> CandidateResult:
        key = tuple(float(value) for value in x)
        if key not in self._cache:
            candidate = self._cache[key] = self.evaluator.evaluate(key)
            self.history.append({"evaluation": len(self.history) + 1, "objective": self._active[0],
                                 "epsilon": dict(self._active[1]), "x": key, "candidate": candidate})
        self._solve_keys.add(key)
        return self._cache[key]

    @staticmethod
    def residuals(candidate: CandidateResult, epsilon: Mapping[str, float]) -> dict[str, float]:
        """Physical constraints plus ``g_epsilon_<metric> = epsilon - metric``; all ``<= 0`` when feasible."""
        values = physical_residuals(candidate)
        for name, bound in epsilon.items():
            values[f"g_epsilon_{name}"] = (FAILED_VIOLATION if candidate.error is not None
                                           else bound - float(getattr(candidate, name)))
        return values

    def _best_seen(self, objective: str, epsilon: Mapping[str, float], tol: float):
        best = None
        for key in self._solve_keys:
            c = self._cache[key]
            if c.feasible and all(v <= tol for v in self.residuals(c, epsilon).values()):
                if best is None or getattr(c, objective) > getattr(best[1], objective):
                    best = (key, c)
        return best

    def solve(self, objective: str, epsilon: Mapping[str, float] | None = None,
              x0: Sequence[float] | None = None) -> SQPRun:
        """Maximise ``objective`` subject to physical constraints and ``metric >= epsilon[metric]``."""
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        epsilon = dict(epsilon or {})
        if objective in epsilon:
            raise ValueError("The objective cannot also carry an epsilon constraint.")
        solver = self.solver
        bounds = self.problem.bounds
        lower, upper = (np.asarray(b) for b in zip(*bounds))
        start = np.clip(self.problem.baseline if x0 is None else np.asarray(x0, dtype=float), lower, upper)
        names = [*RESIDUAL_NAMES, *(f"g_epsilon_{n}" for n in epsilon)]
        self._active, self._solve_keys = (objective, epsilon), set()

        def g(x: Sequence[float]) -> np.ndarray:
            values = self.residuals(self._get(x), epsilon)
            return np.asarray([values[n] for n in names])

        tol = float(solver.get("tol", solver.get("ftol", 1e-8)))
        options: dict[str, Any] = {"maxiter": int(solver.get("max_iter", solver.get("maxiter", 100))),
                                   "ftol": tol, "disp": bool(solver.get("disp", False))}
        if self._step is not None:
            options["eps"] = self._step
        feasibility_tolerance = float(solver.get("feasibility_tolerance", tol))

        initial = self._get(start)
        if initial.error is not None:
            raise RuntimeError(f"Starting design {start.tolist()} failed to evaluate: {initial.error}")
        try:
            opt = minimize(lambda x: -float(getattr(self._get(x), objective)), start, method="SLSQP",
                           bounds=bounds, options=options,
                           constraints=[{"type": "ineq", "fun": lambda x: -g(x)}])
            x_final, message = np.asarray(opt.x, dtype=float), str(opt.message)
            success, status = bool(opt.success), int(opt.status)
            nit, nfev, error = int(getattr(opt, "nit", 0)), int(getattr(opt, "nfev", 0)), None
        except Exception as exc:
            x_final, message, success, status, nit, nfev, error = start, str(exc), False, -1, 0, 0, str(exc)
        final = self._get(x_final)
        feasible = bool(final.feasible and all(v <= feasibility_tolerance for v in g(x_final)))
        if not feasible:
            # Packaging is non-smooth at its feasibility boundary, so SLSQP can stall just
            # outside it; fall back to the best feasible point this solve evaluated.
            best = self._best_seen(objective, epsilon, feasibility_tolerance)
            if best is not None:
                x_final, final, feasible = np.asarray(best[0]), best[1], True
                message = f"best feasible evaluated point (SLSQP: {message})"
        return SQPRun(objective, epsilon, start.tolist(), x_final.tolist(), success, status, message,
                      nit, nfev, initial, final, self.residuals(final, epsilon), feasible, error)

    def run_objective(self, objective: str) -> SQPResult:
        """Solve the epsilon grid for one objective (a single solve when the grid is empty)."""
        grid = epsilon_grid(self.cfg.get("epsilon_constraints"), objective)
        previous, runs = None, []
        for epsilon in grid:
            run = self.solve(objective, epsilon, previous)
            runs.append(run)
            if run.feasible:
                previous = run.x
        feasible = [run for run in runs if run.feasible]
        metrics = [objective, *runs[0].epsilon]
        return SQPResult(objective, runs, non_dominated(feasible, metrics) if len(metrics) > 1 else feasible)

    def run(self) -> list[SQPResult]:
        return [self.run_objective(objective) for objective in self.objectives]


def run_rows(optimizer: SQPOptimizer, runs: Sequence[SQPRun]) -> list[dict[str, Any]]:
    """Flat CSV rows for solves: objective, epsilon point, design, metrics, residuals, solver status."""
    rows = []
    for run in runs:
        row: dict[str, Any] = {"objective": run.objective,
                               **{f"eps_{k}": v for k, v in run.epsilon.items()}}
        row.update(candidate_row(optimizer.variables, run.x, run.candidate))
        row.update({k: v for k, v in run.residuals.items() if k.startswith("g_epsilon_")})
        row.update({"success": run.success, "status": run.status, "nit": run.nit, "nfev": run.nfev,
                    "message": run.message, f"initial_{run.objective}": getattr(run.initial, run.objective)})
        if run.error:
            row["error"] = run.error
        rows.append(row)
    return rows


def history_rows(optimizer: SQPOptimizer) -> list[dict[str, Any]]:
    """Every unique evaluation in order, tagged with its objective and epsilon point."""
    return [{"evaluation": e["evaluation"], "objective": e["objective"],
             **{f"eps_{k}": v for k, v in e["epsilon"].items()},
             **candidate_row(optimizer.variables, e["x"], e["candidate"])}
            for e in optimizer.history]


__all__ = ["SQPOptimizer", "SQPResult", "SQPRun", "epsilon_grid", "history_rows",
           "non_dominated", "run_rows"]
