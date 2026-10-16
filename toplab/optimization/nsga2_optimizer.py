"""NSGA-II multi-objective optimisation on the typed config.

Uses the same design-variable list and :class:`CandidateEvaluator` as the SQP optimiser, so any
number of tanks and any of radius / phi / insulation thickness can be explored. Objectives are
maximised (``eta_g``, ``eta_v``, ``eta_vent``, ``zeta_p``); pymoo minimises, so they are negated.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymoo.core.problem import ElementwiseProblem
import yaml

from toplab.config import load_config
from toplab.optimization.evaluation import (
    CandidateEvaluator, CandidateResult, DesignVariable, parse_design_variables)
from toplab.optimization.sqp_optimizer import SQPOptimizer

OBJECTIVES = ("eta_g", "eta_v", "eta_vent", "zeta_p")
# Penalty for candidates that failed to evaluate; large enough to rank below any real violation.
FAILED_VIOLATION = 1.0e3
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class NSGA2Settings:
    population_size: int = 8
    generations: int = 3
    seed: int | None = None
    eliminate_duplicates: bool = True
    objectives: tuple[str, ...] = ("eta_g", "eta_v", "eta_vent")
    n_workers: int = 1


def constraint_vector(result: CandidateResult) -> list[float]:
    """Inequality constraints ``g <= 0``: discharge shortfall [h], dormancy margin, packaging margin."""
    if result.error is not None:
        return [FAILED_VIOLATION] * 3
    packaging = -result.zeta_p if result.packaging_feasible else max(1.0, -result.zeta_p)
    return [result.discharge_residual / SECONDS_PER_HOUR, result.dormancy_residual, packaging]


def objective_vector(result: CandidateResult, objectives: Sequence[str]) -> list[float]:
    """Minimisation form of the requested (maximised) metrics."""
    if result.error is not None:
        return [0.0] * len(objectives)
    return [-float(getattr(result, name)) for name in objectives]


@dataclass
class NSGA2Result:
    variables: list[DesignVariable]
    objectives: tuple[str, ...]
    pareto_x: np.ndarray
    pareto: list[CandidateResult]
    history: list[tuple[tuple[float, ...], CandidateResult]] = field(default_factory=list)


class TankDesignProblem(ElementwiseProblem):
    def __init__(self, evaluator, variables, objectives, **kwargs):
        self.evaluator, self.objective_names = evaluator, tuple(objectives)
        super().__init__(
            n_var=len(variables), n_obj=len(objectives), n_ieq_constr=3,
            xl=np.array([v.lower for v in variables], dtype=float),
            xu=np.array([v.upper for v in variables], dtype=float), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        candidate = self.evaluator.evaluate(x)
        out["F"] = objective_vector(candidate, self.objective_names)
        out["G"] = constraint_vector(candidate)
        out["candidate"] = candidate


class NSGA2Optimizer:
    """Configure and run NSGA-II from an optimisation YAML (same shape as the SQP one)."""

    def __init__(self, opt_config_path: Path, evaluator: Any = None) -> None:
        self.opt_config_path = Path(opt_config_path)
        with self.opt_config_path.open("r", encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream) or {}
        base_rel = self.cfg.get("optimization", {}).get("base_config", self.cfg.get("base_config"))
        if base_rel is None:
            raise ValueError("Optimization configuration must define base_config.")
        self.base_config_path = (self.opt_config_path.parent / base_rel).resolve()
        self.base_config = load_config(self.base_config_path)
        self.variables = parse_design_variables(self.cfg.get("design_variables", {}))

        raw = self.cfg.get("nsga2", {})
        objectives = tuple(raw.get("objectives", NSGA2Settings.objectives))
        if not objectives or set(objectives) - set(OBJECTIVES):
            raise ValueError(f"nsga2.objectives must be a non-empty subset of {OBJECTIVES}.")
        self.settings = NSGA2Settings(
            population_size=int(raw.get("population_size", 8)),
            generations=int(raw.get("generations", 3)),
            seed=raw.get("seed"),
            eliminate_duplicates=bool(raw.get("eliminate_duplicates", True)),
            objectives=objectives,
            n_workers=int(raw.get("n_workers", 1)))
        self.evaluator = evaluator or CandidateEvaluator(
            self.base_config, self.variables,
            SQPOptimizer._maximum_vented_fraction(self.base_config, self.cfg),
            SQPOptimizer._hold_period_s(self.base_config, self.cfg),
            progress=bool(raw.get("evaluation_progress", True)))

    def initial_population(self) -> np.ndarray:
        """Random sample inside the bounds with the baseline design as the first member."""
        lower = np.array([v.lower for v in self.variables])
        upper = np.array([v.upper for v in self.variables])
        rng = np.random.default_rng(self.settings.seed)
        population = rng.uniform(lower, upper, size=(self.settings.population_size, len(self.variables)))
        population[0] = [v.baseline for v in self.variables]
        return population

    def run(self) -> NSGA2Result:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.callback import Callback
        from pymoo.optimize import minimize

        history: list[tuple[tuple[float, ...], CandidateResult]] = []

        class Collect(Callback):
            def notify(self, algorithm):
                evaluated = algorithm.off if algorithm.off is not None and len(algorithm.off) else algorithm.pop
                for individual in evaluated:
                    history.append((tuple(float(v) for v in individual.X), individual.get("candidate")))

        pool, kwargs = None, {}
        if self.settings.n_workers > 1:
            import multiprocessing
            from pymoo.parallelization.starmap import StarmapParallelization
            pool = multiprocessing.Pool(self.settings.n_workers)
            kwargs["elementwise_runner"] = StarmapParallelization(pool.starmap)
        try:
            problem = TankDesignProblem(self.evaluator, self.variables, self.settings.objectives, **kwargs)
            algorithm = NSGA2(pop_size=self.settings.population_size,
                              eliminate_duplicates=self.settings.eliminate_duplicates,
                              sampling=self.initial_population())
            result = minimize(problem, algorithm, termination=("n_gen", self.settings.generations),
                              seed=self.settings.seed, callback=Collect(), verbose=False)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if result.X is None:
            return NSGA2Result(self.variables, self.settings.objectives,
                               np.empty((0, len(self.variables))), [], history)
        pareto_x = np.atleast_2d(result.X)
        by_x = {x: c for x, c in history}
        pareto = [by_x[tuple(float(v) for v in x)] for x in pareto_x]
        return NSGA2Result(self.variables, self.settings.objectives, pareto_x, pareto, history)


def _row(variables: Sequence[DesignVariable], x: Sequence[float], candidate: CandidateResult) -> dict[str, Any]:
    row = {v.name: float(value) for v, value in zip(variables, x)}
    row.update({name: getattr(candidate, name) for name in OBJECTIVES})
    row.update({"f_vent": candidate.f_vent, "g_dis": candidate.discharge_residual,
                "g_dorm": candidate.dormancy_residual, "packaging_feasible": candidate.packaging_feasible,
                "feasible": candidate.feasible, "error": candidate.error or ""})
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, result: NSGA2Result) -> None:
    """Every evaluated candidate, in evaluation order."""
    _write_rows(path, [{"evaluation": i + 1, **_row(result.variables, x, c)}
                       for i, (x, c) in enumerate(result.history)])


def write_pareto(path: Path, result: NSGA2Result) -> None:
    """The non-dominated feasible set found by the run."""
    _write_rows(path, [_row(result.variables, x, c) for x, c in zip(result.pareto_x, result.pareto)])


__all__ = ["NSGA2Optimizer", "NSGA2Result", "NSGA2Settings", "OBJECTIVES", "constraint_vector",
           "objective_vector", "write_history", "write_pareto"]
