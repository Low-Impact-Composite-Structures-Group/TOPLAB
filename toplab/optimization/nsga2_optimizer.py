"""NSGA-II multi-objective optimisation on the typed config.

Uses the same design-variable list and :class:`CandidateEvaluator` as the SQP optimiser, so any
number of tanks and any of radius / phi / insulation thickness can be explored. Objectives are
maximised (``eta_g``, ``eta_v``, ``eta_vent``, ``zeta_p``); pymoo minimises, so they are negated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from toplab.optimization.evaluation import (
    OBJECTIVES, CandidateResult, DesignVariable, OptimizationProblem,
    candidate_row, load_problem, physical_residuals, write_rows)


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
    g = physical_residuals(result)
    if result.error is None and not result.packaging_feasible:
        # An unplaceable design must rank below every placeable one regardless of zeta_p.
        g["g_packaging"] = max(1.0, g["g_packaging"])
    return list(g.values())


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
        self.problem: OptimizationProblem = load_problem(opt_config_path, evaluator)
        self.cfg = self.problem.cfg
        self.variables = self.problem.variables
        self.base_config = self.problem.base_config
        self.evaluator = self.problem.evaluator

        raw = self.cfg.get("nsga2", {}) or {}
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


def pareto_rows(result: NSGA2Result) -> list[dict[str, Any]]:
    return [candidate_row(result.variables, x, c) for x, c in zip(result.pareto_x, result.pareto)]


def history_rows(result: NSGA2Result) -> list[dict[str, Any]]:
    return [{"evaluation": i + 1, **candidate_row(result.variables, x, c)}
            for i, (x, c) in enumerate(result.history)]


def write_history(path: Path, result: NSGA2Result) -> None:
    """Every evaluated candidate, in evaluation order."""
    write_rows(path, history_rows(result))


def write_pareto(path: Path, result: NSGA2Result) -> None:
    """The non-dominated set returned by the run (feasible whenever any feasible design exists)."""
    write_rows(path, pareto_rows(result))


__all__ = ["NSGA2Optimizer", "NSGA2Result", "NSGA2Settings", "OBJECTIVES", "constraint_vector",
           "history_rows", "objective_vector", "pareto_rows", "write_history", "write_pareto"]
