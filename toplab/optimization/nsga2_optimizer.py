"""NSGA-II multi-objective optimisation on the typed config.

Uses the same design-variable list and :class:`CandidateEvaluator` as the SQP optimiser, so any
number of tanks and any of radius / phi / insulation thickness can be explored. Objectives are
maximised (``eta_g``, ``eta_v``, ``eta_vent``, ``zeta_p``); pymoo minimises, so they are negated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymoo.core.problem import Problem

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
    n_workers: int = 1  # ignored: parallelism comes from the MPI world size (mpirun -n N)


def launcher_world_size() -> int:
    """World size announced by the MPI launcher environment (1 if not launched under MPI)."""
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS"):
        if os.environ.get(key, "").isdigit():
            return int(os.environ[key])
    return 1


def check_mpi_world(launcher_size: int, mpi_size: int | None) -> None:
    """Fail loudly if the launcher started N>1 processes that are not one MPI world.

    Otherwise every process silently runs the full optimisation (same seed, same designs)
    and appends to the same history file, repeating every expensive evaluation N times.
    """
    if launcher_size > 1 and (mpi_size is None or mpi_size != launcher_size):
        raise RuntimeError(
            f"Launcher started {launcher_size} processes but mpi4py reports a world of "
            f"{mpi_size}; the processes are not communicating, so each would run the whole "
            "optimisation independently. Check that mpi4py is built against the mpirun in use.")


def constraint_vector(result: CandidateResult) -> list[float]:
    """Inequality constraints ``g <= 0``: discharge shortfall [h], dormancy margin, packaging margin."""
    g = physical_residuals(result)
    if result.error is None and not result.discharge_failed and not result.packaging_feasible:
        # An unplaceable design must rank below every placeable one regardless of zeta_p.
        g["g_packaging"] = max(1.0, g["g_packaging"])
    return list(g.values())


def objective_vector(result: CandidateResult, objectives: Sequence[str]) -> list[float]:
    """Minimisation form of the requested (maximised) metrics."""
    if result.error is not None or result.discharge_failed:
        # Placeholder worst values: the candidate is rejected through its constraint violation.
        return [0.0] * len(objectives)
    return [-float(getattr(result, name)) for name in objectives]


@dataclass
class NSGA2Result:
    variables: list[DesignVariable]
    objectives: tuple[str, ...]
    pareto_x: np.ndarray
    pareto: list[CandidateResult]
    history: list[tuple[tuple[float, ...], CandidateResult]] = field(default_factory=list)


def _evaluate_batch_serial(evaluator, X: np.ndarray, on_result=None) -> list[CandidateResult]:
    results = []
    for i, x in enumerate(X):
        results.append(evaluator.evaluate(x))
        if on_result is not None:
            on_result(i, results[-1])
    return results


class MPIEvaluator:
    """Master/worker evaluation over MPI with dynamic load balancing.

    Rank 0 runs the optimiser and calls :meth:`map`; every other rank sits in :meth:`serve`
    until rank 0 calls :meth:`shutdown`. Each rank builds its own evaluator, so only the design
    vector and the (picklable) ``CandidateResult`` cross the network.
    """
    TAG_WORK, TAG_STOP = 1, 2

    def __init__(self, evaluator, comm=None) -> None:
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.evaluator, self.comm = evaluator, comm
        self.rank, self.size = comm.Get_rank(), comm.Get_size()

    def serve(self) -> None:
        import traceback
        from mpi4py import MPI
        status = MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == self.TAG_STOP:
                return
            idx, x = task
            try:
                reply = (idx, self.evaluator.evaluate(np.asarray(x)), None)
            except Exception:
                reply = (idx, None, traceback.format_exc())
            self.comm.send(reply, dest=0, tag=self.TAG_WORK)

    def map(self, X: np.ndarray, on_result=None) -> list[CandidateResult]:
        """Evaluate ``X``; ``on_result(index, result)`` fires on rank 0 as each result arrives."""
        if self.size == 1:
            return _evaluate_batch_serial(self.evaluator, X, on_result)
        from mpi4py import MPI
        results: list[Any] = [None] * len(X)
        errors: list[str] = []
        next_task, outstanding, status = 0, 0, MPI.Status()
        for worker in range(1, self.size):
            if next_task < len(X):
                self.comm.send((next_task, X[next_task]), dest=worker, tag=self.TAG_WORK)
                next_task += 1
                outstanding += 1
        while outstanding:
            idx, result, err = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.TAG_WORK, status=status)
            outstanding -= 1
            if err:
                errors.append(err)
            else:
                results[idx] = result
                if on_result is not None:
                    on_result(idx, result)
            if next_task < len(X) and not errors:
                self.comm.send((next_task, X[next_task]), dest=status.Get_source(), tag=self.TAG_WORK)
                next_task += 1
                outstanding += 1
        if errors:
            raise RuntimeError("Worker evaluation failed:\n" + errors[0])
        return results

    def shutdown(self) -> None:
        for worker in range(1, self.size):
            self.comm.send(None, dest=worker, tag=self.TAG_STOP)


class TankDesignProblem(Problem):
    """Batch problem: a whole generation is handed to ``batch_evaluator`` at once."""

    def __init__(self, batch_evaluator, variables, objectives, **kwargs):
        self.batch_evaluator, self.objective_names = batch_evaluator, tuple(objectives)
        self.variables = variables
        super().__init__(
            n_var=len(variables), n_obj=len(objectives), n_ieq_constr=3,
            xl=np.array([v.lower for v in variables], dtype=float),
            xu=np.array([v.upper for v in variables], dtype=float), **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        candidates = self.batch_evaluator(X)
        out["F"] = np.array([objective_vector(c, self.objective_names) for c in candidates], dtype=float)
        out["G"] = np.array([constraint_vector(c) for c in candidates], dtype=float)
        cand = np.empty(len(candidates), dtype=object)
        for i, c in enumerate(candidates):
            cand[i] = c
        out["candidate"] = cand


class NSGA2Optimizer:
    """Configure and run NSGA-II from an optimisation YAML (same shape as the SQP one)."""

    def __init__(self, opt_config_path: Path, evaluator: Any = None) -> None:
        self.opt_config_path = Path(opt_config_path)
        self.problem: OptimizationProblem = load_problem(opt_config_path, evaluator, skip_after_discharge_failure=True)
        self.cfg = self.problem.cfg
        self.variables = self.problem.variables
        self.base_config = self.problem.base_config
        self.evaluator = self.problem.evaluator
        self.live_log = None  # optional RowLogger; every evaluation is appended as it completes

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

    def run(self) -> NSGA2Result | None:
        """Run NSGA-II. Under MPI, non-root ranks serve evaluations and return ``None``."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        history: list[tuple[tuple[float, ...], CandidateResult]] = []

        mpi = None
        try:
            from mpi4py import MPI
            mpi_size = MPI.COMM_WORLD.Get_size()
            if mpi_size > 1:
                mpi = MPIEvaluator(self.evaluator)
        except ImportError:
            mpi_size = None
        check_mpi_world(launcher_world_size(), mpi_size)
        if mpi is not None and mpi.rank != 0:
            mpi.serve()
            return None

        raw_batch = mpi.map if mpi is not None else (
            lambda X, on_result=None: _evaluate_batch_serial(self.evaluator, X, on_result))
        counter = [0]

        def batch(X):
            # Evaluation ids are issued here, on rank 0, in dispatch order; the history is
            # recorded here too, so one executed candidate is exactly one history entry.
            first = counter[0] + 1
            counter[0] += len(X)

            def log(i, candidate):
                candidate.eval_id = first + i
                if self.live_log is not None:
                    self.live_log.append({"evaluation": candidate.eval_id,
                                          **candidate_row(self.variables, X[i], candidate),
                                          "worker_rank": candidate.worker_rank, "pid": candidate.pid})

            results = raw_batch(X, log)
            for i, candidate in enumerate(results):
                candidate.eval_id = first + i
                history.append((tuple(float(v) for v in X[i]), candidate))
            return results

        try:
            problem = TankDesignProblem(batch, self.variables, self.settings.objectives)
            algorithm = NSGA2(pop_size=self.settings.population_size,
                              eliminate_duplicates=self.settings.eliminate_duplicates,
                              sampling=self.initial_population())
            result = minimize(problem, algorithm, termination=("n_gen", self.settings.generations),
                              seed=self.settings.seed, verbose=False)
        finally:
            if mpi is not None:
                mpi.shutdown()

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
    return [{"evaluation": c.eval_id, **candidate_row(result.variables, x, c),
             "worker_rank": c.worker_rank, "pid": c.pid}
            for x, c in result.history]


def write_history(path: Path, result: NSGA2Result) -> None:
    """Every evaluated candidate, in evaluation order."""
    write_rows(path, history_rows(result))


def write_pareto(path: Path, result: NSGA2Result) -> None:
    """The non-dominated set returned by the run (feasible whenever any feasible design exists)."""
    write_rows(path, pareto_rows(result))


__all__ = ["MPIEvaluator", "NSGA2Optimizer", "NSGA2Result", "NSGA2Settings", "OBJECTIVES", "constraint_vector",
           "history_rows", "objective_vector", "pareto_rows", "write_history", "write_pareto"]
