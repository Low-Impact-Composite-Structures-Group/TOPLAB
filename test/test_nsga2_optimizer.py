"""NSGA-II plumbing: objective/constraint mapping, settings validation, and a fake-evaluator run."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from toplab.optimization.evaluation import CandidateResult
from toplab.optimization.nsga2_optimizer import (
    NSGA2Optimizer, constraint_vector, objective_vector, write_pareto,
)

NSGA_YAML = Path(__file__).parent.parent / "examples/optimization_two_tank_cch2/two_tank_cch2_nsga.yaml"


class FakeEvaluator:
    """Analytic trade-off between eta_g and eta_v; always feasible."""

    def evaluate(self, x):
        r = CandidateResult()
        r.eta_g = 0.5 + 0.2 * x[0] - 0.1 * x[1]
        r.eta_v = 0.4 + 0.3 * x[1]
        r.eta_vent = 1.0 - abs(x[0] - x[1])
        r.zeta_p, r.packaging_feasible, r.mission_completed = 0.1, True, True
        return r


def test_objective_vector_negates_maximised_metrics():
    r = CandidateResult(eta_g=0.5, eta_v=0.3, zeta_p=0.9)
    assert objective_vector(r, ("eta_g", "zeta_p")) == [-0.5, -0.9]


def test_failed_candidate_is_heavily_penalised():
    r = CandidateResult(error="boom")
    assert all(g > 100 for g in constraint_vector(r))


def test_constraint_vector_signs():
    ok = CandidateResult(discharge_residual=-3600.0, dormancy_residual=-0.2, zeta_p=0.3, packaging_feasible=True)
    assert constraint_vector(ok) == pytest.approx([-1.0, -0.2, -0.3])
    no_fit = CandidateResult(zeta_p=0.1, packaging_feasible=False)
    assert constraint_vector(no_fit)[2] >= 1.0


def test_initial_population_contains_baseline_and_respects_bounds():
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
    pop = opt.initial_population()
    assert pop.shape == (opt.settings.population_size, len(opt.variables))
    assert np.allclose(pop[0], [v.baseline for v in opt.variables])
    assert np.all(pop >= [v.lower for v in opt.variables]) and np.all(pop <= [v.upper for v in opt.variables])


def test_unknown_objective_rejected(tmp_path):
    text = NSGA_YAML.read_text().replace("zeta_p]", "bogus]", 1)
    bad = NSGA_YAML.parent / "_bad_nsga.yaml"
    bad.write_text(text)
    try:
        with pytest.raises(ValueError):
            NSGA2Optimizer(bad, evaluator=FakeEvaluator())
    finally:
        bad.unlink()


def test_run_with_fake_evaluator_returns_nondominated_set(tmp_path):
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
    opt.settings = dataclasses.replace(opt.settings, population_size=10, generations=3, n_workers=1, seed=3)
    result = opt.run()
    assert len(result.history) >= 10 * 3
    assert len(result.pareto) == len(result.pareto_x) >= 1
    assert all(c.eta_g > 0 for c in result.pareto)

    path = tmp_path / "pareto.csv"
    write_pareto(path, result)
    assert len(path.read_text().strip().splitlines()) == len(result.pareto) + 1


class CountingEvaluator(FakeEvaluator):
    def __init__(self):
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        return super().evaluate(x)


def test_one_history_row_and_unique_id_per_executed_candidate(tmp_path):
    from toplab.optimization.evaluation import RowLogger
    from toplab.optimization.nsga2_optimizer import history_rows

    evaluator = CountingEvaluator()
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=evaluator)
    opt.settings = dataclasses.replace(opt.settings, population_size=10, generations=4, seed=3)
    opt.live_log = RowLogger(tmp_path / "live.csv")
    result = opt.run()
    rows = history_rows(result)
    ids = [r["evaluation"] for r in rows]
    assert evaluator.calls == len(result.history) == len(opt.live_log._rows) == len(rows)
    assert ids == list(range(1, evaluator.calls + 1))
    assert [r["evaluation"] for r in opt.live_log._rows] == ids


def test_mpi_world_mismatch_is_rejected():
    from toplab.optimization.nsga2_optimizer import check_mpi_world

    check_mpi_world(1, 1)
    check_mpi_world(8, 8)
    for mpi_size in (1, None):
        with pytest.raises(RuntimeError, match="not communicating"):
            check_mpi_world(8, mpi_size)


def test_eta_vent_is_not_an_nsga2_objective(tmp_path):
    from toplab.optimization.evaluation import NSGA2_OBJECTIVES
    from toplab.optimization.nsga2_optimizer import NSGA2Settings

    assert NSGA2_OBJECTIVES == ("eta_g", "eta_v", "zeta_p")
    assert NSGA2Settings.objectives == NSGA2_OBJECTIVES
    assert NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator()).settings.objectives == NSGA2_OBJECTIVES

    bad = NSGA_YAML.parent / "_vent_nsga.yaml"
    bad.write_text(NSGA_YAML.read_text().replace("zeta_p]", "zeta_p, eta_vent]", 1))
    try:
        with pytest.raises(ValueError, match="venting"):
            NSGA2Optimizer(bad, evaluator=FakeEvaluator())
    finally:
        bad.unlink()


def test_pymoo_problem_has_three_objectives_and_vent_remains_a_diagnostic(tmp_path):
    from toplab.optimization.evaluation import candidate_row
    from toplab.optimization.nsga2_optimizer import TankDesignProblem

    opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
    problem = TankDesignProblem(lambda X: [opt.evaluator.evaluate(x) for x in X], opt.variables,
                                opt.settings.objectives)
    assert problem.n_obj == 3 and problem.n_ieq_constr == 3
    out = {}
    problem._evaluate(opt.initial_population()[:2], out)
    assert out["F"].shape == (2, 3)

    c = CandidateResult(eta_g=0.5, eta_v=0.4, zeta_p=0.2, f_vent=0.01, vented_mass_kg=0.3,
                        packaging_feasible=True, mission_completed=True)
    row = candidate_row(opt.variables, opt.initial_population()[0], c)
    assert row["f_vent"] == 0.01


def _run(generations, **conv):
    from toplab.optimization.convergence import ConvergenceSettings
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
    opt.settings = dataclasses.replace(opt.settings, population_size=10, generations=generations, seed=3,
                                       convergence=ConvergenceSettings(**conv))
    return opt.run()


def test_fixed_generations_when_convergence_disabled():
    result = _run(4)
    assert result.termination_reason == "max_generations"
    assert result.convergence_trace == []
    assert len(result.history) <= 10 * 4


def test_converges_before_the_cap():
    result = _run(200, enabled=True, tolerance=0.5, period=2)
    assert result.termination_reason == "converged"
    assert len(result.convergence_trace) < 200
    assert result.convergence_trace[-1]["streak"] >= 2


def test_cap_applies_when_not_converged():
    result = _run(3, enabled=True, tolerance=1e-12, period=5)
    assert result.termination_reason == "max_generations"


def test_min_generations_delays_stopping():
    result = _run(200, enabled=True, tolerance=0.5, period=1, min_generations=8)
    assert len(result.convergence_trace) >= 8


def test_convergence_settings_validated():
    from toplab.optimization.convergence import ConvergenceSettings
    for bad in ({"tolerance": 0}, {"period": 0}, {"min_generations": -1}):
        with pytest.raises(ValueError):
            ConvergenceSettings.from_config(bad)


def test_convergence_trace_is_logged_live_to_the_csv(tmp_path):
    from toplab.optimization.convergence import ConvergenceSettings
    from toplab.optimization.evaluation import RowLogger
    path = tmp_path / "nsga2_convergence.csv"
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
    opt.settings = dataclasses.replace(opt.settings, population_size=10, generations=6, seed=3,
                                       convergence=ConvergenceSettings(enabled=True, tolerance=1e-12, period=50))
    opt.convergence_log = RowLogger(path)
    result = opt.run()
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "generation,front_movement,streak"
    assert len(lines) - 1 == len(result.convergence_trace) == 6
