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
    assert constraint_vector(no_fit)[2] > 0.0
    shifted = CandidateResult(zeta_p=-0.1, packaging_feasible=False)
    assert constraint_vector(shifted)[2] == pytest.approx(0.1)


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


def test_front_movement_uses_previous_generation_independent_of_tolerance():
    from toplab.optimization.convergence import _FrontMovement

    class Front:
        def __init__(self, value):
            self.value = value

        def get(self, *keys):
            return np.array([True]), np.array([[self.value]])

    class Algorithm:
        def __init__(self, value):
            self.opt = Front(value)

    traces = []
    for tolerance in (0.001, 0.1):
        movement = _FrontMovement(tol=tolerance, only_feas=True)
        values = []
        for value in (0.0, 0.05, 0.09, 0.14):
            movement.update(Algorithm(value))
            values.append(movement.last_movement)
        traces.append(values)
    assert traces[0] == pytest.approx(traces[1])
    assert traces[0][1:] == pytest.approx([0.05, 0.04, 0.05])


def test_front_history_logs_feasible_front_for_each_generation(tmp_path):
    import csv
    from toplab.optimization.convergence import ConvergenceSettings
    from toplab.optimization.evaluation import RowLogger

    path = tmp_path / "fronts.csv"
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
    opt.settings = dataclasses.replace(
        opt.settings, population_size=10, generations=4, seed=3,
        convergence=ConvergenceSettings(enabled=True, period=10),
        save_front_history=True)
    opt.front_log = RowLogger(path)
    result = opt.run()
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["generation"]) for row in rows} == {1, 2, 3, 4}
    assert all(row["feasible"] == "True" for row in rows)
    history = {c.eval_id: (x, c) for x, c in result.history}
    for row in rows:
        x, candidate = history[int(row["evaluation"])]
        assert float(row["radius_1"]) == pytest.approx(x[0])
        assert float(row["eta_g"]) == pytest.approx(candidate.eta_g)
        assert float(row["eta_v"]) == pytest.approx(candidate.eta_v)
    last = [row for row in rows if int(row["generation"]) == 4]
    assert {int(row["evaluation"]) for row in last} == {c.eval_id for c in result.pareto}
    assert result.front_history == opt.front_log._rows


def test_front_history_marks_empty_generations_and_works_without_convergence(tmp_path):
    import csv
    from toplab.optimization.evaluation import RowLogger

    class InfeasibleEvaluator(FakeEvaluator):
        def evaluate(self, x):
            candidate = super().evaluate(x)
            candidate.packaging_feasible = False
            return candidate

    path = tmp_path / "fronts.csv"
    opt = NSGA2Optimizer(NSGA_YAML, evaluator=InfeasibleEvaluator())
    opt.settings = dataclasses.replace(opt.settings, population_size=6, generations=3, seed=3,
                                       save_front_history=True)
    opt.front_log = RowLogger(path)
    result = opt.run()
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert result.convergence_trace == []
    assert [int(row["generation"]) for row in rows] == [1, 2, 3]
    assert all(row["feasible"] == "False" and not row["eta_g"] for row in rows)


def test_movement_trace_replays_actual_stopping_for_multiple_tolerances():
    from toplab.optimization.convergence import ConvergenceSettings

    def run(tolerance, period):
        opt = NSGA2Optimizer(NSGA_YAML, evaluator=FakeEvaluator())
        opt.settings = dataclasses.replace(
            opt.settings, population_size=10, generations=40, seed=3,
            convergence=ConvergenceSettings(enabled=True, tolerance=tolerance, period=period,
                                            min_generations=4))
        return opt.run()

    reference = run(1e-12, 41)
    for tolerance, period in ((0.01, 2), (0.05, 3), (0.5, 2)):
        streak = 0
        expected = 40
        for row in reference.convergence_trace:
            streak = streak + 1 if row["front_movement"] <= tolerance else 0
            if row["generation"] >= 4 and streak >= period:
                expected = row["generation"]
                break
        actual = run(tolerance, period)
        assert len(actual.convergence_trace) == expected
        assert [row["front_movement"] for row in actual.convergence_trace] == pytest.approx(
            [row["front_movement"] for row in reference.convergence_trace[:expected]])


def test_runner_writes_front_history_when_enabled(tmp_path, monkeypatch):
    import csv
    import toplab.optimization.nsga2_optimizer as nsga2
    from toplab.optimization.convergence import ConvergenceSettings
    from toplab.optimization.runners import run_nsga2

    original = nsga2.NSGA2Optimizer

    def make_optimizer(path):
        opt = original(path, evaluator=FakeEvaluator())
        opt.settings = dataclasses.replace(
            opt.settings, population_size=6, generations=3, seed=3, save_front_history=True,
            convergence=ConvergenceSettings(enabled=True, period=10))
        return opt

    monkeypatch.setattr(nsga2, "NSGA2Optimizer", make_optimizer)
    monkeypatch.setenv("TOPLAB_RESULTS_DIR", str(tmp_path))
    result = run_nsga2(NSGA_YAML, plot=False)
    with (tmp_path / "nsga2_front_history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(result.front_history) > 0
    assert {int(row["generation"]) for row in rows} == {1, 2, 3}
    assert (tmp_path / "nsga2_convergence.csv").exists()


def test_hung_evaluation_is_killed_and_reported_as_error():
    import time
    from toplab.optimization.evaluation import CandidateEvaluator

    ev = CandidateEvaluator.__new__(CandidateEvaluator)
    ev.progress, ev.evaluation_count, ev._stdout, ev.evaluation_timeout_s = False, 0, None, 0.5
    ev._evaluate = lambda values: time.sleep(60)
    started = time.perf_counter()
    result = ev.evaluate([1.0])
    assert time.perf_counter() - started < 10
    assert "time limit" in result.error and not result.feasible
    assert ev.evaluation_count == 1


def test_finished_evaluation_returns_through_timeout_process():
    from toplab.optimization.evaluation import CandidateEvaluator

    ev = CandidateEvaluator.__new__(CandidateEvaluator)
    ev.progress, ev.evaluation_count, ev._stdout, ev.evaluation_timeout_s = False, 0, None, 30.0
    ev._evaluate = lambda values: CandidateResult(eta_g=0.25)
    result = ev.evaluate([1.0])
    assert result.eta_g == 0.25 and result.error is None
