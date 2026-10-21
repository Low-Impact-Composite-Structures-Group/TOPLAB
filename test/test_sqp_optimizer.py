from pathlib import Path

import pytest

from toplab.optimization.evaluation import (
    CandidateResult, dormancy_limits, parse_design_variables, physical_residuals, venting_performance)
from toplab.optimization.sqp_optimizer import SQPOptimizer, epsilon_grid

BASE = Path(__file__).parent.parent / "examples/optimization_two_tank_cch2/two_tank_cch2_config.yaml"


class FakeEvaluator:
    """Analytic trade-off: eta_g rises with radius_1, eta_v falls with it; always feasible."""

    def evaluate(self, x):
        r = CandidateResult()
        r.eta_g = 0.2 + 0.5 * x[0] + (0.1 * x[1] if len(x) > 1 else 0.0)
        r.eta_v = 0.9 - 0.5 * x[0]
        r.eta_vent, r.zeta_p = 0.5, 0.2
        r.packaging_feasible = r.mission_completed = True
        return r


def make_config(tmp_path, variables, objective="eta_g", extra=""):
    path = tmp_path / "opt.yaml"
    path.write_text(f"""
optimization: {{base_config: {BASE}}}
objective: {objective}
design_variables:
{variables}
solver: {{max_iter: 30, tol: 1.0e-9, evaluation_progress: false{extra}}}
output: {{results_dir: out}}
""")
    return path


TWO = "  radius_1: {min: 0.3, max: 0.8, baseline: 0.5}\n  phi_2: {min: 1.0, max: 4.0, baseline: 2.0}"
ONE = "  radius_1: {min: 0.3, max: 0.8, baseline: 0.5}"


def test_signed_venting_performance_is_not_clipped():
    assert venting_performance(0.0, 0.1) == 1.0
    assert venting_performance(0.1, 0.1) == 0.0
    assert venting_performance(0.2, 0.1) == -1.0


def test_physical_residuals_use_canonical_g_le_zero_sign():
    result = CandidateResult(zeta_p=0.2, discharge_residual=-7200.0, dormancy_residual=0.1)
    assert physical_residuals(result) == pytest.approx({"g_dis": -2.0, "g_dorm": 0.1, "g_packaging": -0.2})
    assert all(v > 100 for v in physical_residuals(CandidateResult(error="boom")).values())


def test_epsilon_grid_is_cartesian_and_skips_the_objective():
    raw = {"eta_g": {"values": [0.1]}, "eta_v": {"values": [0.5, 0.6]}, "zeta_p": {"min": 0, "max": 1, "count": 3}}
    grid = epsilon_grid(raw, "eta_g")
    assert len(grid) == 6 and all(set(point) == {"eta_v", "zeta_p"} for point in grid)
    assert epsilon_grid(None, "eta_g") == [{}]
    with pytest.raises(ValueError, match="not one of"):
        epsilon_grid({"bogus": [1]}, "eta_g")


def test_dormancy_limits_accept_every_base_config_form():
    class Base:
        dormancy_check = {"duration_h": 2.0, "maximum_loss_percent": 0.1}
    assert dormancy_limits(Base, {}) == (7200.0, pytest.approx(0.001))
    Base.dormancy_check = {"duration_h": 2.0, "maximum_loss_percent_per_hour": 0.5}
    assert dormancy_limits(Base, {}) == (7200.0, pytest.approx(0.01))
    assert dormancy_limits(Base, {"dormancy": {"hold_period_h": 1, "maximum_vented_percentage": 3}}) == (
        3600.0, pytest.approx(0.03))


def test_bare_design_variable_names_target_tank_one():
    variables = parse_design_variables({"radius": {"min": 0.1, "max": 1, "baseline": 0.5},
                                        "t_ins": {"min": 0.01, "max": 0.1, "baseline": 0.05}})
    assert [v.name for v in variables] == ["radius_1", "insulation_thickness_1"]


def test_design_variable_validation():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_design_variables({"radius": {"min": 0, "max": 1, "baseline": 0.5},
                                "radius_1": {"min": 0, "max": 1, "baseline": 0.5}})
    with pytest.raises(ValueError, match="baseline"):
        parse_design_variables({"phi": {"min": 1, "max": 2, "baseline": 3}})


def test_single_objective_with_variable_subset(tmp_path):
    opt = SQPOptimizer(make_config(tmp_path, ONE), evaluator=FakeEvaluator())
    (result,) = opt.run()
    assert len(result.runs) == 1 and result.runs[0].feasible
    assert result.best.x[0] == pytest.approx(0.8, abs=1e-6)  # eta_g increases with radius_1


def test_any_objective_can_be_maximised(tmp_path):
    opt = SQPOptimizer(make_config(tmp_path, ONE, objective="eta_v"), evaluator=FakeEvaluator())
    (result,) = opt.run()
    assert result.best.x[0] == pytest.approx(0.3, abs=1e-6)


def test_objective_list_runs_each_objective(tmp_path):
    opt = SQPOptimizer(make_config(tmp_path, TWO, objective="[eta_g, eta_v]"), evaluator=FakeEvaluator())
    assert [r.objective for r in opt.run()] == ["eta_g", "eta_v"]


def test_epsilon_sweep_on_chosen_metrics(tmp_path):
    path = make_config(tmp_path, ONE, objective="eta_g")
    path.write_text(path.read_text() + "epsilon_constraints:\n  eta_v: {values: [0.6, 0.7]}\n")
    opt = SQPOptimizer(path, evaluator=FakeEvaluator())
    (result,) = opt.run()
    assert [run.epsilon for run in result.runs] == [{"eta_v": 0.6}, {"eta_v": 0.7}]
    # eta_v = 0.9 - 0.5 x >= eps  ->  x <= 2 (0.9 - eps)
    assert [run.x[0] for run in result.runs] == pytest.approx([0.6, 0.4], abs=1e-5)
    assert len(result.pareto) == 2


def test_infeasible_epsilon_is_reported(tmp_path):
    path = make_config(tmp_path, ONE)
    path.write_text(path.read_text() + "epsilon_constraints:\n  eta_v: {values: [0.95]}\n")
    (result,) = SQPOptimizer(path, evaluator=FakeEvaluator()).run()
    assert not result.runs[0].feasible and result.pareto == []


def test_finite_diff_step_validation(tmp_path):
    with pytest.raises(ValueError, match="finite_diff_step"):
        SQPOptimizer(make_config(tmp_path, ONE, extra=", finite_diff_step: [0.1, 0.1]"), evaluator=FakeEvaluator())
    opt = SQPOptimizer(make_config(tmp_path, TWO, extra=", finite_diff_step: {radius: 0.01, phi: 0.1}"),
                       evaluator=FakeEvaluator())
    assert opt._step == [0.01, 0.1]


def test_failing_start_design_raises_instead_of_reporting_success(tmp_path):
    class Broken:
        def evaluate(self, x):
            return CandidateResult(error="boom")
    with pytest.raises(RuntimeError, match="boom"):
        SQPOptimizer(make_config(tmp_path, ONE), evaluator=Broken()).run()


def test_runner_writes_unified_csvs(tmp_path, monkeypatch):
    from toplab.optimization.runners import run_sqp
    path = make_config(tmp_path, ONE)
    path.write_text(path.read_text() + "epsilon_constraints:\n  eta_v: {values: [0.6]}\n")
    monkeypatch.setattr("toplab.optimization.evaluation.CandidateEvaluator", lambda *a, **k: FakeEvaluator())
    run_sqp(path, plot=False)
    header = (tmp_path / "out/sqp_runs.csv").read_text().splitlines()[0].split(",")
    assert {"objective", "eps_eta_v", "radius_1", "eta_g", "g_epsilon_eta_v", "feasible"} <= set(header)
    assert (tmp_path / "out/sqp_pareto.csv").exists() and (tmp_path / "out/sqp_evaluations.csv").exists()


def test_parse_and_apply_design_variables_do_not_mutate_base():
    from pathlib import Path
    from toplab.config import load_config
    from toplab.optimization.evaluation import apply_design, parse_design_variables

    base = load_config(Path(__file__).parent.parent / "examples/optimization_two_tank_cch2/two_tank_cch2_config.yaml")
    variables = parse_design_variables({
        "radius_1": {"min": 0.3, "max": 0.8, "baseline": 0.5},
        "insulation_thickness_2": {"min": 0.01, "max": 0.1, "baseline": 0.05},
    })
    assert [v.name for v in variables] == ["radius_1", "insulation_thickness_2"]
    candidate = apply_design(base, variables, [0.42, 0.07])
    assert candidate.tanks[0].geometry["radius"] == 0.42
    assert candidate.tanks[1].geometry["radius"] == 0.6
    assert candidate.tanks[1].materials["insulation"]["thickness"] == 0.07
    assert base.tanks[0].geometry["radius"] == 0.6
    assert base.tanks[1].materials["insulation"]["thickness"] == 0.05
    assert candidate.dormancy_check is None and base.dormancy_check["enabled"]


def test_bad_design_variable_name_is_rejected():
    from toplab.optimization.evaluation import parse_design_variables
    with pytest.raises(ValueError, match="<parameter>_<tank>"):
        parse_design_variables({"bogus": {"min": 0, "max": 1, "baseline": 0.5}})
