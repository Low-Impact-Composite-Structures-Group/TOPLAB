import pytest

from toplab.optimization.sqp_optimizer import (
    CandidateResult,
    constraint_residuals,
    EpsilonTriplet,
    epsilon_residuals,
    venting_performance,
)


def test_signed_venting_performance_is_not_clipped():
    assert venting_performance(0.0, 0.1) == 1.0
    assert venting_performance(0.1, 0.1) == 0.0
    assert venting_performance(0.2, 0.1) == -1.0


def test_epsilon_residuals_use_canonical_g_le_zero_sign():
    result = CandidateResult(eta_v=0.4, eta_vent=-0.2, zeta_p=0.6)
    epsilon = EpsilonTriplet(eta_v=0.5, eta_vent=-0.1, zeta_p=0.7)
    assert epsilon_residuals(epsilon, result) == pytest.approx((0.1, 0.1, 0.1))


def test_named_constraint_residuals_describe_feasibility():
    result = CandidateResult(
        eta_v=0.8,
        eta_vent=0.25,
        zeta_p=0.2,
        discharge_residual=-2.0,
        dormancy_residual=0.1,
        packaging_feasible=False,
    )
    residuals = constraint_residuals(result, EpsilonTriplet(0.75, 0.0, 0.1))
    assert residuals == pytest.approx({
        "g_dis": -2.0,
        "g_dorm": 0.1,
        "g_packaging": -0.2,
        "g_epsilon_v": -0.05,
        "g_epsilon_vent": -0.25,
        "g_epsilon_p": -0.1,
    })


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
