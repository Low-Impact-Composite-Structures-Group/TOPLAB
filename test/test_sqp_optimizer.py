import pytest

from toplab.optimization.sqp_optimizer import (
    CandidateResult,
    constraint_residuals,
    EpsilonTriplet,
    decode_design_vector,
    epsilon_residuals,
    venting_performance,
)


def test_decode_design_vector_is_generic_per_tank():
    designs = decode_design_vector([1.0, 2.0, 0.01, 3.0, 4.0, 0.02])
    assert [(design.radius, design.phi, design.insulation_thickness) for design in designs] == [
        (1.0, 2.0, 0.01),
        (3.0, 4.0, 0.02),
    ]


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
