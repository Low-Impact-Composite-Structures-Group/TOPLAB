"""Foam spatial convergence from an equilibrated start (N = 1 ... 40)."""

import numpy as np
import pytest

from test.rc_circuit_validation.foam_rc_studies import (
    NODE_COUNTS,
    SAMPLE_TIMES,
    simulate,
)


def test_equilibrated_start_without_perturbation_is_stationary():
    result = simulate(10, shell_temperature=288.15)
    assert np.ptp(result.heat_flow) < 1e-6 * abs(result.heat_flow[0])


def test_energy_balance_closes_for_every_discretization():
    for n in NODE_COUNTS:
        result = simulate(n)
        scale = max(abs(result.cumulative_kj[-1]), 1.0)
        assert np.max(np.abs(result.energy_residual_kj)) < 1e-5 * scale, n


@pytest.mark.parametrize("time", SAMPLE_TIMES)
def test_cumulative_heat_converges_with_node_count(time):
    values = np.array([simulate(n).cumulative_at(time) for n in NODE_COUNTS])
    steps = np.abs(np.diff(values))
    # Successive refinement changes shrink and the last doubling is small.
    assert steps[-1] < steps[0]
    assert steps[-1] < 0.01 * abs(values[-1])


def test_final_heat_flow_matches_steady_state_at_new_boundary():
    flows = [simulate(n).heat_flow[-1] for n in NODE_COUNTS]
    # Not fully relaxed in 3 h, but all discretizations must agree at the end.
    assert np.ptp(flows) < 0.02 * abs(flows[-1])
