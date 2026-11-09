"""MLI capacitance, Modified Lockheed segmentation and dormancy checks.

Validation-only: no production code is modified. Investigation B (steady
segmentation), A (constituent capacitance) and D (dormancy mission).
"""

import numpy as np
import pytest
from scipy.optimize import root

from test.rc_circuit_validation.mli_capacitance import (
    HALF_MIL,
    QUARTER_MIL,
    TabulatedSpecificHeat,
)
from toplab.materials.insulation_models import make_modified_lockheed_model


def _planar_serial_flux(heat_flux, t_hot, t_cold, thickness, segments):
    if segments == 1:
        return heat_flux(t_hot, t_cold, thickness)

    def residual(interior):
        temperatures = np.concatenate(([t_hot], interior, [t_cold]))
        flux = np.array(
            [
                heat_flux(temperatures[i], temperatures[i + 1], thickness / segments)
                for i in range(segments)
            ]
        )
        return flux[:-1] - flux[1:]

    solution = root(residual, np.linspace(t_hot, t_cold, segments + 1)[1:-1], tol=1e-13)
    assert solution.success
    temperatures = np.concatenate(([t_hot], solution.x, [t_cold]))
    return heat_flux(temperatures[0], temperatures[1], thickness / segments)


def test_modified_lockheed_segmentation_is_bounded_and_converges():
    """Series segments change only the weak f(T_mean) solid term (< 1%).

    Radiation and gas terms are (potential difference)/N and telescope exactly,
    so the mesh dependence stays bounded even for segments thinner than one
    shield spacing (N > layers).
    """
    heat_flux = make_modified_lockheed_model().heat_flux
    full = heat_flux(288.15, 60.0, 0.02)
    deviations = {
        n: _planar_serial_flux(heat_flux, 288.15, 60.0, 0.02, n) / full - 1.0
        for n in (1, 2, 8, 32, 64)
    }
    assert deviations[1] == 0.0
    assert all(abs(value) < 0.01 for value in deviations.values())
    # Monotone and converging: successive refinements change the flux less.
    assert abs(deviations[64] - deviations[32]) < 1e-4
    assert abs(deviations[32] - deviations[8]) < abs(deviations[8] - deviations[2])


@pytest.mark.parametrize("composition", (QUARTER_MIL, HALF_MIL))
def test_constituent_bulk_density_is_far_below_fitted_default(composition):
    assert 20.0 < composition.bulk_density < 60.0
    default = make_modified_lockheed_model().density
    assert default / composition.bulk_density > 5.0


def test_mass_weighted_specific_heat_is_bounded_by_constituents():
    cp = TabulatedSpecificHeat(QUARTER_MIL)
    assert cp(20.0) < cp(100.0) < cp(288.0) < 1300.0
    assert abs(cp.enthalpy(300.0) - cp.enthalpy(200.0) - np.trapezoid(
        [cp(t) for t in np.linspace(200, 300, 2001)], np.linspace(200, 300, 2001)
    )) < 50.0


@pytest.fixture(scope="module")
def harness():
    from test.rc_circuit_validation.mli_dormancy_harness import DormancyHarness

    return DormancyHarness()


def test_dormancy_harness_one_node_reproduces_production_rhs(harness):
    rc_model = harness.rc_model(1)
    y = harness.starting_vector(rc_model)
    for time, perturbed in ((0.0, y), (5.0, y * np.array([1, 1.04, 1, 1.1, 1]))):
        reference = harness.system._rhs(time, perturbed)[0]
        np.testing.assert_allclose(
            harness.derivatives(rc_model, time, perturbed), reference, rtol=1e-12, atol=1e-15
        )


def test_dormancy_one_node_reproduces_production_result(harness):
    result = harness.run(1)
    assert result.vented_mass == pytest.approx(0.3202, abs=5e-4)
    assert result.pressure[-1] == pytest.approx(2.00487e6, rel=1e-5)


def test_dormancy_equilibrated_start_is_nearly_node_independent(harness):
    one, ten = harness.run(1), harness.run(10)
    assert abs(one.vented_mass / ten.vented_mass - 1.0) < 0.10
    assert abs(one.cumulative_h2_heat_kj[-1] / ten.cumulative_h2_heat_kj[-1] - 1.0) < 0.02


def test_dormancy_post_refuel_start_needs_more_than_one_node():
    from test.rc_circuit_validation.mli_dormancy_harness import DormancyHarness

    realistic = DormancyHarness(density=QUARTER_MIL.bulk_density, cp=TabulatedSpecificHeat(QUARTER_MIL))
    one, five, twenty = (realistic.run(n, "post_refuel") for n in (1, 5, 20))
    assert one.vented_mass / twenty.vented_mass < 0.95
    assert abs(five.vented_mass / twenty.vented_mass - 1.0) < 0.03
