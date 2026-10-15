import pytest

from toplab.demand import PiecewiseDemand, ScaledDemand, SumDemand


def test_piecewise_interpolation_boundaries_and_end():
    d = PiecewiseDemand([10.0, 10.0], [[1.0, 3.0], [-5.0]])
    assert d.mass_flow(0.0) == pytest.approx(1.0)
    assert d.mass_flow(5.0) == pytest.approx(2.0)
    assert d.mass_flow(10.0) == pytest.approx(3.0)   # earlier segment wins at a boundary
    assert d.mass_flow(10.5) == pytest.approx(5.0)   # flows are magnitudes
    assert d.mass_flow(20.5) == 0.0
    assert d.duration == 20.0
    assert d.total_mass() == pytest.approx(10 * 2.0 + 10 * 5.0)


def test_multipoint_segment():
    d = PiecewiseDemand([4.0], [[0.0, 2.0, 0.0]])
    assert d.mass_flow(2.0) == pytest.approx(2.0)
    assert d.mass_flow(1.0) == pytest.approx(1.0)
    assert d.total_mass() == pytest.approx(4.0)


def test_scaled_and_sum():
    base = PiecewiseDemand([10.0], [[2.0]])
    mix = SumDemand([base.scaled(0.25), ScaledDemand(base, 0.75)])
    assert mix.mass_flow(3.0) == pytest.approx(2.0)
    assert mix.total_mass() == pytest.approx(20.0)
    with pytest.raises(ValueError):
        base.scaled(-0.1)
