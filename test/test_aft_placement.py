import pytest
import math

from toplab.packaging.aft_placement import (
    AftFuselageDimensions,
    maximize_packaging_compaction,
    place_tanks_in_aft,
)


def test_centreline_angles_are_configured_in_degrees():
    dims = AftFuselageDimensions(
        d1=2.16,
        d2=1.17,
        d3=0.328,
        l1=2.067,
        l2=1.8,
        l3=3.407,
        epsilon=0.05,
        psi_1=12.0,
        psi_2=8.0,
    )

    expected_interface_y = dims.l2 * math.tan(math.radians(12.0))
    assert math.isclose(
        dims.center_y_at(dims.l1 + dims.l2),
        expected_interface_y,
    )
    assert dims.tangent_at(dims.l1 + 0.5)[1] > 0.0


def test_placement_reports_aft_pole_and_lateral_offset():
    dims = AftFuselageDimensions(
        d1=4.0,
        d2=4.0,
        d3=4.0,
        l1=8.0,
        l2=1.0,
        l3=1.0,
        epsilon=0.05,
        psi_1=10.0,
        psi_2=10.0,
    )

    result = place_tanks_in_aft([0.5], [1.0], dims)
    placement = result.placements[0]

    assert placement.s_leftmost_pole > 0.0
    assert placement.lateral_offset == 0.0
    assert placement.center[0] == placement.lateral_offset


def test_nudger_moves_tank_towards_bulkhead_when_aft_is_wider():
    dims = AftFuselageDimensions(
        d1=4.0,
        d2=1.0,
        d3=1.0,
        l1=1.0,
        l2=5.0,
        l3=1.0,
        epsilon=0.05,
    )

    result = place_tanks_in_aft([1.5], [0.1], dims, initial_step=0.01)

    assert result.placements[0].s_leftmost_pole < 1.9


def test_baseline_tank_can_follow_fuselage_tangent_to_feasibility():
    dims = AftFuselageDimensions(
        d1=2.16,
        d2=1.17,
        d3=0.328,
        l1=2.067,
        l2=1.8,
        l3=3.407,
        epsilon=0.05,
        psi_1=12.0,
        psi_2=9.0,
    )

    result = place_tanks_in_aft([0.585], [0.8775], dims)

    assert result.feasible
    assert result.placements[0].max_violation <= 1e-10


def test_single_large_tank_starts_in_cylindrical_section_and_fits():
    dims = AftFuselageDimensions(
        d1=2.16,
        d2=1.17,
        d3=0.328,
        l1=2.067,
        l2=1.8,
        l3=3.407,
        epsilon=0.05,
        psi_1=12.0,
        psi_2=9.0,
    )

    result = place_tanks_in_aft([0.8443], [0.72], dims)

    assert result.feasible
    assert result.placements[0].center[2] < dims.l1
    assert result.placements[0].max_violation <= 1e-10


def _dims():
    return AftFuselageDimensions(
        d1=2.16, d2=1.17, d3=0.328, l1=2.067, l2=1.8, l3=3.407, epsilon=0.05,
    )


def test_infeasible_compaction_shifts_datum_aft():
    dims = _dims()
    # Fits the cylinder radially, but is too long for the unshifted domain.
    result = maximize_packaging_compaction([0.9], [4.0], dims)

    assert not result.feasible
    assert result.z_shift < 0.0
    assert result.zeta_p == result.z_shift / dims.l1
    assert min(p.center[2] - p.half_total_length for p in result.placements) >= result.z_shift - 1e-6
    assert all(p.max_violation <= 1e-10 for p in result.placements)


def test_tank_wider_than_cylinder_raises():
    with pytest.raises(ValueError, match="no datum shift"):
        maximize_packaging_compaction([2.0], [0.1], _dims())
