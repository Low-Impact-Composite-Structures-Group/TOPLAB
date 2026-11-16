import pytest
import math

from toplab.packaging.aft_placement import (
    AftFuselageDimensions,
    TankGeometry,
    _capsule_surface_gap,
    _capsule_radius_at_offset,
    _containment_violation,
    _evaluate_placement,
    _segment_distance,
    _transverse_basis,
    max_required_aft_shift,
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


@pytest.mark.parametrize("point,expected", [((0, 0, 1), 0.0), ((0, 0, 3), 1.0), ((1, 0, 1), 1.0)])
def test_segment_distance_is_symmetric_for_spheres(point, expected):
    a, b = (0, 0, -2), (0, 0, 2)
    assert _segment_distance(a, b, point, point) == pytest.approx(expected)
    assert _segment_distance(point, point, a, b) == pytest.approx(expected)
    assert _segment_distance(point, point, point, point) == pytest.approx(0.0)


def test_sphere_inside_capsule_is_not_accepted():
    dims = AftFuselageDimensions(4, 4, 4, 8, 1, 1, 0.05)
    centers = [(0, 0, 3), (0, 0, 4)]
    geometries = [TankGeometry(0.5, 2), TankGeometry(0.4, 0)]
    axis = (0, 0, 1)
    assert _capsule_surface_gap(centers[0], axis, geometries[0], centers[1], axis, geometries[1]) == pytest.approx(-0.9)
    assert not _evaluate_placement(centers, geometries, dims, 0, 80, 24)[0]


@pytest.mark.parametrize("z", [0.51, 9.49])
def test_end_plane_clearance_is_enforced(z):
    dims = AftFuselageDimensions(4, 4, 4, 8, 1, 1, 0.05)
    feasible, violations = _evaluate_placement([(0, 0, z)], [TankGeometry(0.5, 0)], dims, 0, 80, 24)
    assert not feasible
    assert violations[0] == pytest.approx(0.04)


@pytest.mark.parametrize("center,axis", [
    ((0.1, 0.0, -0.5), (0.0, 0.0, 1.0)),
    ((0.2, 0.3, 2.8), (0.0, math.sin(math.radians(12)), math.cos(math.radians(12)))),
    ((-0.1, 0.5, 6.8), (0.0, math.sin(math.radians(9)), math.cos(math.radians(9)))),
])
def test_vectorized_containment_matches_scalar_sampling(center, axis):
    dims = AftFuselageDimensions(2.16, 1.17, 0.328, 2.067, 1.8, 3.407, 0.05, 12, 9)
    geometry = TankGeometry(0.4, 0.7)
    z_min = -2.0
    extent = abs(axis[2]) * geometry.half_cyl_length + geometry.outer_radius
    expected = max(z_min + dims.epsilon - (center[2] - extent),
                   center[2] + extent + dims.epsilon - dims.total_length)
    e1, e2 = _transverse_basis(axis)
    for i in range(41):
        delta = -geometry.half_total_length + 2 * geometry.half_total_length * i / 40
        radius = _capsule_radius_at_offset(delta, geometry.outer_radius, geometry.half_cyl_length)
        for j in range(24):
            phi = 2 * math.pi * j / 24
            point = tuple(center[k] + axis[k] * delta + radius * (e1[k] * math.cos(phi) +
                                                                    e2[k] * math.sin(phi))
                          for k in range(3))
            z = point[2]
            if z_min <= z <= dims.total_length:
                expected = max(expected, math.hypot(point[0], point[1] - dims.center_y_at(z)) -
                               dims.radius_at(z) + dims.epsilon)
    assert _containment_violation(center, axis, geometry, dims, z_min, 40, 24) == pytest.approx(expected, abs=1e-12)


def test_four_tank_extended_cylinder_has_constructive_placement():
    dims = AftFuselageDimensions(2.16, 1.17, 0.328, 2.067, 1.8, 3.407, 0.05, 12, 9)
    radii, half_lengths = [0.7272] * 4, [0.0] * 4
    shift = max_required_aft_shift(radii, half_lengths, dims)
    result = place_tanks_in_aft(radii, half_lengths, dims, z_min=-shift)
    assert result.feasible
    assert "cylinder_stack" in result.message
    assert all(p.s_leftmost_pole >= -shift + dims.epsilon - 1e-10 for p in result.placements)


def test_two_small_tanks_can_be_placed_in_unshifted_fuselage():
    result = place_tanks_in_aft([0.45, 0.45], [0.2, 0.2], _dims())
    assert result.feasible
    assert all(p.max_violation <= 1e-10 for p in result.placements)


def test_compaction_fallback_returns_infeasible_margin_for_four_spheres():
    dims = AftFuselageDimensions(2.16, 1.17, 0.328, 2.067, 1.8, 3.407, 0.05, 12, 9)
    result = maximize_packaging_compaction([0.7272] * 4, [0.0] * 4, z_tolerance=0.1,
                                          dims=dims, n_axial_samples=20, n_circumferential_samples=12)
    assert not result.feasible
    assert result.zeta_p < 0
    assert all(p.feasible for p in result.placements)
