"""Rigid capsule placement inside a piecewise axisymmetric aft-fuselage volume.

The placement routine is intentionally decoupled from any outer design
optimization. Given a fuselage geometry and a set of tank dimensions, it
searches for feasible rigid-body poses for all tanks.

No generalized coordinates are used. Tank positions are internal Cartesian
coordinates, while tank orientation follows the local fuselage axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Vector3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Basic vector utilities
# ---------------------------------------------------------------------------

def _add(a: Vector3, b: Vector3) -> Vector3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _scale(a: Vector3, s: float) -> Vector3:
    return a[0] * s, a[1] * s, a[2] * s


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vector3) -> Vector3:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return a[0] / n, a[1] / n, a[2] / n


# ---------------------------------------------------------------------------
# Fuselage geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AftFuselageDimensions:
    """Piecewise axisymmetric aft-fuselage geometry.

    The global z-coordinate is used as the physical axial coordinate.
    Each cross-section is circular, with radius ``radius_at(z)`` and centre
    ``(0, center_y_at(z), z)``.
    """

    d1: float       # diameter of cylindrical section [m]
    d2: float       # diameter at first taper transition [m]
    d3: float       # diameter at forward end [m]
    l1: float       # cylindrical-section length [m]
    l2: float       # first tapered-section length [m]
    l3: float       # second tapered-section length [m]
    epsilon: float  # required wall/tank and tank/tank clearance [m]
    psi_1: float = 0.0  # centre-axis inclination in region 2 [deg]
    psi_2: float = 0.0  # centre-axis inclination in region 3 [deg]

    @property
    def total_length(self) -> float:
        return self.l1 + self.l2 + self.l3

    @property
    def interface_1(self) -> float:
        return self.l1

    @property
    def interface_2(self) -> float:
        return self.l1 + self.l2

    def radius_at(self, z: float) -> float:
        """Return fuselage radius at global axial coordinate ``z`` [m]."""
        if not 0.0 <= z <= self.total_length:
            return 0.0

        if z <= self.l1:
            return 0.5 * self.d1

        if z <= self.interface_2:
            fraction = (z - self.l1) / self.l2
            return 0.5 * (self.d1 + fraction * (self.d2 - self.d1))

        fraction = (z - self.interface_2) / self.l3
        return 0.5 * (self.d2 + fraction * (self.d3 - self.d2))

    def center_y_at(self, z: float) -> float:
        """Return y-coordinate of the local cross-section centre [m]."""
        if not 0.0 <= z <= self.total_length:
            raise ValueError(
                f"z = {z} lies outside [0, {self.total_length}]."
            )

        if z <= self.l1:
            return 0.0

        if z <= self.interface_2:
            return (z - self.l1) * math.tan(math.radians(self.psi_1))

        return (
            self.l2 * math.tan(math.radians(self.psi_1))
            + (z - self.interface_2) * math.tan(math.radians(self.psi_2))
        )

    def cross_section_center(self, z: float) -> Vector3:
        """Return the Cartesian centre of the fuselage cross-section at z."""
        return 0.0, self.center_y_at(z), z

    def tangent_at(self, z: float) -> Vector3:
        """Return the preferred local tank-axis direction at ``z``."""
        if not 0.0 <= z <= self.total_length:
            raise ValueError(
                f"z = {z} lies outside [0, {self.total_length}]."
            )

        if z <= self.l1:
            angle = 0.0
        elif z <= self.interface_2:
            angle = math.radians(self.psi_1)
        else:
            angle = math.radians(self.psi_2)

        return 0.0, math.sin(angle), math.cos(angle)


def allowed_radius_at(z: float, dims: AftFuselageDimensions) -> float:
    """Return the usable fuselage radius at ``z`` after clearance deduction.

    This compatibility helper preserves the package-level API used by older
    callers. New placement code should use ``dims.radius_at`` directly when
    it needs the raw fuselage radius.
    """
    if not 0.0 < z <= dims.total_length:
        return 0.0
    return dims.radius_at(z) - dims.epsilon


# ---------------------------------------------------------------------------
# Tank geometry and pose
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TankGeometry:
    """Rigid capsule geometry."""

    outer_radius: float
    half_cyl_length: float

    @property
    def half_total_length(self) -> float:
        return self.half_cyl_length + self.outer_radius

    @property
    def total_length(self) -> float:
        return 2.0 * self.half_total_length


@dataclass
class TankPlacement:
    """Rigid-body placement returned by the packing routine."""

    tank_index: int
    center: Vector3
    axis: Vector3
    half_total_length: float
    feasible: bool
    max_violation: float

    @property
    def x(self) -> float:
        return self.center[0]

    @property
    def y(self) -> float:
        return self.center[1]

    @property
    def z(self) -> float:
        return self.center[2]

    @property
    def lateral_offset(self) -> float:
        """Backward-compatible global-x offset [m]."""
        return self.center[0]

    @property
    def s_leftmost_pole(self) -> float:
        """Backward-compatible aft-most global-z pole coordinate [m]."""
        return self.center[2] - self.half_total_length * self.axis[2]

    @property
    def s_rightmost_pole(self) -> float:
        """Forward-most global-z pole coordinate [m]."""
        return self.center[2] + self.half_total_length * self.axis[2]


@dataclass
class AftPlacementResult:
    """Full placement result."""

    feasible: bool
    placements: list[TankPlacement]
    dims: AftFuselageDimensions
    outer_radii: list[float]
    half_cyl_lengths: list[float]
    half_outer_lengths: list[float]
    message: str


# ---------------------------------------------------------------------------
# Capsule geometry helpers
# ---------------------------------------------------------------------------

def _capsule_radius_at_offset(
    delta: float,
    outer_radius: float,
    half_cyl_length: float,
) -> float:
    """Capsule surface radius at local axial coordinate ``delta``."""
    abs_delta = abs(delta)

    if abs_delta <= half_cyl_length:
        return outer_radius

    cap_distance = abs_delta - half_cyl_length
    return math.sqrt(
        max(0.0, outer_radius**2 - cap_distance**2)
    )


def _transverse_basis(axis: Vector3) -> tuple[Vector3, Vector3]:
    """Return two orthonormal vectors perpendicular to ``axis``.

    For the present geometry the preferred axis lies in the y-z plane, so
    the global x-direction is a natural first transverse basis vector.
    """
    axis = _unit(axis)
    e1 = (1.0, 0.0, 0.0)

    # If a future orientation model makes the axis almost parallel to x,
    # fall back to a different seed vector.
    if abs(_dot(axis, e1)) > 0.99:
        e1 = (0.0, 1.0, 0.0)
        projection = _scale(axis, _dot(e1, axis))
        e1 = _unit(_sub(e1, projection))

    e2 = _unit((
        axis[1] * e1[2] - axis[2] * e1[1],
        axis[2] * e1[0] - axis[0] * e1[2],
        axis[0] * e1[1] - axis[1] * e1[0],
    ))

    return e1, e2


def _axis_segment(
    center: Vector3,
    axis: Vector3,
    half_cyl_length: float,
) -> tuple[Vector3, Vector3]:
    """Return endpoints of the capsule's cylindrical axis segment."""
    offset = _scale(_unit(axis), half_cyl_length)
    return _sub(center, offset), _add(center, offset)


def _segment_distance(
    p1: Vector3,
    q1: Vector3,
    p2: Vector3,
    q2: Vector3,
) -> float:
    """Minimum Euclidean distance between two finite 3-D line segments.

    Based on the standard closest-points formulation with endpoint clamping.
    """
    u = _sub(q1, p1)
    v = _sub(q2, p2)
    w = _sub(p1, p2)

    a = _dot(u, u)
    b = _dot(u, v)
    c = _dot(v, v)
    d = _dot(u, w)
    e = _dot(v, w)

    small = 1e-14
    denom = a * c - b * b

    s_num = 0.0
    s_den = denom
    t_num = 0.0
    t_den = denom

    if denom < small:
        s_num = 0.0
        s_den = 1.0
        t_num = e
        t_den = c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d

        if s_num < 0.0:
            s_num = 0.0
            t_num = e
            t_den = c
        elif s_num > s_den:
            s_num = s_den
            t_num = e + b
            t_den = c

    if t_num < 0.0:
        t_num = 0.0

        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num = -d
            s_den = a

    elif t_num > t_den:
        t_num = t_den

        if (-d + b) < 0.0:
            s_num = 0.0
        elif (-d + b) > a:
            s_num = s_den
        else:
            s_num = -d + b
            s_den = a

    sc = 0.0 if abs(s_num) < small else s_num / s_den
    tc = 0.0 if abs(t_num) < small else t_num / t_den

    closest_delta = _sub(
        _add(w, _scale(u, sc)),
        _scale(v, tc),
    )
    return _norm(closest_delta)


def _capsule_surface_gap(
    center_i: Vector3,
    axis_i: Vector3,
    geometry_i: TankGeometry,
    center_j: Vector3,
    axis_j: Vector3,
    geometry_j: TankGeometry,
) -> float:
    """Exact surface gap between two capsules [m].

    A capsule is the Minkowski sum of its cylindrical axis segment and a
    sphere of radius ``outer_radius``. Hence the capsule-to-capsule surface
    gap is the minimum segment distance minus the two radii.
    """
    p1, q1 = _axis_segment(
        center_i, axis_i, geometry_i.half_cyl_length
    )
    p2, q2 = _axis_segment(
        center_j, axis_j, geometry_j.half_cyl_length
    )

    return (
        _segment_distance(p1, q1, p2, q2)
        - geometry_i.outer_radius
        - geometry_j.outer_radius
    )


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------

def _pose_axis(center: Vector3, dims: AftFuselageDimensions) -> Vector3:
    """Determine tank orientation from its current axial position."""
    z = min(max(center[2], 0.0), dims.total_length)
    return dims.tangent_at(z)


def _containment_violation(
    center: Vector3,
    axis: Vector3,
    geometry: TankGeometry,
    dims: AftFuselageDimensions,
    n_axial_samples: int = 80,
    n_circumferential_samples: int = 24,
) -> float:
    """Maximum tank/fuselage containment violation [m].

    A value <= 0 is feasible. Positive values represent a physical clearance
    violation. The tank surface is sampled directly in Cartesian space.
    """
    e1, e2 = _transverse_basis(axis)
    h = geometry.half_total_length

    max_violation = -float("inf")

    for i in range(n_axial_samples + 1):
        delta = -h + 2.0 * h * i / n_axial_samples
        radius = _capsule_radius_at_offset(
            delta,
            geometry.outer_radius,
            geometry.half_cyl_length,
        )
        section_center = _add(center, _scale(axis, delta))

        # At the poles, one point is sufficient.
        n_phi = 1 if radius <= 1e-14 else n_circumferential_samples

        for j in range(n_phi):
            phi = 0.0 if n_phi == 1 else 2.0 * math.pi * j / n_phi
            radial_vector = _add(
                _scale(e1, radius * math.cos(phi)),
                _scale(e2, radius * math.sin(phi)),
            )
            point = _add(section_center, radial_vector)
            z = point[2]

            if z < 0.0:
                violation = -z + dims.epsilon
            elif z > dims.total_length:
                violation = z - dims.total_length + dims.epsilon
            else:
                local_y = dims.center_y_at(z)
                radial_distance = math.hypot(
                    point[0],
                    point[1] - local_y,
                )
                allowed_radius = dims.radius_at(z) - dims.epsilon
                violation = radial_distance - allowed_radius

            max_violation = max(max_violation, violation)

    return max_violation


def _tank_violation(
    tank_idx: int,
    centers: Sequence[Vector3],
    geometries: Sequence[TankGeometry],
    dims: AftFuselageDimensions,
    n_axial_samples: int,
    n_circumferential_samples: int,
) -> float:
    """Combined containment and pairwise-separation violation for one tank."""
    center_i = centers[tank_idx]
    axis_i = _pose_axis(center_i, dims)

    violation = max(
        0.0,
        _containment_violation(
            center_i,
            axis_i,
            geometries[tank_idx],
            dims,
            n_axial_samples,
            n_circumferential_samples,
        ),
    )

    for j in range(len(centers)):
        if j == tank_idx:
            continue

        axis_j = _pose_axis(centers[j], dims)
        gap = _capsule_surface_gap(
            center_i,
            axis_i,
            geometries[tank_idx],
            centers[j],
            axis_j,
            geometries[j],
        )

        if gap < dims.epsilon:
            violation += dims.epsilon - gap

    return violation


def _evaluate_placement(
    centers: Sequence[Vector3],
    geometries: Sequence[TankGeometry],
    dims: AftFuselageDimensions,
    n_axial_samples: int,
    n_circumferential_samples: int,
) -> tuple[bool, list[float]]:
    """Evaluate all containment and pairwise separation constraints."""
    violations = [
        max(
            0.0,
            _containment_violation(
                centers[i],
                _pose_axis(centers[i], dims),
                geometries[i],
                dims,
                n_axial_samples,
                n_circumferential_samples,
            ),
        )
        for i in range(len(centers))
    ]

    feasible = all(v <= 1e-10 for v in violations)

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            gap = _capsule_surface_gap(
                centers[i],
                _pose_axis(centers[i], dims),
                geometries[i],
                centers[j],
                _pose_axis(centers[j], dims),
                geometries[j],
            )

            pair_violation = max(0.0, dims.epsilon - gap)
            if pair_violation > 0.0:
                feasible = False
                violations[i] += pair_violation
                violations[j] += pair_violation

    return feasible, violations


# ---------------------------------------------------------------------------
# Placement algorithm
# ---------------------------------------------------------------------------

def _initial_centers(
    geometries: Sequence[TankGeometry],
    dims: AftFuselageDimensions,
) -> list[Vector3]:
    """Construct a deterministic centreline-based initial placement.

    The tanks are distributed along the available fuselage length. The packing
    search is free to move them in x, y, and z from these initial positions.
    """
    n = len(geometries)

    if n == 1:
        z_values = [0.5 * dims.total_length]
    else:
        # Keep initial centres away from the exact volume boundaries.
        margin = max(
            dims.epsilon,
            min(g.half_total_length for g in geometries),
        )
        z_lo = min(margin, 0.45 * dims.total_length)
        z_hi = max(
            z_lo,
            dims.total_length - min(margin, 0.45 * dims.total_length),
        )

        if z_hi <= z_lo:
            z_values = [0.5 * dims.total_length] * n
        else:
            z_values = [
                z_lo + i * (z_hi - z_lo) / (n - 1)
                for i in range(n)
            ]

    return [
        (0.0, dims.center_y_at(z), z)
        for z in z_values
    ]


def place_tanks_in_aft(
    outer_radii: Sequence[float],
    half_cyl_lengths: Sequence[float],
    dims: AftFuselageDimensions,
    *,
    max_iterations: int = 1000,
    initial_step: float = 0.10,
    minimum_step: float = 0.002,
    step_reduction: float = 0.5,
    n_axial_samples: int = 80,
    n_circumferential_samples: int = 24,
) -> AftPlacementResult:
    """Find feasible rigid-body poses for a set of capsule tanks.

    The routine is a self-contained packing subroutine. Tank positions are
    *not* design variables of an outer optimization problem.

    Each tank centre is free to move in Cartesian x, y, and z. Its axis is
    aligned with the preferred fuselage direction ``dims.tangent_at(z)`` at
    the tank centre. Feasibility requires:

      1. every sampled tank-surface point to lie inside the fuselage with
         clearance ``epsilon``;
      2. every pair of capsule surfaces to be separated by at least
         ``epsilon``.

    A deterministic coordinate-descent feasibility search is used. The step
    length is reduced when no improving Cartesian move is found.

    Args:
        outer_radii: Tank outer radii [m].
        half_cyl_lengths: Half-lengths of the cylindrical tank portions [m].
        dims: Aft-fuselage geometry.
        max_iterations: Maximum coordinate-descent sweeps.
        initial_step: Initial Cartesian translation step [m].
        minimum_step: Smallest translation step [m].
        step_reduction: Multiplicative step reduction in (0, 1).
        n_axial_samples: Surface samples along each tank axis.
        n_circumferential_samples: Surface samples around each tank section.

    Returns:
        AftPlacementResult containing the rigid pose of every tank.
    """
    if len(outer_radii) != len(half_cyl_lengths):
        raise ValueError(
            "outer_radii and half_cyl_lengths must have the same length."
        )

    if initial_step <= 0.0 or minimum_step <= 0.0:
        raise ValueError("Placement step sizes must be positive.")

    if not 0.0 < step_reduction < 1.0:
        raise ValueError("step_reduction must lie strictly between 0 and 1.")

    n = len(outer_radii)

    if n == 0:
        return AftPlacementResult(
            feasible=True,
            placements=[],
            dims=dims,
            outer_radii=[],
            half_cyl_lengths=[],
            half_outer_lengths=[],
            message="No tanks to place.",
        )

    geometries_original = [
        TankGeometry(float(r), float(h))
        for r, h in zip(outer_radii, half_cyl_lengths)
    ]

    # Pack the more difficult/larger tanks first, but restore original indexing
    # in the returned result.
    order = sorted(
        range(n),
        key=lambda i: (
            geometries_original[i].outer_radius,
            geometries_original[i].total_length,
        ),
        reverse=True,
    )

    geometries = [geometries_original[i] for i in order]
    centers = _initial_centers(geometries, dims)

    step = initial_step
    iteration = 0

    cartesian_directions: tuple[Vector3, ...] = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )

    while iteration < max_iterations and step >= minimum_step:
        iteration += 1
        sweep_improved = False

        for i in range(n):
            current_violation = _tank_violation(
                i,
                centers,
                geometries,
                dims,
                n_axial_samples,
                n_circumferential_samples,
            )

            if current_violation <= 1e-10:
                continue

            best_center = centers[i]
            best_violation = current_violation

            # Include tangent-following moves so a tank can travel along the
            # bent fuselage without first becoming more eccentric in y.
            tangent = _pose_axis(centers[i], dims)
            directions = cartesian_directions + (tangent, _scale(tangent, -1.0))

            for direction in directions:
                candidate = _add(
                    centers[i],
                    _scale(direction, step),
                )

                old_center = centers[i]
                centers[i] = candidate

                candidate_violation = _tank_violation(
                    i,
                    centers,
                    geometries,
                    dims,
                    n_axial_samples,
                    n_circumferential_samples,
                )

                centers[i] = old_center

                if candidate_violation < best_violation - 1e-12:
                    best_violation = candidate_violation
                    best_center = candidate

            if best_center != centers[i]:
                centers[i] = best_center
                sweep_improved = True

        if not sweep_improved:
            step *= step_reduction

    feasible_sorted, violations_sorted = _evaluate_placement(
        centers,
        geometries,
        dims,
        n_axial_samples,
        n_circumferential_samples,
    )

    placements_sorted = [
        TankPlacement(
            tank_index=order[i],
            center=centers[i],
            axis=_pose_axis(centers[i], dims),
            half_total_length=geometries[i].half_total_length,
            feasible=violations_sorted[i] <= 1e-10,
            max_violation=violations_sorted[i],
        )
        for i in range(n)
    ]

    placements = sorted(
        placements_sorted,
        key=lambda p: p.tank_index,
    )

    if feasible_sorted:
        message = (
            f"Placement feasible after {iteration} coordinate-descent sweeps."
        )
    else:
        max_violation = max(violations_sorted)
        message = (
            "Placement infeasible: residual constraint violations remain "
            f"after {iteration} sweeps; maximum accumulated violation "
            f"is {max_violation:.6g} m."
        )

    return AftPlacementResult(
        feasible=feasible_sorted,
        placements=placements,
        dims=dims,
        outer_radii=list(map(float, outer_radii)),
        half_cyl_lengths=list(map(float, half_cyl_lengths)),
        half_outer_lengths=[
            g.half_total_length for g in geometries_original
        ],
        message=message,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fuselage_surface(
    fig,
    dims: AftFuselageDimensions,
    n_theta: int = 64,
    n_z: int = 100,
) -> None:
    """Add the fuselage boundary to a Plotly 3-D figure."""
    import numpy as np
    import plotly.graph_objects as go

    z_values = np.linspace(0.0, dims.total_length, n_z)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    zz, tt = np.meshgrid(z_values, theta)

    rr = np.vectorize(dims.radius_at)(zz)
    yy_center = np.vectorize(dims.center_y_at)(zz)

    fig.add_trace(
        go.Surface(
            x=rr * np.cos(tt),
            y=yy_center + rr * np.sin(tt),
            z=zz,
            surfacecolor=np.zeros_like(zz),
            colorscale=[[0.0, "#8fbcd4"], [1.0, "#8fbcd4"]],
            showscale=False,
            opacity=0.12,
            hoverinfo="skip",
            name="Fuselage",
            showlegend=False,
        )
    )


def _plot_capsule(
    fig,
    placement: TankPlacement,
    geometry: TankGeometry,
    color: str,
    name: str,
) -> None:
    """Add a rigid capsule at the returned Cartesian pose."""
    import numpy as np
    import plotly.graph_objects as go

    e1, e2 = _transverse_basis(placement.axis)

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    delta_values = np.linspace(
        -geometry.half_total_length,
        geometry.half_total_length,
        48,
    )

    tt, dd = np.meshgrid(theta, delta_values)
    xx = np.zeros_like(dd)
    yy = np.zeros_like(dd)
    zz = np.zeros_like(dd)

    for row, delta in enumerate(delta_values):
        radius = _capsule_radius_at_offset(
            float(delta),
            geometry.outer_radius,
            geometry.half_cyl_length,
        )

        section_center = _add(
            placement.center,
            _scale(placement.axis, float(delta)),
        )

        for col, phi in enumerate(theta):
            point = _add(
                section_center,
                _add(
                    _scale(e1, radius * math.cos(float(phi))),
                    _scale(e2, radius * math.sin(float(phi))),
                ),
            )
            xx[row, col], yy[row, col], zz[row, col] = point

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=np.zeros_like(xx),
            colorscale=[[0.0, color], [1.0, color]],
            showscale=False,
            opacity=0.72,
            hovertemplate=(
                f"{name}<br>"
                "x: %{x:.3f} m<br>"
                "y: %{y:.3f} m<br>"
                "z: %{z:.3f} m<extra></extra>"
            ),
            name=name,
            showlegend=False,
        )
    )


def plot_aft_placement(result: AftPlacementResult):
    """Render an interactive Plotly view of the placement."""
    import plotly.graph_objects as go

    colors = [
        "#2d6a4f",
        "#1e3a5f",
        "#7b2d2d",
        "#6b4c11",
        "#4a1060",
    ]

    fig = go.Figure()
    _plot_fuselage_surface(fig, result.dims)

    for placement in result.placements:
        i = placement.tank_index
        geometry = TankGeometry(
            result.outer_radii[i],
            result.half_cyl_lengths[i],
        )
        name = f"Tank_{i + 1}"

        _plot_capsule(
            fig,
            placement,
            geometry,
            colors[i % len(colors)],
            name,
        )

        status = (
            "OK"
            if placement.feasible
            else f"INFEAS {placement.max_violation:.3f} m"
        )

        fig.add_trace(
            go.Scatter3d(
                x=[placement.center[0]],
                y=[placement.center[1]],
                z=[placement.center[2]],
                mode="text",
                text=[f"{name}<br>{status}"],
                textfont=dict(size=11),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        title=result.message,
        margin=dict(l=0, r=0, b=0, t=50),
    )

    return fig
