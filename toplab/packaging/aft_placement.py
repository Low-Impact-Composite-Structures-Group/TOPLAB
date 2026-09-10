"""Centreline-following aft-fuselage tank placement."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Geometry definition
# ---------------------------------------------------------------------------

@dataclass
class AftFuselageDimensions:
    """Geometric definition of the aft fuselage pocket."""

    d1: float       # aft-bulkhead diameter [m]
    d2: float       # intermediate diameter at cone transition [m]
    d3: float       # forward (tip) diameter [m]
    l1: float       # length of cylindrical section [m]
    l2: float       # length of first (aft) conical section [m]
    l3: float       # length of second (fwd) conical section [m]
    epsilon: float  # radial clearance margin deducted from all constraints [m]
    psi_1: float = 0.0  # first cone centreline angle [degrees]
    psi_2: float = 0.0  # second cone centreline angle [degrees]

    @property
    def total_length(self) -> float:
        return self.l1 + self.l2 + self.l3

    @property
    def centreline_interface_1(self) -> float:
        return self.l1

    @property
    def centreline_interface_2(self) -> float:
        return self.l1 + self.l2

    def radius_at(self, s: float) -> float:
        """Return the fuselage radius at global axial coordinate ``s``."""
        if not 0.0 <= s <= self.total_length:
            return 0.0
        if s <= self.l1:
            return self.d1 / 2.0
        if s <= self.centreline_interface_2:
            fraction = (s - self.l1) / self.l2
            return self.d1 / 2.0 + fraction * (self.d2 - self.d1) / 2.0
        fraction = (s - self.centreline_interface_2) / self.l3
        return self.d2 / 2.0 + fraction * (self.d3 - self.d2) / 2.0

    def centreline_at(self, s: float) -> tuple[float, float, float]:
        """Return ``(x, y, z)`` on the piecewise centreline."""
        if not 0.0 <= s <= self.total_length:
            raise ValueError(f"s = {s} lies outside [0, {self.total_length}].")
        if s <= self.l1:
            y = 0.0
        elif s <= self.centreline_interface_2:
            y = (s - self.l1) * math.tan(math.radians(self.psi_1))
        else:
            y = (
                self.l2 * math.tan(math.radians(self.psi_1))
                + (s - self.centreline_interface_2) * math.tan(math.radians(self.psi_2))
            )
        return 0.0, y, s

    def tangent_at(self, s: float) -> tuple[float, float, float]:
        """Return the unit tangent in the global ``(x, y, z)`` frame."""
        angle = self.psi_1 if self.l1 < s <= self.centreline_interface_2 else self.psi_2
        angle_rad = math.radians(angle) if s > self.l1 else 0.0
        norm = math.sqrt(1.0 + math.tan(angle_rad) ** 2)
        return 0.0, math.sin(angle_rad), math.cos(angle_rad)


def allowed_radius_at(s: float, dims: AftFuselageDimensions) -> float:
    """Return the usable fuselage radius at global axial coordinate ``s``."""
    return dims.radius_at(s) - dims.epsilon if 0.0 < s <= dims.total_length else 0.0


def _capsule_radius_at_offset(
    delta: float, outer_radius: float, half_cyl_length: float
) -> float:
    """Return the capsule envelope radius at an axial offset."""
    abs_delta = abs(delta)
    if abs_delta <= half_cyl_length:
        return outer_radius
    cap_dist = abs_delta - half_cyl_length
    return math.sqrt(max(0.0, outer_radius ** 2 - cap_dist ** 2))


def _rigid_tank_pose(
    s_leftmost_pole: float,
    half_total_length: float,
    lateral_offset: float,
    dims: AftFuselageDimensions,
) -> tuple[
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return a rigid tank midpoint, axis, and radial basis."""
    s_mid = s_leftmost_pole + half_total_length
    for _ in range(8):
        _, tangent_y, tangent_z = dims.tangent_at(
            min(max(s_mid, 0.0), dims.total_length)
        )
        s_mid = s_leftmost_pole + half_total_length * tangent_z

    _, centreline_y, _ = dims.centreline_at(s_mid)
    _, tangent_y, tangent_z = dims.tangent_at(s_mid)
    return (
        s_mid,
        (lateral_offset, centreline_y, s_mid),
        (0.0, tangent_y, tangent_z),
        (0.0, tangent_z, -tangent_y),
    )


def _tank_z_end(
    s_leftmost_pole: float,
    half_total_length: float,
    lateral_offset: float,
    dims: AftFuselageDimensions,
) -> float:
    """Return the forward global-z extent of a rigid tank."""
    _, center, axis, _ = _rigid_tank_pose(
        s_leftmost_pole,
        half_total_length,
        lateral_offset,
        dims,
    )
    return center[2] + half_total_length * axis[2]


# ---------------------------------------------------------------------------
# Single-tank violation
# ---------------------------------------------------------------------------

def _tank_violation(
    s_leftmost_pole: float,
    lateral_offset: float,
    outer_radius: float,
    half_cyl_length: float,
    dims: AftFuselageDimensions,
    n_samples: int = 200,
) -> float:
    """Max radial constraint violation for a centreline-following tank.

    A positive return value means the tank protrudes outside the fuselage wall.
    ``float('inf')`` indicates the tank extends beyond the fuselage length.

    Args:
        s_leftmost_pole: Axial position of the aft tank pole [m].
        outer_radius: Tank outer radius [m].
        half_cyl_length: Half the cylindrical section length [m].
        dims: Aft fuselage dimensions.
        n_samples: Number of axial samples for the profile check.

    Returns:
        Max violation >= 0 (positive = infeasible), or inf if out-of-bounds.
    """
    half_total_length = outer_radius + half_cyl_length
    _, center, axis, radial_basis = _rigid_tank_pose(
        s_leftmost_pole,
        half_total_length,
        lateral_offset,
        dims,
    )
    s_start = center[2] - half_total_length * axis[2]
    s_end = center[2] + half_total_length * axis[2]
    if s_start <= 0.0 or s_end > dims.total_length:
        return float("inf")

    max_viol = 0.0
    radial_angles = [2.0 * math.pi * i / 16.0 for i in range(16)]
    for i in range(n_samples + 1):
        t = i / n_samples
        delta = (2.0 * t - 1.0) * half_total_length
        tank_r = _capsule_radius_at_offset(delta, outer_radius, half_cyl_length)
        point_center = tuple(center[j] + delta * axis[j] for j in range(3))
        for angle in radial_angles:
            radial_cos = math.cos(angle) * tank_r
            radial_sin = math.sin(angle) * tank_r
            point = tuple(
                point_center[j]
                + (radial_cos if j == 0 else 0.0)
                + radial_sin * radial_basis[j]
                for j in range(3)
            )
            if not 0.0 <= point[2] <= dims.total_length:
                return float("inf")
            _, fuselage_centerline_y, _ = dims.centreline_at(point[2])
            radial_distance = math.sqrt(
                point[0] ** 2 + (point[1] - fuselage_centerline_y) ** 2
            )
            viol = radial_distance - allowed_radius_at(point[2], dims)
            if viol > max_viol:
                max_viol = viol

    return max_viol


# ---------------------------------------------------------------------------
# Combined violation for nudge optimisation
# ---------------------------------------------------------------------------

def _combined_violation(
    tank_idx: int,
    poles: list[float],
    lateral_offsets: list[float],
    outer_radii: Sequence[float],
    half_cyl_lengths: Sequence[float],
    half_totals: list[float],
    dims: AftFuselageDimensions,
    n_samples: int,
) -> float:
    """Violation for tank *tank_idx*: fuselage constraint + adjacent gap penalties."""
    n = len(poles)
    viol = _tank_violation(
        poles[tank_idx],
        lateral_offsets[tank_idx],
        outer_radii[tank_idx],
        half_cyl_lengths[tank_idx],
        dims,
        n_samples,
    )

    if tank_idx > 0:
        previous_end = _tank_z_end(
            poles[tank_idx - 1],
            half_totals[tank_idx - 1],
            lateral_offsets[tank_idx - 1],
            dims,
        )
        gap = poles[tank_idx] - previous_end
        if gap < dims.epsilon:
            viol += dims.epsilon - gap

    if tank_idx < n - 1:
        current_end = _tank_z_end(
            poles[tank_idx],
            half_totals[tank_idx],
            lateral_offsets[tank_idx],
            dims,
        )
        gap = poles[tank_idx + 1] - current_end
        if gap < dims.epsilon:
            viol += dims.epsilon - gap

    tank_start = poles[tank_idx]
    tank_end = _tank_z_end(
        tank_start,
        half_totals[tank_idx],
        lateral_offsets[tank_idx],
        dims,
    )
    for other_idx in range(n):
        if other_idx == tank_idx:
            continue
        other_start = poles[other_idx]
        other_end = _tank_z_end(
            other_start,
            half_totals[other_idx],
            lateral_offsets[other_idx],
            dims,
        )
        if min(tank_end, other_end) > max(tank_start, other_start):
            lateral_gap = (
                abs(lateral_offsets[tank_idx] - lateral_offsets[other_idx])
                - outer_radii[tank_idx]
                - outer_radii[other_idx]
            )
            if lateral_gap < dims.epsilon:
                viol += dims.epsilon - lateral_gap

    return viol


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TankPlacement:
    """Placement result for a single tank."""

    tank_index: int
    s_leftmost_pole: float  # aft pole axial coordinate [m]
    lateral_offset: float   # offset in global x from the centreline [m]
    feasible: bool
    max_violation: float  # [m], 0 if feasible

    @property
    def x_center(self) -> float:
        """Compatibility alias for the former axial centre field."""
        return self.s_leftmost_pole

    @property
    def center(self) -> tuple[float, float, float]:
        return self.lateral_offset, 0.0, self.s_leftmost_pole


@dataclass
class AftPlacementResult:
    """Full placement result for all tanks."""

    feasible: bool
    placements: list[TankPlacement]
    dims: AftFuselageDimensions
    outer_radii: list[float]
    half_outer_lengths: list[float]  # outer_radius + half_cyl_length per tank
    message: str


# ---------------------------------------------------------------------------
# Placement engine
# ---------------------------------------------------------------------------

def place_tanks_in_aft(
    outer_radii: Sequence[float],
    half_cyl_lengths: Sequence[float],
    dims: AftFuselageDimensions,
    *,
    max_iterations: int = 500,
    nudge_step: float = 0.01,
    n_samples: int = 200,
) -> AftPlacementResult:
    """Place N capsule tanks on-axis inside the aft fuselage.

    Initial positions are evenly spaced along the feasible x range.
    An iterative coordinate-descent nudge then minimises the total
    constraint violation one tank at a time.

    Args:
        outer_radii: Outer radius of each tank [m] (length N).
        half_cyl_lengths: Half the cylindrical section length of each tank [m] (length N).
        dims: Aft fuselage dimensions.
        max_iterations: Maximum nudge iterations before declaring convergence.
        nudge_step: Step size for each positional nudge [m].
        n_samples: Axial sample count for the radial constraint check.

    Returns:
        :class:`AftPlacementResult` describing the final placement.
    """
    n = len(outer_radii)
    if n == 0:
        return AftPlacementResult(
            feasible=True,
            placements=[],
            dims=dims,
            outer_radii=[],
            half_outer_lengths=[],
            message="No tanks to place.",
        )

    # Sort tanks largest-first so the biggest tank gets the aft clearance.
    order = sorted(range(n), key=lambda i: -outer_radii[i])
    inv_order = [0] * n
    for k, orig in enumerate(order):
        inv_order[orig] = k

    s_outer = [outer_radii[order[k]] for k in range(n)]
    s_half_cyl = [half_cyl_lengths[order[k]] for k in range(n)]
    s_half_tot = [s_outer[k] + s_half_cyl[k] for k in range(n)]

    # Per-position x bounds: largest tank clears the aft end, smallest clears the fwd end.
    x_lo = dims.epsilon
    minimum_axis_projection = min(
        math.cos(math.radians(dims.psi_1)),
        math.cos(math.radians(dims.psi_2)),
    )
    x_hi = (
        dims.total_length
        - 2.0 * s_half_tot[-1] * minimum_axis_projection
        - dims.epsilon
    )

    if x_lo > x_hi:
        half_tots_orig = [outer_radii[i] + half_cyl_lengths[i] for i in range(n)]
        placements = [
            TankPlacement(i, dims.total_length / 2.0, 0.0, False, float("inf"))
            for i in range(n)
        ]
        return AftPlacementResult(
            feasible=False,
            placements=placements,
            dims=dims,
            outer_radii=list(outer_radii),
            half_outer_lengths=half_tots_orig,
            message="Tanks too large to fit within aft fuselage length.",
        )

    # Initial positions are expressed by the aft pole of each capsule.
    if n == 1:
        poles = [0.5 * (x_lo + x_hi)]
    else:
        pitch = (x_hi - x_lo) / (n - 1)
        poles = [x_lo + k * pitch for k in range(n)]
    lateral_offsets = [0.0] * n

    # Iterative coordinate-descent nudge on the sorted arrays
    for _ in range(max_iterations):
        improved = False
        for k in range(n):
            cur_viol = _combined_violation(
                k, poles, lateral_offsets, s_outer, s_half_cyl, s_half_tot, dims, n_samples
            )
            if cur_viol <= 0.0:
                continue

            for coordinate in (poles, lateral_offsets):
                for sign in (+1.0, -1.0):
                    coordinate[k] += sign * nudge_step
                    new_viol = _combined_violation(
                        k,
                        poles,
                        lateral_offsets,
                        s_outer,
                        s_half_cyl,
                        s_half_tot,
                        dims,
                        n_samples,
                    )
                    if new_viol < cur_viol - 1e-12:
                        cur_viol = new_viol
                        improved = True
                        break
                    coordinate[k] -= sign * nudge_step
                if improved:
                    break

        if not improved:
            break

    # Build result in ORIGINAL input order
    placements: list[TankPlacement] = []
    overall_feasible = True
    for i in range(n):
        k = inv_order[i]
        raw_viol = _tank_violation(
            poles[k], lateral_offsets[k], s_outer[k], s_half_cyl[k], dims, n_samples
        )
        feasible = raw_viol <= 0.0
        if not feasible:
            overall_feasible = False
        placements.append(TankPlacement(
            i, poles[k], lateral_offsets[k], feasible, max(0.0, raw_viol)
        ))

    gap_violations: list[tuple[int, int, float]] = []
    for k in range(n - 1):
        gap = poles[k + 1] - _tank_z_end(
            poles[k],
            s_half_tot[k],
            lateral_offsets[k],
            dims,
        )
        if gap < dims.epsilon:
            overall_feasible = False
            gap_violations.append((order[k], order[k + 1], gap))

    lateral_violations: list[tuple[int, int, float]] = []
    for first_idx in range(n):
        first_start = poles[first_idx]
        first_end = _tank_z_end(
            first_start,
            s_half_tot[first_idx],
            lateral_offsets[first_idx],
            dims,
        )
        for second_idx in range(first_idx + 1, n):
            second_start = poles[second_idx]
            second_end = _tank_z_end(
                second_start,
                s_half_tot[second_idx],
                lateral_offsets[second_idx],
                dims,
            )
            if min(first_end, second_end) <= max(first_start, second_start):
                continue
            lateral_gap = (
                abs(lateral_offsets[first_idx] - lateral_offsets[second_idx])
                - s_outer[first_idx]
                - s_outer[second_idx]
            )
            if lateral_gap < dims.epsilon:
                overall_feasible = False
                lateral_violations.append((order[first_idx], order[second_idx], lateral_gap))

    if overall_feasible:
        msg = "Placement feasible."
    else:
        msg = "Placement infeasible: constraint violations remain after nudge iterations."
        for orig_a, orig_b, gap in gap_violations:
            msg += (
                f" Gap between tank {orig_a + 1} and tank {orig_b + 1}: "
                f"{gap:.3f} m (min {dims.epsilon:.3f} m required)."
            )
        for orig_a, orig_b, gap in lateral_violations:
            msg += (
                f" Lateral overlap between tank {orig_a + 1} and tank {orig_b + 1}: "
                f"{gap:.3f} m (min {dims.epsilon:.3f} m required)."
            )

    half_tots_orig = [outer_radii[i] + half_cyl_lengths[i] for i in range(n)]
    return AftPlacementResult(
        feasible=overall_feasible,
        placements=placements,
        dims=dims,
        outer_radii=list(outer_radii),
        half_outer_lengths=half_tots_orig,
        message=msg,
    )


# ---------------------------------------------------------------------------
# 3-D visualisation
# ---------------------------------------------------------------------------

def _plot_fuselage_surface(
    fig,
    dims: AftFuselageDimensions,
    n_theta: int = 64,
    n_x: int = 80,
) -> None:
    """Add the aft-fuselage surface to a Plotly 3-D figure."""
    import numpy as np
    import plotly.graph_objects as go

    s_vals = np.linspace(0.0, dims.total_length, n_x)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    ss, tt = np.meshgrid(s_vals, theta)
    rr = np.vectorize(dims.radius_at)(ss)
    centreline_y = np.vectorize(lambda s: dims.centreline_at(s)[1])(ss)

    fig.add_trace(
        go.Surface(
            x=rr * np.cos(tt),
            y=centreline_y + rr * np.sin(tt),
            z=ss,
            surfacecolor=np.zeros_like(ss),
            colorscale=[
                [0.0, "#8fbcd4"],
                [1.0, "#8fbcd4"],
            ],
            showscale=False,
            opacity=0.12,
            hoverinfo="skip",
            name="Fuselage",
            showlegend=False,
        )
    )


def _plot_capsule_along_centreline(
    fig,
    placement: TankPlacement,
    outer_radius: float,
    half_cyl: float,
    dims: AftFuselageDimensions,
    color: str,
    name: str,
) -> None:
    """Add a rigid capsule aligned with the midpoint centreline tangent."""
    import numpy as np
    import plotly.graph_objects as go

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    half_total_length = outer_radius + half_cyl
    _, center, axis, radial_basis = _rigid_tank_pose(
        placement.s_leftmost_pole,
        half_total_length,
        placement.lateral_offset,
        dims,
    )
    distances = np.linspace(-half_total_length, half_total_length, 44)
    tt, dd = np.meshgrid(theta, distances)
    x_values = np.zeros_like(dd)
    y_values = np.zeros_like(dd)
    z_values = np.zeros_like(dd)
    radial = np.zeros_like(dd)

    for row, distance in enumerate(distances):
        radius = _capsule_radius_at_offset(
            distance, outer_radius, half_cyl
        )
        radial[row, :] = radius
        x_values[row, :] = center[0] + radius * np.cos(theta)
        y_values[row, :] = center[1] + distance * axis[1] + radius * radial_basis[1] * np.sin(theta)
        z_values[row, :] = center[2] + distance * axis[2] + radius * radial_basis[2] * np.sin(theta)

    fig.add_trace(go.Surface(
        x=x_values,
        y=y_values,
        z=z_values,
        surfacecolor=np.zeros_like(x_values),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=0.7,
        hovertemplate=f"{name}<br>x: %{{x:.3f}} m<br>y: %{{y:.3f}} m<br>z: %{{z:.3f}} m<extra></extra>",
        name=name,
        showlegend=False,
    ))


def _set_equal_axes(
    fig,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """Set equal numerical ranges on all three Plotly axes."""

    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)

    half = 0.5 * max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[mid_x - half, mid_x + half],
            ),
            yaxis=dict(
                range=[mid_y - half, mid_y + half],
            ),
            zaxis=dict(
                range=[mid_z - half, mid_z + half],
            ),
            aspectmode="cube",
        )
    )

def plot_aft_placement(result: AftPlacementResult):
    """Render an interactive 3-D view of the aft fuselage with placed tanks.

    The fuselage centreline runs along the Plotly z-axis. The plot is
    generated even for infeasible placements so violations remain visible.

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    _COLORS = [
        "#2d6a4f",
        "#1e3a5f",
        "#7b2d2d",
        "#6b4c11",
        "#4a1060",
    ]

    dims = result.dims

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Fuselage
    # ------------------------------------------------------------------

    _plot_fuselage_surface(fig, dims)

    # ------------------------------------------------------------------
    # Tanks
    # ------------------------------------------------------------------

    for p in result.placements:

        i = p.tank_index
        R = result.outer_radii[i]
        half_cyl = result.half_outer_lengths[i] - R
        color = _COLORS[i % len(_COLORS)]

        tank_name = f"Tank_{i + 1}"

        _plot_capsule_along_centreline(
            fig,
            p,
            R,
            half_cyl,
            dims,
            color,
            tank_name,
        )

        status = (
            "OK"
            if p.feasible
            else f"INFEAS {p.max_violation:.3f} m"
        )

        # --------------------------------------------------------------
        # Tank label
        # --------------------------------------------------------------

        fig.add_trace(
            go.Scatter3d(
                x=[p.lateral_offset],
                y=[dims.centreline_at(p.s_leftmost_pole)[1]],
                z=[p.s_leftmost_pole],
                mode="text",
                text=[f"{tank_name}<br>{status}"],
                textfont=dict(size=11),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ------------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------------

    # Determine the complete extent of the geometry.
    max_radius = max([dims.radius_at(s) for s in (0.0, dims.l1, dims.l1 + dims.l2, dims.total_length)] + result.outer_radii)
    centreline_ys = [dims.centreline_at(s)[1] for s in (0.0, dims.l1, dims.l1 + dims.l2, dims.total_length)]

    x_min = -max_radius
    x_max = max_radius
    y_min = min(centreline_ys) - max_radius
    y_max = max(centreline_ys) + max_radius
    z_min = 0.0
    z_max = dims.total_length

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.update_layout(
        title="Aft Fuselage Packaging",
        width=1300,
        height=600,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
        ),
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="s – from aft bulkhead [m]",

            # Equivalent to the equal-scale Matplotlib setup
            aspectmode="cube",

            # Approximate equivalent of:
            # ax.view_init(elev=20, azim=-90)
            camera=dict(
                eye=dict(
                    x=0.0,
                    y=-2.2,
                    z=0.8,
                ),
                center=dict(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                ),
                up=dict(
                    x=0.0,
                    y=0.0,
                    z=1.0,
                ),
            ),
        ),
    )

    _set_equal_axes(
        fig,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )

    return fig
