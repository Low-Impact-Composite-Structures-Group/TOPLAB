"""Aft-fuselage tank placement for packaging studies.

Coordinate convention
---------------------
The aft fuselage is parameterised with:
  - x = 0  at the centre of the aft bulkhead (d1 face)
  - x increasing forward (toward the nose)
  - y = z = 0 on the fuselage symmetry axis

The three segments are:
  1. Cylinder  (0 < x ≤ l1):          inner radius = d1/2
  2. First cone (l1 < x ≤ l1+l2):     radius tapers from d1/2 to d2/2
  3. Second cone (l1+l2 < x ≤ L_tot): radius tapers from d2/2 to d3/2

Tanks are placed on-axis (y = z = 0) with their longitudinal axis along x.
"""
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

    @property
    def total_length(self) -> float:
        return self.l1 + self.l2 + self.l3


def allowed_radius_at(x: float, dims: AftFuselageDimensions) -> float:
    """Max allowed containment radius (epsilon already deducted) at axial position x.

    Returns 0 for x outside the fuselage extent [0, total_length].
    """
    if x <= 0.0 or x > dims.total_length:
        return 0.0
    if x <= dims.l1:
        # Constraint 1 – cylindrical section
        return dims.d1 / 2.0 - dims.epsilon
    if x <= dims.l1 + dims.l2:
        # Constraint 2 – first conical transition (d1 → d2)
        return (
            (dims.l1 + dims.l2 - x) * (dims.d1 - dims.d2) / (2.0 * dims.l2)
            + dims.d2 / 2.0
            - dims.epsilon
        )
    # Constraint 3 – second conical transition (d2 → d3)
    return (
        (dims.l1 + dims.l2 + dims.l3 - x) * (dims.d2 - dims.d3) / (2.0 * dims.l3)
        + dims.d3 / 2.0
        - dims.epsilon
    )


# ---------------------------------------------------------------------------
# Capsule radial profile
# ---------------------------------------------------------------------------

def _capsule_radius_at_offset(
    delta: float, outer_radius: float, half_cyl_length: float
) -> float:
    """Cross-sectional radius of the capsule at axial offset *delta* from its centre.

    Args:
        delta: Signed axial offset from the capsule centre [m].
        outer_radius: Outer radius of the cylindrical section [m].
        half_cyl_length: Half the length of the cylindrical section [m].

    Returns:
        Radial envelope at that offset.
    """
    abs_delta = abs(delta)
    if abs_delta <= half_cyl_length:
        return outer_radius
    cap_dist = abs_delta - half_cyl_length
    return math.sqrt(max(0.0, outer_radius ** 2 - cap_dist ** 2))


# ---------------------------------------------------------------------------
# Single-tank violation
# ---------------------------------------------------------------------------

def _tank_violation(
    x_center: float,
    outer_radius: float,
    half_cyl_length: float,
    dims: AftFuselageDimensions,
    n_samples: int = 200,
) -> float:
    """Max radial constraint violation for one tank placed at *x_center*.

    A positive return value means the tank protrudes outside the fuselage wall.
    ``float('inf')`` indicates the tank extends beyond the fuselage length.

    Args:
        x_center: Axial position of the tank centre [m].
        outer_radius: Tank outer radius [m].
        half_cyl_length: Half the cylindrical section length [m].
        dims: Aft fuselage dimensions.
        n_samples: Number of axial samples for the profile check.

    Returns:
        Max violation ≥ 0 (positive = infeasible), or inf if out-of-bounds.
    """
    half_total = outer_radius + half_cyl_length
    x_start = x_center - half_total
    x_end = x_center + half_total

    if x_start <= 0.0 or x_end > dims.total_length:
        return float("inf")

    max_viol = 0.0
    for i in range(n_samples + 1):
        t = i / n_samples
        x = x_start + t * (x_end - x_start)
        delta = x - x_center
        tank_r = _capsule_radius_at_offset(delta, outer_radius, half_cyl_length)
        allowed = allowed_radius_at(x, dims)
        viol = tank_r - allowed
        if viol > max_viol:
            max_viol = viol

    return max_viol


# ---------------------------------------------------------------------------
# Combined violation for nudge optimisation
# ---------------------------------------------------------------------------

def _combined_violation(
    tank_idx: int,
    centers: list[float],
    outer_radii: Sequence[float],
    half_cyl_lengths: Sequence[float],
    half_totals: list[float],
    dims: AftFuselageDimensions,
    n_samples: int,
) -> float:
    """Violation for tank *tank_idx*: fuselage constraint + adjacent gap penalties."""
    n = len(centers)
    viol = _tank_violation(
        centers[tank_idx],
        outer_radii[tank_idx],
        half_cyl_lengths[tank_idx],
        dims,
        n_samples,
    )

    if tank_idx > 0:
        gap = (
            (centers[tank_idx] - half_totals[tank_idx])
            - (centers[tank_idx - 1] + half_totals[tank_idx - 1])
        )
        if gap < dims.epsilon:
            viol += dims.epsilon - gap

    if tank_idx < n - 1:
        gap = (
            (centers[tank_idx + 1] - half_totals[tank_idx + 1])
            - (centers[tank_idx] + half_totals[tank_idx])
        )
        if gap < dims.epsilon:
            viol += dims.epsilon - gap

    return viol


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TankPlacement:
    """Placement result for a single tank."""

    tank_index: int
    x_center: float       # axial centre position [m]
    feasible: bool
    max_violation: float  # [m], 0 if feasible


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
        nudge_step: Step size for each nudge in x [m].
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

    # Sort tanks largest-first (by outer radius) so the biggest tank is placed
    # nearest the datum where the cylindrical section offers maximum clearance.
    order = sorted(range(n), key=lambda i: -outer_radii[i])
    inv_order = [0] * n
    for k, orig in enumerate(order):
        inv_order[orig] = k

    s_outer = [outer_radii[order[k]] for k in range(n)]
    s_half_cyl = [half_cyl_lengths[order[k]] for k in range(n)]
    s_half_tot = [s_outer[k] + s_half_cyl[k] for k in range(n)]

    # Per-position x bounds: largest tank clears the aft end, smallest clears the fwd end.
    x_lo = s_half_tot[0] + dims.epsilon
    x_hi = dims.total_length - s_half_tot[-1] - dims.epsilon

    if x_lo > x_hi:
        half_tots_orig = [outer_radii[i] + half_cyl_lengths[i] for i in range(n)]
        placements = [
            TankPlacement(i, dims.total_length / 2.0, False, float("inf"))
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

    # Initial positions: evenly spaced in sorted order (k=0 = smallest x = largest tank)
    if n == 1:
        centers = [0.5 * (x_lo + x_hi)]
    else:
        pitch = (x_hi - x_lo) / (n - 1)
        centers = [x_lo + k * pitch for k in range(n)]

    # Iterative coordinate-descent nudge on the sorted arrays
    for _ in range(max_iterations):
        improved = False
        for k in range(n):
            cur_viol = _combined_violation(
                k, centers, s_outer, s_half_cyl, s_half_tot, dims, n_samples
            )
            if cur_viol <= 0.0:
                continue

            for sign in (+1.0, -1.0):
                centers[k] += sign * nudge_step
                new_viol = _combined_violation(
                    k, centers, s_outer, s_half_cyl, s_half_tot, dims, n_samples
                )
                if new_viol < cur_viol - 1e-12:
                    cur_viol = new_viol
                    improved = True
                    break
                else:
                    centers[k] -= sign * nudge_step  # revert

        if not improved:
            break

    # Build result in ORIGINAL input order
    placements: list[TankPlacement] = []
    overall_feasible = True
    for i in range(n):
        k = inv_order[i]
        raw_viol = _tank_violation(centers[k], s_outer[k], s_half_cyl[k], dims, n_samples)
        feasible = raw_viol <= 0.0
        if not feasible:
            overall_feasible = False
        placements.append(TankPlacement(i, centers[k], feasible, max(0.0, raw_viol)))

    gap_violations: list[tuple[int, int, float]] = []
    for k in range(n - 1):
        gap = (centers[k + 1] - s_half_tot[k + 1]) - (centers[k] + s_half_tot[k])
        if gap < dims.epsilon:
            overall_feasible = False
            gap_violations.append((order[k], order[k + 1], gap))

    if overall_feasible:
        msg = "Placement feasible."
    else:
        msg = "Placement infeasible: constraint violations remain after nudge iterations."
        for orig_a, orig_b, gap in gap_violations:
            msg += (
                f" Gap between tank {orig_a + 1} and tank {orig_b + 1}: "
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

def _raw_radius_at(x: float, dims: AftFuselageDimensions) -> float:
    """Aft-fuselage inner-wall radius at x (no epsilon deduction)."""
    if x <= 0.0 or x > dims.total_length:
        return 0.0
    if x <= dims.l1:
        return dims.d1 / 2.0
    if x <= dims.l1 + dims.l2:
        return (
            (dims.l1 + dims.l2 - x) * (dims.d1 - dims.d2) / (2.0 * dims.l2)
            + dims.d2 / 2.0
        )
    return (
        (dims.l1 + dims.l2 + dims.l3 - x) * (dims.d2 - dims.d3) / (2.0 * dims.l3)
        + dims.d3 / 2.0
    )

def _plot_fuselage_surface(
    fig,
    dims: AftFuselageDimensions,
    n_theta: int = 64,
    n_x: int = 80,
) -> None:
    """Add the aft-fuselage surface to a Plotly 3-D figure."""
    import numpy as np
    import plotly.graph_objects as go

    x_vals = np.linspace(0.0, dims.total_length, n_x)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)

    xx, tt = np.meshgrid(x_vals, theta)

    rr = np.vectorize(
        lambda x: _raw_radius_at(x, dims)
    )(xx)

    fig.add_trace(
        go.Surface(
            x=xx,
            y=rr * np.cos(tt),
            z=rr * np.sin(tt),
            surfacecolor=np.zeros_like(xx),
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


def _plot_capsule_along_x(
    fig,
    x_center: float,
    outer_radius: float,
    half_cyl: float,
    color: str,
    name: str,
) -> None:
    """Add a capsule surface with its longitudinal axis along x."""
    import numpy as np
    import plotly.graph_objects as go

    theta = np.linspace(0.0, 2.0 * np.pi, 48)

    # ------------------------------------------------------------------
    # Cylindrical section
    # ------------------------------------------------------------------

    x_cyl = np.linspace(
        x_center - half_cyl,
        x_center + half_cyl,
        22,
    )

    tt, xx = np.meshgrid(theta, x_cyl)

    yy = outer_radius * np.cos(tt)
    zz = outer_radius * np.sin(tt)

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=np.zeros_like(xx),
            colorscale=[
                [0.0, color],
                [1.0, color],
            ],
            showscale=False,
            opacity=0.7,
            hovertemplate=(
                f"{name}"
                "<br>x: %{x:.3f} m"
                "<br>y: %{y:.3f} m"
                "<br>z: %{z:.3f} m"
                "<extra></extra>"
            ),
            name=name,
            showlegend=False,
        )
    )

    # ------------------------------------------------------------------
    # Hemispherical caps
    # ------------------------------------------------------------------

    phi = np.linspace(0.0, np.pi / 2.0, 20)

    pp, tt2 = np.meshgrid(phi, theta)

    radial = outer_radius * np.cos(pp)

    yy_cap = radial * np.cos(tt2)
    zz_cap = radial * np.sin(tt2)

    for sign in (-1.0, +1.0):

        xx_cap = (
            x_center
            + sign * (
                half_cyl
                + outer_radius * np.sin(pp)
            )
        )

        fig.add_trace(
            go.Surface(
                x=xx_cap,
                y=yy_cap,
                z=zz_cap,
                surfacecolor=np.zeros_like(xx_cap),
                colorscale=[
                    [0.0, color],
                    [1.0, color],
                ],
                showscale=False,
                opacity=0.7,
                hovertemplate=(
                    f"{name}"
                    "<br>x: %{x:.3f} m"
                    "<br>y: %{y:.3f} m"
                    "<br>z: %{z:.3f} m"
                    "<extra></extra>"
                ),
                name=name,
                showlegend=False,
            )
        )


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

    The fuselage axis runs along the Plotly x-axis. The plot is generated
    even for infeasible placements so violations remain visible.

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

        _plot_capsule_along_x(
            fig,
            p.x_center,
            R,
            half_cyl,
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
                x=[p.x_center],
                y=[0.0],
                z=[R * 1.05],
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
    x_min = 0.0
    x_max = dims.total_length

    max_radius = max(
        [
            _raw_radius_at(x, dims)
            for x in (
                0.0,
                dims.l1,
                dims.l1 + dims.l2,
                dims.total_length,
            )
        ]
        + result.outer_radii
    )

    y_min = -max_radius
    y_max = max_radius
    z_min = -max_radius
    z_max = max_radius

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
            xaxis_title="x – from aft bulkhead [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",

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
