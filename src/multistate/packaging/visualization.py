"""Visualization helpers for packaging layouts."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import Point3D
from .layout import PackagingLayout
from .primitives import CapsulePrimitive, RegularPrismPrimitive, SpherePrimitive


def _plot_cone(ax, layout: PackagingLayout, n_theta: int = 64, n_z: int = 32) -> None:
    volume = layout.volume
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z_vals = np.linspace(0.0, volume.length, n_z)
    theta_grid, z_grid = np.meshgrid(theta, z_vals)

    r_grid = np.vectorize(volume.radius_at)(z_grid)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    ax.plot_surface(x, y, z_grid, alpha=0.12, linewidth=0.0, color="#8fbcd4")


def _orient_from_z(x, y, z, axis: str):
    """Rotate local +z-aligned coordinates to requested principal axis."""
    if axis == "x":
        return z, x, y
    if axis == "y":
        return x, z, y
    return x, y, z


def _plot_sphere(ax, center: Point3D, primitive: SpherePrimitive, color: str) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 32)
    uu, vv = np.meshgrid(u, v)

    x = primitive.radius * np.cos(uu) * np.sin(vv) + center.x
    y = primitive.radius * np.sin(uu) * np.sin(vv) + center.y
    z = primitive.radius * np.cos(vv) + center.z

    ax.plot_surface(x, y, z, color=color, alpha=0.7, linewidth=0.15, edgecolor="#333333")


def _plot_capsule(ax, center: Point3D, primitive: CapsulePrimitive, color: str) -> None:
    r = primitive.radius
    half_cyl = 0.5 * primitive.cylinder_length

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    z_lin = np.linspace(-half_cyl, half_cyl, 22)
    tt, zz = np.meshgrid(theta, z_lin)

    x0 = r * np.cos(tt)
    y0 = r * np.sin(tt)
    z0 = zz
    xw, yw, zw = _orient_from_z(x0, y0, z0, primitive.axis)
    ax.plot_surface(xw + center.x, yw + center.y, zw + center.z, color=color, alpha=0.7, linewidth=0.15, edgecolor="#333333")

    for sign in (-1.0, 1.0):
        phi = np.linspace(0.0, np.pi / 2.0, 20)
        pp, tt2 = np.meshgrid(phi, theta)

        z_cap = sign * half_cyl + sign * r * np.sin(pp)
        radial = r * np.cos(pp)
        x_cap = radial * np.cos(tt2)
        y_cap = radial * np.sin(tt2)

        xw, yw, zw = _orient_from_z(x_cap, y_cap, z_cap, primitive.axis)
        ax.plot_surface(
            xw + center.x,
            yw + center.y,
            zw + center.z,
            color=color,
            alpha=0.7,
            linewidth=0.15,
            edgecolor="#333333",
        )


def _plot_prism(ax, center: Point3D, primitive: RegularPrismPrimitive, color: str) -> None:
    hx = 0.5 * primitive.width
    hy = 0.5 * primitive.depth
    hz = 0.5 * primitive.height

    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
    )
    vertices[:, 0] += center.x
    vertices[:, 1] += center.y
    vertices[:, 2] += center.z

    face_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    faces = [[vertices[idx] for idx in face] for face in face_indices]

    poly = Poly3DCollection(faces, facecolors=color, alpha=0.7, edgecolors="#333333", linewidths=0.6)
    ax.add_collection3d(poly)


def _set_equal_axes(ax) -> None:
    """Force equal scale in x, y, z so solids are not visually skewed."""
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()
    z_min, z_max = ax.get_zlim3d()

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    half_range = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim3d(x_mid - half_range, x_mid + half_range)
    ax.set_ylim3d(y_mid - half_range, y_mid + half_range)
    ax.set_zlim3d(z_mid - half_range, z_mid + half_range)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_layout(
    layout: PackagingLayout,
    *,
    sample_step: float = 0.08,
    route_points: list[Point3D] | None = None,
    route_point_sets: list[list[Point3D]] | None = None,
    show_quadrants: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a 3D visualization for a packaging layout and optional route."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    _plot_cone(ax, layout)

    colors = ["#33658A", "#55A630", "#BC4749", "#6A4C93", "#A44A3F", "#3A86FF"]

    for i, (component_id, placed) in enumerate(layout.components.items()):
        color = colors[i % len(colors)]
        if isinstance(placed.primitive, SpherePrimitive):
            _plot_sphere(ax, placed.center, placed.primitive, color)
        elif isinstance(placed.primitive, CapsulePrimitive):
            _plot_capsule(ax, placed.center, placed.primitive, color)
        elif isinstance(placed.primitive, RegularPrismPrimitive):
            _plot_prism(ax, placed.center, placed.primitive, color)
        else:
            # Fallback for future primitive types without dedicated surface renderer.
            points = placed.sample_points_world(step=sample_step)
            if points:
                xs = [p.x for p in points]
                ys = [p.y for p in points]
                zs = [p.z for p in points]
                ax.scatter(xs, ys, zs, s=3.0, alpha=0.22, c=color)

        c0, c1 = placed.connection_points()
        ax.scatter([c0.x, c1.x], [c0.y, c1.y], [c0.z, c1.z], s=40.0, c=color, marker="x")

        if show_quadrants:
            q = layout.volume.classify_quadrant(placed.center)
            ax.text(placed.center.x, placed.center.y, placed.center.z, f"{component_id}\n{q}", fontsize=9)
        else:
            ax.text(placed.center.x, placed.center.y, placed.center.z, component_id, fontsize=9)

    all_routes: list[list[Point3D]] = []
    if route_point_sets:
        all_routes.extend(route_point_sets)
    if route_points:
        all_routes.append(route_points)

    for route in all_routes:
        if not route:
            continue
        ax.plot(
            [p.x for p in route],
            [p.y for p in route],
            [p.z for p in route],
            color="#111111",
            linewidth=2.4,
            label="90-degree route",
        )

    max_r = layout.volume.max_radius
    ax.set_xlim(-max_r * 1.1, max_r * 1.1)
    ax.set_ylim(-max_r * 1.1, max_r * 1.1)
    ax.set_zlim(0.0, layout.volume.length)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Packaging Prototype: Truncated-Cone Volume")
    _set_equal_axes(ax)
    ax.view_init(elev=20.0, azim=35.0)

    return fig, ax
