"""Volume and efficiency metrics for packaging studies."""

from __future__ import annotations

import math

from .geometry import AxisymmetricVolume, Point3D
from .layout import PackagingLayout


def route_length(route: list[Point3D]) -> float:
    """Compute polyline length for one routed pipe centerline."""
    if len(route) < 2:
        return 0.0

    total = 0.0
    for p0, p1 in zip(route[:-1], route[1:]):
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        dz = p1.z - p0.z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def components_volume(layout: PackagingLayout) -> float:
    """Sum exact primitive volumes for all placed components."""
    return sum(placed.primitive.volume() for placed in layout.components.values())


def piping_volume(routes: list[list[Point3D]], pipe_radius: float) -> float:
    """Pipe volume using V = pi*r^2*L on routed centerlines."""
    if pipe_radius < 0.0:
        raise ValueError("pipe_radius must be >= 0")
    total_length = sum(route_length(route) for route in routes)
    return math.pi * pipe_radius * pipe_radius * total_length


def packaged_volume(layout: PackagingLayout, routes: list[list[Point3D]], pipe_radius: float) -> float:
    """Total occupied volume = components + piping."""
    return components_volume(layout) + piping_volume(routes, pipe_radius)


def packaging_efficiency(
    layout: PackagingLayout,
    routes: list[list[Point3D]],
    pipe_radius: float,
    outer_volume: AxisymmetricVolume | None = None,
) -> float:
    """Efficiency defined as occupied volume divided by outer volume."""
    volume_container = layout.volume if outer_volume is None else outer_volume
    if volume_container.volume <= 0.0:
        raise ValueError("outer volume must be > 0")
    return packaged_volume(layout, routes, pipe_radius) / volume_container.volume
