"""Primitive solids for packaging studies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal
import math

from .geometry import Point3D

Axis = Literal["x", "y", "z"]


def _iter_grid(min_p: Point3D, max_p: Point3D, step: float) -> Iterable[Point3D]:
    x = min_p.x
    while x <= max_p.x + 1e-12:
        y = min_p.y
        while y <= max_p.y + 1e-12:
            z = min_p.z
            while z <= max_p.z + 1e-12:
                yield Point3D(x, y, z)
                z += step
            y += step
        x += step


class Primitive(ABC):
    """Abstract local-space primitive centered at origin."""

    @abstractmethod
    def contains_local(self, point: Point3D, margin: float = 0.0) -> bool:
        pass

    @abstractmethod
    def connection_points_local(self) -> tuple[Point3D, Point3D]:
        pass

    @abstractmethod
    def bounding_box_local(self) -> tuple[Point3D, Point3D]:
        pass

    def sample_points_local(self, step: float, margin: float = 0.0) -> list[Point3D]:
        if step <= 0.0:
            raise ValueError("step must be > 0")
        bb_min, bb_max = self.bounding_box_local()
        points: list[Point3D] = []
        for p in _iter_grid(bb_min, bb_max, step):
            if self.contains_local(p, margin=margin):
                points.append(p)
        return points


@dataclass(frozen=True)
class CapsulePrimitive(Primitive):
    """Cylinder with hemispherical end caps."""

    cylinder_length: float
    radius: float
    axis: Axis = "z"

    def __post_init__(self) -> None:
        if self.cylinder_length < 0.0:
            raise ValueError("cylinder_length must be >= 0")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

    @property
    def half_total_length(self) -> float:
        return 0.5 * self.cylinder_length + self.radius

    def _split_axes(self, point: Point3D) -> tuple[float, float, float]:
        if self.axis == "x":
            return point.x, point.y, point.z
        if self.axis == "y":
            return point.y, point.x, point.z
        return point.z, point.x, point.y

    def contains_local(self, point: Point3D, margin: float = 0.0) -> bool:
        axis_pos, r1, r2 = self._split_axes(point)
        allowed_r = self.radius + margin
        radial = math.hypot(r1, r2)
        half_cyl = 0.5 * self.cylinder_length

        if abs(axis_pos) <= half_cyl:
            return radial <= allowed_r

        cap_center = half_cyl if axis_pos > 0.0 else -half_cyl
        cap_dist = math.sqrt((axis_pos - cap_center) ** 2 + radial ** 2)
        return cap_dist <= allowed_r

    def connection_points_local(self) -> tuple[Point3D, Point3D]:
        d = self.half_total_length
        if self.axis == "x":
            return Point3D(-d, 0.0, 0.0), Point3D(d, 0.0, 0.0)
        if self.axis == "y":
            return Point3D(0.0, -d, 0.0), Point3D(0.0, d, 0.0)
        return Point3D(0.0, 0.0, -d), Point3D(0.0, 0.0, d)

    def bounding_box_local(self) -> tuple[Point3D, Point3D]:
        d = self.half_total_length
        r = self.radius
        if self.axis == "x":
            return Point3D(-d, -r, -r), Point3D(d, r, r)
        if self.axis == "y":
            return Point3D(-r, -d, -r), Point3D(r, d, r)
        return Point3D(-r, -r, -d), Point3D(r, r, d)


@dataclass(frozen=True)
class SpherePrimitive(Primitive):
    radius: float
    connection_axis: Axis = "z"

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

    def contains_local(self, point: Point3D, margin: float = 0.0) -> bool:
        allowed_r = self.radius + margin
        return math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2) <= allowed_r

    def connection_points_local(self) -> tuple[Point3D, Point3D]:
        r = self.radius
        if self.connection_axis == "x":
            return Point3D(-r, 0.0, 0.0), Point3D(r, 0.0, 0.0)
        if self.connection_axis == "y":
            return Point3D(0.0, -r, 0.0), Point3D(0.0, r, 0.0)
        return Point3D(0.0, 0.0, -r), Point3D(0.0, 0.0, r)

    def bounding_box_local(self) -> tuple[Point3D, Point3D]:
        r = self.radius
        return Point3D(-r, -r, -r), Point3D(r, r, r)


@dataclass(frozen=True)
class RegularPrismPrimitive(Primitive):
    """Axis-aligned rectangular prism with two face-center ports."""

    width: float
    depth: float
    height: float
    connection_axis: Axis = "z"

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.depth <= 0.0 or self.height <= 0.0:
            raise ValueError("width, depth, and height must be > 0")

    def contains_local(self, point: Point3D, margin: float = 0.0) -> bool:
        hx = 0.5 * self.width + margin
        hy = 0.5 * self.depth + margin
        hz = 0.5 * self.height + margin
        return abs(point.x) <= hx and abs(point.y) <= hy and abs(point.z) <= hz

    def connection_points_local(self) -> tuple[Point3D, Point3D]:
        if self.connection_axis == "x":
            return Point3D(-0.5 * self.width, 0.0, 0.0), Point3D(0.5 * self.width, 0.0, 0.0)
        if self.connection_axis == "y":
            return Point3D(0.0, -0.5 * self.depth, 0.0), Point3D(0.0, 0.5 * self.depth, 0.0)
        return Point3D(0.0, 0.0, -0.5 * self.height), Point3D(0.0, 0.0, 0.5 * self.height)

    def bounding_box_local(self) -> tuple[Point3D, Point3D]:
        return (
            Point3D(-0.5 * self.width, -0.5 * self.depth, -0.5 * self.height),
            Point3D(0.5 * self.width, 0.5 * self.depth, 0.5 * self.height),
        )
