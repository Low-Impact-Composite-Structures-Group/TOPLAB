"""Core geometry definitions for packaging studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class AxisymmetricVolume(Protocol):
    """Interface for axisymmetric volumes aligned with +z."""

    @property
    def length(self) -> float:
        ...

    @property
    def max_radius(self) -> float:
        ...

    def radius_at(self, z: float) -> float:
        ...

    def contains_point(self, point: "Point3D", margin: float = 0.0) -> bool:
        ...

    def classify_quadrant(self, point: "Point3D") -> str:
        ...


@dataclass(frozen=True)
class Point3D:
    """Simple immutable 3D point/vector helper."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class TruncatedConeVolume:
    """Axis-aligned truncated cone (frustum) with axis along +z."""

    length: float
    radius_z0: float
    radius_z1: float

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("length must be > 0")
        if self.radius_z0 <= 0.0 or self.radius_z1 <= 0.0:
            raise ValueError("radii must be > 0")

    @property
    def max_radius(self) -> float:
        return max(self.radius_z0, self.radius_z1)

    def radius_at(self, z: float) -> float:
        if z < 0.0 or z > self.length:
            raise ValueError("z must be within [0, length]")
        t = z / self.length
        return self.radius_z0 * (1.0 - t) + self.radius_z1 * t

    def contains_point(self, point: Point3D, margin: float = 0.0) -> bool:
        """Return True if point is inside the frustum with optional inward margin."""
        if point.z < margin or point.z > (self.length - margin):
            return False
        r_allowed = self.radius_at(point.z) - margin
        if r_allowed < 0.0:
            return False
        r_xy = math.hypot(point.x, point.y)
        return r_xy <= r_allowed

    def classify_quadrant(self, point: Point3D) -> str:
        """Return an 8-region label split by x/y signs and fore/aft z-half."""
        if not self.contains_point(point):
            raise ValueError("point is outside the truncated cone")

        if point.x >= 0.0 and point.y >= 0.0:
            base = "Q1"
        elif point.x < 0.0 <= point.y:
            base = "Q2"
        elif point.x < 0.0 and point.y < 0.0:
            base = "Q3"
        else:
            base = "Q4"

        axial = "FORE" if point.z <= 0.5 * self.length else "AFT"
        return f"{base}_{axial}"


@dataclass(frozen=True)
class AxialSegment:
    """Piece of an axisymmetric profile with linear radius interpolation."""

    length: float
    radius_start: float
    radius_end: float

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("segment length must be > 0")
        if self.radius_start <= 0.0 or self.radius_end <= 0.0:
            raise ValueError("segment radii must be > 0")


@dataclass(frozen=True)
class CompositeAxisymmetricVolume:
    """Axisymmetric volume made from chained linear-radius segments."""

    segments: tuple[AxialSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("at least one segment is required")

    @property
    def length(self) -> float:
        return sum(seg.length for seg in self.segments)

    @property
    def max_radius(self) -> float:
        return max(max(seg.radius_start, seg.radius_end) for seg in self.segments)

    def radius_at(self, z: float) -> float:
        if z < 0.0 or z > self.length:
            raise ValueError("z must be within [0, length]")

        z_cursor = 0.0
        for seg in self.segments:
            z_next = z_cursor + seg.length
            if z <= z_next + 1e-12:
                local_z = min(max(z - z_cursor, 0.0), seg.length)
                t = local_z / seg.length
                return seg.radius_start * (1.0 - t) + seg.radius_end * t
            z_cursor = z_next

        # Numerical guard for z == length.
        last = self.segments[-1]
        return last.radius_end

    def contains_point(self, point: Point3D, margin: float = 0.0) -> bool:
        if point.z < margin or point.z > (self.length - margin):
            return False
        r_allowed = self.radius_at(point.z) - margin
        if r_allowed < 0.0:
            return False
        r_xy = math.hypot(point.x, point.y)
        return r_xy <= r_allowed

    def classify_quadrant(self, point: Point3D) -> str:
        if not self.contains_point(point):
            raise ValueError("point is outside the composite volume")

        if point.x >= 0.0 and point.y >= 0.0:
            base = "Q1"
        elif point.x < 0.0 <= point.y:
            base = "Q2"
        elif point.x < 0.0 and point.y < 0.0:
            base = "Q3"
        else:
            base = "Q4"

        axial = "FORE" if point.z <= 0.5 * self.length else "AFT"
        return f"{base}_{axial}"
