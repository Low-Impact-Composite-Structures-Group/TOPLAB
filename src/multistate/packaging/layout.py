"""Placement engine for packaging primitives in a truncated-cone volume."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .geometry import AxisymmetricVolume, Point3D
from .primitives import Primitive


class PlacementError(RuntimeError):
    """Raised when a placement violates volume or overlap constraints."""


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


@dataclass(frozen=True)
class PlacedPrimitive:
    component_id: str
    primitive: Primitive
    center: Point3D

    def to_local(self, point_world: Point3D) -> Point3D:
        return point_world - self.center

    def to_world(self, point_local: Point3D) -> Point3D:
        return point_local + self.center

    def contains_point(self, point_world: Point3D, margin: float = 0.0) -> bool:
        return self.primitive.contains_local(self.to_local(point_world), margin=margin)

    def connection_points(self) -> tuple[Point3D, Point3D]:
        p0, p1 = self.primitive.connection_points_local()
        return self.to_world(p0), self.to_world(p1)

    def connection_directions(self) -> tuple[Point3D, Point3D]:
        """Outward unit vectors for each connection point."""
        c0, c1 = self.connection_points()

        def _unit_from_center(p: Point3D) -> Point3D:
            vx = p.x - self.center.x
            vy = p.y - self.center.y
            vz = p.z - self.center.z
            mag = math.sqrt(vx * vx + vy * vy + vz * vz)
            if mag <= 0.0:
                raise ValueError("connection point is coincident with component center")
            return Point3D(vx / mag, vy / mag, vz / mag)

        return _unit_from_center(c0), _unit_from_center(c1)

    def bounding_box_world(self) -> tuple[Point3D, Point3D]:
        bb_min, bb_max = self.primitive.bounding_box_local()
        return self.to_world(bb_min), self.to_world(bb_max)

    def sample_points_world(self, step: float) -> list[Point3D]:
        return [self.to_world(p) for p in self.primitive.sample_points_local(step=step)]


class PackagingLayout:
    """Container for component placements and geometric validity checks."""

    def __init__(self, volume: AxisymmetricVolume):
        self.volume = volume
        self.components: dict[str, PlacedPrimitive] = {}

    def add_component(
        self,
        component_id: str,
        primitive: Primitive,
        center: Point3D,
        *,
        sampling_step: float = 0.05,
        clearance: float = 0.0,
    ) -> PlacedPrimitive:
        if component_id in self.components:
            raise PlacementError(f"component id already exists: {component_id}")

        placed = PlacedPrimitive(component_id=component_id, primitive=primitive, center=center)

        if not self._is_fully_inside_volume(placed, sampling_step=sampling_step, clearance=clearance):
            raise PlacementError(f"component {component_id} is not fully inside truncated cone")

        for existing in self.components.values():
            if self._components_overlap(placed, existing, sampling_step=sampling_step, clearance=clearance):
                raise PlacementError(
                    f"component {component_id} overlaps with existing component {existing.component_id}"
                )

        self.components[component_id] = placed
        return placed

    def get_component(self, component_id: str) -> PlacedPrimitive:
        return self.components[component_id]

    def get_connection_point(self, component_id: str, connection_index: int) -> Point3D:
        if connection_index not in (0, 1):
            raise ValueError("connection_index must be 0 or 1")
        return self.get_component(component_id).connection_points()[connection_index]

    def get_connection_direction(self, component_id: str, connection_index: int) -> Point3D:
        if connection_index not in (0, 1):
            raise ValueError("connection_index must be 0 or 1")
        return self.get_component(component_id).connection_directions()[connection_index]

    def get_component_quadrants(self) -> dict[str, str]:
        return {
            component_id: self.volume.classify_quadrant(component.center)
            for component_id, component in self.components.items()
        }

    def _is_fully_inside_volume(
        self,
        component: PlacedPrimitive,
        *,
        sampling_step: float,
        clearance: float,
    ) -> bool:
        for point in component.sample_points_world(step=sampling_step):
            if not self.volume.contains_point(point, margin=clearance):
                return False
        return True

    @staticmethod
    def _aabb_overlap(a: tuple[Point3D, Point3D], b: tuple[Point3D, Point3D]) -> bool:
        a_min, a_max = a
        b_min, b_max = b
        return (
            a_min.x <= b_max.x and a_max.x >= b_min.x
            and a_min.y <= b_max.y and a_max.y >= b_min.y
            and a_min.z <= b_max.z and a_max.z >= b_min.z
        )

    def _components_overlap(
        self,
        c1: PlacedPrimitive,
        c2: PlacedPrimitive,
        *,
        sampling_step: float,
        clearance: float,
    ) -> bool:
        bb1 = c1.bounding_box_world()
        bb2 = c2.bounding_box_world()
        if not self._aabb_overlap(bb1, bb2):
            return False

        bb_min = Point3D(
            max(bb1[0].x, bb2[0].x),
            max(bb1[0].y, bb2[0].y),
            max(bb1[0].z, bb2[0].z),
        )
        bb_max = Point3D(
            min(bb1[1].x, bb2[1].x),
            min(bb1[1].y, bb2[1].y),
            min(bb1[1].z, bb2[1].z),
        )

        for p in _iter_grid(bb_min, bb_max, sampling_step):
            if c1.contains_point(p, margin=clearance) and c2.contains_point(p, margin=clearance):
                return True
        return False
