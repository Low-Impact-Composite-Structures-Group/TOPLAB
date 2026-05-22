"""Orthogonal path routing inside a truncated-cone packaging volume."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from .geometry import AxisymmetricVolume, Point3D
from .layout import PlacedPrimitive


class RoutingError(RuntimeError):
    """Raised when a valid route cannot be found."""


GridIndex = tuple[int, int, int]
State = tuple[GridIndex, int]


class GridRouter:
    """Simple 3D A* router constrained to axis-aligned (90-degree) motion."""

    _DIRS: tuple[GridIndex, ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    def __init__(self, grid_step: float = 0.05, turn_penalty: float = 0.1):
        if grid_step <= 0.0:
            raise ValueError("grid_step must be > 0")
        self.grid_step = grid_step
        self.turn_penalty = max(0.0, turn_penalty)

    def _to_grid(self, point: Point3D) -> GridIndex:
        s = self.grid_step
        return (
            int(round(point.x / s)),
            int(round(point.y / s)),
            int(round(point.z / s)),
        )

    def _to_point(self, index: GridIndex) -> Point3D:
        s = self.grid_step
        return Point3D(index[0] * s, index[1] * s, index[2] * s)

    def _neighbors(self, idx: GridIndex) -> Iterable[tuple[GridIndex, int]]:
        for i, d in enumerate(self._DIRS):
            yield (idx[0] + d[0], idx[1] + d[1], idx[2] + d[2]), i

    def _heuristic(self, a: GridIndex, b: GridIndex) -> float:
        return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) * self.grid_step

    def _direction_to_index(self, direction: Point3D) -> int:
        comps = [direction.x, direction.y, direction.z]
        abs_comps = [abs(c) for c in comps]
        axis = int(max(range(3), key=lambda i: abs_comps[i]))

        if abs_comps[axis] < 1e-9:
            raise RoutingError("direction vector must be non-zero")

        # Port directions are expected to be axis aligned in this prototype.
        off_axes = [abs_comps[i] for i in range(3) if i != axis]
        if any(v > 1e-6 for v in off_axes):
            raise RoutingError("direction vector must be axis-aligned")

        sign_positive = comps[axis] >= 0.0
        if axis == 0:
            return 0 if sign_positive else 1
        if axis == 1:
            return 2 if sign_positive else 3
        return 4 if sign_positive else 5

    def route(
        self,
        start: Point3D,
        end: Point3D,
        *,
        volume: AxisymmetricVolume,
        obstacles: list[PlacedPrimitive],
        pipe_radius: float = 0.0,
        start_direction: Point3D | None = None,
        end_direction: Point3D | None = None,
        min_straight_length: float = 0.0,
        max_expansions: int = 250000,
    ) -> list[Point3D]:
        """Route between two points; movement is axis-aligned so turns are 90-degree only."""
        if min_straight_length < 0.0:
            raise ValueError("min_straight_length must be >= 0")

        start_idx = self._to_grid(start)
        end_idx = self._to_grid(end)

        cache_free: dict[GridIndex, bool] = {}

        def is_free(idx: GridIndex) -> bool:
            if idx in cache_free:
                return cache_free[idx]
            p = self._to_point(idx)
            if not volume.contains_point(p, margin=pipe_radius):
                cache_free[idx] = False
                return False
            for obstacle in obstacles:
                if obstacle.contains_point(p, margin=pipe_radius):
                    cache_free[idx] = False
                    return False
            cache_free[idx] = True
            return True

        if not is_free(start_idx):
            raise RoutingError("start point is blocked or outside volume")
        if not is_free(end_idx):
            raise RoutingError("end point is blocked or outside volume")

        straight_steps = int(math.ceil(min_straight_length / self.grid_step - 1e-12))

        search_start_idx = start_idx
        start_dir_idx = -1
        forced_start_segment: list[GridIndex] = [start_idx]

        if start_direction is not None and straight_steps > 0:
            start_dir_idx = self._direction_to_index(start_direction)
            step = self._DIRS[start_dir_idx]
            for _ in range(straight_steps):
                search_start_idx = (
                    search_start_idx[0] + step[0],
                    search_start_idx[1] + step[1],
                    search_start_idx[2] + step[2],
                )
                if not is_free(search_start_idx):
                    raise RoutingError("required straight run from start is blocked or outside volume")
                forced_start_segment.append(search_start_idx)

        search_end_idx = end_idx
        forced_end_segment: list[GridIndex] = [end_idx]

        if end_direction is not None and straight_steps > 0:
            end_dir_idx = self._direction_to_index(end_direction)
            step = self._DIRS[end_dir_idx]
            for _ in range(straight_steps):
                search_end_idx = (
                    search_end_idx[0] - step[0],
                    search_end_idx[1] - step[1],
                    search_end_idx[2] - step[2],
                )
                if not is_free(search_end_idx):
                    raise RoutingError("required straight run into end is blocked or outside volume")
                forced_end_segment.append(search_end_idx)
            forced_end_segment.reverse()

        start_state: State = (search_start_idx, start_dir_idx)
        open_heap: list[tuple[float, State]] = []
        heapq.heappush(open_heap, (0.0, start_state))

        g_cost: dict[State, float] = {start_state: 0.0}
        came_from: dict[State, State] = {}
        best_goal_state: State | None = None

        expansions = 0
        while open_heap:
            _, current = heapq.heappop(open_heap)
            current_idx, current_dir = current

            if current_idx == search_end_idx:
                best_goal_state = current
                break

            expansions += 1
            if expansions > max_expansions:
                raise RoutingError("routing search exceeded max_expansions")

            current_g = g_cost[current]
            for next_idx, next_dir in self._neighbors(current_idx):
                if not is_free(next_idx):
                    continue

                turn_cost = self.turn_penalty if (current_dir != -1 and current_dir != next_dir) else 0.0
                step_cost = self.grid_step + turn_cost
                next_state: State = (next_idx, next_dir)
                tentative_g = current_g + step_cost

                if tentative_g >= g_cost.get(next_state, float("inf")):
                    continue

                g_cost[next_state] = tentative_g
                came_from[next_state] = current
                f_cost = tentative_g + self._heuristic(next_idx, end_idx)
                heapq.heappush(open_heap, (f_cost, next_state))

        if best_goal_state is None:
            raise RoutingError("no valid orthogonal route found")

        path_indices: list[GridIndex] = []
        trace_state = best_goal_state
        while True:
            path_indices.append(trace_state[0])
            if trace_state == start_state:
                break
            trace_state = came_from[trace_state]

        path_indices.reverse()

        full_indices = forced_start_segment[:-1] + path_indices
        if forced_end_segment[0] == search_end_idx and forced_end_segment[-1] == end_idx:
            full_indices.extend(forced_end_segment[1:])

        return [self._to_point(idx) for idx in full_indices]
