"""Discrete insulation sizing: the thinnest admissible insulation that meets the dormancy limit.

The admissible thicknesses form the grid ``t_min + k * resolution`` up to ``t_max``. Venting is
assumed to decrease monotonically with thickness, so the smallest feasible grid value is found by
binary search with at most ``2 + ceil(log2(N - 1))`` dormancy simulations for ``N`` grid values
(the two end points plus the bisection decisions).

The routine only needs a callable ``thickness -> DormancyPoint``; it knows nothing about tanks, so a
multi-tank sizing policy can be added by supplying a different callable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# Status values of :class:`SizingResult`.
AT_MINIMUM = "at_minimum"          # the thinnest admissible insulation already satisfies dormancy
BRACKETED = "bracketed"            # found by binary search between t_min and t_max
INFEASIBLE = "dormancy_infeasible"  # even t_max violates dormancy


@dataclass(frozen=True)
class DormancyPoint:
    """Outcome of one dormancy simulation. ``g_dorm = f_vent / f_vent_max - 1`` (<= 0 is feasible)."""
    g_dorm: float
    f_vent: float
    data: Any = field(default=None, compare=False)  # full analysis payload, reused by the caller


@dataclass(frozen=True)
class InsulationGrid:
    """Admissible insulation thicknesses [m] ``t_min, t_min + resolution, ..., t_max``."""
    t_min: float
    t_max: float
    resolution: float = 0.0025

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("insulation_sizing.resolution must be positive.")
        if self.t_min < 0.0 or self.t_max < self.t_min:
            raise ValueError("insulation_sizing needs 0 <= t_min <= t_max.")
        steps = (self.t_max - self.t_min) / self.resolution
        if abs(steps - round(steps)) > 1.0e-6:
            raise ValueError(
                f"insulation_sizing: (t_max - t_min) = {self.t_max - self.t_min:g} m is not a whole "
                f"multiple of resolution = {self.resolution:g} m.")

    @property
    def size(self) -> int:
        return int(round((self.t_max - self.t_min) / self.resolution)) + 1

    def thickness(self, index: int) -> float:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return round(self.t_min + index * self.resolution, 12)

    @property
    def values(self) -> list[float]:
        return [self.thickness(i) for i in range(self.size)]

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "InsulationGrid":
        if not raw or "t_min" not in raw or "t_max" not in raw:
            raise ValueError("Define insulation_sizing.t_min and insulation_sizing.t_max "
                             "(and optionally resolution) in the optimisation YAML.")
        return cls(float(raw["t_min"]), float(raw["t_max"]), float(raw.get("resolution", 0.0025)))


@dataclass(frozen=True)
class SizingResult:
    status: str
    t_ins: float          # t* (the thickness at t_max when infeasible, i.e. the last one tried)
    point: DormancyPoint  # dormancy outcome at ``t_ins``
    n_evaluations: int    # number of distinct thicknesses passed to ``evaluate``

    @property
    def feasible(self) -> bool:
        return self.status != INFEASIBLE

    @property
    def f_vent(self) -> float:
        return self.point.f_vent

    @property
    def g_dorm(self) -> float:
        return self.point.g_dorm


def size_insulation(evaluate: Callable[[float], DormancyPoint], grid: InsulationGrid) -> SizingResult:
    """Return the smallest grid thickness with ``g_dorm <= 0`` (see the module docstring).

    Every grid index is simulated at most once. On a bracketed result the point just below ``t*``
    has been simulated and violates dormancy, so ``t*`` is the exact discrete minimum provided the
    behaviour is monotonic. That assumption cannot be checked from the few points a binary search
    visits; use :func:`check_monotonic` to verify a model offline.
    """
    points: dict[int, DormancyPoint] = {}

    def at(index: int) -> DormancyPoint:
        if index not in points:
            points[index] = evaluate(grid.thickness(index))
        return points[index]

    def done(status: str, index: int) -> SizingResult:
        return SizingResult(status, grid.thickness(index), points[index], len(points))

    last = grid.size - 1
    if at(0).g_dorm <= 0.0:
        return done(AT_MINIMUM, 0)
    if last == 0 or at(last).g_dorm > 0.0:
        return done(INFEASIBLE, last)
    low, high = 0, last  # invariant: g(low) > 0 and g(high) <= 0
    while high - low > 1:
        middle = (low + high) // 2
        if at(middle).g_dorm <= 0.0:
            high = middle
        else:
            low = middle
    return done(BRACKETED, high)


def check_monotonic(evaluate: Callable[[float], DormancyPoint], grid: InsulationGrid,
                    tolerance: float = 1.0e-9) -> tuple[bool, list[DormancyPoint]]:
    """Simulate every grid thickness and report whether ``g_dorm`` never increases with thickness."""
    points = [evaluate(t) for t in grid.values]
    g = [p.g_dorm for p in points]
    return all(later <= earlier + tolerance for earlier, later in zip(g, g[1:])), points


def max_evaluations(grid: InsulationGrid) -> int:
    """Upper bound on dormancy simulations: both end points plus the binary-search decisions."""
    return 1 if grid.size == 1 else 2 + math.ceil(math.log2(grid.size - 1))
