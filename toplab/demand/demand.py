"""
Demand: a mass-flow-versus-time function, the single representation of what a
tank or node is asked to deliver (or receive).

Everything downstream (ODE right-hand side, sizing, post-processing) asks a
Demand for `mass_flow(t)` and never walks mission sections itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

import numpy as np


class Demand(ABC):
    """Non-negative mass flow rate [kg/s] as a function of time [s]."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """Time [s] after which the demand is zero."""

    @abstractmethod
    def mass_flow(self, t: float) -> float:
        """Mass flow rate [kg/s] at time t."""

    @abstractmethod
    def total_mass(self) -> float:
        """Integral of the flow over the whole demand [kg]."""

    def __call__(self, t: float) -> float:
        return self.mass_flow(t)

    def scaled(self, fraction: float) -> "Demand":
        return ScaledDemand(self, fraction)


class ZeroDemand(Demand):
    duration = 0.0

    def mass_flow(self, t: float) -> float:
        return 0.0

    def total_mass(self) -> float:
        return 0.0


class PiecewiseDemand(Demand):
    """
    Demand made of consecutive segments. Segment k spans [starts[k], ends[k]] and
    carries `values[k]`, flow samples evenly spaced across the segment and
    linearly interpolated (one sample = constant). At a boundary the earlier
    segment applies; after the last segment the flow is zero.
    """

    def __init__(self, durations: Sequence[float], values: Sequence[Sequence[float]]):
        if len(durations) != len(values):
            raise ValueError("durations and values must have the same length")
        self._durations = np.asarray(durations, dtype=float)
        self._ends = np.cumsum(self._durations)
        self._starts = self._ends - self._durations
        self._values = [np.abs(np.asarray(v, dtype=float)) for v in values]
        if any(v.size == 0 for v in self._values):
            raise ValueError("every segment needs at least one flow sample")
        self._positions = [np.linspace(0.0, 1.0, v.size) for v in self._values]

    @property
    def duration(self) -> float:
        return float(self._ends[-1]) if self._ends.size else 0.0

    def mass_flow(self, t: float) -> float:
        k = int(np.searchsorted(self._ends, t, side="left"))
        if k >= self._ends.size:
            return 0.0
        v = self._values[k]
        if v.size == 1 or self._durations[k] <= 0.0:
            return float(v[0])
        progress = min(max((t - self._starts[k]) / self._durations[k], 0.0), 1.0)
        return float(np.interp(progress, self._positions[k], v))

    def total_mass(self) -> float:
        return float(sum(
            d * np.trapz(v, p) if v.size > 1 else d * v[0]
            for d, v, p in zip(self._durations, self._values, self._positions)
        ))


class ScaledDemand(Demand):
    def __init__(self, base: Demand, fraction: float):
        if fraction < 0.0:
            raise ValueError(f"fraction must be >= 0, got {fraction}")
        self.base = base
        self.fraction = float(fraction)

    @property
    def duration(self) -> float:
        return self.base.duration

    def mass_flow(self, t: float) -> float:
        return self.fraction * self.base.mass_flow(t)

    def total_mass(self) -> float:
        return self.fraction * self.base.total_mass()


class SumDemand(Demand):
    def __init__(self, parts: Iterable[Demand]):
        self.parts: List[Demand] = list(parts)

    @property
    def duration(self) -> float:
        return max((p.duration for p in self.parts), default=0.0)

    def mass_flow(self, t: float) -> float:
        return sum(p.mass_flow(t) for p in self.parts)

    def total_mass(self) -> float:
        return sum(p.total_mass() for p in self.parts)
