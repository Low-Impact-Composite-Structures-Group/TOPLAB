"""Convergence-based termination for NSGA-II.

Each generation the feasible non-dominated front is compared with the previous generation's in
normalised objective space. The movement is the largest of the ideal-point shift, the nadir-point
shift and the IGD between the two fronts (pymoo's ``MultiObjectiveSpaceTermination``). The run is
converged once the movement has stayed at or below ``tolerance`` for ``period`` consecutive
generations. A hard cap (``max_generations``) always applies, and nothing can stop the run before
``min_generations`` or before a feasible design exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pymoo.core.termination import Termination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

CONVERGED = "converged"
MAX_GENERATIONS = "max_generations"


@dataclass(frozen=True)
class ConvergenceSettings:
    enabled: bool = False
    tolerance: float = 0.005   # normalised front movement per generation
    period: int = 10           # consecutive generations that must stay within the tolerance
    min_generations: int = 0

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("nsga2.convergence.tolerance must be positive.")
        if self.period < 1:
            raise ValueError("nsga2.convergence.period must be at least 1.")
        if self.min_generations < 0:
            raise ValueError("nsga2.convergence.min_generations must not be negative.")

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "ConvergenceSettings":
        raw = raw or {}
        defaults = cls()
        return cls(enabled=bool(raw.get("enabled", defaults.enabled)),
                   tolerance=float(raw.get("tolerance", defaults.tolerance)),
                   period=int(raw.get("period", defaults.period)),
                   min_generations=int(raw.get("min_generations", defaults.min_generations)))


class _FrontMovement(MultiObjectiveSpaceTermination):
    """pymoo's front-movement criterion that also reports the movement it measured."""

    last_movement = math.inf

    def _update(self, algorithm):
        self.last_movement = math.inf  # stays infinite when there is nothing to compare
        return super()._update(algorithm)

    def _delta(self, prev, current):
        self.last_movement = float(super()._delta(prev, current))
        return self.last_movement


class NSGA2Termination(Termination):
    """Stop at ``max_generations`` or, if enabled, when the front has converged."""

    def __init__(self, max_generations: int, convergence: ConvergenceSettings | None = None,
                 on_generation: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__()
        self.on_generation = on_generation  # called with each trace row as soon as it is known
        if max_generations < 1:
            raise ValueError("nsga2.generations must be at least 1.")
        self.max_generations = int(max_generations)
        self.convergence = convergence or ConvergenceSettings()
        self.reason = ""
        self.streak = 0
        self.trace: list[dict[str, Any]] = []  # per generation: movement and current streak
        self._movement = _FrontMovement(tol=self.convergence.tolerance, only_feas=True)

    def _update(self, algorithm) -> float:
        generation = int(algorithm.n_gen)
        settings = self.convergence
        if settings.enabled:
            within = self._movement.update(algorithm) >= 1.0
            self.streak = self.streak + 1 if within else 0
            row = {"generation": generation, "front_movement": self._movement.last_movement,
                   "streak": self.streak}
            self.trace.append(row)
            if self.on_generation is not None:
                self.on_generation(row)
        if settings.enabled and generation >= settings.min_generations and self.streak >= settings.period:
            self.reason = CONVERGED
            return 1.0
        if generation >= self.max_generations:
            self.reason = MAX_GENERATIONS
            return 1.0
        return min(0.99, generation / self.max_generations)
