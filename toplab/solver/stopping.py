"""Stopping criteria: per-tank minimum pressure/density and an optional target density."""

from typing import List, Optional, Sequence

import numpy as np
from CoolProp.CoolProp import PropsSI

from toplab.fluids.coolprop_safe import safe_pressure_from_T_rho

STATES_PER_TANK = 5   # [m, T_H2, T_struct, T_ins, T_shell]; mass and gas temperature come first


class StoppingCriteria:
    """Terminal events for a tank system, plus a fallback check and a stop-reason report.

    CH2 tanks stop at their minimum pressure, all other tanks at their minimum density.
    Events are ordered: one per tank, then the target-density event if configured.
    """

    def __init__(self, volumes: Sequence[float], fluids: Sequence[str], is_gas: Sequence[bool],
                 min_pressures: Sequence[float], min_densities: Sequence[float],
                 target_density: Optional[float] = None):
        self.volumes = list(volumes)
        self.fluids = list(fluids)
        self.is_gas = list(is_gas)
        self.min_pressures = list(min_pressures)
        self.min_densities = list(min_densities)
        self.target_density = target_density
        self.n_tanks = len(self.volumes)

    def _pressure(self, i: int, y) -> float:
        rho = float(y[i * STATES_PER_TANK]) / max(self.volumes[i], 1e-9)
        return PropsSI("P", "T", max(float(y[i * STATES_PER_TANK + 1]), 14.0), "Dmass", max(rho, 1e-9), self.fluids[i])

    def _margin(self, i: int, y) -> float:
        """Positive while tank ``i`` is above its minimum; raises on property failure."""
        if self.is_gas[i]:
            return self._pressure(i, y) - self.min_pressures[i]
        return float(y[i * STATES_PER_TANK]) / self.volumes[i] - self.min_densities[i]

    def _min_event(self, i: int):
        def event(t, y):
            try:
                return self._margin(i, y)
            except Exception:
                return 1.0   # a CoolProp failure must not falsely stop the run
        event.terminal = True
        event.direction = -1
        return event

    def _target_event(self):
        def event(t, y):
            for i in range(self.n_tanks):
                m = y[i * STATES_PER_TANK]
                if m > 0 and m / self.volumes[i] >= self.target_density:
                    return m / self.volumes[i] - self.target_density
            return -1.0
        event.terminal = True
        event.direction = 1
        return event

    def events(self) -> List:
        events = [self._min_event(i) for i in range(self.n_tanks)]
        if self.target_density is not None:
            events.append(self._target_event())
        return events

    def truncate_if_missed(self, solution) -> Optional[str]:
        """If no minimum event fired but a stored point violates one, cut the solution there.

        Returns a warning message when the solution was truncated.
        """
        if any(len(te) > 0 for te in solution.t_events[:self.n_tanks]):
            return None
        for step in range(solution.t.size):
            for i in range(self.n_tanks):
                try:
                    violated = self._margin(i, solution.y[:, step]) < 0.0
                except Exception:
                    continue
                if violated:
                    solution.t = solution.t[:step + 1]
                    solution.y = solution.y[:, :step + 1]
                    solution.edge_flows = solution.edge_flows[:, :step + 1]
                    solution.extras = solution.extras[:, :step + 1]
                    t = solution.t[-1]
                    return (f"stopping event was not caught by solver; truncating solution at "
                            f"Tank {i + 1} stopping criterion (t={t:.1f}s, {t / 3600:.3f}h).")
        return None

    def stop_reasons(self, solution) -> List[str]:
        """Human-readable reasons the run ended early (empty if the mission completed)."""
        reasons = []
        for i in range(self.n_tanks):
            if len(solution.t_events[i]) > 0:
                t = solution.t_events[i][0]
                reasons.append(f"Tank {i + 1} reached its minimum before completing the mission "
                               f"(t={t:.1f}s, {t / 3600:.3f}h).")
        if self.target_density is not None and len(solution.t_events[self.n_tanks]) > 0:
            t = solution.t_events[self.n_tanks][0]
            reasons.append(f"Target density reached at t={t:.1f}s ({t / 3600:.3f}h).")
        return reasons
