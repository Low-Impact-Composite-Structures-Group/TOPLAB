"""Hybrid integrator: mode switches fire as events and the right-hand side stays pure."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from toplab.coupling.base import ModeSwitch
from toplab.coupling.pressure_triggered_valve import PressureTriggeredValve
from toplab.fluids.valve_flow import ValvePath
from toplab.solver.hybrid import integrate_with_mode_switches


class _Solver:
    def __init__(self, rhs):
        self.rhs = rhs

    def integrate_full(self, t_span, y0, events, **kw):
        return solve_ivp(self.rhs, t_span, y0, events=events, rtol=1e-8, atol=1e-10)


class _Thermostat:
    """Heater on below 1.0, off above 2.0 (state y[0] is the temperature)."""
    n_states = 0

    def __init__(self):
        self.on = True

    def mode_switches(self):
        if self.on:
            return [ModeSwitch(0, 2.0, +1, lambda: setattr(self, "on", False), "off")]
        return [ModeSwitch(0, 1.0, -1, lambda: setattr(self, "on", True), "on")]


def _event(switch):
    def event(t, y):
        return y[0] - switch.pressure_pa
    event.terminal, event.direction = True, switch.direction
    return event


def test_thermostat_cycles_between_thresholds():
    edge = _Thermostat()
    rhs = lambda t, y: [1.0 if edge.on else -1.0]
    sol = integrate_with_mode_switches(
        _Solver(rhs), [edge], [], (0.0, 10.0), np.array([1.5]),
        switch_event=_event,
        edge_flows=lambda t, y: [1.0 if edge.on else 0.0],
    )
    assert sol.y[0].min() >= 1.0 - 1e-6 and sol.y[0].max() <= 2.0 + 1e-6
    assert sol.t[-1] == pytest.approx(10.0)
    assert np.all(np.diff(sol.t) > 0)          # no duplicated restart points
    assert sol.edge_flows.shape == (1, sol.t.size)


def test_switch_event_attributes_come_from_the_caller():
    edge = _Thermostat()
    (switch,) = edge.mode_switches()
    assert (switch.tank_idx, switch.pressure_pa, switch.direction) == (0, 2.0, 1)


def test_valve_modes_and_states():
    valve = PressureTriggeredValve(0, 1, p_open=16e5, p_close=30e5, path=ValvePath(orifice_diameter_m=0.005),
                                   valve_time_constant_s=0.5)
    assert valve.n_states == 1
    valve.is_active = False
    (opens,) = valve.mode_switches()
    assert (opens.tank_idx, opens.pressure_pa, opens.direction) == (1, 16e5, -1)
    opens.apply()
    (closes,) = valve.mode_switches()
    assert (closes.pressure_pa, closes.direction) == (30e5, +1)
    # The opening relaxes toward the commanded position
    assert valve.state_derivatives(0.0, [], [0.0]) == [pytest.approx(2.0)]

    instant = PressureTriggeredValve(0, 1, 16e5, 30e5, ValvePath(orifice_diameter_m=0.005), valve_time_constant_s=0.0)
    assert instant.n_states == 0
