"""Hybrid integration: continuous ODE segments separated by discrete edge-mode switches.

Edges change mode (valve open/closed, ...) only when a terminal event fires. The integration is then
restarted from the event with the new mode, so the right-hand side stays a pure function of (t, y).
"""

from types import SimpleNamespace
from typing import Callable, List

import numpy as np

MAX_SWITCHES = 100_000


def integrate_with_mode_switches(
    solver,
    edges: List,
    stop_events: List[Callable],
    t_span: tuple,
    y0: np.ndarray,
    switch_event: Callable,
    edge_flows: Callable,
    extras: Callable = None,
    log: Callable[[str], None] = lambda msg: None,
    **solver_kwargs,
):
    """Integrate until the end of ``t_span`` or until a stop event fires.

    ``switch_event(mode_switch)`` builds a terminal event for an edge mode switch and
    ``edge_flows(t, y)`` returns the per-edge flows [kg/s] at an accepted point (evaluated with the
    mode that was active during the segment). Returns an object with ``t``, ``y``, ``edge_flows``
    (n_edges x n_points), ``extras`` (n_rows x n_points, from the optional
    ``extras(t, y)`` callback evaluated at the same points) and ``t_events`` (one array per stop event).
    """
    t0, t_end = float(t_span[0]), float(t_span[1])
    n_stop = len(stop_events)
    stop_times = [[] for _ in stop_events]
    times, states, flows, extra_rows = [], [], [], []

    for _ in range(MAX_SWITCHES):
        switches = [s for edge in edges for s in edge.mode_switches()]
        events = list(stop_events) + [switch_event(s) for s in switches]
        sol = solver.integrate_full(t_span=(t0, t_end), y0=y0, events=events, **solver_kwargs)

        # A restart repeats the switch point, which the previous segment already recorded.
        keep = slice(1, None) if times else slice(None)
        seg_flows = np.array([edge_flows(sol.t[i], sol.y[:, i]) for i in range(sol.t.size)]).reshape(sol.t.size, len(edges)).T
        seg_extras = (np.array([extras(sol.t[i], sol.y[:, i]) for i in range(sol.t.size)]).reshape(sol.t.size, -1).T
                      if extras else np.zeros((0, sol.t.size)))
        extra_rows.append(seg_extras[:, keep])
        times.append(sol.t[keep])
        states.append(sol.y[:, keep])
        flows.append(seg_flows[:, keep])

        stopped = False
        for k in range(n_stop):
            te = sol.t_events[k]
            stop_times[k].extend(te)
            stopped = stopped or len(te) > 0
        fired = [s for k, s in enumerate(switches) if len(sol.t_events[n_stop + k]) > 0]
        if stopped or not fired or sol.t[-1] >= t_end - 1e-9:
            break

        for s in fired:
            s.apply()
            log(f"   t={sol.t[-1] / 3600:.3f} h: {s.label}")
        t0, y0 = float(sol.t[-1]), sol.y[:, -1].copy()
    else:
        raise RuntimeError(f"More than {MAX_SWITCHES} edge mode switches; the valve logic is chattering.")

    return SimpleNamespace(
        t=np.concatenate(times),
        y=np.concatenate(states, axis=1),
        edge_flows=np.concatenate(flows, axis=1),
        extras=np.concatenate(extra_rows, axis=1),
        t_events=[np.asarray(t) for t in stop_times],
    )
