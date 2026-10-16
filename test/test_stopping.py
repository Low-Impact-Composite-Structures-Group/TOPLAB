"""StoppingCriteria: events per tank, target density, and the missed-event fallback."""

from types import SimpleNamespace

import numpy as np

from toplab.solver.stopping import StoppingCriteria


def _criteria(target=None):
    return StoppingCriteria(volumes=[2.0], fluids=["Hydrogen"], is_gas=[False],
                            min_pressures=[0.0], min_densities=[5.0], target_density=target)


def test_events_are_one_per_tank_plus_target():
    assert len(_criteria().events()) == 1
    assert len(_criteria(target=60.0).events()) == 2


def test_min_density_event_crosses_downward():
    (event,) = _criteria().events()
    assert event.terminal and event.direction == -1
    assert event(0.0, np.array([20.0, 30, 30, 30, 30])) > 0     # 10 kg/m3
    assert event(0.0, np.array([8.0, 30, 30, 30, 30])) < 0      # 4 kg/m3


def test_fallback_truncates_at_first_violation_and_reports_reasons():
    c = _criteria()
    y = np.array([[20.0, 12.0, 8.0, 6.0], [30.0] * 4, [30.0] * 4, [30.0] * 4, [30.0] * 4])
    sol = SimpleNamespace(t=np.arange(4.0), y=y, edge_flows=np.zeros((0, 4)), extras=np.zeros((0, 4)), t_events=[np.array([])])
    assert "Tank 1" in c.truncate_if_missed(sol)
    assert sol.t.size == 3 and sol.y.shape[1] == 3
    assert c.stop_reasons(SimpleNamespace(t_events=[np.array([42.0])]))[0].startswith("Tank 1 reached")
