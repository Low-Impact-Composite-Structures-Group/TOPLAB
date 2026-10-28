"""Discharge-infeasible candidates must skip dormancy and packaging and be rejected via g_dis."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from toplab.optimization import evaluation
from toplab.optimization.evaluation import CandidateEvaluator, CandidateResult, candidate_row
from toplab.optimization.nsga2_optimizer import constraint_vector, objective_vector

MISSION_S = 3600.0


class _State:
    def get_tank_state(self, i):
        return SimpleNamespace(fuel_mass=10.0)


class _Orchestrator:
    actual_s = MISSION_S

    def __init__(self, config, verbosity="quiet"):
        self.tank_geometries = [SimpleNamespace(radius=0.5, phi=2.0)]
        self.tank_system = SimpleNamespace(
            config=SimpleNamespace(MISSION_DURATION=MISSION_S),
            _cached_tank_properties=[{"liner_mass": 1.0, "wall_mass": 4.0, "foam_mass": 1.0, "shell_mass": 1.0, "volume": 1.0,
                                      "outer_volume": 1.25, "outer_diameter": 1.0,
                                      "cylindrical_section_length": 1.0}])

    def run_simulation(self):
        return SimpleNamespace(times=[0.0, self.actual_s],
                               multi_tank_states=[_State(), _State()])


class _Candidate(SimpleNamespace):
    pass


@pytest.fixture
def calls(monkeypatch):
    log = {"dormancy": 0, "packaging": 0}
    candidate = SimpleNamespace(tanks=[SimpleNamespace(materials={"insulation": {"thickness": 0.01}})],
                                dormancy_check=None)

    def dormancy(case, silent=True):
        log["dormancy"] += 1
        return {"tanks": [{"initial_mass_kg": 10.0, "vented_mass_kg": 0.1}]}

    def packaging(radii, half_lengths, dims):
        log["packaging"] += 1
        return SimpleNamespace(feasible=True, z_shift=0.0, zeta_p=0.2, placements=[])

    monkeypatch.setattr(evaluation, "apply_design", lambda *a: candidate)
    monkeypatch.setattr(evaluation, "replace", lambda obj, **kw: obj)
    monkeypatch.setattr(evaluation, "maximize_packaging_compaction", packaging)
    monkeypatch.setattr("toplab.orchestration.system_orchestrator.SystemOrchestrator", _Orchestrator)
    monkeypatch.setattr("toplab.orchestration.run_analysis.run_dormancy_check", dormancy)
    return log


def _evaluator(skip):
    ev = object.__new__(CandidateEvaluator)
    ev.base_config, ev.variables = None, []
    ev.packaging_dimensions = None
    ev.maximum_vented_fraction, ev.hold_period_s = 0.05, 3600.0
    ev.progress, ev.evaluation_count, ev._stdout = False, 0, None
    ev.skip_after_discharge_failure = skip
    return ev


def test_discharge_failure_skips_dormancy_and_packaging(calls, monkeypatch):
    monkeypatch.setattr(_Orchestrator, "actual_s", 0.5 * MISSION_S)
    result = _evaluator(True).evaluate([0.5, 2.0])
    assert calls == {"dormancy": 0, "packaging": 0}
    assert result.discharge_failed and not result.feasible and result.error is None
    g = constraint_vector(result)
    assert g[0] > 0 and g[1:] == [0.0, 0.0]
    assert g[0] == pytest.approx(0.5)  # shortfall in hours
    assert objective_vector(result, ("eta_g", "eta_v", "zeta_p")) == [0.0, 0.0, 0.0]


def test_discharge_failure_row_marks_unavailable_quantities(calls, monkeypatch):
    monkeypatch.setattr(_Orchestrator, "actual_s", 0.5 * MISSION_S)
    result = _evaluator(True).evaluate([0.5, 2.0])
    row = candidate_row([], [], result)
    assert row["discharge_failed"] is True and row["feasible"] is False
    assert row["g_dis"] > 0 and row["eta_g"] > 0
    for name in ("zeta_p", "g_dorm", "g_packaging", "f_vent"):
        assert row[name] == ""


def test_successful_discharge_runs_all_analyses(calls):
    result = _evaluator(True).evaluate([0.5, 2.0])
    assert calls == {"dormancy": 1, "packaging": 1}
    assert not result.discharge_failed and result.feasible


def test_short_circuit_is_opt_in(calls, monkeypatch):
    monkeypatch.setattr(_Orchestrator, "actual_s", 0.5 * MISSION_S)
    result = _evaluator(False).evaluate([0.5, 2.0])
    assert calls == {"dormancy": 1, "packaging": 1}
    assert not result.discharge_failed


def test_infeasible_placeholder_cannot_dominate_feasible():
    bad = CandidateResult(discharge_failed=True, discharge_residual=60.0)
    good = CandidateResult(eta_g=0.1, eta_v=0.1, zeta_p=0.01, packaging_feasible=True,
                           mission_completed=True)
    cv = lambda r: sum(max(0.0, g) for g in constraint_vector(r))
    assert cv(bad) > 0 >= cv(good)
