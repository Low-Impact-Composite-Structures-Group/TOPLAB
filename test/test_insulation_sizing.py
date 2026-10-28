"""Discrete insulation sizing and its integration in the NSGA-II candidate evaluation."""

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from toplab.optimization import evaluation
from toplab.optimization.evaluation import (
    CandidateEvaluator, CandidateResult, candidate_row, load_problem, parse_design_variables)
from toplab.optimization.insulation_sizing import (
    AT_MINIMUM, BRACKETED, INFEASIBLE, DormancyPoint, InsulationGrid, check_monotonic, max_evaluations, size_insulation)
from toplab.optimization.nsga2_optimizer import NSGA2Optimizer, history_rows

NSGA_YAML = Path(__file__).parent.parent / "examples/optimization_two_tank_cch2/two_tank_cch2_nsga.yaml"
GRID = InsulationGrid(0.0, 0.050, 0.0025)
LIMIT = 0.05


def vent_model(t):
    """Venting fraction that decreases monotonically with thickness (infinite for a bare tank)."""
    return math.inf if t <= 0.0 else 0.001 / t


class Recorder:
    def __init__(self, model=vent_model, limit=LIMIT):
        self.model, self.limit, self.thicknesses = model, limit, []

    def __call__(self, t):
        self.thicknesses.append(t)
        f = self.model(t)
        return DormancyPoint(f / self.limit - 1.0, f)


# --- grid -------------------------------------------------------------------------------------

def test_grid_has_19_values_for_5_to_50_mm_and_21_from_zero():
    five = InsulationGrid(0.005, 0.050, 0.0025)
    assert five.size == 19 and five.values[0] == 0.005 and five.values[-1] == 0.05
    assert GRID.size == 21 and GRID.thickness(5) == 0.0125


def test_grid_resolution_is_configurable_and_validated():
    assert InsulationGrid(0.0, 0.05, 0.005).size == 11
    assert InsulationGrid.from_config({"t_min": 0.01, "t_max": 0.02}).resolution == 0.0025
    with pytest.raises(ValueError, match="multiple"):
        InsulationGrid(0.0, 0.05, 0.003)
    with pytest.raises(ValueError):
        InsulationGrid(0.0, 0.05, 0.0)
    with pytest.raises(ValueError, match="t_min"):
        InsulationGrid.from_config({"t_max": 0.05})


# --- sizing algorithm -------------------------------------------------------------------------

def test_dormancy_met_at_minimum_returns_minimum_with_one_simulation():
    grid = InsulationGrid(0.005, 0.050, 0.0025)
    recorder = Recorder(lambda t: 0.0)
    result = size_insulation(recorder, grid)
    assert result.status == AT_MINIMUM and result.t_ins == 0.005
    assert result.n_evaluations == 1 and recorder.thicknesses == [0.005]


def test_bracketed_solution_returns_smallest_feasible_grid_value():
    recorder = Recorder()
    result = size_insulation(recorder, GRID)
    assert result.status == BRACKETED and result.feasible
    assert result.t_ins == 0.02  # 0.0175 gives 5.7 % > 5 %, 0.02 gives exactly 5 %
    assert result.g_dorm <= 0.0 and result.g_dorm == pytest.approx(0.0)
    assert result.t_ins in GRID.values
    assert recorder.model(result.t_ins - GRID.resolution) / LIMIT - 1.0 > 0.0
    assert result.f_vent == pytest.approx(0.05)


def test_example_from_the_specification_selects_15mm_not_something_between():
    result = size_insulation(Recorder(lambda t: 0.04 if t >= 0.015 else 0.06), GRID)
    assert result.t_ins == 0.015


@pytest.mark.parametrize("threshold", range(1, 21))
def test_binary_search_matches_exhaustive_search_within_bound(threshold):
    values = GRID.values
    recorder = Recorder(lambda t, th=values[threshold]: 0.04 if t >= th else 0.06)
    result = size_insulation(recorder, GRID)
    expected = next(t for t in values if recorder.model(t) <= LIMIT)
    assert result.t_ins == expected
    assert result.n_evaluations == len(set(recorder.thicknesses)) <= max_evaluations(GRID)
    assert result.n_evaluations < GRID.size


def test_evaluation_bound_is_logarithmic():
    assert max_evaluations(GRID) == 7
    assert max_evaluations(InsulationGrid(0.005, 0.05, 0.0025)) == 7
    assert max_evaluations(InsulationGrid(0.0, 0.0, 0.0025)) == 1


def test_violation_at_maximum_is_dormancy_infeasible():
    recorder = Recorder(lambda t: 0.5)
    result = size_insulation(recorder, GRID)
    assert result.status == INFEASIBLE and not result.feasible
    assert result.t_ins == GRID.t_max and result.g_dorm > 0.0
    assert result.n_evaluations == 2 and recorder.thicknesses == [0.0, 0.05]


def test_check_monotonic_detects_non_monotonic_models():
    ok, points = check_monotonic(Recorder(), GRID)
    assert ok and len(points) == GRID.size
    # Feasible at 12.5 mm but violating at 20 mm: a binary search could not see this
    bad, _ = check_monotonic(Recorder(lambda t: 0.04 if t in (0.0125, 0.05) else 0.06), GRID)
    assert not bad


# --- evaluator integration (stubbed analyses) -------------------------------------------------

MISSION_S = 3600.0


class _State:
    def get_tank_state(self, i):
        return SimpleNamespace(fuel_mass=10.0)


def _thickness(config):
    return config.tanks[0].materials["insulation"]["thickness"]


class _Orchestrator:
    runs = []
    actual_s = MISSION_S

    def __init__(self, config, verbosity="quiet"):
        t = _thickness(config)
        self.t = t
        self.tank_geometries = [SimpleNamespace(radius=0.5, phi=2.0)]
        self.tank_system = SimpleNamespace(
            config=SimpleNamespace(MISSION_DURATION=MISSION_S),
            _cached_tank_properties=[{"liner_mass": 1.0, "wall_mass": 4.0, "foam_mass": 50.0 * t,
                                      "shell_mass": 2.0 + 10.0 * t, "volume": 1.0,
                                      "outer_volume": 1.0 + 10.0 * t, "outer_diameter": 1.0 + 2.0 * t,
                                      "cylindrical_section_length": 1.0}])

    def run_simulation(self):
        _Orchestrator.runs.append(self.t)
        return SimpleNamespace(times=[0.0, _Orchestrator.actual_s], multi_tank_states=[_State(), _State()])


@pytest.fixture
def stubs(monkeypatch):
    log = SimpleNamespace(dormancy=[], packaging=[], discharge=_Orchestrator.runs)
    _Orchestrator.runs.clear()
    _Orchestrator.actual_s = MISSION_S

    def dormancy(case, silent=True):
        t = _thickness(case)
        log.dormancy.append(t)
        f = vent_model(t)
        return {"tanks": [{"initial_mass_kg": 10.0, "vented_mass_kg": 10.0 * min(f, 1.0)}]}

    def packaging(radii, half_lengths, dims):
        log.packaging.append(list(radii))
        return SimpleNamespace(feasible=True, z_shift=0.0, zeta_p=0.2, placements=[])

    base = SimpleNamespace(tanks=[SimpleNamespace(materials={"insulation": {"thickness": 0.026}})],
                           dormancy_check=None)
    monkeypatch.setattr(evaluation, "apply_design", lambda *a: base)
    monkeypatch.setattr(evaluation, "replace", lambda obj, **kw: SimpleNamespace(**{**vars(obj), **kw}))
    monkeypatch.setattr(evaluation, "maximize_packaging_compaction", packaging)
    monkeypatch.setattr("toplab.orchestration.system_orchestrator.SystemOrchestrator", _Orchestrator)
    monkeypatch.setattr("toplab.orchestration.run_analysis.run_dormancy_check", dormancy)
    return log


def _evaluator(grid=GRID, skip=True):
    ev = object.__new__(CandidateEvaluator)
    ev.base_config, ev.variables = None, []
    ev.packaging_dimensions = None
    ev.maximum_vented_fraction, ev.hold_period_s = LIMIT, 3600.0
    ev.progress, ev.evaluation_count, ev._stdout = False, 0, None
    ev.skip_after_discharge_failure, ev.insulation_grid = skip, grid
    return ev


def test_sized_thickness_is_used_for_eta_g_eta_v_and_packaging(stubs):
    result = _evaluator().evaluate([0.5, 2.0])
    t = 0.02
    assert result.error is None and result.t_ins_star == t and result.sizing_status == BRACKETED
    assert result.designs[0].insulation_thickness == t
    # eta_g / eta_v come from the discharge run at t*, packaging gets the outer radius at t*
    assert stubs.discharge[-1] == t
    structure = 1.0 + 4.0 + 50.0 * t + 2.0 + 10.0 * t  # liner + wall + insulation + shell
    assert result.eta_g == pytest.approx(10.0 / (10.0 + structure))
    assert result.eta_v == pytest.approx(1.0 / (1.0 + 10.0 * t))
    assert stubs.packaging == [[pytest.approx((1.0 + 2.0 * t) / 2.0)]]
    assert result.f_vent == pytest.approx(0.05) and result.dormancy_residual <= 0.0
    assert result.n_dormancy_evals == len(stubs.dormancy) and result.feasible


def test_dormancy_simulations_stay_within_the_logarithmic_bound(stubs):
    result = _evaluator().evaluate([0.5, 2.0])
    assert 2 <= result.n_dormancy_evals <= max_evaluations(GRID)
    assert len(set(stubs.dormancy)) == len(stubs.dormancy)  # no thickness is simulated twice
    assert 0.0 not in stubs.dormancy


def test_minimum_thickness_design_needs_one_dormancy_and_one_discharge(stubs):
    grid = InsulationGrid(0.04, 0.05, 0.0025)  # already feasible at the thinnest value
    result = _evaluator(grid).evaluate([0.5, 2.0])
    assert result.t_ins_star == 0.04 and result.sizing_status == AT_MINIMUM
    assert result.n_dormancy_evals == 1 and stubs.dormancy == [0.04]
    assert stubs.discharge == [0.04]  # screening discharge reused, not repeated


def test_dormancy_infeasible_design_is_rejected_by_constraint_not_objectives(stubs):
    result = _evaluator(InsulationGrid(0.0025, 0.005, 0.0025)).evaluate([0.5, 2.0])
    assert result.sizing_status == INFEASIBLE and result.dormancy_residual > 0.0
    assert result.t_ins_star == 0.005 and not result.feasible and not result.discharge_failed
    assert result.designs[0].insulation_thickness == 0.005 and len(stubs.packaging) == 1
    assert evaluation.physical_residuals(result)["g_dorm"] > 0.0


def test_discharge_infeasible_candidate_runs_no_dormancy_and_no_packaging(stubs):
    _Orchestrator.actual_s = 0.5 * MISSION_S
    result = _evaluator().evaluate([0.5, 2.0])
    assert stubs.dormancy == [] and stubs.packaging == []
    assert result.discharge_failed and result.n_dormancy_evals == 0
    assert evaluation.physical_residuals(result)["g_dis"] > 0.0


def test_thickness_zero_is_never_simulated_but_counts_as_violating(stubs):
    _evaluator().evaluate([0.5, 2.0])
    assert 0.0 not in stubs.discharge and 0.0 not in stubs.dormancy
    assert stubs.discharge[0] == 0.0025  # screening at the thinnest simulable value


def test_history_row_records_sizing_columns(stubs):
    result = _evaluator().evaluate([0.5, 2.0])
    row = candidate_row(parse_design_variables({"radius": {"min": .1, "max": 1, "baseline": .3}}), [0.5], result)
    assert row["t_ins_star"] == 0.02 and row["sizing_status"] == BRACKETED
    assert row["n_dormancy_evals"] == result.n_dormancy_evals == len(stubs.dormancy) and row["f_vent"] == pytest.approx(0.05)
    assert row["g_dorm"] == pytest.approx(0.0) and "g_dis" in row and "g_packaging" in row


# --- NSGA-II problem definition ---------------------------------------------------------------

def test_insulation_thickness_is_not_an_nsga2_decision_variable():
    optimizer = NSGA2Optimizer(NSGA_YAML, evaluator=object())
    assert all(v.parameter != "insulation_thickness" for v in optimizer.variables)
    assert optimizer.problem.cfg["insulation_sizing"]["resolution"] == 0.0025

    bad = NSGA_YAML.parent / "_bad_tins_nsga.yaml"
    bad.write_text(NSGA_YAML.read_text().replace(
        "design_variables:\n", "design_variables:\n  t_ins:\n    min: 0.005\n    max: 0.05\n    baseline: 0.02\n", 1))
    try:
        with pytest.raises(ValueError, match="sized internally"):
            NSGA2Optimizer(bad, evaluator=object())
    finally:
        bad.unlink()


def test_sizing_section_is_required_for_nsga2():
    text = NSGA_YAML.read_text()
    start = text.index("insulation_sizing:")
    stripped = NSGA_YAML.parent / "_nosizing_nsga.yaml"
    stripped.write_text(text[:start] + text[text.index("dormancy:"):])
    try:
        with pytest.raises(ValueError, match="insulation_sizing"):
            NSGA2Optimizer(stripped, evaluator=object())
    finally:
        stripped.unlink()


def test_sqp_loading_keeps_insulation_as_a_variable_and_needs_no_sizing():
    problem = load_problem(NSGA_YAML, evaluator=object())
    assert problem.evaluator is not None


class _SizedFake:
    """Evaluator double: records one call per candidate and reports sizing."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        r = CandidateResult(eta_g=0.5 + 0.1 * x[0], eta_v=0.4 + 0.1 * x[1], zeta_p=0.1,
                            packaging_feasible=True, mission_completed=True,
                            t_ins_star=0.0025 * (1 + int(x[0] * 100) % 20), sizing_status=BRACKETED,
                            n_dormancy_evals=5)
        return r


def test_history_has_one_row_per_evaluation_and_records_t_ins_star():
    fake = _SizedFake()
    optimizer = NSGA2Optimizer(NSGA_YAML, evaluator=fake)
    optimizer.settings = dataclasses.replace(optimizer.settings, population_size=8, generations=3, seed=1)
    result = optimizer.run()
    rows = history_rows(result)
    assert len(rows) == fake.calls == len(result.history)
    assert [r["evaluation"] for r in rows] == list(range(1, fake.calls + 1))
    assert all("t_ins_star" in r and r["t_ins_star"] in {0.0025 * k for k in range(1, 21)} for r in rows)
    assert all(not any(k.startswith("insulation_thickness") for k in r) for r in rows)


def test_eta_g_decreases_with_insulation_thickness(stubs):
    thin = _evaluator(InsulationGrid(0.0025, 0.0025, 0.0025)).evaluate([0.5, 2.0])
    thick = _evaluator(InsulationGrid(0.04, 0.04, 0.0025)).evaluate([0.5, 2.0])
    assert thick.eta_g < thin.eta_g
    assert thick.dry_masses_kg[0] == pytest.approx(
        thick.structural_masses_kg[0] + thick.insulation_masses_kg[0] + thick.shell_masses_kg[0])


# --- history layout -----------------------------------------------------------------------------

def test_history_columns_are_grouped_and_unavailable_values_are_blank():
    from toplab.optimization.nsga2_optimizer import candidate_status, nsga2_row

    variables = parse_design_variables({"radius": {"min": .1, "max": 1, "baseline": .3},
                                        "phi": {"min": 1, "max": 4, "baseline": 3}})
    ok = CandidateResult(eta_g=0.4, eta_v=0.8, zeta_p=0.5, f_vent=0.01, eta_vent=0.5, t_ins_star=0.02,
                         sizing_status=BRACKETED, n_dormancy_evals=5, dormancy_residual=-0.1,
                         packaging_feasible=True, mission_completed=True, eval_id=3)
    keys = list(nsga2_row(variables, [0.5, 2.0], ok))
    groups = [["evaluation"], ["radius_1", "phi_1"], ["t_ins_star"], ["eta_g", "eta_v", "zeta_p"],
              ["eta_vent", "f_vent", "n_dormancy_evals", "sizing_status"],
              ["g_dis", "g_dorm", "g_packaging"], ["status", "feasible"]]
    positions = [[keys.index(k) for k in g] for g in groups]
    assert all(p == sorted(p) for p in positions)
    assert [max(p) for p in positions[:-1]] < [min(p) for p in positions[1:]]
    assert nsga2_row(variables, [0.5, 2.0], ok)["status"] == "feasible"

    failed = CandidateResult(discharge_failed=True, discharge_residual=60.0, eta_g=0.3, eta_v=0.7, eval_id=4)
    row = nsga2_row(variables, [0.1, 2.0], failed)
    assert row["status"] == "discharge_infeasible" == candidate_status(failed)
    assert row["zeta_p"] == row["g_dorm"] == row["g_packaging"] == row["f_vent"] == ""
    assert candidate_status(CandidateResult(error="x")) == "error"
    assert candidate_status(dataclasses.replace(ok, dormancy_residual=0.3)) == "dormancy_infeasible"
    assert candidate_status(dataclasses.replace(ok, packaging_feasible=False)) == "packaging_infeasible"
