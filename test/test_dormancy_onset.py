"""Dormancy onset and coupled insulation sizing regression tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from toplab.config import load_config
from toplab.config.schema import EdgeSpec
from toplab.optimization.evaluation import CandidateEvaluator, CandidateResult, candidate_row, dormancy_limits
from toplab.optimization.insulation_sizing import DormancyPoint, InsulationGrid, size_tank_insulations
from toplab.orchestration.run_analysis import run_dormancy_check


CONFIG = (Path(__file__).resolve().parents[2] /
          "toplab-apps/analysis/novacom/cch2_two_tanks/config_cch2_two_tanks.yaml")


def _point(values, residuals):
    return DormancyPoint(max(residuals), 0.0, {"residuals": residuals})


def test_heterogeneous_tanks_size_independently():
    grid = InsulationGrid(0.01, 0.05, 0.01)
    result = size_tank_insulations(
        lambda t: _point(t, [0.02 - t[0], 0.04 - t[1]]), grid, 2)
    assert result.status == "converged"
    assert result.thicknesses == (0.02, 0.04)
    assert result.point.g_dorm <= 0


def test_coupled_sizing_preserves_full_system_feasibility():
    grid = InsulationGrid(0.01, 0.05, 0.01)
    # Tank 2 needs additional insulation when tank 1 is thin.
    def evaluate(t):
        return _point(t, [0.02 - t[0], 0.06 - t[0] - t[1]])
    result = size_tank_insulations(evaluate, grid, 2)
    assert result.status == "converged"
    assert result.point.g_dorm <= 1e-12
    for i, t in enumerate(result.thicknesses):
        if t > grid.t_min:
            trial = list(result.thicknesses)
            trial[i] = round(t - grid.resolution, 12)
            assert evaluate(trial).g_dorm > 0


def test_infeasible_sizing_is_not_claimed_as_feasible():
    grid = InsulationGrid(0.01, 0.05, 0.01)
    result = size_tank_insulations(lambda t: _point(t, [0.06 - t[0], 0.02 - t[1]]), grid, 2)
    assert result.status == "dormancy_infeasible"
    assert result.point.g_dorm > 0


def test_zero_limit_overrides_legacy_base_and_missing_limit_defaults_to_zero():
    base = type("Base", (), {"dormancy_check": {"duration_h": 2, "maximum_loss_percent": 1}})()
    assert dormancy_limits(base, {"dormancy": {"maximum_vented_percentage": 0}}) == (7200, 0)
    base.dormancy_check = {"duration_h": 2}
    assert dormancy_limits(base, {}) == (7200, 0)


def test_real_dormancy_tracks_initial_onset_and_censored_tank():
    config = load_config(CONFIG)
    first, second = config.tanks
    first = replace(first, operating_limits={**first.operating_limits, "venting_pressure": 39.9e6})
    config = replace(config, tanks=[first, second],
                     dormancy_check={"enabled": True, "duration_h": 0.01})
    result = run_dormancy_check(config, silent=True)
    assert not result["passed"]
    assert result["tanks"][0]["vent_onset_s"] == 0.0
    assert result["tanks"][0]["vented_mass_kg"] > 0
    assert result["tanks"][1]["vent_onset_s"] is None
    assert result["tanks"][1]["vent_onset_censored"]
    assert result["tanks"][1]["passed"]


def test_early_stopping_is_not_a_pass():
    config = load_config(CONFIG)
    config = replace(config, solver={**config.solver, "max_simulation_time": 1},
                     dormancy_check={"enabled": True, "duration_h": 0.01})
    with pytest.raises(RuntimeError, match="stopped early"):
        run_dormancy_check(config, silent=True)


def test_early_stop_is_a_failure_when_every_tank_already_vented():
    config = load_config(CONFIG)
    tanks = [replace(tank, operating_limits={**tank.operating_limits, "venting_pressure": 39.9e6})
             for tank in config.tanks]
    config = replace(config, tanks=tanks, solver={**config.solver, "max_simulation_time": 1},
                     dormancy_check={"enabled": True, "duration_h": 0.01})
    result = run_dormancy_check(config, silent=True)
    assert not result["complete"] and not result["passed"]
    assert all(not tank["passed"] and tank["loss_is_lower_bound"] for tank in result["tanks"])


def test_real_dormancy_detects_crossing_and_legacy_limit():
    config = load_config(CONFIG)
    first, second = config.tanks
    first = replace(first, operating_limits={**first.operating_limits, "venting_pressure": 40000100.0})
    config = replace(config, tanks=[first, second],
                     dormancy_check={"enabled": True, "duration_h": 0.1})
    result = run_dormancy_check(config, silent=True)
    tank = result["tanks"][0]
    assert 0 < tank["vent_onset_s"] < 360
    assert tank["vented_mass_kg"] > 0
    assert not tank["passed"]
    legacy = replace(config, dormancy_check={**config.dormancy_check, "maximum_loss_percent": 1})
    assert run_dormancy_check(legacy, silent=True)["passed"]


def test_initial_pressure_equal_to_vent_threshold_is_zero_dormancy():
    config = load_config(CONFIG)
    first, second = config.tanks
    first = replace(first, operating_limits={**first.operating_limits, "venting_pressure": 40e6})
    config = replace(config, tanks=[first, second],
                     dormancy_check={"enabled": True, "duration_h": 0.001})
    result = run_dormancy_check(config, silent=True)
    assert result["tanks"][0]["vent_onset_s"] == 0
    assert not result["tanks"][0]["passed"]


def test_real_coupling_mass_transfer_is_not_counted_as_venting():
    config = load_config(CONFIG)
    first, second = config.tanks
    second = replace(second, initial_conditions={**second.initial_conditions, "pressure": 35e6})
    edge = EdgeSpec("transfer", "pressure_compensation", 1, 2, options={
        "activation_conditions": {"pressure_open_bar": 360, "pressure_close_bar": 390},
        "flow_parameters": {"orifice_diameter_m": 0.001},
    })
    config = replace(config, tanks=[first, second], edges=[*config.edges, edge],
                     dormancy_check={"enabled": True, "duration_h": 0.001,
                                     "maximum_loss_percent": 0.001})
    result = run_dormancy_check(config, silent=True)
    assert result["passed"]
    assert result["tanks"][0]["initial_mass_kg"] > result["tanks"][0]["final_mass_kg"]
    assert result["tanks"][1]["initial_mass_kg"] < result["tanks"][1]["final_mass_kg"]
    assert result["tanks"][0]["vented_mass_kg"] == pytest.approx(0.0, abs=1e-10)


def test_sized_vector_is_used_by_final_discharge_packaging_and_history(monkeypatch):
    config = load_config(CONFIG)
    evaluator = object.__new__(CandidateEvaluator)
    evaluator.insulation_grid = InsulationGrid(0.01, 0.05, 0.01)
    evaluator.maximum_vented_fraction = 0.0
    evaluator.skip_after_discharge_failure = True
    evaluator.progress = False
    calls = []

    def thicknesses(case):
        return tuple(t.materials["insulation"]["thickness"] for t in case.tanks)

    def discharge(case, result):
        calls.append(("discharge", thicknesses(case)))
        result.mission_completed = True
        return case

    def dormancy(case):
        t = thicknesses(case)
        residuals = [0.02 - t[0], 0.04 - t[1]]
        return DormancyPoint(max(residuals), 0.0, {
            "residuals": residuals, "vented_mass_kg": 0.0,
            "per_tank_vented_fraction": [0.0, 0.0], "vent_onset_s": [None, None]})

    monkeypatch.setattr(evaluator, "_run_discharge", discharge)
    monkeypatch.setattr(evaluator, "_run_dormancy", dormancy)
    monkeypatch.setattr(evaluator, "_run_packaging",
                        lambda case, result: calls.append(("packaging", thicknesses(case))))
    monkeypatch.setattr(evaluator, "_designs", lambda case, orchestrator: [])
    result = CandidateResult()
    evaluator._evaluate_sized(config, result)
    assert result.t_ins_stars == [0.02, 0.04]
    assert calls[-2:] == [("discharge", (0.02, 0.04)), ("packaging", (0.02, 0.04))]
    row = candidate_row([], [], result)
    assert row["t_ins_star_1"] == 0.02 and row["t_ins_star_2"] == 0.04
    assert row["vent_onset_censored_1"] and row["vent_onset_s_1"] == ""


def test_fixed_step_solver_is_rejected_for_dormancy():
    config = load_config(CONFIG)
    config = replace(config, solver={**config.solver, "method": "RK4_FIXED"},
                     dormancy_check={"enabled": True, "duration_h": 1})
    with pytest.raises(ValueError, match="root-located"):
        run_dormancy_check(config, silent=True)
