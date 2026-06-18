from __future__ import annotations

import pytest

from src.configuration.scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator
from test.peripheral_component_test_utils import (
    run_component_benchmark,
    write_yaml_smoke_config,
)


@pytest.mark.integration
@pytest.mark.coupling
def test_compressor_component_matches_constant_flow_steady_state():
    benchmark = run_component_benchmark("compressor")

    assert benchmark["component_name"] == benchmark["expected_component_name"]
    assert benchmark["coupling_flow_history"]
    assert benchmark["final_source_mass"] < benchmark["initial_source_mass"]
    assert benchmark["final_target_mass"] == pytest.approx(benchmark["initial_target_mass"], rel=5e-3)
    # Enthalpy is sensitive to EOS/integration details; keep this check strict enough
    # to catch regressions without flaking across minor numerical implementation drift.
    assert benchmark["target_enthalpy"] == pytest.approx(benchmark["expected_enthalpy"], rel=1.5e-2)
    assert benchmark["target_temperature"] == pytest.approx(benchmark["expected_temperature"], abs=3.5)


@pytest.mark.integration
@pytest.mark.coupling
def test_heat_exchanger_component_matches_constant_flow_steady_state():
    benchmark = run_component_benchmark("ideal_heat_exchanger")

    assert benchmark["component_name"] == benchmark["expected_component_name"]
    assert benchmark["coupling_flow_history"]
    assert benchmark["final_source_mass"] < benchmark["initial_source_mass"]
    assert benchmark["final_target_mass"] == pytest.approx(benchmark["initial_target_mass"], rel=5e-3)
    # Enthalpy is sensitive to EOS/integration details; keep this check strict enough
    # to catch regressions without flaking across minor numerical implementation drift.
    assert benchmark["target_enthalpy"] == pytest.approx(benchmark["expected_enthalpy"], rel=1.5e-2)
    assert benchmark["target_temperature"] == pytest.approx(benchmark["expected_temperature"], abs=3.5)


@pytest.mark.integration
@pytest.mark.coupling
def test_cryopump_component_matches_constant_flow_steady_state():
    benchmark = run_component_benchmark("cryopump")

    assert benchmark["component_name"] == benchmark["expected_component_name"]
    assert benchmark["coupling_flow_history"]
    assert benchmark["final_source_mass"] < benchmark["initial_source_mass"]
    assert benchmark["final_target_mass"] == pytest.approx(benchmark["initial_target_mass"], rel=1e-2)
    assert benchmark["target_enthalpy"] == pytest.approx(benchmark["expected_enthalpy"], rel=5e-2)
    assert benchmark["target_temperature"] == pytest.approx(benchmark["expected_temperature"], abs=2.0)

@pytest.mark.integration
@pytest.mark.coupling
def test_yaml_backed_peripheral_component_smoke(tmp_path):
    config_path = tmp_path / "yaml_hx_smoke.yaml"
    write_yaml_smoke_config(config_path)

    config = ScenarioConfig.from_yaml(config_path)
    orchestrator = SystemOrchestrator(config)

    valve = orchestrator.tank_system.coupling_valves[0]
    assert len(valve.component_chain) == 1
    assert type(valve.component_chain[0]).__name__ == "IdealHeatExchanger"

    results = orchestrator.run_simulation(
        "LSODA",
        {
            "time_step": 1.0,
            "rtol": 1e-5,
            "atol": 1e-7,
            "max_step": 1.0,
            "max_simulation_time": 40.0,
        },
    )

    target_initial = results.multi_tank_states[0].get_tank_state(1)
    target_final = results.multi_tank_states[-1].get_tank_state(1)
    mission_only_final_mass = target_initial.fuel_mass - (0.01 * 40.0)

    assert orchestrator.tank_system.coupling_flow_history
    assert max(orchestrator.tank_system.coupling_flow_history) > 0.0
    assert target_final.temperature > target_initial.temperature
    assert target_final.fuel_mass > mission_only_final_mass