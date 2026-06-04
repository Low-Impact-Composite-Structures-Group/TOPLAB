import pytest

from src.multistate.configuration.scenario_configuration import ScenarioConfig
from src.multistate.orchestration.system_orchestrator import SystemOrchestrator


@pytest.mark.orchestrator
@pytest.mark.config
@pytest.mark.unit
def test_all_main_configs_load_and_wire(main_analysis_configs):
    for name, path in main_analysis_configs.items():
        assert path.exists(), f"Missing config for {name}: {path}"
        config = ScenarioConfig.from_yaml(path)

        assert config.config_format == "new", f"Expected new-format config for {name}"
        assert config.get_tank_count() >= 1

        orchestrator = SystemOrchestrator(config)
        assert len(orchestrator.tank_system.tanks) == config.get_tank_count()

        if name.startswith("coupled_"):
            assert len(orchestrator.tank_system.tanks) >= 2
            assert len(orchestrator.tank_system.coupling_valves) >= 1
        else:
            assert len(orchestrator.tank_system.tanks) == 1


@pytest.mark.orchestrator
@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.parametrize("analysis_key", ["coupled_ch2_cch2", "coupled_ch2_lh2"])
def test_coupled_systems_short_simulation_smoke(main_analysis_configs, analysis_key):
    config_path = main_analysis_configs[analysis_key]
    config = ScenarioConfig.from_yaml(config_path)
    orchestrator = SystemOrchestrator(config)

    solver_method = config.config_dict.get("solver", {}).get("method", "LSODA")

    # Keep this intentionally short: we only need to verify the coupled solver path executes.
    # TankSystem now honors max_simulation_time.
    solver_config = {
        "time_step": 2.0,
        "rtol": 1e-4,
        "atol": 1e-7,
        "max_step": 2.0,
        "max_simulation_time": 10.0,
    }

    results = orchestrator.run_simulation(solver_method, solver_config)

    assert len(results.times) > 0, "No time data returned"
    assert results.times[-1] > 0.0
    assert len(results.multi_tank_states) == len(results.times)

    # Basic sanity: no negative masses
    final_state = results.multi_tank_states[-1]
    for tank_state in final_state.tank_states:
        assert tank_state.fuel_mass >= 0.0
