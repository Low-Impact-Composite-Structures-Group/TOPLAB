import pytest

from toplab.configuration.scenario_configuration import ScenarioConfig
from toplab.orchestration.system_orchestrator import SystemOrchestrator


@pytest.mark.plotting
@pytest.mark.integration
@pytest.mark.regression
def test_coupled_ch2_cch2_report_generation_with_blank_density(main_analysis_configs, tmp_path):
    """Regression: coupled_ch2_cch2 has a blank initial density field in YAML.

    This should not crash report generation.
    """
    config_path = main_analysis_configs["coupled_ch2_cch2"]
    config = ScenarioConfig.from_yaml(config_path)
    orchestrator = SystemOrchestrator(config)

    solver_method = config.config_dict.get("solver", {}).get("method", "LSODA")
    solver_config = {
        "time_step": 2.0,
        "rtol": 1e-4,
        "atol": 1e-7,
        "max_step": 2.0,
        "max_simulation_time": 10.0,
    }

    orchestrator.run_simulation(solver_method, solver_config)

    # Make outputs go to a temp dir to avoid polluting the workspace
    original_output = orchestrator.scenario_config.config_dict.get("output", {})
    orchestrator.scenario_config.config_dict["output"] = {
        **original_output,
        "results_directory": str(tmp_path / "results"),
        "plots_directory": str(tmp_path / "plots"),
    }

    report_path = orchestrator.save_comprehensive_results()
    assert report_path is not None


@pytest.mark.plotting
@pytest.mark.unit
def test_generate_plots_callable(main_analysis_configs):
    for name, path in main_analysis_configs.items():
        config = ScenarioConfig.from_yaml(path)
        orchestrator = SystemOrchestrator(config)
        assert hasattr(orchestrator, "generate_plots")
        assert callable(orchestrator.generate_plots)
