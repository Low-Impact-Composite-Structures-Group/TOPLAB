import pytest

from src.configuration.scenario_configuration import ScenarioConfig


@pytest.mark.config
@pytest.mark.unit
def test_main_analysis_mission_sections_parse(main_analysis_configs):
    for name, path in main_analysis_configs.items():
        assert path.exists(), f"Missing config for {name}: {path}"
        config = ScenarioConfig.from_yaml(path)

        assert config.mission_sequence.missions, f"No missions parsed for {name}"
        mission = config.mission_sequence.missions[0]

        # All five primary analyses use discharge + csv today
        assert mission.type == "discharge", f"Unexpected mission type for {name}: {mission.type}"
        assert mission.profile == "csv", f"Unexpected mission profile for {name}: {mission.profile}"


@pytest.mark.config
@pytest.mark.unit
def test_assigned_to_node_is_present_and_valid(main_analysis_configs):
    for name, path in main_analysis_configs.items():
        config = ScenarioConfig.from_yaml(path)
        raw = config.config_dict

        assigned_to = raw.get("mission", {}).get("assigned_to_node")
        assert assigned_to is not None, f"mission.assigned_to_node missing for {name}"

        node_ids = [n.get("node_id") for n in raw.get("network", {}).get("nodes", [])]
        assert assigned_to in node_ids, f"assigned_to_node={assigned_to} not in node_ids={node_ids} for {name}"


@pytest.mark.config
@pytest.mark.unit
def test_csv_mission_file_exists(main_analysis_configs):
    for name, path in main_analysis_configs.items():
        config = ScenarioConfig.from_yaml(path)
        mission = config.mission_sequence.missions[0]

        csv_file = mission.parameters.get("csv_file")
        assert csv_file, f"No mission.parameters.csv_file for {name}"

        csv_path = (path.parent / csv_file).resolve()
        assert csv_path.exists(), f"CSV mission file missing for {name}: {csv_path}"
