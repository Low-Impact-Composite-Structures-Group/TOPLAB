import pytest

from toplab.configuration.scenario_configuration import ScenarioConfig


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


# ---------------------------------------------------------------------------
# Dual-flow / multi-node mission parsing
# ---------------------------------------------------------------------------

@pytest.mark.config
@pytest.mark.unit
def test_dual_flow_config_exists(dual_flow_config):
    assert dual_flow_config.exists(), f"Dual-flow config not found: {dual_flow_config}"


@pytest.mark.config
@pytest.mark.unit
def test_dual_flow_missions_plural_parsed(dual_flow_config):
    """The 'missions:' list produces two MissionConfig objects."""
    config = ScenarioConfig.from_yaml(dual_flow_config)
    missions = config.mission_sequence.missions

    assert len(missions) == 2, f"Expected 2 missions, got {len(missions)}"


@pytest.mark.config
@pytest.mark.unit
def test_dual_flow_missions_assigned_to_distinct_nodes(dual_flow_config):
    """Each mission is assigned to a different node."""
    config = ScenarioConfig.from_yaml(dual_flow_config)
    missions = config.mission_sequence.missions

    assigned = [m.assigned_to for m in missions]
    assert all(a is not None for a in assigned), "Some missions are missing assigned_to"
    assert len(set(assigned)) == len(assigned), f"Duplicate node assignments: {assigned}"


@pytest.mark.config
@pytest.mark.unit
def test_dual_flow_missions_have_column_name(dual_flow_config):
    """Both missions declare a column_name pointing to distinct CSV columns."""
    config = ScenarioConfig.from_yaml(dual_flow_config)
    missions = config.mission_sequence.missions

    columns = [m.parameters.get("column_name") for m in missions]
    assert all(c for c in columns), f"Some missions missing column_name: {columns}"
    assert len(set(columns)) == 2, f"Expected two distinct columns, got: {columns}"


@pytest.mark.config
@pytest.mark.unit
def test_dual_flow_csv_file_exists(dual_flow_config):
    """The shared dual-flow CSV file is present on disk."""
    config = ScenarioConfig.from_yaml(dual_flow_config)
    mission = config.mission_sequence.missions[0]

    csv_file = mission.parameters.get("csv_file")
    assert csv_file, "No csv_file in first mission parameters"

    csv_path = (dual_flow_config.parent / csv_file).resolve()
    assert csv_path.exists(), f"Dual-flow CSV not found: {csv_path}"


@pytest.mark.config
@pytest.mark.unit
def test_dual_flow_assigned_nodes_exist_in_network(dual_flow_config):
    """Each mission's assigned_to_node matches a node_id in the network."""
    config = ScenarioConfig.from_yaml(dual_flow_config)
    raw = config.config_dict
    node_ids = {n.get("node_id") for n in raw.get("network", {}).get("nodes", [])}

    for m in config.mission_sequence.missions:
        assert m.assigned_to in node_ids, (
            f"Mission assigned_to={m.assigned_to} not in network node_ids={node_ids}"
        )


@pytest.mark.unit
def test_from_csv_with_column_name_produces_sections(dual_flow_config):
    """Mission.from_csv with column_name returns a non-empty section list."""
    from toplab.missions.mission import Mission

    config = ScenarioConfig.from_yaml(dual_flow_config)
    missions_cfg = config.mission_sequence.missions
    csv_file = missions_cfg[0].parameters["csv_file"]
    csv_path = (dual_flow_config.parent / csv_file).resolve()

    for m in missions_cfg:
        col = m.parameters["column_name"]
        mission = Mission.from_csv(str(csv_path), column_name=col)
        assert len(mission.sections) > 0, f"No sections parsed for column '{col}'"


@pytest.mark.unit
def test_from_csv_column_name_produces_independent_missions(dual_flow_config):
    """FC and GT missions loaded from the same CSV produce different fuel totals."""
    from toplab.missions.mission import Mission

    config = ScenarioConfig.from_yaml(dual_flow_config)
    missions_cfg = config.mission_sequence.missions
    csv_file = missions_cfg[0].parameters["csv_file"]
    csv_path = (dual_flow_config.parent / csv_file).resolve()

    fc_col = next(m.parameters["column_name"] for m in missions_cfg if m.assigned_to == 1)
    gt_col = next(m.parameters["column_name"] for m in missions_cfg if m.assigned_to == 2)

    fc_mission = Mission.from_csv(str(csv_path), column_name=fc_col)
    gt_mission = Mission.from_csv(str(csv_path), column_name=gt_col)

    fc_fuel = fc_mission.required_fuel
    gt_fuel = gt_mission.required_fuel

    assert fc_fuel > 0, "FC mission has zero required fuel"
    assert gt_fuel > 0, "GT mission has zero required fuel"
    # GT demand is substantially larger than FC (≈3–4×)
    assert gt_fuel > fc_fuel, (
        f"Expected GT fuel ({gt_fuel:.3f} kg) > FC fuel ({fc_fuel:.3f} kg)"
    )


@pytest.mark.unit
def test_from_csv_unknown_column_name_raises(dual_flow_config):
    """from_csv raises ValueError for a non-existent column_name."""
    from toplab.missions.mission import Mission

    config = ScenarioConfig.from_yaml(dual_flow_config)
    csv_file = config.mission_sequence.missions[0].parameters["csv_file"]
    csv_path = (dual_flow_config.parent / csv_file).resolve()

    with pytest.raises(ValueError, match="not found in CSV"):
        Mission.from_csv(str(csv_path), column_name="Non-existent column [kg/s]")


@pytest.mark.unit
def test_dual_flow_missions_share_time_axis_length(dual_flow_config):
    """Both missions are built from the same CSV rows and have the same section count."""
    from toplab.missions.mission import Mission

    config = ScenarioConfig.from_yaml(dual_flow_config)
    missions_cfg = config.mission_sequence.missions
    csv_file = missions_cfg[0].parameters["csv_file"]
    csv_path = (dual_flow_config.parent / csv_file).resolve()

    section_counts = []
    for m in missions_cfg:
        mission = Mission.from_csv(str(csv_path), column_name=m.parameters["column_name"])
        section_counts.append(len(mission.sections))

    assert len(set(section_counts)) == 1, (
        f"FC and GT missions have different section counts: {section_counts}. "
        "They must share the same time axis."
    )
