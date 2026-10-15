"""
Tests for flow-distribution mixers: N tanks feed a mixer junction and the mission
demand assigned to the mixer is split across the tanks by `flow_fraction`.
"""

import copy
from pathlib import Path

import pytest
import yaml

from toplab.configuration.scenario_configuration import ScenarioConfig
from toplab.orchestration.system_orchestrator import SystemOrchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "examples" / "single_tank_cch2" / "single_tank_cch2_config.yaml"


def _write_mixer_config(tmp_path, fractions=None, n_tanks=2):
    """Derive an N-tank mixer config from the single-tank CcH2 example."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    tank = cfg["network"]["nodes"][0]
    mixer_id, sink_id = n_tanks + 1, n_tanks + 2

    nodes = []
    for i in range(1, n_tanks + 1):
        node = copy.deepcopy(tank)
        node["node_id"] = i
        nodes.append(node)
    nodes.append({"node_id": mixer_id, "type": "junction", "junction_role": "mixer"})
    nodes.append({"node_id": sink_id, "type": "sink"})

    edges = []
    for i in range(1, n_tanks + 1):
        edge = {
            "edge_id": f"tank{i}_to_mixer",
            "connection_type": "discharge",
            "participants": {"source": i, "target": mixer_id},
        }
        if fractions is not None:
            edge["flow_fraction"] = fractions[i - 1]
        edges.append(edge)
    edges.append({
        "edge_id": "mixer_to_powerplant",
        "connection_type": "discharge",
        "participants": {"source": mixer_id, "target": sink_id},
    })

    cfg["network"] = {"nodes": nodes, "edges": edges}
    cfg["mission"]["assigned_to_node"] = mixer_id
    cfg["mission"]["parameters"]["csv_file"] = str((REPO_ROOT / "missions" / "atr72_fuel_flows.csv"))

    path = tmp_path / "mixer.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_default_fractions_are_even(tmp_path):
    config = ScenarioConfig.from_yaml(_write_mixer_config(tmp_path, n_tanks=3))
    (group,) = config.mixer_groups
    assert group["flow_fractions"] == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
    # Mixer edges must not create tank-to-tank coupling rules
    assert config.config_dict["coupling_rules"] == []


@pytest.mark.parametrize("fractions", [[0.5, 0.4], [-0.2, 1.2]])
def test_fractions_must_sum_to_one_and_be_nonnegative(tmp_path, fractions):
    with pytest.raises(ValueError, match="flow_fraction"):
        ScenarioConfig.from_yaml(_write_mixer_config(tmp_path, fractions=fractions))


def test_partial_fractions_rejected(tmp_path):
    path = _write_mixer_config(tmp_path, fractions=[0.7, 0.3])
    cfg = yaml.safe_load(path.read_text())
    del cfg["network"]["edges"][1]["flow_fraction"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="all incoming edges or on none"):
        ScenarioConfig.from_yaml(path)


def test_mixer_splits_mission_flow_and_sizes_tanks(tmp_path):
    fractions = [0.7, 0.3]
    orchestrator = SystemOrchestrator(ScenarioConfig.from_yaml(_write_mixer_config(tmp_path, fractions)))
    system = orchestrator.tank_system

    t = 500.0
    from toplab.demand import outflow_demand
    total = outflow_demand(orchestrator.mission_profile).mass_flow(t)
    assert total > 0.0
    assert system._get_outflow_rate(t, 0) == pytest.approx(fractions[0] * total)
    assert system._get_outflow_rate(t, 1) == pytest.approx(fractions[1] * total)

    # Same fluid state and shape, so tank volume scales with its share of mission fuel
    v1, v2 = (g.volume for g in orchestrator.tank_geometries)
    assert v1 / v2 == pytest.approx(fractions[0] / fractions[1], rel=1e-6)


def test_mixer_without_assigned_mission_is_an_error(tmp_path):
    path = _write_mixer_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["mission"]["assigned_to_node"] = 1
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="has no mission"):
        SystemOrchestrator(ScenarioConfig.from_yaml(path))
