"""Typed config loading and Network topology (tank index map, demand routing)."""

from pathlib import Path

import pytest

from toplab.config import ConfigError, load_config
from toplab.network import Network

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    return load_config(EXAMPLES / name / f"{name}_config.yaml")


def test_single_tank_network():
    net = Network(_load("single_tank_ch2"))
    assert net.tank_ids == [1]
    assert net.demand_targets(1) == {0: 1.0}
    assert net.coupling_edges() == []


def test_coupled_network_has_coupling_edge_and_index_map():
    config = _load("coupled_ch2_cch2")
    net = Network(config)
    assert [net.index_of(t) for t in net.tank_ids] == list(range(len(net.tank_ids)))
    assert len(net.coupling_edges()) >= 1


def test_demand_target_on_non_flow_node_raises():
    net = Network(_load("single_tank_ch2"))
    with pytest.raises(ConfigError):
        net.demand_targets(999)


def test_missing_network_section_rejected():
    from toplab.config import parse_config

    with pytest.raises(ConfigError):
        parse_config({"analysis": {"name": "x"}}, None)
