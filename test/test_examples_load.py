"""Every shipped example parses into a typed config and a consistent network."""

from pathlib import Path

import pytest

from toplab.config import load_config
from toplab.network import Network

EXAMPLES = Path(__file__).parent.parent / "examples"
CONFIGS = sorted(p for p in EXAMPLES.glob("*/*_config.yaml"))


def test_examples_are_discovered():
    assert len(CONFIGS) >= 6


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.parent.name)
def test_example_config_builds_network(path):
    config = load_config(path)
    network = Network(config)
    assert network.tank_count == len(config.tanks) >= 1
    assert config.missions
    for mission in config.missions:
        shares = network.demand_targets(mission.assigned_to_node)
        assert sum(shares.values()) == pytest.approx(1.0)
        assert all(0 <= index < network.tank_count for index in shares)
