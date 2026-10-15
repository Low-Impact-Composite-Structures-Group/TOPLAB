"""Per-tank fluid resolution."""

import pytest

from toplab.config import ConfigError, parse_config
from toplab.fluids.fluid import resolve_coolprop_fluid


def test_known_labels_resolve():
    for label in ("CH2", "CcH2", "SLH2", "LH2"):
        assert resolve_coolprop_fluid(label) == "PARAHYD"


def test_override_wins():
    assert resolve_coolprop_fluid("CH2", "Hydrogen") == "Hydrogen"


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        resolve_coolprop_fluid("N2O")


def test_unknown_tank_fluid_rejected_at_load():
    raw = {
        "analysis": {"name": "x"},
        "network": {"nodes": [{"node_id": 1, "type": "tank", "fluid": "XYZ"}], "edges": []},
    }
    with pytest.raises(ConfigError, match="fluid"):
        parse_config(raw, None)
