"""Edge registry: edge types parse and validate their own options."""

from pathlib import Path

import pytest

from toplab.config import ConfigError, load_config
from toplab.coupling import (EDGE_TYPES, EdgeContext, InterTankCoupling, PressureTriggeredValve,
                             build_coupling, register_edge)
from toplab.network import Network
from toplab.network.network import CouplingEdge

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _ctx():
    config = load_config(EXAMPLES / "coupled_ch2_cch2" / "coupled_ch2_cch2_config.yaml")
    return config, EdgeContext(minimum_pressures={0: 1e5, 1: 1e5}, fluids={0: "PARAHYD", 1: "PARAHYD"})


def test_example_edge_builds_valve_with_configured_thresholds():
    config, ctx = _ctx()
    edge = Network(config).coupling_edges()[0]
    valve = build_coupling(edge, ctx)
    assert isinstance(valve, PressureTriggeredValve)
    assert (valve.source_tank, valve.target_tank) == (edge.source, edge.target)
    assert valve.p_open == pytest.approx(16e5) and valve.p_close == pytest.approx(30e5)


def test_unknown_type_and_bad_options_fail_loudly():
    _, ctx = _ctx()
    with pytest.raises(ConfigError, match="unknown connection_type"):
        build_coupling(CouplingEdge("e", "nope", 0, 1, {}), ctx)
    with pytest.raises(ConfigError, match="activation_conditions"):
        build_coupling(CouplingEdge("e", "pressure_compensation", 0, 1, {}), ctx)
    inverted = {"activation_conditions": {"pressure_open_bar": 30, "pressure_close_bar": 16},
                "valve": {"orifice_diameter_m": 0.005}}
    with pytest.raises(ConfigError):
        build_coupling(CouplingEdge("e", "pressure_compensation", 0, 1, inverted), ctx)


def test_new_edge_type_needs_only_a_registration():
    @register_edge("_test_edge")
    class _Edge(InterTankCoupling):
        @classmethod
        def from_edge(cls, edge, ctx):
            return cls(edge.source, edge.target, edge.edge_id)

    try:
        _, ctx = _ctx()
        coupling = build_coupling(CouplingEdge("x", "_test_edge", 0, 1, {}), ctx)
        assert isinstance(coupling, _Edge) and coupling.coupling_id == "x"
    finally:
        EDGE_TYPES.pop("_test_edge")
