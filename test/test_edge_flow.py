import pytest

from src.multistate.dynamics.edge_flow import EdgeFlow


@pytest.mark.unit
@pytest.mark.physics
def test_mass_contribution_sign_convention_for_internal_edge():
    edge = EdgeFlow(
        mdot=0.05,
        h=3.2e6,
        edge_type="coupling",
        from_node=0,
        to_node=1,
    )

    assert edge.mass_contribution(0) == pytest.approx(-0.05)
    assert edge.mass_contribution(1) == pytest.approx(0.05)
    assert edge.mass_contribution(2) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.physics
def test_enthalpy_contribution_is_zero_for_outflow_with_same_tank_enthalpy():
    h_tank = 4.1e6
    edge = EdgeFlow(
        mdot=0.08,
        h=h_tank,
        edge_type="discharge",
        from_node=0,
        to_node=-1,
    )

    assert edge.is_outflow_for(0)
    assert edge.enthalpy_contribution(0, h_tank) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.physics
def test_enthalpy_contribution_for_external_refuel_inflow():
    edge = EdgeFlow(
        mdot=0.02,
        h=5.0e6,
        edge_type="refuel",
        from_node=-1,
        to_node=1,
    )

    h_tank = 4.6e6
    expected = 0.02 * (5.0e6 - 4.6e6)

    assert edge.is_inflow_for(1)
    assert edge.enthalpy_contribution(1, h_tank) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.physics
def test_direction_helpers_for_environment_edges():
    vent = EdgeFlow(mdot=0.01, h=4.0e6, edge_type="vent", from_node=2, to_node=-1)
    refuel = EdgeFlow(mdot=0.03, h=4.9e6, edge_type="refuel", from_node=-1, to_node=2)

    assert vent.is_outflow_for(2)
    assert not vent.is_inflow_for(2)

    assert refuel.is_inflow_for(2)
    assert not refuel.is_outflow_for(2)
