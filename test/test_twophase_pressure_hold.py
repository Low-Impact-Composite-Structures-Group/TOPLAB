"""Two-phase Config B (pressure hold) must report iHEX heat when the config is replayed post-hoc."""

from types import SimpleNamespace

from CoolProp.CoolProp import PropsSI

from toplab.dynamics.edge_flow import EdgeFlow
from toplab.dynamics.isochoric_dynamic_models import TwoPhaseIsochoricModel
from toplab.thermodynamics.tank_states import IsochoricTankState

FLUID = "PARAHYD"
VOLUME = 0.5
T = 25.0
MDOT = 0.01


def _state():
    rho = 0.5 * (PropsSI("Dmass", "T", T, "Q", 0, FLUID) + PropsSI("Dmass", "T", T, "Q", 1, FLUID))
    state = IsochoricTankState(
        tank=SimpleNamespace(volume=VOLUME),
        fuel_mass=rho * VOLUME,
        h2_temperature=T,
        structure_temperature=T,
        insulation_temperature=T,
        shell_temperature=T,
        fluid=FLUID,
    )
    state.pressure = PropsSI("P", "T", T, "Q", 0, FLUID)
    return state


def _model():
    # p_sat sits inside the 5% band below p_min, so a fresh latch picks A but B is wanted.
    p_sat = PropsSI("P", "T", T, "Q", 0, FLUID)
    return TwoPhaseIsochoricModel("DISCHARGE", p_min=p_sat * 1.03, tank_volume=VOLUME)


def _discharge_heat(model, state):
    flows = [EdgeFlow(mdot=MDOT, h=0.0, edge_type="discharge", from_node=0, to_node=-1)]
    return model.compute_state_derivatives(0.0, state, flows, Q_structure_to_h2=0.0).discharge_heat_flux


def test_latched_two_phase_config_hides_pressure_hold_heat():
    model = _model()
    state = _state()
    assert model.is_applicable(state)
    assert _discharge_heat(model, state) == 0.0
    assert model._last_config == "A"


def test_stateless_replay_reports_two_phase_pressure_hold_heat():
    model = _model()
    model.stateless_config = True
    state = _state()
    assert model.is_applicable(state)
    assert _discharge_heat(model, state) > 0.0
    assert state.configuration == "B"
    assert not hasattr(model, "_last_config")
