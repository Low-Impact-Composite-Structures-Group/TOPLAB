import pytest
import CoolProp.CoolProp as CP

from toplab.fluids.valve_flow import PipeSpec, ValvePath, path_from_options, valve_mass_flow

FLUID = "PARAHYD"
T1, P1 = 290.0, 300e5
RHO1 = CP.PropsSI("D", "T", T1, "P", P1, FLUID)


def _flow(path, p2, opening=1.0):
    return valve_mass_flow(path, FLUID, P1, T1, RHO1, p2, opening)


def test_lossless_choked_flow_matches_ideal_gas_estimate():
    path = ValvePath(orifice_diameter_m=0.005, discharge_coefficient=0.6)
    # Cd*A*p0*sqrt(gamma/(R T0))*(2/(gamma+1))^((gamma+1)/(2(gamma-1))) ~ 0.22 kg/s for ideal H2
    assert _flow(path, 16e5) == pytest.approx(0.22, rel=0.1)
    assert _flow(path, 16e5) == pytest.approx(_flow(path, 100e5), rel=1e-3)  # choked: independent of back pressure


def test_pipes_reduce_flow_and_closed_or_no_dp_gives_zero():
    bare = ValvePath(0.005)
    piped = ValvePath(0.005, upstream=PipeSpec(0.01, 2.0, 0.02), downstream=PipeSpec(0.01, 2.0, 0.02))
    assert 0.0 < _flow(piped, 100e5) < _flow(bare, 100e5)
    assert _flow(piped, 100e5, opening=0.0) == 0.0
    assert _flow(piped, P1) == 0.0


def test_smaller_opening_gives_smaller_flow():
    path = ValvePath(0.005)
    assert _flow(path, 100e5, 0.5) == pytest.approx(0.5 * _flow(path, 100e5, 1.0), rel=1e-3)


def test_path_from_options_accepts_area_or_diameter():
    a = path_from_options({"valve": {"orifice_area_m2": 1.0e-4}})
    assert a.orifice_area == pytest.approx(1.0e-4)
    with pytest.raises(ValueError):
        path_from_options({"valve": {}})
