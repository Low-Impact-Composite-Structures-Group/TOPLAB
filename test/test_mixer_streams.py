"""Mixer stream physics: mass and enthalpy conservation of the merged stream."""

import numpy as np
import pytest
import yaml
from CoolProp.CoolProp import PropsSI

from toplab.config import ConfigError, load_config
from toplab.network import Network
from toplab.post.mixer import mix_streams

from .test_mixer_flow_distribution import _write_mixer_config

FLUID = "PARAHYD"


def _stream(T, P):
    return PropsSI("Hmass", "T", T, "P", P, FLUID)


def test_mix_conserves_mass_and_enthalpy():
    p = [30e5, 12e5]
    h = [_stream(60.0, p[0]), _stream(150.0, p[1])]
    m, h_out, p_out, T_out = mix_streams([0.02, 0.03], h, p, FLUID, outlet_pressure=5e5)
    assert m == pytest.approx(0.05)
    assert h_out == pytest.approx((0.02 * h[0] + 0.03 * h[1]) / 0.05)
    assert p_out == 5e5
    # Mixed state reproduces the enthalpy flux of the inlets
    assert m * PropsSI("Hmass", "P", p_out, "T", T_out, FLUID) == pytest.approx(0.02 * h[0] + 0.03 * h[1], rel=1e-6)
    assert 60.0 < T_out < 150.0


def test_outlet_pressure_never_exceeds_lowest_inlet():
    p = [30e5, 12e5]
    h = [_stream(60.0, p[0]), _stream(60.0, p[1])]
    assert mix_streams([0.01, 0.01], h, p, FLUID, outlet_pressure=20e5)[2] == 12e5
    assert mix_streams([0.01, 0.01], h, p, FLUID)[2] == 12e5


def test_idle_stream_is_ignored_when_choosing_pressure_and_enthalpy():
    p = [30e5, 2e5]
    h = [_stream(60.0, p[0]), _stream(60.0, p[1])]
    m, h_out, p_out, _ = mix_streams([0.01, 0.0], h, p, FLUID)
    assert (m, h_out, p_out) == (0.01, h[0], 30e5)


def test_no_flow_returns_none():
    assert mix_streams([0.0, np.nan], [1.0, 1.0], [1e5, 1e5], FLUID) is None


def test_outlet_pressure_option_is_read_in_pa(tmp_path):
    path = _write_mixer_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    next(n for n in cfg["network"]["nodes"] if n.get("junction_role") == "mixer")["outlet_pressure_bar"] = 3.0
    path.write_text(yaml.safe_dump(cfg))
    (mixer,) = Network(load_config(path)).mixers.values()
    assert mixer.outlet_pressure == pytest.approx(3.0e5)


def test_nonpositive_outlet_pressure_rejected(tmp_path):
    path = _write_mixer_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    next(n for n in cfg["network"]["nodes"] if n.get("junction_role") == "mixer")["outlet_pressure_bar"] = 0
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="outlet_pressure_bar"):
        Network(load_config(path))
