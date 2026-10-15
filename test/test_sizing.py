import copy
import glob
import math

import pytest

from toplab.config import ConfigError, load_config
from toplab.sizing import size_tank


def _spec():
    path = glob.glob("examples/single_tank_cch2/*_config.yaml")[0]
    return load_config(path).tanks[0]


def test_size_tank_is_pure_and_mission_sized():
    spec = _spec()
    before = copy.deepcopy(spec.geometry_data())
    design = size_tank(spec, required_fuel=100.0)
    assert spec.geometry_data() == before
    assert design.required_fuel == 100.0
    assert design.geometry.radius > 0
    assert design.initial_density is None or design.initial_density > 0
    assert design.venting_pressure > design.initial_pressure * 0.99


def test_larger_fuel_gives_larger_tank():
    spec = _spec()
    a = size_tank(spec, required_fuel=50.0).geometry.radius
    b = size_tank(spec, required_fuel=100.0).geometry.radius
    assert b > a


def test_zero_fuel_without_radius_raises():
    spec = _spec()
    if "radius" in spec.geometry_data():
        pytest.skip("example specifies a radius")
    with pytest.raises(ConfigError):
        size_tank(spec, required_fuel=0.0)
