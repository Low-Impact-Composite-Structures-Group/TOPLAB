import copy
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from toplab.config import ConfigError, load_config
from toplab.optimization.evaluation import maximum_outer_radii
from toplab.sizing import size_tank
from toplab.system.tank_system import TankSystem
from toplab.tank_design.structural_models import CompositeCylinder, CompositeSphericalEndCap


def _spec():
    path = Path(__file__).parent.parent / "examples/single_tank_cch2/single_tank_cch2_config.yaml"
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


def test_structural_wall_uses_venting_pressure():
    spec = _spec()
    design = size_tank(spec, required_fuel=100.0)
    tank_system = TankSystem.__new__(TankSystem)
    config = load_config(
        Path(__file__).parent.parent
        / "examples/single_tank_cch2/single_tank_cch2_config.yaml"
    )
    tank_system.analysis_config = replace(config, tanks=[spec])
    tank_system.verbosity = "quiet"

    properties = tank_system._get_tank_properties(design.geometry, tank_index=0)
    material = spec.nist_materials()["composite"]
    section = SimpleNamespace(radius=design.geometry.radius, material=material)
    safety_factor = float(spec.materials["safety_margin"])
    expected_thickness = max(
        CompositeCylinder().compute_thickness(
            section, design.venting_pressure * safety_factor
        ),
        CompositeSphericalEndCap().compute_thickness(
            section, design.venting_pressure * safety_factor
        ),
    )
    initial_pressure_thickness = max(
        CompositeCylinder().compute_thickness(
            section, design.initial_pressure * safety_factor
        ),
        CompositeSphericalEndCap().compute_thickness(
            section, design.initial_pressure * safety_factor
        ),
    )

    assert properties["wall_thickness"] == pytest.approx(expected_thickness)
    assert properties["wall_thickness"] > initial_pressure_thickness
    assert design.venting_pressure > design.initial_pressure


def test_optimization_packaging_bound_uses_venting_pressure():
    spec = _spec()
    spec = replace(
        spec,
        geometry={**spec.geometry, "radius": 0.4},
        operating_limits={**spec.operating_limits, "venting_pressure": 45e6},
    )
    config = load_config(
        Path(__file__).parent.parent
        / "examples/single_tank_cch2/single_tank_cch2_config.yaml"
    )
    config = replace(config, tanks=[spec])
    lower_bound = maximum_outer_radii(config, [])[0]

    higher_pressure_spec = replace(
        spec,
        operating_limits={**spec.operating_limits, "venting_pressure": 90e6},
    )
    higher_pressure_config = replace(config, tanks=[higher_pressure_spec])

    assert maximum_outer_radii(higher_pressure_config, [])[0] > lower_bound
