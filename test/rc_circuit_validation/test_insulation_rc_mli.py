"""Validation of the diagnostic multi-node insulation network with MLI."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import root

from toplab.materials.insulation_models import make_modified_lockheed_model
from toplab.materials.nist_materials import NISTComposite, NISTMetal
from toplab.thermodynamics.insulation_rc_diagnostic import InsulationRCModel
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel


_L_CYLINDER = 1.5
_T_SHELL_BOUNDARY = 288.15
_T_STRUCTURE_BOUNDARY = 60.0
_MLI = make_modified_lockheed_model()


def _layer_mass(density, inner_radius, outer_radius):
    volume = (
        math.pi * (outer_radius**2 - inner_radius**2) * _L_CYLINDER
        + (4.0 * math.pi / 3.0) * (outer_radius**3 - inner_radius**3)
    )
    return density * volume


def _make_model():
    liner = NISTMetal.aluminum_6061T6_nist()
    wall = NISTComposite.carbon_epoxy_nist()
    r_inner = 0.5
    r_structure = r_inner + 0.003 + 0.015
    r_shell = r_structure + 0.05
    return InsulatedTankThermalModel(
        tank_volume=math.pi * r_inner**3 * (4.0 / 3.0 + _L_CYLINDER / r_inner),
        inner_surface_area=4.0 * math.pi * r_inner**2
        + 2.0 * math.pi * r_inner * _L_CYLINDER,
        inner_diameter=2.0 * r_inner,
        r_structure=r_structure,
        r_shell=r_shell,
        cylinder_length=_L_CYLINDER,
        liner_mass=_layer_mass(liner.density, r_inner, r_inner + 0.003),
        wall_mass=_layer_mass(wall.density, r_inner + 0.003, r_structure),
        foam_mass=_layer_mass(_MLI.density, r_structure, r_shell),
        shell_mass=_layer_mass(liner.density, r_shell, r_shell + 0.002),
        ambient_temperature=_T_SHELL_BOUNDARY,
        alpha_amb=5.0,
        emissivity_shell=0.05,
        liner_material=liner,
        wall_material=wall,
        shell_material=liner,
        insulation_model=_MLI,
    )


def _steady_temperatures(rc_model):
    initial = np.linspace(
        _T_SHELL_BOUNDARY, _T_STRUCTURE_BOUNDARY, rc_model.node_count + 2
    )[1:-1]

    def balance(temperatures):
        flows = rc_model.heat_flows(
            _T_SHELL_BOUNDARY, temperatures, _T_STRUCTURE_BOUNDARY
        )
        return flows[:-1] - flows[1:]

    solution = root(balance, initial)
    assert solution.success, solution.message
    return solution.x


def test_rc_model_accepts_mli_and_conserves_mass():
    tank_model = _make_model()
    for node_count in (1, 2, 5, 20):
        rc_model = InsulationRCModel(tank_model, node_count)
        assert rc_model.total_insulation_mass == pytest.approx(
            tank_model.m_foam, rel=1e-12
        )


def test_mli_heat_flow_sign_convention_and_equal_temperature_limit():
    rc_model = InsulationRCModel(_make_model(), 4)
    forward = rc_model.heat_flows(288.0, [240.0, 190.0, 130.0, 90.0], 60.0)
    reverse = rc_model.heat_flows(60.0, [90.0, 130.0, 190.0, 240.0], 288.0)
    equal = rc_model.heat_flows(150.0, [150.0] * 4, 150.0)

    assert np.all(forward > 0.0)
    assert np.all(reverse < 0.0)
    assert equal == pytest.approx(np.zeros(5), abs=1e-12)


def test_mli_one_node_reproduces_production_heat_flows_and_derivative():
    tank_model = _make_model()
    rc_model = InsulationRCModel(tank_model, 1)
    shell, insulation, structure = 288.0, 174.0, 60.0
    state = SimpleNamespace(
        shell_temperature=shell,
        insulation_temperature=insulation,
        structure_temperature=structure,
    )
    expected = np.array(
        [
            tank_model.compute_shell_to_insulation_heat_flux(shell, insulation),
            tank_model.compute_insulation_to_structure_heat_flux(
                insulation, structure
            ),
        ]
    )
    assert rc_model.heat_flows(shell, [insulation], structure) == pytest.approx(
        expected, rel=1e-13
    )
    assert rc_model.temperature_derivatives(shell, [insulation], structure)[
        0
    ] == pytest.approx(
        tank_model.compute_insulation_temperature_derivative(0.0, state),
        rel=1e-13,
    )


def test_mli_energy_residual_is_zero():
    rc_model = InsulationRCModel(_make_model(), 5)
    temperatures = np.array([270.0, 235.0, 190.0, 130.0, 85.0])
    assert rc_model.energy_residual(
        _T_SHELL_BOUNDARY, temperatures, _T_STRUCTURE_BOUNDARY
    ) == pytest.approx(0.0, abs=1e-10)


def test_mli_steady_state_heat_flow_converges_with_node_count():
    tank_model = _make_model()
    flows = {}
    for node_count in (1, 2, 4, 8, 16, 32):
        rc_model = InsulationRCModel(tank_model, node_count)
        temperatures = _steady_temperatures(rc_model)
        steady = rc_model.heat_flows(
            _T_SHELL_BOUNDARY, temperatures, _T_STRUCTURE_BOUNDARY
        )
        assert steady == pytest.approx(np.full_like(steady, steady[0]), rel=1e-8)
        flows[node_count] = steady[0]

    reference = flows[32]
    assert reference > 0.0
    errors = [abs(flows[n] - reference) / reference for n in (2, 4, 8, 16)]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.02


def test_mli_transient_relaxes_to_steady_state():
    rc_model = InsulationRCModel(_make_model(), 8)
    steady = _steady_temperatures(rc_model)
    solution = solve_ivp(
        lambda _t, y: rc_model.temperature_derivatives(
            _T_SHELL_BOUNDARY, y, _T_STRUCTURE_BOUNDARY
        ),
        (0.0, 5000.0 * 3600.0),
        np.full(8, _T_SHELL_BOUNDARY),
        method="Radau",
        rtol=1e-8,
        atol=1e-9,
    )
    assert solution.success, solution.message
    assert solution.y[:, -1] == pytest.approx(steady, abs=0.5)
