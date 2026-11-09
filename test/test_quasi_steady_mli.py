"""Tests for the optional quasi-steady MLI mode (``insulation.thermal_mode: quasi_steady``).

E1 backwards compatibility, E2 steady-state equivalence, E3 zero-capacitance limit,
E4 energy conservation, E5 thickness sensitivity. Run from ``TOPLAB/``:
``python -m pytest test/test_quasi_steady_mli.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent / "rc_circuit_validation"))

from mli_dormancy_harness import DormancyHarness, run_production  # noqa: E402
from toplab.materials.insulation_models import ROHACELL  # noqa: E402
from toplab.thermodynamics.isochoric_thermal_model import (  # noqa: E402
    QUASI_STEADY,
    TRANSIENT_LUMPED,
    InsulatedTankThermalModel,
)

T_SHELL, T_STRUCT = 288.0, 25.0


@pytest.fixture(scope="module")
def harness():
    return DormancyHarness()


@pytest.fixture(scope="module")
def production_runs():
    return {mode: run_production(mode) for mode in (TRANSIENT_LUMPED, QUASI_STEADY)}


def test_e1_default_mode_is_transient_lumped(production_runs):
    thermal = production_runs[TRANSIENT_LUMPED]["thermal_model"]
    assert thermal.insulation_mode == TRANSIENT_LUMPED
    assert not thermal.is_quasi_steady


def test_e1_default_dormancy_unchanged(production_runs):
    run = production_runs[TRANSIENT_LUMPED]
    # Reference values recorded before the quasi-steady mode was added.
    assert run["vented"][-1] == pytest.approx(0.3202380, rel=1e-5)
    assert run["pressure"][-1] == pytest.approx(2004867.15, rel=1e-6)
    assert run["h2_temperature"][-1] == pytest.approx(25.57312, abs=1e-4)


def test_invalid_mode_rejected(harness):
    thermal = harness.thermal
    with pytest.raises(ValueError):
        _rebuild(thermal, "bogus")


def test_foam_with_quasi_steady_is_rejected(harness):
    with pytest.raises(ValueError):
        _rebuild(harness.thermal, QUASI_STEADY, insulation_model=ROHACELL)


def _rebuild(thermal, mode, insulation_model=None):
    return InsulatedTankThermalModel(
        tank_volume=thermal.V_t,
        inner_surface_area=thermal.A_in,
        inner_diameter=thermal.diameter,
        r_structure=thermal.r_structure,
        r_shell=thermal.r_shell,
        cylinder_length=thermal.L,
        liner_mass=thermal.m_liner,
        wall_mass=thermal.m_wall,
        foam_mass=thermal.m_foam,
        shell_mass=thermal.m_shell,
        ambient_temperature=thermal.T_amb,
        alpha_amb=thermal.alpha_amb,
        emissivity_shell=thermal.eps_shell,
        liner_material=thermal.liner_material,
        wall_material=thermal.wall_material,
        shell_material=thermal.shell_material,
        insulation_model=insulation_model or thermal.insulation_model,
        insulation_mode=mode,
    )


def test_e2_production_quasi_steady_matches_model_c(production_runs, harness):
    production = production_runs[QUASI_STEADY]
    model_c = harness.run_quasi_steady("C")
    assert production["vented"][-1] == pytest.approx(
        model_c.y[0, 0] - model_c.y[0, -1], rel=2e-2
    )
    assert production["pressure"][-1] == pytest.approx(model_c.pressure[-1], rel=1e-4)


def test_e2_quasi_steady_equals_transient_steady_state(harness):
    """At steady state the one-node transient flux equals the full-blanket flux (segmentation <~1 %)."""
    thermal = harness.thermal
    q_full = thermal.compute_blanket_heat_flow(T_SHELL, T_STRUCT)
    t_ins = thermal.reconstruct_insulation_temperature(T_SHELL, T_STRUCT)
    q_in = thermal.compute_shell_to_insulation_heat_flux(T_SHELL, t_ins)
    q_out = thermal.compute_insulation_to_structure_heat_flux(t_ins, T_STRUCT)
    assert q_in == pytest.approx(q_out, rel=1e-6)
    assert q_full == pytest.approx(q_out, rel=0.02)


def test_e3_zero_capacitance_limit(harness):
    """Transient one-node T_ins relaxes to the algebraic value as the capacitance -> 0.

    At the historical rho=300 (eps=1) the insulation has not equilibrated after 12 h,
    which is the physical origin of the transient/quasi-steady difference.
    """
    thermal = harness.thermal
    t_alg = thermal.reconstruct_insulation_temperature(T_SHELL, T_STRUCT)
    errors = {}
    for epsilon in (1.0, 0.1, 0.01, 0.001):
        capacity = thermal.m_foam * thermal.insulation_model.specific_heat(250.0) * epsilon

        def rhs(_, y):
            q_in = thermal.compute_shell_to_insulation_heat_flux(T_SHELL, y[0])
            q_out = thermal.compute_insulation_to_structure_heat_flux(y[0], T_STRUCT)
            return [(q_in - q_out) / capacity]

        sol = solve_ivp(rhs, (0, 12 * 3600.0), [250.0], method="LSODA", rtol=1e-8, atol=1e-8)
        assert sol.success
        errors[epsilon] = abs(sol.y[0, -1] - t_alg)
    values = [errors[e] for e in (1.0, 0.1, 0.01, 0.001)]
    assert values == sorted(values, reverse=True)
    assert errors[0.01] < 0.05 and errors[0.001] < 0.05


def test_e3_dormancy_converges_to_quasi_steady_as_capacitance_vanishes():
    quasi = DormancyHarness().run_quasi_steady("C")
    vented = quasi.y[0, 0] - quasi.y[0, -1]
    errors = []
    for epsilon in (1.0, 0.1, 0.01):
        harness = DormancyHarness(density=300.0 * epsilon, cp=lambda _: 1000.0)
        result = harness.run(1, "equilibrated")
        errors.append(abs(result.vented_mass - vented))
    assert errors[-1] <= errors[0] + 1e-9
    assert errors[-1] < 0.02 * max(vented, 1e-9) + 5e-3


def test_e4_energy_conservation_quasi_steady(harness):
    """Structure + H2 enthalpy change equals blanket heat in minus nothing else (no vent heat tracked)."""
    result = harness.run_quasi_steady("C", output_step=30.0)
    q_blanket = result.heat_to_structure
    q_h2 = result.cumulative_h2_heat_kj * 1000.0
    times = result.times
    cumulative_blanket = np.concatenate(
        ([0.0], np.cumsum(0.5 * (q_blanket[1:] + q_blanket[:-1]) * np.diff(times)))
    )
    t_struct = result.y[2]
    structure_enthalpy = np.concatenate(
        (
            [0.0],
            np.cumsum(
                [
                    harness._capacities(0.5 * (a + b), 288.0)[0] * (b - a)
                    for a, b in zip(t_struct[:-1], t_struct[1:])
                ]
            ),
        )
    )
    residual = cumulative_blanket - q_h2 - structure_enthalpy
    scale = max(abs(cumulative_blanket[-1]), 1.0)
    assert abs(residual[-1]) / scale < 0.02


@pytest.mark.parametrize("thickness", [0.010, 0.020, 0.030, 0.040, 0.050])
def test_e5_thickness_sensitivity_flux_consistency(thickness):
    harness = DormancyHarness(thickness=thickness)
    thermal = harness.thermal
    q_full = thermal.compute_blanket_heat_flow(T_SHELL, T_STRUCT)
    t_ins = thermal.reconstruct_insulation_temperature(T_SHELL, T_STRUCT)
    q_half = thermal.compute_shell_to_insulation_heat_flux(T_SHELL, t_ins)
    assert q_full > 0.0
    assert q_full == pytest.approx(q_half, rel=0.03)
