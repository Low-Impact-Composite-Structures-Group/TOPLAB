"""
Tests for the four-layer insulation thermal model and the 4-state DAE paradigm.

Covers:
- Rohacell 51A material properties
- InsulatedTankThermalModel physics (Q_structure, Q_insulation, Q_amb)
- IsochoricTankState 4-element state vector
- MultiTankState stride-4 assembly / disassembly
- ODE sign convention and energy balance directions
"""

import math
import pytest
import numpy as np

from toplab.materials.rohacell_properties import thermal_conductivity as rohacell_k, DENSITY as ROHACELL_DENSITY
from toplab.materials.nist_materials import NISTMetal, NISTComposite
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel
from toplab.thermodynamics.tank_states import (
    IsochoricTankState,
    IsochoricStateDerivatives,
    IsochoricInitialState,
)
from toplab.system.state_management import MultiTankState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thermal_model(
    r_inner=0.5,
    t_insulation=0.05,
    t_shell=0.002,
    L=1.5,
    alpha_amb=5.0,
    emissivity=0.05,
    T_amb=288.15,
):
    liner = NISTMetal.aluminum_6061T6_nist()
    wall = NISTComposite.carbon_epoxy_nist()
    r_structure = r_inner + 0.003 + 0.015  # liner + wall
    r_shell = r_structure + t_insulation
    A_in = 4.0 * math.pi * r_inner**2 + 2.0 * math.pi * r_inner * L

    def _layer_mass(rho, r_a, r_b):
        cyl = math.pi * (r_b**2 - r_a**2) * L
        sph = (4.0 / 3.0) * math.pi * (r_b**3 - r_a**3)
        return rho * (cyl + sph)

    liner_mass = _layer_mass(liner.density, r_inner, r_inner + 0.003)
    wall_mass  = _layer_mass(wall.density,  r_inner + 0.003, r_structure)
    shell_mass = _layer_mass(liner.density, r_shell, r_shell + t_shell)

    return InsulatedTankThermalModel(
        tank_volume=math.pi * r_inner**3 * (4.0 / 3.0 + L / r_inner),
        inner_surface_area=A_in,
        inner_diameter=2.0 * r_inner,
        r_structure=r_structure,
        r_shell=r_shell,
        cylinder_length=L,
        liner_mass=liner_mass,
        wall_mass=wall_mass,
        shell_mass=shell_mass,
        ambient_temperature=T_amb,
        alpha_amb=alpha_amb,
        emissivity_shell=emissivity,
        liner_material=liner,
        wall_material=wall,
        shell_material=liner,
    )


class _MockTank:
    def __init__(self, volume):
        self.volume = volume

    def compute_fuel_height(self, fuel_volume):
        return 0.0


def _make_state(T_fluid=54.0, T_structure=60.0, T_shell=288.0, volume=2.5, mass=None):
    tank = _MockTank(volume)
    if mass is None:
        mass = 70.0 * volume  # ~70 kg/m³
    return IsochoricTankState(
        tank=tank,
        fuel_mass=mass,
        temperature=T_fluid,
        solid_temperature=T_structure,
        shell_temperature=T_shell,
    )


# ---------------------------------------------------------------------------
# Rohacell material
# ---------------------------------------------------------------------------

class TestRohacellProperties:

    def test_density(self):
        assert ROHACELL_DENSITY == pytest.approx(51.1)

    def test_conductivity_at_low_temperature(self):
        k = rohacell_k(20.0)
        assert 0.004 < k < 0.006  # ~0.005 W/mK at 20 K

    def test_conductivity_at_ambient(self):
        k = rohacell_k(288.0)
        assert 0.025 < k < 0.035  # ~0.029 W/mK at 288 K

    def test_conductivity_increases_with_temperature(self):
        assert rohacell_k(50.0) < rohacell_k(150.0) < rohacell_k(280.0)

    def test_conductivity_clamped_below_data_range(self):
        assert rohacell_k(1.0) == rohacell_k(20.0)

    def test_conductivity_clamped_above_data_range(self):
        # data runs to ~324.5 K; values above and far above should be identical
        assert rohacell_k(400.0) == rohacell_k(350.0)


# ---------------------------------------------------------------------------
# InsulatedTankThermalModel
# ---------------------------------------------------------------------------

class TestInsulatedTankThermalModel:

    def test_construction_raises_if_r_shell_leq_r_structure(self):
        liner = NISTMetal.aluminum_6061T6_nist()
        wall = NISTComposite.carbon_epoxy_nist()
        with pytest.raises(ValueError, match="Shell radius"):
            InsulatedTankThermalModel(
                tank_volume=1.0,
                inner_surface_area=4.0,
                inner_diameter=1.0,
                r_structure=0.6,
                r_shell=0.6,      # equal → invalid
                cylinder_length=1.0,
                liner_mass=50.0,
                wall_mass=100.0,
                shell_mass=20.0,
                ambient_temperature=288.15,
                alpha_amb=5.0,
                emissivity_shell=0.05,
                liner_material=liner,
                wall_material=wall,
                shell_material=liner,
            )

    def test_a_shell_formula(self):
        model = _make_thermal_model(r_inner=0.5, t_insulation=0.05, L=1.5)
        r_sh = model.r_shell
        L = model.L
        expected = 2.0 * math.pi * r_sh * L + 4.0 * math.pi * r_sh**2
        assert model.A_shell == pytest.approx(expected, rel=1e-9)

    def test_q_amb_positive_when_shell_colder_than_ambient(self):
        model = _make_thermal_model(T_amb=288.15)
        Q = model.compute_ambient_heat_flux(T_shell=250.0)
        assert Q > 0.0

    def test_q_amb_negative_when_shell_hotter_than_ambient(self):
        model = _make_thermal_model(T_amb=288.15)
        Q = model.compute_ambient_heat_flux(T_shell=350.0)
        assert Q < 0.0

    def test_q_insulation_positive_when_shell_hotter_than_structure(self):
        model = _make_thermal_model()
        Q = model.compute_insulation_heat_flux(T_structure=60.0, T_shell=288.0)
        assert Q > 0.0

    def test_q_insulation_zero_when_equal_temperatures(self):
        model = _make_thermal_model()
        Q = model.compute_insulation_heat_flux(T_structure=100.0, T_shell=100.0)
        assert Q == pytest.approx(0.0, abs=1e-10)

    def test_q_insulation_scales_with_conductivity_and_geometry(self):
        """Thicker insulation → lower Q_insulation for same ΔT."""
        model_thin = _make_thermal_model(t_insulation=0.02)
        model_thick = _make_thermal_model(t_insulation=0.10)
        Q_thin  = model_thin.compute_insulation_heat_flux(60.0, 288.0)
        Q_thick = model_thick.compute_insulation_heat_flux(60.0, 288.0)
        assert Q_thick < Q_thin

    def test_q_structure_positive_when_structure_hotter_than_fluid(self):
        model = _make_thermal_model()
        state = _make_state(T_fluid=54.0, T_structure=60.0, T_shell=288.0)
        Q = model.compute_heat_flux(0.0, state)
        assert Q > 0.0

    def test_structure_ode_sign_convention(self):
        """dT_structure/dt > 0 when Q_insulation > Q_structure (net heating)."""
        model = _make_thermal_model()
        # Large T gradient across insulation → large Q_insulation
        # Small T gradient fluid↔structure → small Q_structure
        state = _make_state(T_fluid=54.0, T_structure=54.1, T_shell=288.0)
        dTs = model.compute_structure_temperature_derivative(0.0, state)
        assert dTs > 0.0  # structure warms: Q_insulation dominates

    def test_shell_ode_sign_convention(self):
        """dT_shell/dt > 0 when Q_amb > Q_insulation (net heating of shell)."""
        model = _make_thermal_model(T_amb=288.15)
        # Shell slightly below ambient → Q_amb > 0; small insulation load
        state = _make_state(T_fluid=286.0, T_structure=286.5, T_shell=287.0)
        dTsh = model.compute_shell_temperature_derivative(0.0, state)
        assert dTsh > 0.0

    def test_legacy_alias_matches_structure_derivative(self):
        model = _make_thermal_model()
        state = _make_state()
        assert model.compute_solid_temperature_derivative(0.0, state) == pytest.approx(
            model.compute_structure_temperature_derivative(0.0, state)
        )


# ---------------------------------------------------------------------------
# IsochoricTankState — 4-element state vector
# ---------------------------------------------------------------------------

class TestIsochoricTankState4State:

    def test_state_vector_has_four_elements(self):
        state = _make_state()
        assert len(state.state_vector) == 4

    def test_state_vector_order(self):
        state = _make_state(T_fluid=54.0, T_structure=60.0, T_shell=288.0)
        sv = state.state_vector
        assert sv[1] == pytest.approx(54.0)
        assert sv[2] == pytest.approx(60.0)
        assert sv[3] == pytest.approx(288.0)

    def test_from_state_vector_round_trip(self):
        state = _make_state(T_fluid=54.0, T_structure=61.0, T_shell=285.0, mass=180.0)
        recovered = IsochoricTankState.from_state_vector(state.tank, state.state_vector)
        assert recovered.temperature      == pytest.approx(54.0)
        assert recovered.solid_temperature == pytest.approx(61.0)
        assert recovered.shell_temperature == pytest.approx(285.0)
        assert recovered.fuel_mass         == pytest.approx(180.0)

    def test_update_from_state_vector(self):
        state = _make_state()
        new_sv = [150.0, 55.0, 62.0, 290.0]
        state.update_from_state_vector(new_sv)
        assert state.fuel_mass         == pytest.approx(150.0)
        assert state.temperature       == pytest.approx(55.0)
        assert state.solid_temperature == pytest.approx(62.0)
        assert state.shell_temperature == pytest.approx(290.0)

    def test_initial_state_get_state_vector(self):
        init = IsochoricInitialState(
            fuel_mass=200.0, temperature=54.0, solid_temperature=54.1, shell_temperature=288.15
        )
        sv = init.get_state_vector()
        assert len(sv) == 4
        assert sv[3] == pytest.approx(288.15)


# ---------------------------------------------------------------------------
# IsochoricStateDerivatives — shell_temperature_derivative field
# ---------------------------------------------------------------------------

class TestIsochoricStateDerivatives4State:

    def test_derivative_vector_has_four_elements(self):
        d = IsochoricStateDerivatives(
            fuel_mass_derivative=-0.01,
            temperature_derivative=0.001,
            solid_temperature_derivative=0.0005,
            shell_temperature_derivative=-0.0001,
        )
        assert len(d.state_derivative_vector) == 4
        assert d.state_derivative_vector[3] == pytest.approx(-0.0001)


# ---------------------------------------------------------------------------
# MultiTankState — stride-4 assembly
# ---------------------------------------------------------------------------

class TestMultiTankStateStride4:

    def _make_tanks(self, n):
        return [_MockTank(2.5) for _ in range(n)]

    def test_state_vector_length_is_4n(self):
        tanks = self._make_tanks(2)
        states = [_make_state(volume=2.5) for _ in tanks]
        ms = MultiTankState(tank_states=states)
        assert len(ms.state_vector) == 8

    def test_from_state_vector_stride_4(self):
        tanks = self._make_tanks(2)
        y = np.array([100.0, 54.0, 60.0, 288.0,
                      200.0, 30.0, 35.0, 280.0])
        ms = MultiTankState.from_state_vector(y, tanks)
        assert ms.tank_states[0].fuel_mass         == pytest.approx(100.0)
        assert ms.tank_states[0].temperature       == pytest.approx(54.0)
        assert ms.tank_states[0].solid_temperature == pytest.approx(60.0)
        assert ms.tank_states[0].shell_temperature == pytest.approx(288.0)
        assert ms.tank_states[1].fuel_mass         == pytest.approx(200.0)
        assert ms.tank_states[1].shell_temperature == pytest.approx(280.0)

    def test_wrong_length_raises(self):
        tanks = self._make_tanks(2)
        with pytest.raises(ValueError, match="4 \\* 2"):
            MultiTankState.from_state_vector(np.zeros(6), tanks)

    def test_state_vector_round_trip(self):
        tanks = self._make_tanks(1)
        y = np.array([180.0, 54.0, 61.0, 287.5])
        ms = MultiTankState.from_state_vector(y, tanks)
        recovered = ms.state_vector
        np.testing.assert_allclose(recovered, y, rtol=1e-6)
