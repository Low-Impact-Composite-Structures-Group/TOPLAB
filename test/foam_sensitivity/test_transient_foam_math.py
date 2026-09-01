"""
Mathematical verification tests for the foam sensitivity study helpers.

These tests verify the internal correctness of foam_sensitivity_study.py
(geometry, resistance splitting, energy balance).  They do NOT independently
test the production InsulatedTankThermalModel — that is covered end-to-end
by D2/D3/D4 in dormancy_discharge_verification/ and by
test_insulation_analytical_verification.py.

Tests
-----
T0  Q_foam_qs_split exactly matches InsulatedTankThermalModel.compute_insulation_heat_flux.
T1  Foam mass formula matches _layer_mass for all sweep thicknesses.
T2  Foam thermal capacitance is positive over the cryogenic-to-ambient range.
T3  Cylindrical midpoint radius gives equal log half-resistances.
T4  Spherical midpoint radius gives equal inverse half-resistances.
T5  Half conductances equal twice the total conductance (constant-k identity).
T6  Series combination of half conductances reconstructs the total (constant k).
T7  Foam energy balance residual is near zero during a short TF simulation.
T8  At steady state (long run) both halves carry the same heat flow.
T9  QS and TF models agree on total conductance for constant k.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

# Allow import from the sibling study module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from foam_sensitivity_study import (
    R_STRUCTURE, L_CYL, T_STRUCT_FIXED, T_EVAL,
    ROHACELL_DENSITY, R_INNER, T_LINER, T_SHELL_THK, A_IN,
    build_geometry, TankGeometry,
    _layer_mass,
    cap_foam,
    Q_foam_qs_split, Q_shell_to_foam, Q_foam_to_structure,
    _G_half, _integrate,
    _tf_rhs_a,
    SWEEP_THICKNESSES_M,
)
from toplab.materials.rohacell_properties import (
    thermal_conductivity as rohacell_k,
    specific_heat as rohacell_cp,
)


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def geo50() -> TankGeometry:
    return build_geometry(0.050)


# ══════════════════════════════════════════════════════════════════════════════
# T0 — Production formula alignment
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("t_ins", SWEEP_THICKNESSES_M)
def test_q_foam_qs_split_matches_production_insulation_model(t_ins: float) -> None:
    """
    Q_foam_qs_split must reproduce InsulatedTankThermalModel.compute_insulation_heat_flux
    for identical geometry — both solve the same brentq problem independently.
    """
    from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel
    from toplab.materials.nist_materials import NISTMetal, NISTComposite

    geo   = build_geometry(t_ins)
    liner = NISTMetal.aluminum_6061T6_nist()
    wall  = NISTComposite.carbon_epoxy_nist()
    liner_mass = _layer_mass(liner.density, R_INNER, R_INNER + T_LINER)
    wall_mass  = _layer_mass(wall.density,  R_INNER + T_LINER, R_STRUCTURE)
    shell_mass = _layer_mass(liner.density, geo.r_shell, geo.r_shell + T_SHELL_THK)

    model = InsulatedTankThermalModel(
        tank_volume         = 1.0,          # unused by insulation formula
        inner_surface_area  = A_IN,
        inner_diameter      = 2.0 * R_INNER,
        r_structure         = R_STRUCTURE,
        r_shell             = geo.r_shell,
        cylinder_length     = L_CYL,
        liner_mass          = liner_mass,
        wall_mass           = wall_mass,
        shell_mass          = shell_mass,
        ambient_temperature = 288.15,
        alpha_amb           = 5.0,
        emissivity_shell    = 0.05,
        liner_material      = liner,
        wall_material       = wall,
        shell_material      = liner,
    )

    for T_s, T_sh in [(60.0, 250.0), (30.0, 100.0), (20.0, 288.0)]:
        Q_study = Q_foam_qs_split(T_sh, T_s, geo)               # study: (T_sh, T_s)
        Q_prod  = model.compute_insulation_heat_flux(T_s, T_sh)  # model: (T_s, T_sh)
        assert Q_study == pytest.approx(Q_prod, rel=1e-6), (
            f"t_ins={t_ins*1e3:.0f} mm, T_s={T_s}, T_sh={T_sh}: "
            f"Q_foam_qs_split={Q_study:.6f} W ≠ InsulatedTankThermalModel={Q_prod:.6f} W"
        )


# ══════════════════════════════════════════════════════════════════════════════
# T1 — Foam mass formula
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("t_ins", SWEEP_THICKNESSES_M)
def test_foam_mass_positive_and_finite(t_ins: float) -> None:
    geo = build_geometry(t_ins)
    assert geo.m_foam > 0.0, f"Foam mass non-positive at t_ins={t_ins*1000:.0f} mm"
    assert math.isfinite(geo.m_foam), "Foam mass is not finite"


@pytest.mark.parametrize("t_ins", SWEEP_THICKNESSES_M)
def test_foam_mass_matches_layer_mass_formula(t_ins: float) -> None:
    geo = build_geometry(t_ins)
    expected = _layer_mass(ROHACELL_DENSITY, R_STRUCTURE, geo.r_shell)
    assert geo.m_foam == pytest.approx(expected, rel=1e-12)


def test_foam_mass_increases_with_thickness() -> None:
    masses = [build_geometry(t).m_foam for t in SWEEP_THICKNESSES_M]
    for i in range(len(masses) - 1):
        assert masses[i] < masses[i + 1], "Foam mass should increase with thickness"


# ══════════════════════════════════════════════════════════════════════════════
# T2 — Foam thermal capacitance
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T_f", [30.0, 60.0, 100.0, 150.0, 200.0, 288.0])
def test_foam_capacitance_positive(T_f: float, geo50: TankGeometry) -> None:
    C = cap_foam(T_f, geo50)
    assert C > 0.0, f"C_foam non-positive at T={T_f:.0f} K"
    assert math.isfinite(C), f"C_foam not finite at T={T_f:.0f} K"


# ══════════════════════════════════════════════════════════════════════════════
# T3 — Cylindrical equal-resistance midpoint
# ══════════════════════════════════════════════════════════════════════════════

def test_cylindrical_midpoint_equal_log_resistances(geo50: TankGeometry) -> None:
    """ln(r_m/r_s) must equal ln(r_sh/r_m) — the defining property of r_m_cyl."""
    r_s   = R_STRUCTURE
    r_sh  = geo50.r_shell
    r_m   = geo50.r_m_cyl
    ln_in  = math.log(r_m  / r_s)
    ln_out = math.log(r_sh / r_m)
    assert ln_in == pytest.approx(ln_out, rel=1e-12), (
        f"Cylindrical half-log-resistances differ: ln_in={ln_in:.6e}, ln_out={ln_out:.6e}"
    )


def test_cylindrical_midpoint_is_geometric_mean(geo50: TankGeometry) -> None:
    assert geo50.r_m_cyl == pytest.approx(
        math.sqrt(R_STRUCTURE * geo50.r_shell), rel=1e-12
    )


# ══════════════════════════════════════════════════════════════════════════════
# T4 — Spherical equal-resistance midpoint
# ══════════════════════════════════════════════════════════════════════════════

def test_spherical_midpoint_equal_inverse_resistances(geo50: TankGeometry) -> None:
    """1/r_s − 1/r_m must equal 1/r_m − 1/r_sh — the defining property of r_m_sph."""
    r_s   = R_STRUCTURE
    r_sh  = geo50.r_shell
    r_m   = geo50.r_m_sph
    R_in  = 1.0 / r_s  - 1.0 / r_m
    R_out = 1.0 / r_m  - 1.0 / r_sh
    assert R_in == pytest.approx(R_out, rel=1e-12), (
        f"Spherical half-inverse-resistances differ: R_in={R_in:.6e}, R_out={R_out:.6e}"
    )


def test_spherical_midpoint_is_harmonic_mean(geo50: TankGeometry) -> None:
    r_s, r_sh = R_STRUCTURE, geo50.r_shell
    expected = 2.0 * r_s * r_sh / (r_s + r_sh)
    assert geo50.r_m_sph == pytest.approx(expected, rel=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# T5 — Half conductances equal twice the total (constant-k identity)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("t_ins", SWEEP_THICKNESSES_M)
def test_half_conductances_equal_twice_total_for_constant_k(t_ins: float) -> None:
    """For constant k, each half conductance should be 2 × the whole-layer conductance."""
    geo = build_geometry(t_ins)
    k = 0.020  # W/mK, arbitrary constant

    r_s, r_sh = R_STRUCTURE, geo.r_shell
    G_cyl_total = 2.0 * math.pi * L_CYL * k / math.log(r_sh / r_s)
    G_cap_total = 4.0 * math.pi * k * r_s * r_sh / (r_sh - r_s)
    G_total = G_cyl_total + G_cap_total

    G_out = _G_half(geo.r_m_cyl, r_sh, geo.r_m_sph, r_sh, k)
    G_in  = _G_half(r_s, geo.r_m_cyl, r_s, geo.r_m_sph, k)

    assert G_out == pytest.approx(2.0 * G_total, rel=1e-10), (
        f"G_out={G_out:.6e} ≠ 2×G_total={2*G_total:.6e}"
    )
    assert G_in == pytest.approx(2.0 * G_total, rel=1e-10), (
        f"G_in={G_in:.6e} ≠ 2×G_total={2*G_total:.6e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# T6 — Series half-conductances reconstruct total conductance (constant k)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("t_ins", SWEEP_THICKNESSES_M)
def test_series_half_conductances_reconstruct_total(t_ins: float) -> None:
    """1/G_total = 1/G_out + 1/G_in for constant k (series thermal resistances)."""
    geo = build_geometry(t_ins)
    k = 0.020

    r_s, r_sh = R_STRUCTURE, geo.r_shell
    G_total = (2.0 * math.pi * L_CYL * k / math.log(r_sh / r_s)
               + 4.0 * math.pi * k * r_s * r_sh / (r_sh - r_s))
    G_out = _G_half(geo.r_m_cyl, r_sh, geo.r_m_sph, r_sh, k)
    G_in  = _G_half(r_s, geo.r_m_cyl, r_s, geo.r_m_sph, k)

    G_series = 1.0 / (1.0 / G_out + 1.0 / G_in)
    assert G_series == pytest.approx(G_total, rel=1e-10), (
        f"G_series={G_series:.6e} ≠ G_total={G_total:.6e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# T7 — Foam energy balance residual is near zero during TF simulation
# ══════════════════════════════════════════════════════════════════════════════

def test_foam_ode_energy_balance_residual(geo50: TankGeometry) -> None:
    """
    Numerically verify  C_foam * dT_foam/dt ≈ Q_shell_to_foam − Q_foam_to_structure.

    This identity holds by construction (it is the ODE). The residual should be
    bounded by solver tolerance plus numerical differentiation error (numpy.gradient).
    """
    T_sh0, T_f0 = 250.0, 155.0
    T_amb = 288.15

    # Short dense grid to make np.gradient accurate
    t_short = np.linspace(0.0, 5_000.0, 501)
    y_tf = _integrate(_tf_rhs_a, [T_sh0, T_f0], geo50, T_amb, t_eval=t_short)
    T_sh_tf, T_f_tf = y_tf[0], y_tf[1]

    n = len(t_short)
    Qsf = np.array([Q_shell_to_foam(T_sh_tf[i], T_f_tf[i], geo50) for i in range(n)])
    Qfs = np.array([Q_foam_to_structure(T_f_tf[i], T_STRUCT_FIXED, geo50) for i in range(n)])
    C_f = np.array([cap_foam(T_f_tf[i], geo50) for i in range(n)])

    dTdt   = np.gradient(T_f_tf, t_short)
    resid  = C_f * dTdt - (Qsf - Qfs)
    Q_scale = np.max(np.abs(Qsf)) + 1e-10

    # Allow 1 % relative residual (dominated by np.gradient endpoint accuracy)
    rel_resid = np.abs(resid) / Q_scale
    # Exclude endpoints where np.gradient is less accurate
    assert np.max(rel_resid[2:-2]) < 0.01, (
        f"Foam energy balance residual exceeded 1 %: "
        f"max rel = {np.max(rel_resid[2:-2]):.3e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# T8 — At steady state both foam halves carry the same heat flow
# ══════════════════════════════════════════════════════════════════════════════

def test_foam_halves_balance_at_steady_state(geo50: TankGeometry) -> None:
    """
    At long time (dT_foam/dt → 0) the two TF heat flows must converge:
        Q_shell_to_foam ≈ Q_foam_to_structure.
    """
    T_sh0, T_f0 = 250.0, 155.0
    T_amb = 288.15
    t_long = np.linspace(0.0, 8.0 * T_EVAL[-1], 501)  # ~48 h

    y_tf = _integrate(_tf_rhs_a, [T_sh0, T_f0], geo50, T_amb, t_eval=t_long)
    T_sh_tf, T_f_tf = y_tf[0], y_tf[1]

    n = len(t_long)
    Qsf = np.array([Q_shell_to_foam(T_sh_tf[i], T_f_tf[i], geo50) for i in range(n)])
    Qfs = np.array([Q_foam_to_structure(T_f_tf[i], T_STRUCT_FIXED, geo50) for i in range(n)])

    # Use the last 5 % of the run for the steady-state check
    ss = t_long >= 0.95 * t_long[-1]
    Q_mean = np.mean(np.abs(Qsf[ss])) + 1e-10
    imbalance = np.max(np.abs(Qsf[ss] - Qfs[ss])) / Q_mean

    assert imbalance < 1e-4, (
        f"TF foam halves not balanced at steady state: "
        f"max relative imbalance = {imbalance:.3e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# T9 — QS and TF models agree on total conductance (algebraic, constant k)
# ══════════════════════════════════════════════════════════════════════════════

def test_qs_and_tf_agree_on_total_conductance_constant_k(geo50: TankGeometry) -> None:
    """
    For constant k, the TF split model at steady state (T_foam = (T_sh+T_s)/2)
    delivers Q = G_total * ΔT, identical to the full-layer analytical formula.
    Confirms that T5/T6 (equal-resistance split) is algebraically self-consistent.
    """
    T_sh, T_s = 288.0, 60.0
    k_const = 0.020  # W/mK

    # QS conductance (full layer)
    r_s, r_sh = R_STRUCTURE, geo50.r_shell
    G_total = (2.0 * math.pi * L_CYL * k_const / math.log(r_sh / r_s)
               + 4.0 * math.pi * k_const * r_s * r_sh / (r_sh - r_s))
    Q_QS = G_total * (T_sh - T_s)

    # TF at steady-state foam temperature T_foam = (T_sh + T_s)/2
    T_f_ss = 0.5 * (T_sh + T_s)
    G_out = _G_half(geo50.r_m_cyl, r_sh, geo50.r_m_sph, r_sh, k_const)
    G_in  = _G_half(r_s, geo50.r_m_cyl, r_s, geo50.r_m_sph, k_const)
    Q_TF_out = G_out * (T_sh - T_f_ss)
    Q_TF_in  = G_in  * (T_f_ss - T_s)

    # Both halves should carry the same heat (energy balance at SS)
    assert Q_TF_out == pytest.approx(Q_TF_in, rel=1e-10), (
        "TF heat flows unbalanced at analytical SS: "
        f"Q_out={Q_TF_out:.6f} W, Q_in={Q_TF_in:.6f} W"
    )
    # Total should match QS
    assert Q_TF_out == pytest.approx(Q_QS, rel=1e-10), (
        f"TF SS heat flow {Q_TF_out:.6f} W ≠ QS {Q_QS:.6f} W for constant k"
    )



