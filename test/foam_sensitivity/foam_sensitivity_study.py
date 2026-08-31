"""
Sensitivity study: quasi-steady (QS) vs one-state transient-foam (TF) thermal models.

Cases
-----
A  Step change in T_amb; T_structure fixed.  Isolates shell–foam coupling.
B  Full shell + structure network; T_fluid fixed.  Tests structure-side heat input.

Foam-thickness sweep (25–100 mm) and initial-gradient sensitivity are also included.

Usage:  python foam_sensitivity_study.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq

from toplab.materials.rohacell_properties import (
    thermal_conductivity as rohacell_k,
    specific_heat as rohacell_cp,
    DENSITY as ROHACELL_DENSITY,
)
from toplab.materials.nist_materials import NISTMetal, NISTComposite

# ── Physical constant ─────────────────────────────────────────────────────────
SIGMA = 5.67e-8  # W/m²K⁴

# ── Reference geometry (mirrors test_insulation_analytical_verification.py) ───
R_INNER     = 0.500                           # m, fluid-facing inner radius
T_LINER     = 0.003                           # m
T_WALL      = 0.015                           # m
R_STRUCTURE = R_INNER + T_LINER + T_WALL      # 0.518 m, outer liner+wall radius
L_CYL       = 1.500                           # m, cylindrical section length
T_SHELL_THK = 0.002                           # m, thin Al outer shell
A_IN = 4.0 * math.pi * R_INNER**2 + 2.0 * math.pi * R_INNER * L_CYL

# ── External-boundary and inner-HTC parameters ────────────────────────────────
ALPHA_AMB  = 5.0    # W/m²K, ambient convective HTC
EMISSIVITY = 0.05   # outer-shell emissivity
ALPHA_S    = 10.0   # W/m²K, fixed inner natural-convection HTC (Cases A, B)
T_FLUID    = 54.0   # K, fixed hydrogen temperature (Cases A, B)

# ── Case A boundary conditions ────────────────────────────────────────────────
T_AMB_INIT     = 250.0   # K, pre-step ambient temperature
T_AMB_STEP     = 288.15  # K, post-step ambient temperature
T_STRUCT_FIXED = 60.0    # K, fixed structure temperature (Case A only)

# ── Simulation output grid ────────────────────────────────────────────────────
T_END  = 21_600.0    # s (6 h)
T_EVAL = np.concatenate([
    np.arange(0.0, 3_601.0, 30.0),       # dense:  every 30 s for first hour
    np.arange(3_720.0, T_END + 1.0, 120.0),  # coarse: every 2 min thereafter
])

# ── Foam thicknesses for the sweep ────────────────────────────────────────────
SWEEP_THICKNESSES_M = [0.025, 0.050, 0.075, 0.100]

# ── Delft colour palette ──────────────────────────────────────────────────────
_C = {
    "qs_shell":   "#0C2340",  # dark blue
    "tf_shell":   "#0076C2",  # royal blue
    "tf_foam":    "#00B8C8",  # teal
    "qs_struct":  "#A50034",  # bordeaux
    "tf_struct":  "#E03C31",  # red
    "Q_qs":       "#0C2340",
    "Q_tf_outer": "#0076C2",
    "Q_tf_inner": "#E03C31",
    "delta":      "#6F1D77",  # purple
    "E_qs":         "#0C2340",
    "E_tf":         "#E03C31",
    "residual":     "#EC6842",  # orange
    "qs_split_shell": "#6CC24A",  # green — QS-split model
    "Q_qs_split":    "#6CC24A",
}

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ══════════════════════════════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════════════════════════════

def _layer_mass(rho: float, r_a: float, r_b: float) -> float:
    """Mass of a cylindrical+spherical shell layer."""
    cyl = math.pi * (r_b**2 - r_a**2) * L_CYL
    sph = (4.0 / 3.0) * math.pi * (r_b**3 - r_a**3)
    return rho * (cyl + sph)


@dataclass
class TankGeometry:
    t_insul:   float   # foam layer thickness [m]
    r_shell:   float   # outer foam surface radius [m]
    A_shell:   float   # outer shell surface area [m²]
    m_liner:   float   # [kg]
    m_wall:    float   # [kg]
    m_shell:   float   # [kg]
    m_foam:    float   # full foam mass [kg]  — assigned to single TF node
    r_m_cyl:   float   # cylindrical equal-resistance midpoint radius [m]
    r_m_sph:   float   # spherical equal-resistance midpoint radius [m]
    liner_mat: object  # NISTMetal (Al 6061-T6)
    wall_mat:  object  # NISTComposite (carbon-epoxy)
    shell_mat: object  # NISTMetal (Al 6061-T6, same alloy as liner)


def build_geometry(t_insul: float = 0.050) -> TankGeometry:
    liner = NISTMetal.aluminum_6061T6_nist()
    wall  = NISTComposite.carbon_epoxy_nist()
    r_sh  = R_STRUCTURE + t_insul
    r_sh_outer = r_sh + T_SHELL_THK
    A_shell = 2.0 * math.pi * r_sh * L_CYL + 4.0 * math.pi * r_sh**2

    # Equal-resistance midpoint radii (Sections 4, 7 of request)
    # Cylindrical: ln(r_m/r_s) = ln(r_sh/r_m) → r_m = sqrt(r_s * r_sh)
    r_m_cyl = math.sqrt(R_STRUCTURE * r_sh)
    # Spherical:   1/r_s - 1/r_m = 1/r_m - 1/r_sh → r_m = 2*r_s*r_sh/(r_s+r_sh)
    r_m_sph = 2.0 * R_STRUCTURE * r_sh / (R_STRUCTURE + r_sh)

    return TankGeometry(
        t_insul   = t_insul,
        r_shell   = r_sh,
        A_shell   = A_shell,
        m_liner   = _layer_mass(liner.density,      R_INNER,          R_INNER + T_LINER),
        m_wall    = _layer_mass(wall.density,        R_INNER + T_LINER, R_STRUCTURE),
        m_shell   = _layer_mass(liner.density,       r_sh,             r_sh_outer),
        m_foam    = _layer_mass(ROHACELL_DENSITY,    R_STRUCTURE,      r_sh),
        r_m_cyl   = r_m_cyl,
        r_m_sph   = r_m_sph,
        liner_mat = liner,
        wall_mat  = wall,
        shell_mat = liner,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Thermal capacitances
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(T: float, lo: float = 4.0, hi: float = 400.0) -> float:
    return max(lo, min(float(T), hi))


def cap_shell(T: float, geo: TankGeometry) -> float:
    return geo.m_shell * geo.shell_mat.determine_specific_heat(_clamp(T))


def cap_structure(T: float, geo: TankGeometry) -> float:
    Tc = _clamp(T)
    return (geo.m_liner * geo.liner_mat.determine_specific_heat(Tc)
            + geo.m_wall * geo.wall_mat.determine_specific_heat(Tc))


def cap_foam(T: float, geo: TankGeometry) -> float:
    """Full foam mass assigned to single TF node (Section 5 of request)."""
    return geo.m_foam * rohacell_cp(_clamp(T))


# ══════════════════════════════════════════════════════════════════════════════
# Heat fluxes
# ══════════════════════════════════════════════════════════════════════════════

def Q_amb(T_sh: float, T_amb: float, geo: TankGeometry) -> float:
    """Ambient → shell: natural convection + radiation [W]. Positive when T_amb > T_sh."""
    return (ALPHA_AMB * geo.A_shell * (T_amb - T_sh)
            + EMISSIVITY * SIGMA * geo.A_shell * (T_amb**4 - T_sh**4))


def Q_foam_qs(T_s: float, T_sh: float, geo: TankGeometry) -> float:
    """QS: single-layer Fourier radial conduction [W] (production formula).
    Positive when T_sh > T_s."""
    T_m = 0.5 * (T_s + T_sh)
    k   = rohacell_k(_clamp(T_m))
    r_s, r_sh = R_STRUCTURE, geo.r_shell
    G_cyl = 2.0 * math.pi * L_CYL * k / math.log(r_sh / r_s)
    G_cap = 4.0 * math.pi * k * r_s * r_sh / (r_sh - r_s)
    return (G_cyl + G_cap) * (T_sh - T_s)


def _G_half(r_cyl_in: float, r_cyl_out: float,
             r_sph_in: float, r_sph_out: float, k: float) -> float:
    """Combined cylindrical + spherical conductance for one foam half-layer [W/K]."""
    G_cyl = 2.0 * math.pi * L_CYL * k / math.log(r_cyl_out / r_cyl_in)
    G_cap = 4.0 * math.pi * k * r_sph_in * r_sph_out / (r_sph_out - r_sph_in)
    return G_cyl + G_cap


def Q_shell_to_foam(T_sh: float, T_f: float, geo: TankGeometry) -> float:
    """TF outer half: shell → foam node [W]. Uses T_m_out = (T_sh + T_f)/2."""
    k = rohacell_k(_clamp(0.5 * (T_sh + T_f)))
    return _G_half(geo.r_m_cyl, geo.r_shell, geo.r_m_sph, geo.r_shell, k) * (T_sh - T_f)


def Q_foam_to_structure(T_f: float, T_s: float, geo: TankGeometry) -> float:
    """TF inner half: foam node → structure [W]. Uses T_m_in = (T_f + T_s)/2."""
    k = rohacell_k(_clamp(0.5 * (T_f + T_s)))
    return _G_half(R_STRUCTURE, geo.r_m_cyl, R_STRUCTURE, geo.r_m_sph, k) * (T_f - T_s)


def Q_struct_to_fluid(T_s: float) -> float:
    """Structure → fluid (fixed α_s, fixed T_fluid) [W]."""
    return ALPHA_S * A_IN * (T_s - T_FLUID)


def _solve_foam_qs_split(T_sh: float, T_s: float, geo: TankGeometry) -> float:
    """Algebraically solve for the foam temperature where Q_shell_to_foam = Q_foam_to_structure."""
    if abs(T_sh - T_s) < 1e-8:
        return 0.5 * (T_sh + T_s)
    lo, hi = min(T_sh, T_s) + 1e-6, max(T_sh, T_s) - 1e-6
    def res(T_f: float) -> float:
        return Q_shell_to_foam(T_sh, T_f, geo) - Q_foam_to_structure(T_f, T_s, geo)
    return brentq(res, lo, hi, xtol=1e-8)


def Q_foam_qs_split(T_sh: float, T_s: float, geo: TankGeometry) -> float:
    """QS-split heat flow: split-resistance geometry with zero foam capacitance [W]."""
    return Q_shell_to_foam(T_sh, _solve_foam_qs_split(T_sh, T_s, geo), geo)


# ══════════════════════════════════════════════════════════════════════════════
# ODE right-hand sides
# ══════════════════════════════════════════════════════════════════════════════

def _qs_rhs_a(t: float, y: np.ndarray, geo: TankGeometry, T_amb: float) -> list:
    """QS Case A: state = [T_shell]; T_structure pinned."""
    T_sh = y[0]
    return [(Q_amb(T_sh, T_amb, geo) - Q_foam_qs(T_STRUCT_FIXED, T_sh, geo)) / cap_shell(T_sh, geo)]


def _tf_rhs_a(t: float, y: np.ndarray, geo: TankGeometry, T_amb: float) -> list:
    """TF Case A: state = [T_shell, T_foam]; T_structure pinned."""
    T_sh, T_f = y
    Qsf = Q_shell_to_foam(T_sh, T_f, geo)
    Qfs = Q_foam_to_structure(T_f, T_STRUCT_FIXED, geo)
    return [
        (Q_amb(T_sh, T_amb, geo) - Qsf) / cap_shell(T_sh, geo),
        (Qsf - Qfs) / cap_foam(T_f, geo),
    ]


def _qs_rhs_b(t: float, y: np.ndarray, geo: TankGeometry, T_amb: float) -> list:
    """QS Case B: state = [T_shell, T_structure]."""
    T_sh, T_s = y
    Qf = Q_foam_qs(T_s, T_sh, geo)
    return [
        (Q_amb(T_sh, T_amb, geo) - Qf) / cap_shell(T_sh, geo),
        (Qf - Q_struct_to_fluid(T_s)) / cap_structure(T_s, geo),
    ]


def _tf_rhs_b(t: float, y: np.ndarray, geo: TankGeometry, T_amb: float) -> list:
    """TF Case B: state = [T_shell, T_foam, T_structure]."""
    T_sh, T_f, T_s = y
    Qsf = Q_shell_to_foam(T_sh, T_f, geo)
    Qfs = Q_foam_to_structure(T_f, T_s, geo)
    return [
        (Q_amb(T_sh, T_amb, geo) - Qsf) / cap_shell(T_sh, geo),
        (Qsf - Qfs) / cap_foam(T_f, geo),
        (Qfs - Q_struct_to_fluid(T_s)) / cap_structure(T_s, geo),
    ]


def _qs_split_rhs_a(t: float, y: np.ndarray, geo: TankGeometry, T_amb: float) -> list:
    """QS-split Case A: split-resistance geometry, T_foam algebraic, T_structure pinned."""
    T_sh = y[0]
    Qf = Q_foam_qs_split(T_sh, T_STRUCT_FIXED, geo)
    return [(Q_amb(T_sh, T_amb, geo) - Qf) / cap_shell(T_sh, geo)]


def _qs_split_rhs_b(t: float, y: np.ndarray, geo: TankGeometry, T_amb: float) -> list:
    """QS-split Case B: split-resistance geometry, T_foam algebraic."""
    T_sh, T_s = y
    Qf = Q_foam_qs_split(T_sh, T_s, geo)
    return [
        (Q_amb(T_sh, T_amb, geo) - Qf) / cap_shell(T_sh, geo),
        (Qf - Q_struct_to_fluid(T_s)) / cap_structure(T_s, geo),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Integration
# ══════════════════════════════════════════════════════════════════════════════

def _integrate(rhs, y0: list, geo: TankGeometry, T_amb: float,
               t_eval: np.ndarray = T_EVAL) -> np.ndarray:
    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="LSODA",
        args=(geo, T_amb),
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.y  # shape (n_states, n_t)


def _find_shell_eq_a(T_amb_eq: float, geo: TankGeometry) -> float:
    """Equilibrium shell temperature for Case A: Q_amb(T_sh) = Q_foam_qs(T_STRUCT_FIXED, T_sh)."""
    def res(T_sh: float) -> float:
        return Q_amb(T_sh, T_amb_eq, geo) - Q_foam_qs(T_STRUCT_FIXED, T_sh, geo)
    lo = T_STRUCT_FIXED + 0.1
    hi = max(T_amb_eq, T_STRUCT_FIXED) + 30.0
    return brentq(res, lo, hi, xtol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Case runners
# ══════════════════════════════════════════════════════════════════════════════

def run_case_a(
    geo: TankGeometry,
    T_amb_init: float = T_AMB_INIT,
    T_amb_final: float = T_AMB_STEP,
    T_sh_init: float | None = None,
    t_eval: np.ndarray = T_EVAL,
) -> dict:
    """
    Case A: step T_amb_init → T_amb_final at t=0; T_structure fixed.

    Three models run on the same grid:
      QS-production: single k(T_m), production formula.
      QS-split:      split-resistance geometry, T_foam algebraic (zero capacitance).
      TF-split:      split-resistance geometry, T_foam as ODE state.

    TF foam initial condition = QS-split algebraic equilibrium, so the only
    difference between QS-split and TF-split is the foam thermal capacitance.
    """
    if T_sh_init is None:
        T_sh_init = _find_shell_eq_a(T_amb_init, geo)
    T_f_init = _solve_foam_qs_split(T_sh_init, T_STRUCT_FIXED, geo)

    y_qs  = _integrate(_qs_rhs_a,       [T_sh_init],           geo, T_amb_final, t_eval)
    y_qss = _integrate(_qs_split_rhs_a, [T_sh_init],           geo, T_amb_final, t_eval)
    y_tf  = _integrate(_tf_rhs_a,       [T_sh_init, T_f_init], geo, T_amb_final, t_eval)

    T_sh_qs  = y_qs[0]
    T_sh_qss = y_qss[0]
    T_sh_tf, T_f_tf = y_tf[0], y_tf[1]
    n = len(t_eval)

    Q_qs  = np.array([Q_foam_qs(T_STRUCT_FIXED, T_sh_qs[i], geo) for i in range(n)])
    Q_qss = np.array([Q_foam_qs_split(T_sh_qss[i], T_STRUCT_FIXED, geo) for i in range(n)])
    Qsf   = np.array([Q_shell_to_foam(T_sh_tf[i], T_f_tf[i], geo) for i in range(n)])
    Qfs   = np.array([Q_foam_to_structure(T_f_tf[i], T_STRUCT_FIXED, geo) for i in range(n)])

    return dict(
        t=t_eval, geo=geo,
        qs=dict(
            T_shell=T_sh_qs,
            T_structure=np.full(n, T_STRUCT_FIXED),
            Q_into_structure=Q_qs,
        ),
        qs_split=dict(
            T_shell=T_sh_qss,
            T_structure=np.full(n, T_STRUCT_FIXED),
            Q_into_structure=Q_qss,
        ),
        tf=dict(
            T_shell=T_sh_tf,
            T_foam=T_f_tf,
            T_structure=np.full(n, T_STRUCT_FIXED),
            Q_shell_to_foam=Qsf,
            Q_foam_to_structure=Qfs,
            Q_into_structure=Qfs,
        ),
    )


def run_case_b(
    geo: TankGeometry,
    T_amb_final: float = T_AMB_STEP,
    T_sh_init: float = T_AMB_INIT,
    T_s_init: float = 60.0,
    t_eval: np.ndarray = T_EVAL,
) -> dict:
    """
    Case B: full shell + structure network; T_fluid fixed.
    Three models: QS-production, QS-split, TF-split.
    """
    T_f_init = _solve_foam_qs_split(T_sh_init, T_s_init, geo)

    y_qs  = _integrate(_qs_rhs_b,       [T_sh_init, T_s_init],           geo, T_amb_final, t_eval)
    y_qss = _integrate(_qs_split_rhs_b, [T_sh_init, T_s_init],           geo, T_amb_final, t_eval)
    y_tf  = _integrate(_tf_rhs_b,       [T_sh_init, T_f_init, T_s_init], geo, T_amb_final, t_eval)

    T_sh_qs,  T_s_qs  = y_qs[0],  y_qs[1]
    T_sh_qss, T_s_qss = y_qss[0], y_qss[1]
    T_sh_tf, T_f_tf, T_s_tf = y_tf[0], y_tf[1], y_tf[2]
    n = len(t_eval)

    Q_foam_qs_arr  = np.array([Q_foam_qs(T_s_qs[i], T_sh_qs[i], geo) for i in range(n)])
    Q_str_qs       = np.array([Q_struct_to_fluid(T_s_qs[i]) for i in range(n)])
    Q_qss_arr      = np.array([Q_foam_qs_split(T_sh_qss[i], T_s_qss[i], geo) for i in range(n)])
    Q_str_qss      = np.array([Q_struct_to_fluid(T_s_qss[i]) for i in range(n)])
    Qsf            = np.array([Q_shell_to_foam(T_sh_tf[i], T_f_tf[i], geo) for i in range(n)])
    Qfs            = np.array([Q_foam_to_structure(T_f_tf[i], T_s_tf[i], geo) for i in range(n)])
    Q_str_tf       = np.array([Q_struct_to_fluid(T_s_tf[i]) for i in range(n)])

    return dict(
        t=t_eval, geo=geo,
        qs=dict(
            T_shell=T_sh_qs, T_structure=T_s_qs,
            Q_into_structure=Q_foam_qs_arr, Q_structure=Q_str_qs,
        ),
        qs_split=dict(
            T_shell=T_sh_qss, T_structure=T_s_qss,
            Q_into_structure=Q_qss_arr, Q_structure=Q_str_qss,
        ),
        tf=dict(
            T_shell=T_sh_tf, T_foam=T_f_tf, T_structure=T_s_tf,
            Q_shell_to_foam=Qsf, Q_foam_to_structure=Qfs,
            Q_into_structure=Qfs, Q_structure=Q_str_tf,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(result: dict) -> dict:
    t    = result["t"]
    qs   = result["qs"]
    tf   = result["tf"]
    qs_s = result.get("qs_split")

    E_qs = cumulative_trapezoid(qs["Q_into_structure"], t, initial=0.0)
    E_tf = cumulative_trapezoid(tf["Q_into_structure"], t, initial=0.0)
    out: dict = {"E_qs": E_qs, "E_tf": E_tf, "dE": E_tf - E_qs}

    if qs_s is not None:
        E_qss = cumulative_trapezoid(qs_s["Q_into_structure"], t, initial=0.0)
        out["E_qs_split"]  = E_qss
        out["dE_form"]     = E_qss - E_qs     # k(T) discretisation only
        out["dE_inertia"]  = E_tf  - E_qss    # thermal inertia only

    dT_sh_tot = tf["T_shell"]          - qs["T_shell"]
    dT_s_tot  = tf["T_structure"]      - qs["T_structure"]
    dQ_tot    = tf["Q_into_structure"] - qs["Q_into_structure"]

    windows = {"1h": 3_600, "3h": 10_800, "6h": int(T_END)}
    for label, t_hi in windows.items():
        m = (t >= 0.0) & (t <= t_hi)
        entry = dict(
            max_dT_shell  = float(np.max(np.abs(dT_sh_tot[m]))),
            max_dT_struct = float(np.max(np.abs(dT_s_tot[m]))),
            max_dQ_struct = float(np.max(np.abs(dQ_tot[m]))),
            dE_final      = float(np.abs(out["dE"][m][-1])),
        )
        if qs_s is not None:
            dQ_form    = qs_s["Q_into_structure"] - qs["Q_into_structure"]
            dQ_inertia = tf["Q_into_structure"]   - qs_s["Q_into_structure"]
            dT_sh_in   = tf["T_shell"]            - qs_s["T_shell"]
            dT_s_in    = tf["T_structure"]        - qs_s["T_structure"]
            entry["max_dQ_form"]        = float(np.max(np.abs(dQ_form[m])))
            entry["max_dQ_inertia"]     = float(np.max(np.abs(dQ_inertia[m])))
            entry["max_dT_sh_inertia"]  = float(np.max(np.abs(dT_sh_in[m])))
            entry["max_dT_s_inertia"]   = float(np.max(np.abs(dT_s_in[m])))
            entry["dE_form_final"]      = float(np.abs(out["dE_form"][m][-1]))
            entry["dE_inertia_final"]   = float(np.abs(out["dE_inertia"][m][-1]))
        out[label] = entry

    ss = t >= 0.9 * T_END
    Q_QS_ss = float(np.mean(qs["Q_into_structure"][ss]))
    Q_TF_ss = float(np.mean(tf["Q_into_structure"][ss]))
    ss_entry = dict(
        Q_QS    = Q_QS_ss,
        Q_TF    = Q_TF_ss,
        dQ      = abs(Q_TF_ss - Q_QS_ss),
        dQ_rel  = abs(Q_TF_ss - Q_QS_ss) / max(abs(Q_QS_ss), 1e-10),
    )
    if qs_s is not None:
        Q_split_ss = float(np.mean(qs_s["Q_into_structure"][ss]))
        ss_entry["Q_QS_split"]    = Q_split_ss
        ss_entry["dQ_form"]       = abs(Q_split_ss - Q_QS_ss)
        ss_entry["dQ_form_rel"]   = abs(Q_split_ss - Q_QS_ss) / max(abs(Q_QS_ss), 1e-10)
        ss_entry["dQ_inertia"]    = abs(Q_TF_ss - Q_split_ss)
        ss_entry["dQ_inertia_rel"]= abs(Q_TF_ss - Q_split_ss) / max(abs(Q_split_ss), 1e-10)
    out["ss"] = ss_entry
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Timescale diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def compute_timescales(geo: TankGeometry,
                       T_sh: float = 288.0,
                       T_s: float = T_STRUCT_FIXED) -> dict:
    """Compute foam, shell, and structure thermal timescales at given temperatures."""
    G_foam = Q_foam_qs(T_s, T_sh, geo) / (T_sh - T_s)
    h_A    = ALPHA_AMB * geo.A_shell
    C_sh   = cap_shell(T_sh, geo)
    C_str  = cap_structure(T_s, geo)
    C_f    = cap_foam(0.5 * (T_s + T_sh), geo)

    T_mean  = 0.5 * (T_s + T_sh)
    k_mean  = rohacell_k(_clamp(T_mean))
    cp_mean = rohacell_cp(_clamp(T_mean))
    alpha_d = k_mean / (ROHACELL_DENSITY * cp_mean)

    return dict(
        tau_foam    = geo.t_insul**2 / alpha_d,
        tau_shell   = C_sh / (h_A + G_foam),
        tau_struct  = C_str / G_foam,
        G_foam      = G_foam,
        C_shell     = C_sh,
        C_structure = C_str,
        C_foam      = C_f,
        m_foam      = geo.m_foam,
        k_mean      = k_mean,
        cp_mean     = cp_mean,
        alpha_diff  = alpha_d,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _hours(t: np.ndarray) -> np.ndarray:
    return t / 3600.0


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(_OUTDIR, exist_ok=True)
    path = os.path.join(_OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_temperatures(result: dict, case_label: str) -> None:
    """Plot 1: temperatures for QS-production, QS-split, and TF-split models."""
    t, qs, tf = _hours(result["t"]), result["qs"], result["tf"]
    qs_s = result.get("qs_split")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, qs["T_shell"],   color=_C["qs_shell"],  lw=1.8, label="QS-prod $T_{shell}$")
    if qs_s is not None:
        ax.plot(t, qs_s["T_shell"], color=_C["qs_split_shell"], lw=1.5,
                ls=(0, (3, 1, 1, 1)), label="QS-split $T_{shell}$")
    ax.plot(t, tf["T_shell"],   color=_C["tf_shell"],  lw=1.8, ls="--",
            label="TF-split $T_{shell}$")
    ax.plot(t, tf["T_foam"],    color=_C["tf_foam"],   lw=1.8, ls="-.",
            label="TF $T_{foam}$")
    ax.plot(t, qs["T_structure"], color=_C["qs_struct"], lw=1.8,
            label="QS-prod $T_{structure}$")
    if not np.allclose(tf["T_structure"], qs["T_structure"], atol=0.0):
        if qs_s is not None:
            ax.plot(t, qs_s["T_structure"], color=_C["qs_split_shell"], lw=1.2,
                    ls=":", label="QS-split $T_{structure}$")
        ax.plot(t, tf["T_structure"], color=_C["tf_struct"], lw=1.8, ls="--",
                label="TF-split $T_{structure}$")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"Case {case_label} — Temperatures")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, f"case{case_label}_temperatures.png")


def plot_heat_flows(result: dict, case_label: str) -> None:
    """Plot 2: foam heat flows for all three models."""
    t, qs, tf = _hours(result["t"]), result["qs"], result["tf"]
    qs_s = result.get("qs_split")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, qs["Q_into_structure"], color=_C["Q_qs"], lw=1.8,
            label="QS-prod $\\dot{Q}_{foam}$")
    if qs_s is not None:
        ax.plot(t, qs_s["Q_into_structure"], color=_C["Q_qs_split"], lw=1.5,
                ls=(0, (3, 1, 1, 1)), label="QS-split $\\dot{Q}_{foam}$")
    ax.plot(t, tf["Q_shell_to_foam"],   color=_C["Q_tf_outer"], lw=1.8, ls="--",
            label="TF $\\dot{Q}_{shell\\to foam}$")
    ax.plot(t, tf["Q_foam_to_structure"], color=_C["Q_tf_inner"], lw=1.8, ls="-.",
            label="TF $\\dot{Q}_{foam\\to structure}$")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Heat flow [W]")
    ax.set_title(f"Case {case_label} — Foam heat transfer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, f"case{case_label}_heat_flows.png")


def plot_differences(result: dict, case_label: str) -> None:
    """Plot 3: decompose ΔQ into k(T)-formulation effect and thermal-inertia effect."""
    t   = _hours(result["t"])
    qs  = result["qs"]
    tf  = result["tf"]
    qs_s = result.get("qs_split")

    dQ_tot = tf["Q_into_structure"] - qs["Q_into_structure"]
    dT_sh  = tf["T_shell"]          - qs["T_shell"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel 1: shell temperature difference (inertia is the relevant driver here)
    axes[0].plot(t, dT_sh, color=_C["delta"], lw=1.5,
                 label="TF−QS-prod (total)")
    if qs_s is not None:
        dT_sh_in = tf["T_shell"] - qs_s["T_shell"]
        axes[0].plot(t, dT_sh_in, color=_C["Q_tf_outer"], lw=1.5, ls="--",
                     label="TF−QS-split (inertia only)")
    axes[0].axhline(0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("$\\Delta T_{shell}$ [K]")
    axes[0].set_title(f"Case {case_label} — Effect decomposition: formulation vs inertia")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: heat flow difference decomposed into two contributions
    axes[1].plot(t, dQ_tot, color=_C["delta"], lw=1.8,
                 label="TF−QS-prod (total)")
    if qs_s is not None:
        dQ_form    = qs_s["Q_into_structure"] - qs["Q_into_structure"]
        dQ_inertia = tf["Q_into_structure"]   - qs_s["Q_into_structure"]
        axes[1].plot(t, dQ_form,    color=_C["Q_qs_split"], lw=1.5,
                     ls=(0, (3, 1, 1, 1)), label="QS-split−QS-prod (formulation, static)")
        axes[1].plot(t, dQ_inertia, color=_C["Q_tf_outer"], lw=1.5, ls="--",
                     label="TF−QS-split (inertia, dynamic)")
    axes[1].axhline(0, color="gray", lw=0.8, ls=":")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("$\\Delta \\dot{Q}_{structure}$ [W]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f"case{case_label}_differences.png")


def plot_cumulative_energy(result: dict, metrics: dict, case_label: str) -> None:
    """Plot 4: cumulative energy into structure for all three models."""
    t    = _hours(result["t"])
    E_qs = metrics["E_qs"] / 1e3
    E_tf = metrics["E_tf"] / 1e3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(t, E_qs, color=_C["E_qs"], lw=1.8, label="QS-prod $E_{structure}$")
    if "E_qs_split" in metrics:
        ax1.plot(t, metrics["E_qs_split"] / 1e3, color=_C["Q_qs_split"], lw=1.5,
                 ls=(0, (3, 1, 1, 1)), label="QS-split $E_{structure}$")
    ax1.plot(t, E_tf, color=_C["E_tf"], lw=1.8, ls="--", label="TF-split $E_{structure}$")
    ax1.set_ylabel("Cumulative heat [kJ]")
    ax1.set_title(f"Case {case_label} — Cumulative heat into structure")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, metrics["dE"] / 1e3, color=_C["delta"], lw=1.5,
             label="TF−QS-prod (total)")
    if "dE_form" in metrics:
        ax2.plot(t, metrics["dE_form"] / 1e3, color=_C["Q_qs_split"], lw=1.5,
                 ls=(0, (3, 1, 1, 1)), label="QS-split−QS-prod (formulation)")
        ax2.plot(t, metrics["dE_inertia"] / 1e3, color=_C["Q_tf_outer"], lw=1.5,
                 ls="--", label="TF−QS-split (inertia)")
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("$\\Delta E_{structure}$ [kJ]")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f"case{case_label}_cumulative_energy.png")


def plot_foam_energy_balance(result: dict, geo: TankGeometry, case_label: str) -> None:
    """Verification: residual R = C_foam*dT_foam/dt - (Q_sf - Q_fs) should be near zero."""
    t  = result["t"]
    tf = result["tf"]
    T_f = tf["T_foam"]

    C_f   = np.array([cap_foam(T_f[i], geo) for i in range(len(t))])
    dTdt  = np.gradient(T_f, t)   # numerical derivative; has endpoint noise
    resid = C_f * dTdt - (tf["Q_shell_to_foam"] - tf["Q_foam_to_structure"])
    scale = np.max(np.abs(tf["Q_shell_to_foam"])) + 1e-10

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_hours(t), resid, color=_C["residual"], lw=1.2,
            label="Residual [W]")
    ax.plot(_hours(t), resid / scale, color=_C["delta"], lw=1.2, ls="--",
            label="Residual / max|Q_sf| [−]")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual [W] / [−]")
    ax.set_title(f"Case {case_label} — Foam energy balance residual "
                 r"$\mathcal{R} = C_f\,\dot{T}_f - (\dot{Q}_{sf} - \dot{Q}_{fs})$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, f"case{case_label}_foam_energy_balance.png")


# ══════════════════════════════════════════════════════════════════════════════
# Foam-thickness sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_thickness_sweep(case: str = "A") -> list[dict]:
    """Run Case A or B for each foam thickness in SWEEP_THICKNESSES_M."""
    records = []
    print(f"\n{'='*65}")
    print(f"Foam-thickness sensitivity sweep — Case {case}")
    print(f"{'='*65}")

    for t_ins in SWEEP_THICKNESSES_M:
        geo = build_geometry(t_ins)
        ts  = compute_timescales(geo)
        print(f"\n  t_foam = {t_ins * 1000:.0f} mm  |  "
              f"τ_foam = {ts['tau_foam']:.0f} s  |  "
              f"τ_shell = {ts['tau_shell']:.0f} s  |  "
              f"τ_foam/τ_shell = {ts['tau_foam']/ts['tau_shell']:.2f}")

        res = run_case_a(geo) if case == "A" else run_case_b(geo)
        m   = compute_metrics(res)

        records.append(dict(
            t_ins_mm        = t_ins * 1000,
            tau_foam        = ts["tau_foam"],
            tau_shell       = ts["tau_shell"],
            tau_struct      = ts["tau_struct"],
            tau_ratio       = ts["tau_foam"] / ts["tau_shell"],
            max_dT_shell_1h  = m["1h"]["max_dT_shell"],
            max_dT_struct_1h = m["1h"]["max_dT_struct"],
            max_dQ_1h        = m["1h"]["max_dQ_struct"],
            max_dQ_form_1h   = m["1h"].get("max_dQ_form",    float("nan")),
            max_dQ_iner_1h   = m["1h"].get("max_dQ_inertia", float("nan")),
            dE_6h            = m["6h"]["dE_final"],
            ss_Q_QS          = m["ss"]["Q_QS"],
            ss_Q_TF          = m["ss"]["Q_TF"],
            ss_dQ_rel        = m["ss"]["dQ_rel"],
            ss_dQ_form_rel   = m["ss"].get("dQ_form_rel",    float("nan")),
            ss_dQ_iner_rel   = m["ss"].get("dQ_inertia_rel", float("nan")),
        ))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Initial-gradient sensitivity (Case A)
# ══════════════════════════════════════════════════════════════════════════════

def run_gradient_sensitivity(geo: TankGeometry | None = None) -> list[dict]:
    """Repeat Case A for small, moderate, and large initial shell–structure gradients."""
    if geo is None:
        geo = build_geometry(0.050)

    scenarios = [
        (65.0,  "small (ΔT ≈ 5 K)"),
        (150.0, "moderate (ΔT = 90 K)"),
        (250.0, "large cryogenic (ΔT = 190 K)"),
    ]
    records = []
    print(f"\n{'='*65}")
    print("Initial-gradient sensitivity — Case A, 50 mm foam")
    print(f"{'='*65}")

    for T_sh0, label in scenarios:
        res = run_case_a(geo, T_sh_init=T_sh0)
        m   = compute_metrics(res)
        print(f"\n  {label}  T_sh0 = {T_sh0:.0f} K")
        print(f"    max ΔT_shell (1h)   = {m['1h']['max_dT_shell']:.4f} K")
        print(f"    max ΔQ_struct (1h)  = {m['1h']['max_dQ_struct']:.4f} W")
        print(f"    ΔE_struct (6h)      = {m['6h']['dE_final']/1e3:.4f} kJ")
        records.append(dict(
            label=label,
            T_sh0=T_sh0,
            **{f"1h_{k}": v for k, v in m["1h"].items()},
            **{f"6h_{k}": v for k, v in m["6h"].items()},
        ))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(records: list[dict], case: str) -> None:
    print(f"\n{'='*112}")
    print(f"Summary — foam-thickness sweep, Case {case}  "
          "[total = QS-prod→TF, form = QS-prod→QS-split, iner = QS-split→TF]")
    print(f"{'='*112}")
    hdr = (f"{'t_foam':>8}  {'τ/τ_s':>6}  "
           f"{'maxΔQ_tot(1h)':>14}  {'maxΔQ_form(1h)':>14}  {'maxΔQ_iner(1h)':>14}  "
           f"{'ΔE(6h)':>8}  {'ss_tot':>7}  {'ss_form':>7}  {'ss_iner':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        def _pct(v: float) -> str:
            return f"{v*100:.2f}%" if not math.isnan(v) else "  n/a  "
        print(
            f"{r['t_ins_mm']:>7.0f}mm  "
            f"{r['tau_ratio']:>6.2f}  "
            f"{r['max_dQ_1h']:>13.2f}W  "
            f"{r['max_dQ_form_1h']:>13.2f}W  "
            f"{r['max_dQ_iner_1h']:>13.2f}W  "
            f"{r['dE_6h']/1e3:>7.2f}kJ  "
            f"{_pct(r['ss_dQ_rel']):>7}  "
            f"{_pct(r['ss_dQ_form_rel']):>7}  "
            f"{_pct(r['ss_dQ_iner_rel']):>7}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(_OUTDIR, exist_ok=True)

    geo = build_geometry(0.050)
    ts  = compute_timescales(geo)

    print("\n" + "=" * 65)
    print("Quasi-steady vs transient-foam sensitivity study")
    print("=" * 65)
    print(f"Geometry:  R_structure = {R_STRUCTURE:.4f} m,  "
          f"r_shell = {geo.r_shell:.4f} m,  t_foam = 50 mm")
    print(f"Foam mass: {geo.m_foam:.2f} kg  "
          f"(ρ={ROHACELL_DENSITY:.1f} kg/m³, cylindrical+spherical volume)")
    print(f"Midpoints: r_m_cyl = {geo.r_m_cyl:.5f} m,  "
          f"r_m_sph = {geo.r_m_sph:.5f} m")
    print(f"\nTimescales at T_sh={288:.0f} K, T_s={T_STRUCT_FIXED:.0f} K:")
    print(f"  τ_foam      = {ts['tau_foam']:.0f} s  "
          f"(k={ts['k_mean']:.4f} W/mK, cp={ts['cp_mean']:.0f} J/kgK, "
          f"α={ts['alpha_diff']:.2e} m²/s)")
    print(f"  τ_shell     = {ts['tau_shell']:.0f} s")
    print(f"  τ_structure = {ts['tau_struct']:.0f} s")
    print(f"  τ_foam/τ_shell     = {ts['tau_foam']/ts['tau_shell']:.2f}")
    print(f"  τ_foam/τ_structure = {ts['tau_foam']/ts['tau_struct']:.3f}")

    # ── Case A ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*50}")
    print("Case A: step T_amb 250→288.15 K, T_structure fixed at 60 K")
    res_a = run_case_a(geo)
    m_a   = compute_metrics(res_a)

    for win in ("1h", "3h", "6h"):
        mx = m_a[win]
        print(f"  [{win}]  ΔQ total={mx['max_dQ_struct']:7.2f}W  "
              f"form={mx.get('max_dQ_form', float('nan')):7.2f}W  "
              f"inertia={mx.get('max_dQ_inertia', float('nan')):7.2f}W  "
              f"ΔE={mx['dE_final']/1e3:.3f}kJ")
    ss = m_a["ss"]
    print(f"  [SS]  Q_prod={ss['Q_QS']:.3f}W  "
          f"Q_split={ss.get('Q_QS_split', float('nan')):.3f}W  "
          f"Q_TF={ss['Q_TF']:.3f}W")
    if "dQ_form_rel" in ss:
        print(f"        formulation Δ={ss['dQ_form']:.3f}W ({ss['dQ_form_rel']*100:.3f}%)  "
              "[static k(T) discretisation]")
        print(f"        inertia     Δ={ss['dQ_inertia']:.3f}W ({ss['dQ_inertia_rel']*100:.3f}%)  "
              "[→ 0 at steady state]")

    plot_temperatures(res_a, "A")
    plot_heat_flows(res_a, "A")
    plot_differences(res_a, "A")
    plot_cumulative_energy(res_a, m_a, "A")
    plot_foam_energy_balance(res_a, geo, "A")

    # ── Case B ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*50}")
    print("Case B: full shell + structure, T_fluid fixed at 54 K")
    res_b = run_case_b(geo)
    m_b   = compute_metrics(res_b)

    for win in ("1h", "3h", "6h"):
        mx = m_b[win]
        print(f"  [{win}]  ΔT_struct total={mx['max_dT_struct']:6.4f}K  "
              f"inertia={mx.get('max_dT_s_inertia', float('nan')):6.4f}K  "
              f"ΔQ total={mx['max_dQ_struct']:7.2f}W  "
              f"form={mx.get('max_dQ_form', float('nan')):7.2f}W  "
              f"inertia={mx.get('max_dQ_inertia', float('nan')):7.2f}W")
    ss = m_b["ss"]
    print(f"  [SS]  Q_prod={ss['Q_QS']:.3f}W  "
          f"Q_split={ss.get('Q_QS_split', float('nan')):.3f}W  "
          f"Q_TF={ss['Q_TF']:.3f}W")
    if "dQ_form_rel" in ss:
        print(f"        formulation Δ={ss['dQ_form']:.3f}W ({ss['dQ_form_rel']*100:.3f}%)")
        print(f"        inertia     Δ={ss['dQ_inertia']:.3f}W ({ss['dQ_inertia_rel']*100:.3f}%)")

    plot_temperatures(res_b, "B")
    plot_heat_flows(res_b, "B")
    plot_differences(res_b, "B")
    plot_cumulative_energy(res_b, m_b, "B")
    plot_foam_energy_balance(res_b, geo, "B")

    # ── Thickness sweeps ──────────────────────────────────────────────────────
    sweep_a = run_thickness_sweep("A")
    print_summary_table(sweep_a, "A")

    sweep_b = run_thickness_sweep("B")
    print_summary_table(sweep_b, "B")

    # ── Initial-gradient sensitivity ──────────────────────────────────────────
    run_gradient_sensitivity(geo)

    print(f"\nStudy complete. Figures saved to: {_OUTDIR}")


if __name__ == "__main__":
    main()
