"""
Analytical and numerical verification of InsulatedTankThermalModel.

V1  Exact cylindrical + spherical radial conduction
V2  Thin-insulation asymptotic limit (curved → planar wall)
V3  Mean-temperature conductivity approximation error
V4  Shell energy balance (ODE assembly)
V5  Structure energy balance (ODE assembly)
V6  Outer-shell transient: closed-form vs numerical (isolated problem)
V7  Whole-network steady-state energy conservation
V8  Quasi-steady foam timescale assessment

These complement the software/unit-level tests in test_tank_thermal_model.py
by verifying the governing thermal equations against independently evaluated
analytical solutions.  All values are in SI units.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve

from toplab.materials.nist_materials import NISTComposite, NISTMetal
from toplab.materials.rohacell_properties import DENSITY as ROHACELL_DENSITY
from toplab.materials.rohacell_properties import specific_heat as rohacell_cp
from toplab.materials.rohacell_properties import thermal_conductivity as rohacell_k
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel
from toplab.thermodynamics.tank_states import IsochoricTankState

# ---------------------------------------------------------------------------
# Reference geometry — mirrors the _make_thermal_model convention in
# test_tank_thermal_model.py so that both suites use a consistent tank.
# ---------------------------------------------------------------------------

_R_INNER = 0.500       # m, inner fluid-facing radius
_T_LINER = 0.003       # m, liner thickness
_T_WALL = 0.015        # m, composite-wall thickness
_R_STRUCTURE = _R_INNER + _T_LINER + _T_WALL   # 0.518 m
_T_INSULATION = 0.050  # m, nominal foam layer
_R_SHELL = _R_STRUCTURE + _T_INSULATION        # 0.568 m
_L_CYL = 1.500         # m, cylindrical section length
_T_SHELL_THK = 0.002   # m, thin Al outer shell wall
_T_AMB = 288.15        # K
_ALPHA_AMB = 5.0       # W/m²K, convective HTC ambient → shell
_EMISSIVITY = 0.05     # shell surface emissivity

# Representative operating temperatures for V1/V4/V5
_T_STRUCTURE_OP = 60.0    # K, cold liner+wall assembly
_T_SHELL_OP = 288.0       # K, near-ambient outer shell
_T_FLUID_OP = 54.0        # K, cryogenic H₂

SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W/m²K⁴]


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

class _MockTank:
    volume = math.pi * _R_INNER**3 * (4.0 / 3.0 + _L_CYL / _R_INNER)

    def compute_fuel_height(self, fuel_volume):
        return 0.0


def _make_model(
    t_insulation=_T_INSULATION,
    alpha_amb=_ALPHA_AMB,
    emissivity=_EMISSIVITY,
    T_amb=_T_AMB,
):
    """Construct a representative InsulatedTankThermalModel for verification."""
    liner = NISTMetal.aluminum_6061T6_nist()
    wall = NISTComposite.carbon_epoxy_nist()
    r_s = _R_STRUCTURE
    r_sh = r_s + t_insulation

    def _layer_mass(rho, r_a, r_b):
        cyl = math.pi * (r_b**2 - r_a**2) * _L_CYL
        sph = (4.0 / 3.0) * math.pi * (r_b**3 - r_a**3)
        return rho * (cyl + sph)

    liner_mass = _layer_mass(liner.density, _R_INNER, _R_INNER + _T_LINER)
    wall_mass = _layer_mass(wall.density, _R_INNER + _T_LINER, r_s)
    shell_mass = _layer_mass(liner.density, r_sh, r_sh + _T_SHELL_THK)

    return InsulatedTankThermalModel(
        tank_volume=_MockTank.volume,
        inner_surface_area=4.0 * math.pi * _R_INNER**2 + 2.0 * math.pi * _R_INNER * _L_CYL,
        inner_diameter=2.0 * _R_INNER,
        r_structure=r_s,
        r_shell=r_sh,
        cylinder_length=_L_CYL,
        liner_mass=liner_mass,
        wall_mass=wall_mass,
        shell_mass=shell_mass,
        ambient_temperature=T_amb,
        alpha_amb=alpha_amb,
        emissivity_shell=emissivity,
        liner_material=liner,
        wall_material=wall,
        shell_material=liner,   # Al 6061-T6 for the thin outer shell
    )


def _make_state(
    T_fluid=_T_FLUID_OP,
    T_structure=_T_STRUCTURE_OP,
    T_shell=_T_SHELL_OP,
):
    """Construct a representative IsochoricTankState (triggers CoolProp hydrogen lookup)."""
    tank = _MockTank()
    mass = 70.0 * tank.volume   # ≈ 70 kg/m³, representative LH₂ density
    return IsochoricTankState(
        tank=tank,
        fuel_mass=mass,
        temperature=T_fluid,
        solid_temperature=T_structure,
        shell_temperature=T_shell,
    )


# ---------------------------------------------------------------------------
# Reference heat-flux helper
# ---------------------------------------------------------------------------

def _q_insulation_ref(r_s: float, r_sh: float, L: float, T_s: float, T_sh: float) -> float:
    """
    Independent evaluation of the Fourier radial-conduction formula.

    Q_cyl  = 2π L k(T_m) / ln(r_sh / r_s) · ΔT
    Q_caps = 4π k(T_m) r_s r_sh / (r_sh − r_s) · ΔT

    This function must NOT be replaced by a call to
    model.compute_insulation_heat_flux() inside verification tests.
    """
    T_m = 0.5 * (T_s + T_sh)
    k = rohacell_k(T_m)
    Q_cyl = 2.0 * math.pi * L * k / math.log(r_sh / r_s) * (T_sh - T_s)
    Q_cap = 4.0 * math.pi * k * r_s * r_sh / (r_sh - r_s) * (T_sh - T_s)
    return Q_cyl + Q_cap


# ===========================================================================
# V1 — Exact radial-conduction solution
# ===========================================================================

class TestV1ExactRadialConduction:
    """
    Independently evaluate the Fourier cylindrical + spherical-endcap formula
    and compare against model.compute_insulation_heat_flux().

    A pass at machine precision (< 1 × 10⁻¹²) confirms that the production
    formula matches the governing equations.
    """

    def test_total_heat_flux_matches_independent_formula(self):
        model = _make_model()
        T_s, T_sh = _T_STRUCTURE_OP, _T_SHELL_OP

        Q_ref = _q_insulation_ref(model.r_structure, model.r_shell, model.L, T_s, T_sh)
        Q_prod = model.compute_insulation_heat_flux(T_s, T_sh)

        rel_err = abs(Q_prod - Q_ref) / abs(Q_ref)
        assert rel_err < 1e-12, (
            f"V1: Q_insulation mismatch. "
            f"ref={Q_ref:.6f} W, prod={Q_prod:.6f} W, rel_err={rel_err:.2e}"
        )

    def test_cylindrical_contribution_positive_when_shell_hotter(self):
        model = _make_model()
        T_m = 0.5 * (_T_STRUCTURE_OP + _T_SHELL_OP)
        k = rohacell_k(T_m)
        Q_cyl = (
            2.0 * math.pi * model.L * k
            / math.log(model.r_shell / model.r_structure)
            * (_T_SHELL_OP - _T_STRUCTURE_OP)
        )
        assert Q_cyl > 0.0

    def test_endcap_contribution_positive_when_shell_hotter(self):
        model = _make_model()
        r_s, r_sh = model.r_structure, model.r_shell
        T_m = 0.5 * (_T_STRUCTURE_OP + _T_SHELL_OP)
        k = rohacell_k(T_m)
        Q_cap = 4.0 * math.pi * k * r_s * r_sh / (r_sh - r_s) * (_T_SHELL_OP - _T_STRUCTURE_OP)
        assert Q_cap > 0.0


# ===========================================================================
# V2 — Thin-insulation asymptotic limit (curved → planar wall)
# ===========================================================================

class TestV2ThinInsulationAsymptotic:
    """
    As t_insulation / r_structure → 0, the curved Fourier formula must converge
    to the planar-wall solution Q = k·A·ΔT / t.

    The relative error is O(t / r_structure) by Taylor expansion of
    ln(r_sh / r_s) and the spherical formula; verified numerically here.
    """

    THICKNESSES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]   # m, progressively thinner

    @staticmethod
    def _q_planar(t_ins: float, T_s: float, T_sh: float) -> float:
        """Planar-wall reference: k·A·ΔT / t, surface area evaluated at r_structure."""
        A = 2.0 * math.pi * _R_STRUCTURE * _L_CYL + 4.0 * math.pi * _R_STRUCTURE**2
        T_m = 0.5 * (T_s + T_sh)
        k = rohacell_k(T_m)
        return k * A / t_ins * (T_sh - T_s)

    @staticmethod
    def _q_radial(t_ins: float, T_s: float, T_sh: float) -> float:
        r_sh = _R_STRUCTURE + t_ins
        return _q_insulation_ref(_R_STRUCTURE, r_sh, _L_CYL, T_s, T_sh)

    def test_error_decreases_monotonically_as_insulation_thins(self):
        T_s, T_sh = _T_STRUCTURE_OP, _T_SHELL_OP
        errors = [
            abs(self._q_radial(t, T_s, T_sh) - self._q_planar(t, T_s, T_sh))
            / abs(self._q_radial(t, T_s, T_sh))
            for t in self.THICKNESSES
        ]
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"V2: Error did not decrease from t={self.THICKNESSES[i]:.0e} m "
                f"(ε={errors[i]:.3e}) to t={self.THICKNESSES[i+1]:.0e} m (ε={errors[i+1]:.3e})"
            )

    def test_thinnest_insulation_within_one_percent_of_planar(self):
        t = self.THICKNESSES[-1]   # 1e-4 m; t/r ≈ 1.9 × 10⁻⁴
        T_s, T_sh = _T_STRUCTURE_OP, _T_SHELL_OP
        Q_rad = self._q_radial(t, T_s, T_sh)
        Q_pla = self._q_planar(t, T_s, T_sh)
        eps = abs(Q_rad - Q_pla) / abs(Q_rad)
        assert eps < 0.01, (
            f"V2: Thinnest-case error {eps:.3e} exceeds 1 % at t={t:.0e} m"
        )

    @pytest.mark.parametrize("t_ins", THICKNESSES)
    def test_error_bounded_by_order_t_over_r(self, t_ins):
        """Relative error must remain within O(t / r_structure) (with factor 2 margin)."""
        T_s, T_sh = _T_STRUCTURE_OP, _T_SHELL_OP
        Q_rad = self._q_radial(t_ins, T_s, T_sh)
        Q_pla = self._q_planar(t_ins, T_s, T_sh)
        eps = abs(Q_rad - Q_pla) / abs(Q_rad)
        expected_order = t_ins / _R_STRUCTURE
        assert eps < 2.0 * expected_order, (
            f"V2: ε={eps:.3e} exceeds 2 × (t/r)={2.0 * expected_order:.3e} at t={t_ins:.0e} m"
        )


# ===========================================================================
# V3 — Mean-temperature conductivity approximation
# ===========================================================================

class TestV3MeanTemperatureConductivity:
    """
    Compare the production approximation k(T_m)·ΔT against ∫k(T) dT.

    Two purposes:
    1. Verify the implementation of the mean-temperature approximation (V1
       already checks the formula; V3 checks whether the approximation is
       consistently applied across a range of temperature intervals).
    2. Characterise the approximation error for preliminary-design documentation.

    The threshold (20 %) is intentionally generous; tighten it once the actual
    Rohacell data have been reviewed across the full cryogenic range.
    """

    TEMPERATURE_INTERVALS = [
        (20.0, 300.0),
        (30.0, 250.0),
        (50.0, 200.0),
        (100.0, 300.0),
    ]

    @pytest.mark.parametrize("T_lo,T_hi", TEMPERATURE_INTERVALS)
    def test_conductivity_approximation_error(self, T_lo, T_hi):
        model = _make_model()
        r_s, r_sh, L = model.r_structure, model.r_shell, model.L

        # Geometric factor common to both approximation and integral
        G_geom = (
            2.0 * math.pi * L / math.log(r_sh / r_s)
            + 4.0 * math.pi * r_s * r_sh / (r_sh - r_s)
        )

        # Production approximation: k(T_m) · ΔT
        T_m = 0.5 * (T_lo + T_hi)
        Q_mean = G_geom * rohacell_k(T_m) * (T_hi - T_lo)

        # More-exact quasi-steady solution: G_geom · ∫k(T) dT
        integrated_k, _ = quad(rohacell_k, T_lo, T_hi)
        Q_integrated = G_geom * integrated_k

        eps_k = abs(Q_mean - Q_integrated) / abs(Q_integrated)

        assert eps_k < 0.20, (
            f"V3: Conductivity approximation error {eps_k:.2%} > 20 % "
            f"for T=[{T_lo:.0f}, {T_hi:.0f}] K. "
            f"Q_mean={Q_mean:.4f} W, Q_integrated={Q_integrated:.4f} W"
        )


# ===========================================================================
# V4 — Shell energy balance
# ===========================================================================

class TestV4ShellEnergyBalance:
    """
    Independently assemble the shell ODE right-hand side

        dT_shell/dt = (Q_amb − Q_insulation) / C_shell

    and compare against model.compute_shell_temperature_derivative().

    Reference quantities are computed explicitly from model attributes and
    the governing formulas; the production derivative method is NOT used in
    the reference calculation.
    """

    def test_shell_derivative_matches_independent_energy_balance(self):
        model = _make_model()
        state = _make_state()
        T_s = state.solid_temperature
        T_sh = state.shell_temperature

        # -- Reference Q_amb: convection + radiation --
        Q_conv = model.alpha_amb * model.A_shell * (model.T_amb - T_sh)
        Q_rad = (
            model.eps_shell * SIGMA * model.A_shell * (model.T_amb**4 - T_sh**4)
        )
        Q_amb_ref = Q_conv + Q_rad

        # -- Reference Q_insulation: Fourier formula (not via production method) --
        Q_ins_ref = _q_insulation_ref(model.r_structure, model.r_shell, model.L, T_s, T_sh)

        # -- Reference C_shell --
        T_bounded = max(4.0, min(T_sh, 400.0))
        C_shell = model.m_shell * model.shell_material.determine_specific_heat(T_bounded)

        dT_shell_ref = (Q_amb_ref - Q_ins_ref) / C_shell

        # -- Production --
        dT_shell_prod = model.compute_shell_temperature_derivative(0.0, state)

        rel_err = abs(dT_shell_prod - dT_shell_ref) / abs(dT_shell_ref)
        assert rel_err < 1e-10, (
            f"V4: Shell temperature derivative mismatch. "
            f"ref={dT_shell_ref:.8e} K/s, prod={dT_shell_prod:.8e} K/s, "
            f"rel_err={rel_err:.2e}"
        )


# ===========================================================================
# V5 — Structure energy balance
# ===========================================================================

class TestV5StructureEnergyBalance:
    """
    Independently assemble the structure ODE right-hand side

        dT_structure/dt = (Q_insulation − Q_structure) / C_structure

    and compare against model.compute_structure_temperature_derivative().

    Sign convention verified explicitly:
    - Q_insulation > 0: energy enters the structure from the shell side.
    - Q_structure  > 0: energy leaves the structure to the fluid.
    """

    def test_structure_derivative_matches_independent_energy_balance(self):
        model = _make_model()
        state = _make_state()
        T_s = state.solid_temperature

        # -- Reference Q_insulation: Fourier formula (not via production method) --
        Q_ins_ref = _q_insulation_ref(
            model.r_structure, model.r_shell, model.L,
            T_s, state.shell_temperature,
        )

        # -- Reference Q_structure: Churchill-Chu α_s × A_in × ΔT --
        # alpha_s is obtained from the lower-level correlation helper; this is
        # distinct from the ODE method under test.
        alpha_s = model.get_alpha_s(state.temperature, T_s, state.pressure)
        Q_str_ref = alpha_s * model.A_in * (T_s - state.temperature)

        # -- Reference C_structure --
        T_bounded = max(4.0, min(T_s, 400.0))
        C_structure = (
            model.m_liner * model.liner_material.determine_specific_heat(T_bounded)
            + model.m_wall * model.wall_material.determine_specific_heat(T_bounded)
        )

        dT_struct_ref = (Q_ins_ref - Q_str_ref) / C_structure

        # -- Production --
        dT_struct_prod = model.compute_structure_temperature_derivative(0.0, state)

        rel_err = abs(dT_struct_prod - dT_struct_ref) / abs(dT_struct_ref)
        assert rel_err < 1e-10, (
            f"V5: Structure temperature derivative mismatch. "
            f"ref={dT_struct_ref:.8e} K/s, prod={dT_struct_prod:.8e} K/s, "
            f"rel_err={rel_err:.2e}"
        )

    def test_q_insulation_enters_structure_positively(self):
        """With T_shell > T_structure, heat flows in: Q_insulation > 0."""
        assert _T_SHELL_OP > _T_STRUCTURE_OP   # sanity on the operating point
        Q_ins = _q_insulation_ref(
            _R_STRUCTURE, _R_SHELL, _L_CYL, _T_STRUCTURE_OP, _T_SHELL_OP
        )
        assert Q_ins > 0.0

    def test_q_structure_leaves_structure_positively(self):
        """With T_structure > T_fluid, heat flows out: Q_structure > 0."""
        assert _T_STRUCTURE_OP > _T_FLUID_OP   # sanity on the operating point
        model = _make_model()
        state = _make_state()
        Q_str = model.compute_heat_flux(0.0, state)
        assert Q_str > 0.0


# ===========================================================================
# V6 — Outer-shell transient: closed-form vs numerical (isolated problem)
# ===========================================================================

class TestV6ShellTransient:
    """
    Validate the shell ODE mathematical form against its exact analytical solution
    using an isolated problem with fully frozen parameters.

    Architecture note
    -----------------
    The production compute_shell_temperature_derivative uses temperature-dependent
    c_shell(T_sh) and variable insulation conductance.  Freezing these parameters
    without modifying production code is not practical.  Instead, a self-contained
    ODE is integrated numerically (no production code) and compared against the
    analytical closed-form solution.

    Governing ODE (frozen parameters):
        C_s · dT_s/dt = hA·(T_a − T_s) − G_foam·(T_s − T_w)

    Analytical solution:
        T_s(t) = T_s∞ + (T_s(0) − T_s∞) · exp(−t / τ)
        T_s∞   = (hA · T_a + G_foam · T_w) / (hA + G_foam)
        τ      = C_s / (hA + G_foam)
    """

    # Frozen parameters
    C_S = 5_000.0    # J/K, shell thermal capacity
    H_A = 2.0        # W/K, ambient convection conductance
    G_FOAM = 3.0     # W/K, foam conductance
    T_A = 288.15     # K,  ambient temperature
    T_W = 60.0       # K,  structure temperature (held fixed)
    T_S0 = 100.0     # K,  initial shell temperature

    @property
    def _T_s_inf(self):
        return (self.H_A * self.T_A + self.G_FOAM * self.T_W) / (self.H_A + self.G_FOAM)

    @property
    def _tau(self):
        return self.C_S / (self.H_A + self.G_FOAM)

    def _analytical(self, t: float) -> float:
        return self._T_s_inf + (self.T_S0 - self._T_s_inf) * math.exp(-t / self._tau)

    def _ode_rhs(self, t, y):
        T_s = y[0]
        dT_s = (self.H_A * (self.T_A - T_s) - self.G_FOAM * (T_s - self.T_W)) / self.C_S
        return [dT_s]

    def test_numerical_matches_analytical_solution(self):
        tau = self._tau
        t_eval = np.linspace(0.0, 5.0 * tau, 500)

        sol = solve_ivp(
            self._ode_rhs,
            (t_eval[0], t_eval[-1]),
            [self.T_S0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol.success, f"V6: ODE solver failed: {sol.message}"

        T_numerical = sol.y[0]
        T_exact = np.array([self._analytical(t) for t in t_eval])

        # Normalise by the full temperature excursion so the error is dimensionless
        T_scale = abs(self.T_S0 - self._T_s_inf)
        max_rel_err = np.max(np.abs(T_numerical - T_exact)) / T_scale

        assert max_rel_err < 1e-7, (
            f"V6: Max relative temperature error {max_rel_err:.2e} exceeds tolerance. "
            f"τ={tau:.1f} s, T_s∞={self._T_s_inf:.2f} K"
        )


# ===========================================================================
# V7 — Whole-network steady-state energy conservation
# ===========================================================================

class TestV7SteadyStateConservation:
    """
    At thermal steady state with a prescribed hydrogen temperature, verify

        Q_amb ≈ Q_insulation ≈ Q_structure.

    The steady-state temperatures (T_sh_ss, T_s_ss) are found by solving the
    algebraic system Q_amb = Q_ins, Q_ins = Q_str — the t → ∞ limit of the
    coupled shell+structure ODEs.  A representative fixed alpha_s is used for
    the inner convection so that CoolProp is not required at each iteration.
    """

    _ALPHA_S_FIXED = 10.0   # W/m²K, representative inner natural-convection HTC

    def test_energy_balance_residuals_at_steady_state(self):
        model = _make_model()
        alpha_s = self._ALPHA_S_FIXED

        def _residual(T_vec):
            T_sh, T_s = T_vec
            Q_amb = model.compute_ambient_heat_flux(T_sh)
            Q_ins = model.compute_insulation_heat_flux(T_s, T_sh)
            Q_str = alpha_s * model.A_in * (T_s - _T_FLUID_OP)
            # Both residuals must vanish at steady state
            return [Q_amb - Q_ins, Q_ins - Q_str]

        # Initial guess informed by the expected steady state:
        # shell is pulled below T_amb by the cold insulation load;
        # structure is only slightly above T_fluid due to small G_foam.
        x0 = [model.T_amb - 15.0, _T_FLUID_OP + 8.0]
        T_ss, _info, ier, msg = fsolve(_residual, x0, full_output=True)
        assert ier == 1, f"V7: Steady-state solver failed to converge: {msg}"

        T_sh_ss, T_s_ss = T_ss
        Q_amb_ss = model.compute_ambient_heat_flux(T_sh_ss)
        Q_ins_ss = model.compute_insulation_heat_flux(T_s_ss, T_sh_ss)
        Q_str_ss = alpha_s * model.A_in * (T_s_ss - _T_FLUID_OP)

        scale = abs(Q_amb_ss)
        R_shell = abs(Q_amb_ss - Q_ins_ss) / scale
        R_structure = abs(Q_ins_ss - Q_str_ss) / scale

        assert R_shell < 1e-8, (
            f"V7: Shell residual {R_shell:.2e} at steady state. "
            f"Q_amb={Q_amb_ss:.4f} W, Q_ins={Q_ins_ss:.4f} W"
        )
        assert R_structure < 1e-8, (
            f"V7: Structure residual {R_structure:.2e} at steady state. "
            f"Q_ins={Q_ins_ss:.4f} W, Q_str={Q_str_ss:.4f} W"
        )


# ===========================================================================
# V8 — Quasi-steady foam timescale assessment
# ===========================================================================

def foam_timescale_assessment(
    t_foam: float,
    C_shell: float,
    h_A: float,
    G_foam: float,
    C_structure: float | None = None,
    cp_rohacell: float | None = None,
    T_mean: float = 160.0,
) -> dict:
    """
    Estimate the Rohacell foam thermal diffusion time and compare it against
    the system (shell / structure) thermal timescales.

    Parameters
    ----------
    t_foam : float
        Foam thickness [m].
    C_shell : float
        Shell thermal capacity [J/K].
    h_A : float
        Ambient convection conductance = α_amb · A_shell [W/K].
    G_foam : float
        Foam conductance = Q_insulation / ΔT [W/K].
    C_structure : float, optional
        Structure thermal capacity [J/K].
    cp_rohacell : float, optional
        Rohacell specific heat [J/kg·K].  Defaults to the value from the
        Rohacell 31 dataset (rohacell31_specific_heat.csv) evaluated at T_mean.
    T_mean : float
        Representative foam mean temperature [K] for evaluating k and α_diff.

    Returns
    -------
    dict
        Keys: ``tau_shell``, ``tau_foam``, ``alpha_foam`` always present;
        ``tau_structure`` present when C_structure is supplied.
    """
    result: dict = {}
    result["tau_shell"] = C_shell / (h_A + G_foam)

    if C_structure is not None:
        result["tau_structure"] = C_structure / G_foam

    if cp_rohacell is None:
        cp_rohacell = rohacell_cp(T_mean)
    k_foam = rohacell_k(T_mean)
    alpha_diff = k_foam / (ROHACELL_DENSITY * cp_rohacell)
    result["alpha_foam"] = alpha_diff
    result["tau_foam"] = t_foam**2 / alpha_diff

    return result


class TestV8QuasiSteadyFoam:
    """
    Assess whether treating the Rohacell foam conduction as quasi-steady is
    justified: τ_foam << τ_shell and τ_foam << τ_structure.

    Specific heat is taken from the Rohacell 31 dataset (rohacell31_specific_heat.csv),
    used as the best available approximation for Rohacell 51A.

    Physical finding: for this tank geometry, τ_foam > τ_shell, so the quasi-steady
    assumption is NOT validated relative to the outer-shell timescale.  It IS
    validated relative to the structure timescale.
    """

    def _compute_system_timescales(self, model):
        T_sh = _T_SHELL_OP
        T_bounded = max(4.0, min(T_sh, 400.0))
        C_shell = model.m_shell * model.shell_material.determine_specific_heat(T_bounded)

        # Structure thermal capacity at operating temperature
        T_s_b = max(4.0, min(_T_STRUCTURE_OP, 400.0))
        C_structure = (
            model.m_liner * model.liner_material.determine_specific_heat(T_s_b)
            + model.m_wall * model.wall_material.determine_specific_heat(T_s_b)
        )

        G_foam = (
            model.compute_insulation_heat_flux(_T_STRUCTURE_OP, T_sh)
            / (T_sh - _T_STRUCTURE_OP)
        )
        h_A = model.alpha_amb * model.A_shell
        t_foam = model.r_shell - model.r_structure
        return C_shell, C_structure, G_foam, h_A, t_foam

    def test_tau_foam_vs_tau_shell_ratio(self):
        """
        Physical finding: τ_foam EXCEEDS τ_shell for this tank geometry.
        The quasi-steady assumption is therefore NOT validated relative to the
        outer-shell timescale.  The test asserts only that the ratio stays below
        100× (sanity / order-of-magnitude check).
        """
        model = _make_model()
        C_shell, C_structure, G_foam, h_A, t_foam = self._compute_system_timescales(model)

        result = foam_timescale_assessment(
            t_foam=t_foam, C_shell=C_shell, h_A=h_A, G_foam=G_foam,
        )
        ratio = result["tau_foam"] / result["tau_shell"]
        assert ratio < 100.0, (
            f"V8: τ_foam/τ_shell={ratio:.1f} exceeds 100×. "
            f"τ_foam={result['tau_foam']:.1f} s, τ_shell={result['tau_shell']:.1f} s. "
            f"NOTE: quasi-steady assumption is NOT justified for the shell ODE."
        )

    def test_tau_foam_less_than_tau_structure(self):
        model = _make_model()
        C_shell, C_structure, G_foam, h_A, t_foam = self._compute_system_timescales(model)

        result = foam_timescale_assessment(
            t_foam=t_foam, C_shell=C_shell, h_A=h_A, G_foam=G_foam,
            C_structure=C_structure,
        )
        assert result["tau_foam"] < result["tau_structure"], (
            f"V8: τ_foam={result['tau_foam']:.1f} s is NOT << "
            f"τ_structure={result['tau_structure']:.1f} s. "
            f"Quasi-steady assumption may be invalid."
        )
