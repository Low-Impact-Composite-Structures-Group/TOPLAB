"""Validation tests for the diagnostic multi-node insulation network."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import root

from toplab.materials.insulation_models import ROHACELL
from toplab.materials.nist_materials import NISTComposite, NISTMetal
from toplab.thermodynamics.insulation_rc_diagnostic import (
    InsulationRCModel,
    insulation_diffusion_times,
)
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel


_R_STRUCTURE = 0.518
_T_INSULATION = 0.05
_L_CYLINDER = 1.5
_T_SHELL = 0.002
_T_SHELL_BOUNDARY = 288.15
_T_STRUCTURE_BOUNDARY = 60.0


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
    r_shell = r_structure + _T_INSULATION
    tank_volume = math.pi * r_inner**3 * (
        4.0 / 3.0 + _L_CYLINDER / r_inner
    )
    return InsulatedTankThermalModel(
        tank_volume=tank_volume,
        inner_surface_area=4.0 * math.pi * r_inner**2
        + 2.0 * math.pi * r_inner * _L_CYLINDER,
        inner_diameter=2.0 * r_inner,
        r_structure=r_structure,
        r_shell=r_shell,
        cylinder_length=_L_CYLINDER,
        liner_mass=_layer_mass(liner.density, r_inner, r_inner + 0.003),
        wall_mass=_layer_mass(wall.density, r_inner + 0.003, r_structure),
        foam_mass=_layer_mass(ROHACELL.density, r_structure, r_shell),
        shell_mass=_layer_mass(
            liner.density, r_shell, r_shell + _T_SHELL
        ),
        ambient_temperature=_T_SHELL_BOUNDARY,
        alpha_amb=5.0,
        emissivity_shell=0.05,
        liner_material=liner,
        wall_material=wall,
        shell_material=liner,
    )


def _steady_temperatures(rc_model, shell_temperature, structure_temperature):
    initial = np.linspace(
        shell_temperature,
        structure_temperature,
        rc_model.node_count + 2,
    )[1:-1]

    def balance(temperatures):
        flows = rc_model.heat_flows(
            shell_temperature, temperatures, structure_temperature
        )
        return flows[:-1] - flows[1:]

    solution = root(balance, initial)
    assert solution.success, solution.message
    return solution.x


def _integrate_fixed_boundaries(
    rc_model, initial_temperatures, duration_seconds, output_count=301
):
    times = np.linspace(0.0, duration_seconds, output_count)
    solution = solve_ivp(
        lambda _time, temperatures: rc_model.temperature_derivatives(
            _T_SHELL_BOUNDARY, temperatures, _T_STRUCTURE_BOUNDARY
        ),
        (0.0, duration_seconds),
        initial_temperatures,
        method="DOP853",
        t_eval=times,
        rtol=1e-9,
        atol=1e-10,
        max_step=300.0,
    )
    assert solution.success, solution.message
    return times, solution.y.T


def test_insulation_mass_is_conserved_for_each_discretization():
    tank_model = _make_model()
    for node_count in (1, 2, 3, 5, 10, 20, 40, 80):
        rc_model = InsulationRCModel(tank_model, node_count)
        assert rc_model.total_insulation_mass == pytest.approx(
            tank_model.m_foam, rel=1e-12
        )
        assert np.all(rc_model.node_masses > 0.0)


def test_energy_balance_residual_is_zero_and_integrated_balance_closes():
    rc_model = InsulationRCModel(_make_model(), 5)
    temperatures = np.array([270.0, 235.0, 190.0, 130.0, 85.0])
    assert rc_model.energy_residual(
        _T_SHELL_BOUNDARY, temperatures, _T_STRUCTURE_BOUNDARY
    ) == pytest.approx(0.0, abs=1e-10)

    times, history = _integrate_fixed_boundaries(
        rc_model,
        np.full(5, _T_SHELL_BOUNDARY),
        3.0 * 3600.0,
        output_count=1201,
    )
    flow_history = np.asarray(
        [
            rc_model.heat_flows(
                _T_SHELL_BOUNDARY, state, _T_STRUCTURE_BOUNDARY
            )
            for state in history
        ]
    )
    energy_in = simpson(flow_history[:, 0], x=times)
    energy_out = simpson(flow_history[:, -1], x=times)
    energy_change = rc_model.energy(history[-1]) - rc_model.energy(history[0])
    residual = energy_in - energy_out - energy_change
    relative_error = abs(residual) / max(abs(energy_in), abs(energy_out), 1.0)
    assert relative_error < 2e-5


def test_heat_flow_sign_convention_and_equal_temperature_limit():
    rc_model = InsulationRCModel(_make_model(), 4)
    forward = rc_model.heat_flows(288.0, [240.0, 190.0, 130.0, 90.0], 60.0)
    reverse = rc_model.heat_flows(60.0, [90.0, 130.0, 190.0, 240.0], 288.0)
    equal = rc_model.heat_flows(150.0, [150.0] * 4, 150.0)

    assert np.all(forward > 0.0)
    assert np.all(reverse < 0.0)
    assert equal == pytest.approx(np.zeros(5), abs=1e-12)


def test_one_node_reproduces_production_insulation_heat_flows_and_derivative():
    tank_model = _make_model()
    rc_model = InsulationRCModel(tank_model, 1)
    shell_temperature, insulation_temperature, structure_temperature = (
        288.0,
        174.0,
        60.0,
    )
    state = SimpleNamespace(
        shell_temperature=shell_temperature,
        insulation_temperature=insulation_temperature,
        structure_temperature=structure_temperature,
    )
    expected_flows = np.array(
        [
            tank_model.compute_shell_to_insulation_heat_flux(
                shell_temperature, insulation_temperature
            ),
            tank_model.compute_insulation_to_structure_heat_flux(
                insulation_temperature, structure_temperature
            ),
        ]
    )
    assert rc_model.heat_flows(
        shell_temperature, [insulation_temperature], structure_temperature
    ) == pytest.approx(expected_flows, rel=1e-13)
    assert rc_model.temperature_derivatives(
        shell_temperature, [insulation_temperature], structure_temperature
    )[0] == pytest.approx(
        tank_model.compute_insulation_temperature_derivative(0.0, state),
        rel=1e-13,
    )


def test_transient_converges_as_insulation_node_count_increases():
    duration = 3.0 * 3600.0
    results = {}
    tank_model = _make_model()
    for node_count in (1, 2, 3, 10, 20, 40, 80):
        rc_model = InsulationRCModel(tank_model, node_count)
        initial = _steady_temperatures(
            rc_model, _T_SHELL_BOUNDARY, _T_STRUCTURE_BOUNDARY
        )
        # A suddenly warm insulation layer exposes transient differences.
        initial[:] = _T_SHELL_BOUNDARY
        times, history = _integrate_fixed_boundaries(
            rc_model, initial, duration, output_count=181
        )
        structure_heat_flow = np.asarray(
            [
                rc_model.heat_flows(
                    _T_SHELL_BOUNDARY, state, _T_STRUCTURE_BOUNDARY
                )[-1]
                for state in history
            ]
        )
        results[node_count] = (times, history, structure_heat_flow)

    reference_flow = results[20][2]
    errors = {
        node_count: {
            "rms": float(np.sqrt(np.mean((result[2] - reference_flow) ** 2))),
            "max": float(np.max(np.abs(result[2] - reference_flow))),
        }
        for node_count, result in results.items()
    }
    assert errors[10]["rms"] < errors[3]["rms"] < errors[2]["rms"]
    assert errors[3]["rms"] < errors[1]["rms"]

    sample_indices = {
        "10 min": 10,
        "1 h": 60,
        "3 h": 180,
    }
    cumulative_heat_kj = {}
    for node_count, (model_times, _, heat_flow) in results.items():
        cumulative_heat_kj[node_count] = {
            label: float(simpson(heat_flow[: index + 1], x=model_times[: index + 1]) / 1000.0)
            for label, index in sample_indices.items()
        }

    relative_rms = {
        node_count: 100.0 * error["rms"] / np.sqrt(np.mean(reference_flow**2))
        for node_count, error in errors.items()
    }
    diffusion_rows = insulation_diffusion_times(
        tank_model, (50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    )
    print("\nThermal diffusion times")
    print("-----------------------")
    print(f"{'T [K]':>6}  {'alpha [m2/s]':>14}  {'t_diff [h]':>12}")
    for row in diffusion_rows:
        print(
            f"{row['temperature_K']:>6.0f}  "
            f"{row['diffusivity_m2_s']:>14.6g}  "
            f"{row['diffusion_time_h']:>12.6g}"
        )

    print("\nWarm-start transient, N=20 reference")
    print("------------------------------------")
    print(
        f"{'N':>2}  {'RMS Q error [W]':>16}  "
        f"{'Max Q error [W]':>16}  {'Relative RMS [%]':>17}"
    )
    for node_count in (1, 2, 3, 10, 20, 40, 80):
        print(
            # f"{node_count:>2}  {errors[node_count]['rms']:>16.6g}  "
            f"{errors[node_count]['max']:>16.6g}  "
            f"{relative_rms[node_count]:>17.6g}"
        )

    print("\nCumulative heat to structure after:")
    print("-----------------------------------")
    print(
        f"{'N':>2}  {'10 min [kJ]':>12}  {'1 h [kJ]':>12}  "
        f"{'3 h [kJ]':>12}"
    )
    for node_count in (1, 2, 3, 10, 20, 40, 80):
        heat = cumulative_heat_kj[node_count]
        print(
            f"{node_count:>2}  {heat['10 min']:>12.6g}  "
            f"{heat['1 h']:>12.6g}  {heat['3 h']:>12.6g}"
        )



def test_constant_property_rc_discretization_invariants():
    """Verify the diagnostic mesh independently with constant k and constant cp.

    This isolates the spatial RC discretization from the Rohacell property
    correlations.  For every mesh order, the test checks:

    1. total insulation mass/capacitance,
    2. steady-state heat flow against the exact whole-layer solution,
    3. monotonic transient convergence toward the finest mesh.

    The production thermal model is not modified.
    """
    node_counts = (1, 2, 3, 5, 10, 20, 40, 80)
    reference_node_count = 80

    tank_model = _make_model()

    # Representative frozen properties.
    T_mean = 0.5 * (_T_SHELL_BOUNDARY + _T_STRUCTURE_BOUNDARY)
    k_const = float(
        tank_model.insulation_model.integrated_conductivity(
            T_mean - 0.5, T_mean + 0.5
        )
    )
    cp_const = float(tank_model.insulation_model.specific_heat(T_mean))
    rho_const = float(tank_model.insulation_model.density)

    r_inner = tank_model.r_structure
    r_outer = tank_model.r_shell
    length = tank_model.L

    # Exact whole-layer conductances.  Cylinder and spherical endcaps are
    # parallel heat-flow paths, so their conductances add.
    G_cyl = 2.0 * math.pi * length * k_const / math.log(r_outer / r_inner)
    G_sph = (
        4.0
        * math.pi
        * k_const
        * r_inner
        * r_outer
        / (r_outer - r_inner)
    )
    G_exact = G_cyl + G_sph

    delta_T = _T_SHELL_BOUNDARY - _T_STRUCTURE_BOUNDARY
    Q_exact = G_exact * delta_T

    insulation_volume = (
        math.pi * length * (r_outer**2 - r_inner**2)
        + (4.0 * math.pi / 3.0) * (r_outer**3 - r_inner**3)
    )
    mass_exact = rho_const * insulation_volume
    C_exact = mass_exact * cp_const

    def segment_heat_flow(
        inner_radius,
        outer_radius,
        inner_temperature,
        outer_temperature,
    ):
        geometry_factor = (
            2.0 * math.pi * length / math.log(outer_radius / inner_radius)
            + 4.0
            * math.pi
            * inner_radius
            * outer_radius
            / (outer_radius - inner_radius)
        )
        return (
            geometry_factor
            * k_const
            * (outer_temperature - inner_temperature)
        )

    def constant_heat_flows(rc_model, node_temperatures):
        temperatures = np.asarray(node_temperatures, dtype=float)

        boundary_temperatures = np.concatenate(
            (
                [_T_SHELL_BOUNDARY],
                temperatures,
                [_T_STRUCTURE_BOUNDARY],
            )
        )
        boundary_radii = np.concatenate(
            (
                [tank_model.r_shell],
                rc_model.node_radii,
                [tank_model.r_structure],
            )
        )

        return np.asarray(
            [
                segment_heat_flow(
                    boundary_radii[index + 1],
                    boundary_radii[index],
                    boundary_temperatures[index + 1],
                    boundary_temperatures[index],
                )
                for index in range(rc_model.node_count + 1)
            ],
            dtype=float,
        )

    def constant_derivatives(rc_model, node_temperatures):
        flows = constant_heat_flows(rc_model, node_temperatures)
        capacities = rc_model.node_masses * cp_const
        return (flows[:-1] - flows[1:]) / capacities

    def steady_temperatures_constant(rc_model):
        initial = np.linspace(
            _T_SHELL_BOUNDARY,
            _T_STRUCTURE_BOUNDARY,
            rc_model.node_count + 2,
        )[1:-1]

        solution = root(
            lambda temperatures: (
                constant_heat_flows(rc_model, temperatures)[:-1]
                - constant_heat_flows(rc_model, temperatures)[1:]
            ),
            initial,
        )
        assert solution.success, solution.message
        return solution.x

    steady_results = {}
    transient_results = {}

    duration = 3.0 * 3600.0
    output_count = 181
    times = np.linspace(0.0, duration, output_count)

    print("\nConstant-property RC discretization verification")
    print("------------------------------------------------")
    print(f"k_const  = {k_const:.8g} W/(m K)")
    print(f"cp_const = {cp_const:.8g} J/(kg K)")
    print(f"m_exact  = {mass_exact:.8g} kg")
    print(f"C_exact  = {C_exact:.8g} J/K")
    print(f"Q_exact  = {Q_exact:.8g} W")

    print("\nSteady-state / capacitance invariants")
    print("------------------------------------")
    print(
        f"{'N':>3}  "
        f"{'mass [kg]':>12}  "
        f"{'C_total [J/K]':>15}  "
        f"{'Q_ss [W]':>14}  "
        f"{'Q error [%]':>12}  "
        f"{'C error [%]':>12}"
    )

    for node_count in node_counts:
        rc_model = InsulationRCModel(tank_model, node_count)

        mass_n = float(np.sum(rc_model.node_masses))
        C_n = float(np.sum(rc_model.node_masses * cp_const))

        steady_temperatures = steady_temperatures_constant(rc_model)
        steady_flows = constant_heat_flows(rc_model, steady_temperatures)
        Q_n = float(np.mean(steady_flows))

        q_error_percent = 100.0 * (Q_n - Q_exact) / Q_exact
        c_error_percent = 100.0 * (C_n - C_exact) / C_exact

        steady_results[node_count] = {
            "mass": mass_n,
            "capacity": C_n,
            "flow": Q_n,
        }

        print(
            f"{node_count:>3}  "
            f"{mass_n:>12.6g}  "
            f"{C_n:>15.6g}  "
            f"{Q_n:>14.6g}  "
            f"{q_error_percent:>12.6g}  "
            f"{c_error_percent:>12.6g}"
        )

        # These are invariants of a consistent discretization, not merely
        # asymptotic convergence targets.
        assert mass_n == pytest.approx(mass_exact, rel=1e-12)
        assert C_n == pytest.approx(C_exact, rel=1e-12)
        relative_q_error = (
            np.max(np.abs(steady_flows - Q_exact))
            / abs(Q_exact)
        )
        assert relative_q_error < 0.05

        # Warm-start transient with the same fixed boundary temperatures used
        # by the original convergence diagnostic.
        initial = np.full(node_count, _T_SHELL_BOUNDARY, dtype=float)

        solution = solve_ivp(
            lambda _time, temperatures: constant_derivatives(
                rc_model, temperatures
            ),
            (0.0, duration),
            initial,
            method="DOP853",
            t_eval=times,
            rtol=1e-9,
            atol=1e-10,
            max_step=300.0,
        )
        assert solution.success, solution.message

        history = solution.y.T
        structure_heat_flow = np.asarray(
            [
                constant_heat_flows(rc_model, state)[-1]
                for state in history
            ]
        )
        transient_results[node_count] = structure_heat_flow

    reference_flow = transient_results[reference_node_count]

    transient_errors = {
        node_count: {
            "rms": float(
                np.sqrt(
                    np.mean(
                        (heat_flow - reference_flow) ** 2
                    )
                )
            ),
            "max": float(
                np.max(np.abs(heat_flow - reference_flow))
            ),
        }
        for node_count, heat_flow in transient_results.items()
    }

    print(
        f"\nConstant-property warm-start transient, "
        f"N={reference_node_count} reference"
    )
    print("--------------------------------------------------")
    print(
        f"{'N':>3}  "
        f"{'RMS Q error [W]':>16}  "
        f"{'Max Q error [W]':>16}"
    )
    for node_count in node_counts:
        print(
            f"{node_count:>3}  "
            f"{transient_errors[node_count]['rms']:>16.6g}  "
            f"{transient_errors[node_count]['max']:>16.6g}"
        )

    # For a well-behaved cell-centred discretization, refinement should reduce
    # the transient error toward the finest available reference.
    for coarse, fine in zip(node_counts[:-2], node_counts[1:-1]):
        assert transient_errors[fine]["rms"] < transient_errors[coarse]["rms"], (
            f"Constant-property transient did not converge monotonically "
            f"from N={coarse} to N={fine}: "
            f"{transient_errors[coarse]['rms']:.6g} -> "
            f"{transient_errors[fine]['rms']:.6g} W"
        )

def test_quasi_steady_heat_flows_match_across_segments_and_model_orders():
    tank_model = _make_model()
    steady_flows = {}
    for node_count in (1, 2, 3, 10, 20, 40, 80):
        rc_model = InsulationRCModel(tank_model, node_count)
        temperatures = _steady_temperatures(
            rc_model, _T_SHELL_BOUNDARY, _T_STRUCTURE_BOUNDARY
        )
        flows = rc_model.heat_flows(
            _T_SHELL_BOUNDARY, temperatures, _T_STRUCTURE_BOUNDARY
        )
        assert np.max(np.abs(flows - flows[0])) / abs(flows[0]) < 1e-7
        steady_flows[node_count] = flows[0]

    assert steady_flows[10] == pytest.approx(steady_flows[1], rel=0.05)


def test_diffusion_time_table_uses_temperature_dependent_properties():
    model = _make_model()
    temperatures = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    rows = insulation_diffusion_times(model, temperatures)

    assert [row["temperature_K"] for row in rows] == list(temperatures)
    for row in rows:
        assert row["conductivity_W_mK"] > 0.0
        assert row["specific_heat_J_kgK"] > 0.0
        assert row["density_kg_m3"] == pytest.approx(ROHACELL.density)
        assert row["diffusivity_m2_s"] == pytest.approx(
            row["conductivity_W_mK"]
            / (row["density_kg_m3"] * row["specific_heat_J_kgK"])
        )
        assert row["diffusion_time_h"] == pytest.approx(
            row["diffusion_time_s"] / 3600.0
        )


def test_quasi_steady_initialization_and_warm_initialization_are_supported():
    rc_model = InsulationRCModel(_make_model(), 3)
    equilibrated = _steady_temperatures(
        rc_model, _T_SHELL_BOUNDARY, _T_STRUCTURE_BOUNDARY
    )
    equilibrium_derivatives = rc_model.temperature_derivatives(
        _T_SHELL_BOUNDARY, equilibrated, _T_STRUCTURE_BOUNDARY
    )
    assert np.max(np.abs(equilibrium_derivatives)) < 1e-9

    warm = np.full(3, _T_SHELL_BOUNDARY)
    warm_derivatives = rc_model.temperature_derivatives(
        _T_SHELL_BOUNDARY, warm, _T_STRUCTURE_BOUNDARY
    )
    assert np.any(np.abs(warm_derivatives) > 0.0)
    assert np.all(warm_derivatives <= 0.0)
