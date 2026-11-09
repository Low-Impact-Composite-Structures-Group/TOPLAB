"""Validation of the diagnostic multi-node insulation network with MLI.

The Modified Lockheed correlation is applied independently to every RC
segment. The effective MLI density and specific heat are model defaults, not
measured transient properties, hence the heat-capacity sensitivity study.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from test.rc_circuit_validation.mli_rc_studies import (
    DEFAULT_DENSITY,
    DURATION,
    PRODUCTION_TOLERANCES,
    REFINED_TOLERANCES,
    SAMPLE_TIMES,
    T_SHELL_BOUNDARY,
    T_STRUCTURE_BOUNDARY,
    TIGHT_TOLERANCES,
    UNIFORM_STEPS,
    make_model,
    simulate,
    steady_temperatures,
)
from toplab.thermodynamics.insulation_rc_diagnostic import InsulationRCModel


SPATIAL_NODE_COUNTS = (1, 10, 20, 50, 100, 150)
SPATIAL_REFERENCE = 150
DENSITIES = (60.0, DEFAULT_DENSITY, 1500.0)
DENSITY_NODE_COUNTS = (1, 5, 10, 20, 50, 100)
DENSITY_REFERENCE = 100


def _percent_error(value, reference):
    return 100.0 * (value - reference) / reference


def test_rc_model_accepts_mli_and_conserves_mass():
    tank_model = make_model()
    for node_count in (1, 2, 5, 20):
        rc_model = InsulationRCModel(tank_model, node_count)
        assert rc_model.total_insulation_mass == pytest.approx(
            tank_model.m_foam, rel=1e-12
        )


def test_mli_heat_flow_sign_convention_and_equal_temperature_limit():
    rc_model = InsulationRCModel(make_model(), 4)
    forward = rc_model.heat_flows(288.0, [240.0, 190.0, 130.0, 90.0], 60.0)
    reverse = rc_model.heat_flows(60.0, [90.0, 130.0, 190.0, 240.0], 288.0)
    equal = rc_model.heat_flows(150.0, [150.0] * 4, 150.0)

    assert np.all(forward > 0.0)
    assert np.all(reverse < 0.0)
    assert equal == pytest.approx(np.zeros(5), abs=1e-12)


def test_mli_one_node_reproduces_production_heat_flows_and_derivative():
    tank_model = make_model()
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
    rc_model = InsulationRCModel(make_model(), 5)
    temperatures = np.array([270.0, 235.0, 190.0, 130.0, 85.0])
    assert rc_model.energy_residual(
        T_SHELL_BOUNDARY, temperatures, T_STRUCTURE_BOUNDARY
    ) == pytest.approx(0.0, abs=1e-10)


def test_mli_steady_state_heat_flow_converges_with_node_count():
    tank_model = make_model()
    flows = {}
    for node_count in (1, 2, 4, 8, 16, 32):
        rc_model = InsulationRCModel(tank_model, node_count)
        temperatures = steady_temperatures(rc_model)
        steady = rc_model.heat_flows(
            T_SHELL_BOUNDARY, temperatures, T_STRUCTURE_BOUNDARY
        )
        assert steady == pytest.approx(np.full_like(steady, steady[0]), rel=1e-8)
        flows[node_count] = steady[0]

    reference = flows[32]
    assert reference > 0.0
    errors = [abs(flows[n] - reference) / reference for n in (2, 4, 8, 16)]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.02


def test_mli_transient_relaxes_to_steady_state():
    rc_model = InsulationRCModel(make_model(), 8)
    steady = steady_temperatures(rc_model)
    solution = solve_ivp(
        lambda _t, y: rc_model.temperature_derivatives(
            T_SHELL_BOUNDARY, y, T_STRUCTURE_BOUNDARY
        ),
        (0.0, 5000.0 * 3600.0),
        np.full(8, T_SHELL_BOUNDARY),
        method="Radau",
        rtol=1e-8,
        atol=1e-9,
    )
    assert solution.success, solution.message
    assert solution.y[:, -1] == pytest.approx(steady, abs=0.5)


def test_mli_transient_spatial_convergence():
    """Step start (all nodes warm vs. cold structure): absolute cumulative heat."""
    results = {n: simulate(n) for n in SPATIAL_NODE_COUNTS}
    reference = results[SPATIAL_REFERENCE]

    print("\nMLI step-start spatial convergence (20 mm, default rho*cp)")
    print(f"{'N':>4} " + " ".join(f"{f'Q {t / 60:g} min [kJ]':>16}" for t in SAMPLE_TIMES)
          + f" {'q(1 s) [W]':>12} {'q(10 min) [W]':>14}")
    for n, result in results.items():
        print(
            f"{n:>4} "
            + " ".join(f"{result.cumulative_at(t):>16.4f}" for t in SAMPLE_TIMES)
            + f" {result.heat_flow_at(1.0):>12.3f} {result.heat_flow_at(600.0):>14.3f}"
        )
    print(f"\nDeviation from N={SPATIAL_REFERENCE} [%] (reference is the finest "
          "mesh, not an exact solution)")
    for n, result in results.items():
        print(
            f"{n:>4} "
            + " ".join(
                f"{_percent_error(result.cumulative_at(t), reference.cumulative_at(t)):>16.3f}"
                for t in SAMPLE_TIMES
            )
        )

    for t in SAMPLE_TIMES:
        errors = {
            n: abs(results[n].cumulative_at(t) - reference.cumulative_at(t))
            for n in (1, 10, 20, 50)
        }
        assert errors[1] > errors[10] > errors[20] > errors[50]


def test_mli_heat_capacity_sensitivity():
    """Required node count and transient heat depend strongly on rho*cp."""
    results = {
        (rho, n): simulate(n, density=rho)
        for rho in DENSITIES
        for n in DENSITY_NODE_COUNTS
    }

    print("\nMLI rho*cp sensitivity (cp=1000 J/kg/K, step start, 20 mm)")
    for horizon in SAMPLE_TIMES:
        print(f"\nCumulative heat at {horizon / 60:g} min [kJ] (deviation from N={DENSITY_REFERENCE} [%])")
        print(f"{'rho':>8} " + " ".join(f"{f'N={n}':>18}" for n in DENSITY_NODE_COUNTS))
        for rho in DENSITIES:
            reference = results[(rho, DENSITY_REFERENCE)].cumulative_at(horizon)
            cells = []
            for n in DENSITY_NODE_COUNTS:
                value = results[(rho, n)].cumulative_at(horizon)
                cells.append(f"{value:>9.2f}({_percent_error(value, reference):+6.1f}%)")
            print(f"{rho:>8g} " + " ".join(cells))

    one_hour = 3600.0
    converged = [results[(rho, DENSITY_REFERENCE)].cumulative_at(one_hour) for rho in DENSITIES]
    assert converged[0] < converged[1] < converged[2]

    mesh_errors = [
        abs(
            results[(rho, 20)].cumulative_at(one_hour)
            - results[(rho, DENSITY_REFERENCE)].cumulative_at(one_hour)
        )
        / results[(rho, DENSITY_REFERENCE)].cumulative_at(one_hour)
        for rho in DENSITIES
    ]
    assert mesh_errors[0] < mesh_errors[1] < mesh_errors[2]


def test_mli_time_accuracy():
    """Separate solver-tolerance error from output-grid (quadrature) error."""
    node_count = 50
    tolerance_sets = {
        "production 1e-6/1e-9": PRODUCTION_TOLERANCES,
        "refined 1e-9/1e-10": REFINED_TOLERANCES,
        "tight 1e-11/1e-12": TIGHT_TOLERANCES,
    }
    reference = simulate(node_count, grid="very_fine", tolerances=TIGHT_TOLERANCES)

    print(f"\nMLI time accuracy, N={node_count}; reference: very_fine grid, tight tolerances")
    print("\nSolver tolerance (refined output grid): deviation of cumulative heat [%]")
    print(f"{'tolerances':>24} " + " ".join(f"{f'{t / 60:g} min':>10}" for t in SAMPLE_TIMES))
    tolerance_errors = {}
    for label, tolerances in tolerance_sets.items():
        result = simulate(node_count, tolerances=tolerances)
        errors = [
            _percent_error(result.cumulative_at(t), reference.cumulative_at(t))
            for t in SAMPLE_TIMES
        ]
        tolerance_errors[label] = errors
        print(f"{label:>24} " + " ".join(f"{e:>10.2e}" for e in errors))

    print("\nOutput grid (refined tolerances): deviation of cumulative heat [%]")
    print(f"{'grid':>24} " + " ".join(f"{f'{t / 60:g} min':>10}" for t in SAMPLE_TIMES))
    grid_errors = {}
    for grid in ("refined",) + tuple(f"uniform_{s:g}s" for s in UNIFORM_STEPS):
        result = simulate(node_count, grid=grid)
        errors = [
            _percent_error(result.cumulative_at(t), reference.cumulative_at(t))
            for t in SAMPLE_TIMES
        ]
        grid_errors[grid] = errors
        print(f"{grid:>24} " + " ".join(f"{e:>10.2e}" for e in errors))

    # Integrated energy balance on the refined grid: shell in - structure out
    # equals the change of stored insulation energy.
    balance_model = simulate(20)
    shell_in = np.trapezoid(balance_model.shell_heat_flow, balance_model.times)
    structure_out = np.trapezoid(balance_model.heat_flow, balance_model.times)
    stored = balance_model.stored_energy[-1] - balance_model.stored_energy[0]
    closure = (shell_in - structure_out - stored) / max(abs(stored), 1.0)
    print(f"\nIntegrated energy balance, N=20: closure {closure:.3e} (relative to stored energy)")

    assert max(abs(e) for e in tolerance_errors["refined 1e-9/1e-10"]) < 0.01
    assert max(abs(e) for e in grid_errors["refined"]) < 0.01
    assert abs(grid_errors["uniform_60s"][0]) > 10.0 * abs(grid_errors["refined"][0])
    assert abs(closure) < 1e-3


def test_mli_temperature_transients():
    """Mass-averaged and structure-side insulation temperatures vs. node count."""
    node_counts = (1, 10, 20, 50, 100, 150)
    results = {n: simulate(n) for n in node_counts}
    reference = results[150]

    print("\nMLI mass-averaged insulation temperature [K], step start")
    print(f"{'N':>4} " + " ".join(f"{f'{t / 60:g} min':>10}" for t in (60.0, 600.0, 3600.0, 10800.0)))
    for n, result in results.items():
        mean = result.mean_temperature()
        print(f"{n:>4} " + " ".join(f"{np.interp(t, result.times, mean):>10.3f}" for t in (60.0, 600.0, 3600.0, 10800.0)))

    print("\nStructure-side node temperature [K]")
    for n, result in results.items():
        print(f"{n:>4} " + " ".join(f"{np.interp(t, result.times, result.temperatures[-1]):>10.3f}" for t in (60.0, 600.0, 3600.0, 10800.0)))

    # Insulation must cool monotonically from the warm start.
    for n, result in results.items():
        assert np.all(np.diff(result.mean_temperature()) <= 1e-9), n

    for t in (3600.0, 10800.0):
        mean_ref = np.interp(t, reference.times, reference.mean_temperature())
        deviations = {
            n: abs(np.interp(t, results[n].times, results[n].mean_temperature()) - mean_ref)
            for n in (1, 10, 20, 50)
        }
        assert deviations[1] > deviations[10] > deviations[20] > deviations[50]


def test_mli_steady_start_perturbation_is_insensitive_to_node_count():
    """Production-like transient: steady start, shell boundary steps 288 -> 250 K."""
    from test.rc_circuit_validation.mli_rc_studies import T_SHELL_PERTURBED

    node_counts = (1, 5, 10, 20, 50, 100)
    results = {
        n: simulate(n, initial="steady", shell_temperature=T_SHELL_PERTURBED)
        for n in node_counts
    }
    reference = results[100]

    print("\nMLI steady-start, shell 288.15 -> 250 K, cumulative heat deviation from N=100 [%]")
    print(f"{'N':>4} {'q(0) [W]':>10} " + " ".join(f"{f'{t / 60:g} min':>10}" for t in SAMPLE_TIMES))
    for n, result in results.items():
        print(
            f"{n:>4} {result.heat_flow[0]:>10.3f} "
            + " ".join(
                f"{_percent_error(result.cumulative_at(t), reference.cumulative_at(t)):>10.3f}"
                for t in SAMPLE_TIMES
            )
        )

    for n, result in results.items():
        for t in SAMPLE_TIMES:
            assert abs(
                _percent_error(result.cumulative_at(t), reference.cumulative_at(t))
            ) < 1.0, (n, t)
