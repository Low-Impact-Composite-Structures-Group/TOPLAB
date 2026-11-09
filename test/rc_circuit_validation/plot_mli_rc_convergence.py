"""Figures for the MLI RC validation studies (linear axes, no arbitrary reference).

Run from the TOPLAB repository root:
    python -m test.rc_circuit_validation.plot_mli_rc_convergence            # all
    python -m test.rc_circuit_validation.plot_mli_rc_convergence spatial    # subset

Studies: spatial, rho_cp, time, temperature, steady_start. Cases come from
``mli_rc_studies`` so the figures match the assertions in
``test_insulation_rc_mli.py``.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from test.rc_circuit_validation.mli_rc_studies import (
    DEFAULT_DENSITY,
    DEFAULT_SPECIFIC_HEAT,
    PRODUCTION_TOLERANCES,
    SAMPLE_TIMES,
    T_SHELL_BOUNDARY,
    T_SHELL_PERTURBED,
    TIGHT_TOLERANCES,
    UNIFORM_STEPS,
    metadata,
    simulate,
)


FIGURE_DIRECTORY = Path("test/rc_circuit_validation/figures")
SPATIAL_NODE_COUNTS = (1, 10, 20, 50, 100, 150)
DENSITIES = (60.0, DEFAULT_DENSITY, 1500.0)
DENSITY_NODE_COUNTS = (1, 5, 10, 20, 50, 100)
STEP_START = (
    f"all nodes start at {T_SHELL_BOUNDARY:g} K, "
    f"rho={DEFAULT_DENSITY:g} kg/m3, cp={DEFAULT_SPECIFIC_HEAT:g} J/kg/K"
)


def _finish(figure, name, title, extra):
    figure.suptitle(f"{title}\n{metadata(extra)}", fontsize=9)
    figure.tight_layout()
    path = FIGURE_DIRECTORY / name
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Generated {path}")


def plot_spatial():
    results = {n: simulate(n) for n in SPATIAL_NODE_COUNTS}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for n, result in results.items():
        axes[0].plot(result.times, result.heat_flow, label=f"N={n}")
        axes[1].plot(result.times, result.cumulative_kj, label=f"N={n}")
    axes[0].set_xlim(0.0, 600.0)
    axes[0].set_ylabel("Heat flow into structure [W]")
    axes[0].set_title("Instantaneous (first 10 min)")
    axes[1].set_ylabel("Cumulative heat into structure [kJ]")
    axes[1].set_title("Cumulative")
    for horizon, label in zip(SAMPLE_TIMES, ("10 min", "1 h", "3 h")):
        axes[2].plot(
            SPATIAL_NODE_COUNTS,
            [results[n].cumulative_at(horizon) for n in SPATIAL_NODE_COUNTS],
            marker="o",
            label=label,
        )
    axes[2].set_xlabel("Insulation nodes, N")
    axes[2].set_ylabel("Cumulative heat into structure [kJ]")
    axes[2].set_title("Plateau with node count")
    axes[2].legend(title="Horizon")
    for axis in axes[:2]:
        axis.set_xlabel("Time [s]")
        axis.legend(ncol=2, title="Insulation nodes")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    _finish(figure, "mli_spatial_convergence.png", "MLI step-start spatial convergence", STEP_START)


def plot_rho_cp():
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for rho in DENSITIES:
        reference = simulate(100, density=rho)
        axes[0].plot(
            reference.times, reference.cumulative_kj, label=f"rho={rho:g} kg/m3"
        )
        axes[1].plot(
            DENSITY_NODE_COUNTS,
            [
                simulate(n, density=rho).cumulative_at(3600.0)
                for n in DENSITY_NODE_COUNTS
            ],
            marker="o",
            label=f"rho={rho:g} kg/m3",
        )
        axes[2].plot(
            DENSITY_NODE_COUNTS,
            [
                simulate(n, density=rho).cumulative_at(600.0)
                for n in DENSITY_NODE_COUNTS
            ],
            marker="o",
            label=f"rho={rho:g} kg/m3",
        )
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Cumulative heat into structure [kJ]")
    axes[0].set_title("N=100 mesh")
    axes[1].set_title("Cumulative heat at 1 h vs mesh")
    axes[2].set_title("Cumulative heat at 10 min vs mesh")
    for axis in axes[1:]:
        axis.set_xlabel("Insulation nodes, N")
        axis.set_ylabel("Cumulative heat into structure [kJ]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    _finish(
        figure,
        "mli_rho_cp_sensitivity.png",
        "MLI sensitivity to effective heat capacity (rho*cp)",
        f"step start at {T_SHELL_BOUNDARY:g} K, cp={DEFAULT_SPECIFIC_HEAT:g} J/kg/K, rho varied",
    )


def plot_time_accuracy():
    node_count = 50
    reference = simulate(node_count, grid="very_fine", tolerances=TIGHT_TOLERANCES)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for grid, label in (
        ("very_fine", "very fine (reference)"),
        ("uniform_1s", "1 s"),
        ("uniform_10s", "10 s"),
        ("uniform_60s", "60 s"),
    ):
        result = simulate(node_count, grid=grid)
        mask = result.times <= 300.0
        axes[0].plot(result.times[mask], result.heat_flow[mask], label=label, marker="." if grid.startswith("uniform_60") else None)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Heat flow into structure [W]")
    axes[0].set_title(f"Early transient by output grid (N={node_count})")
    axes[0].legend(title="Output step")

    for n in (20, 50, 100):
        ref = simulate(n, grid="very_fine", tolerances=TIGHT_TOLERANCES).cumulative_at(600.0)
        values = [
            100.0 * (simulate(n, grid=f"uniform_{s:g}s").cumulative_at(600.0) - ref) / ref
            for s in UNIFORM_STEPS
        ]
        axes[1].plot(UNIFORM_STEPS, values, marker="o", label=f"N={n}")
    axes[1].set_xlabel("Uniform output step [s]")
    axes[1].set_ylabel("Cumulative heat error at 10 min [%]")
    axes[1].set_title("Output-grid (quadrature) error")
    axes[1].legend()

    labels = ("1e-6/1e-9\n(production)", "1e-9/1e-10", "1e-11/1e-12")
    for horizon, name in zip(SAMPLE_TIMES, ("10 min", "1 h", "3 h")):
        values = [
            100.0
            * (simulate(node_count, tolerances=tol).cumulative_at(horizon) - reference.cumulative_at(horizon))
            / reference.cumulative_at(horizon)
            for tol in (PRODUCTION_TOLERANCES, (1e-9, 1e-10), TIGHT_TOLERANCES)
        ]
        axes[2].plot(labels, values, marker="o", label=name)
    axes[2].set_xlabel("Solver rtol/atol")
    axes[2].set_ylabel("Cumulative heat error [%]")
    axes[2].set_title(f"Solver-tolerance error (N={node_count})")
    axes[2].legend(title="Horizon")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    _finish(
        figure,
        "mli_time_accuracy.png",
        "MLI temporal accuracy: output grid vs solver tolerance",
        STEP_START + "; reference = very fine grid, rtol 1e-11",
    )


def plot_temperatures():
    node_counts = (1, 10, 20, 50, 100, 150)
    results = {n: simulate(n) for n in node_counts}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for n, result in results.items():
        axes[0].plot(result.times, result.mean_temperature(), label=f"N={n}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Mass-averaged insulation temperature [K]")
    axes[0].set_title("Mass-averaged temperature")
    axes[0].legend(title="Insulation nodes")

    for n in (10, 50, 150):
        result = results[n]
        radius = result.node_radii
        position = (radius - radius[0]) / (radius[-1] - radius[0])
        for horizon, style in ((600.0, "--"), (3600.0, "-")):
            index = int(np.searchsorted(result.times, horizon))
            axes[1].plot(
                position,
                result.temperatures[:, index],
                linestyle=style,
                color=f"C{node_counts.index(n)}",
                label=f"N={n}, {horizon / 60:g} min",
            )
    axes[1].set_xlabel("Normalised position (0 = shell side, 1 = structure side)")
    axes[1].set_ylabel("Node temperature [K]")
    axes[1].set_title("Temperature profile")
    axes[1].legend(fontsize=8)

    for n in (10, 50, 150):
        result = results[n]
        axes[2].plot(result.times, result.temperatures[-1], label=f"N={n}")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Structure-side node temperature [K]")
    axes[2].set_title("Node nearest the structure (position shifts with N)")
    axes[2].legend(title="Insulation nodes")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    _finish(figure, "mli_temperature_transients.png", "MLI insulation temperature transients", STEP_START)


def plot_steady_start():
    node_counts = (1, 5, 10, 20, 50, 100)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for n in node_counts:
        result = simulate(n, initial="steady", shell_temperature=T_SHELL_PERTURBED)
        axes[0].plot(result.times, result.heat_flow, label=f"N={n}")
        axes[1].plot(result.times, result.cumulative_kj, label=f"N={n}")
    axes[0].set_ylabel("Heat flow into structure [W]")
    axes[1].set_ylabel("Cumulative heat into structure [kJ]")
    for axis in axes:
        axis.set_xlabel("Time [s]")
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, title="Insulation nodes")
    _finish(
        figure,
        "mli_steady_start_perturbation.png",
        "MLI near-steady transient (steady start, shell step)",
        f"steady profile at {T_SHELL_BOUNDARY:g} K shell, then shell held at {T_SHELL_PERTURBED:g} K",
    )


STUDIES = {
    "spatial": plot_spatial,
    "rho_cp": plot_rho_cp,
    "time": plot_time_accuracy,
    "temperature": plot_temperatures,
    "steady_start": plot_steady_start,
}


def main(names):
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name in names or STUDIES:
        if name not in STUDIES:
            raise SystemExit(f"Unknown study '{name}'. Choose from: {', '.join(STUDIES)}")
        STUDIES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
