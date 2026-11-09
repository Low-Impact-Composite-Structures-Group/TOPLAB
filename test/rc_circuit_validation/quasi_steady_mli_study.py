"""Quasi-steady MLI study: Models A (one-node transient), B (two half-blankets), C (full blanket).

Writes ``quasi_steady_results.json`` and figures to ``test/rc_circuit_validation/figures/``.
Run from ``TOPLAB/``: ``python -m test.rc_circuit_validation.quasi_steady_mli_study`` (or run the file directly).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from mli_dormancy_harness import DormancyHarness, run_production  # noqa: E402

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
LIMIT_FRACTION = 1e-3  # 0.1 % of the initial hydrogen mass
THICKNESS_SWEEP_MM = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
SIZING_INCREMENT_MM = 0.5


def summarise(result, kind):
    if kind == "A":
        mass = result.y[0]
        t_ins = result.y[3]
    else:
        mass = result.y[0]
        t_ins = None
    vented = mass[0] - mass
    onset_idx = np.argmax(vented > 1e-6) if np.any(vented > 1e-6) else None
    return dict(
        vented_kg=float(vented[-1]),
        vented_fraction=float(vented[-1] / mass[0]),
        onset_h=float(result.times[onset_idx] / 3600.0) if onset_idx is not None else None,
        final_pressure_mpa=float(result.pressure[-1] / 1e6),
        final_t_h2_k=float(result.y[1, -1]),
        final_t_struct_k=float(result.y[2, -1]),
        cumulative_h2_heat_kj=float(result.cumulative_h2_heat_kj[-1]),
        cpu_s=float(result.cpu_seconds),
        final_t_ins_k=None if t_ins is None else float(t_ins[-1]),
    )


def run_models(thickness=None, initial="equilibrated", duration=None):
    kwargs = {}
    if thickness is not None:
        kwargs["thickness"] = thickness
    if duration is not None:
        kwargs["duration"] = duration
    harness = DormancyHarness(**kwargs)
    runs = {
        "A": harness.run(1, initial, output_step=300.0),
        "B": harness.run_quasi_steady("B", output_step=300.0),
        "C": harness.run_quasi_steady("C", output_step=300.0),
    }
    return harness, runs


def vented_fraction_c_or_a(thickness, model):
    harness = DormancyHarness(thickness=thickness)
    result = (
        harness.run(1, "equilibrated") if model == "A" else harness.run_quasi_steady("C")
    )
    return float((result.y[0, 0] - result.y[0, -1]) / result.y[0, 0])


def bisect_minimum_thickness(model, low=0.001, high=0.060):
    if vented_fraction_c_or_a(high, model) > LIMIT_FRACTION:
        return None
    while (high - low) * 1000.0 > SIZING_INCREMENT_MM / 5.0:
        mid = 0.5 * (low + high)
        if vented_fraction_c_or_a(mid, model) <= LIMIT_FRACTION:
            high = mid
        else:
            low = mid
    return high * 1000.0


def main():
    results = {"limit_fraction": LIMIT_FRACTION}

    harness, equilibrated = run_models()
    results["F1_equilibrated"] = {k: summarise(v, k) for k, v in equilibrated.items()}
    results["F1_production"] = {}
    for mode in ("transient_lumped", "quasi_steady"):
        run = run_production(mode)
        results["F1_production"][mode] = dict(
            vented_kg=float(run["vented"][-1]),
            final_pressure_mpa=float(run["pressure"][-1] / 1e6),
            final_t_h2_k=float(run["h2_temperature"][-1]),
            cpu_s=float(run["cpu_seconds"]),
        )

    _, post_refuel = run_models(initial="post_refuel")
    results["F2_post_refuel"] = {k: summarise(v, k) for k, v in post_refuel.items()}

    sweep = {"thickness_mm": THICKNESS_SWEEP_MM, "A": [], "C": []}
    for t_mm in THICKNESS_SWEEP_MM:
        for model in ("A", "C"):
            sweep[model].append(vented_fraction_c_or_a(t_mm / 1000.0, model))
    results["F3_sweep_vented_fraction"] = sweep
    results["F3_minimum_thickness_mm"] = {
        model: bisect_minimum_thickness(model) for model in ("A", "C")
    }

    (OUT / "quasi_steady_results.json").write_text(json.dumps(results, indent=2))
    make_figures(harness, equilibrated, post_refuel, sweep, results)
    print(json.dumps(results["F3_minimum_thickness_mm"]))


def make_figures(harness, equilibrated, post_refuel, sweep, results):
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    labels = {
        "A": "A: one-node transient (ρ=300, cp=1000)",
        "B": "B: quasi-steady, two half-blankets",
        "C": "C: quasi-steady, full blanket",
    }
    mass0 = equilibrated["A"].y[0, 0]
    for tag, runs in (("equilibrated", equilibrated), ("post_refuel", post_refuel)):
        title = f"slh2_baseline dormancy, 12 h, {tag.replace('_', ' ')} start"
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, run in runs.items():
            ax.plot(run.times / 3600, run.heat_to_structure, color=colors[key], label=labels[key])
        ax.set(xlabel="Time [h]", ylabel="Heat into structure [W]", title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"qs_heat_ingress_{tag}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for key, run in runs.items():
            ax.plot(run.times / 3600, run.pressure / 1e6, color=colors[key], label=labels[key])
        ax.axhline(2.0, color="k", ls="--", label="Vent pressure 2.0 MPa")
        ax.set(xlabel="Time [h]", ylabel="Tank pressure [MPa]", title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"qs_pressure_{tag}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for key, run in runs.items():
            ax.plot(run.times / 3600, run.y[0, 0] - run.y[0], color=colors[key], label=labels[key])
        ax.axhline(LIMIT_FRACTION * mass0, color="k", ls="--", label="0.1 % of initial H₂ mass")
        ax.set(xlabel="Time [h]", ylabel="Cumulative vented mass [kg]", title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"qs_vented_mass_{tag}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["thickness_mm"], 100 * np.array(sweep["A"]), "o-", color=colors["A"], label="A: transient one-node")
    ax.plot(sweep["thickness_mm"], 100 * np.array(sweep["C"]), "s-", color=colors["C"], label="C: quasi-steady")
    ax.axhline(100 * LIMIT_FRACTION, color="k", ls="--", label="0.1 % limit")
    ax.set(
        xlabel="MLI thickness [mm]",
        ylabel="Vented mass [% of initial H₂]",
        title="slh2_baseline, 12 h dormancy, equilibrated start",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "qs_thickness_vs_vented.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
