#!/usr/bin/env python3
"""
Quick visualisation of pressure-buffer sensitivity study results.

Produces a single figure saved to the output directory (and shown interactively
if a display is available).  Run from any directory:

    python optimization/presure_buffer_opt/plot_sensitivity_results.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

# import src.plotting.plot_style_sb as plot_style


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).parent / "output"
SUMMARY_CSV = OUTPUT_DIR / "pressure_buffer_sensitivity_summary.csv"
FIGURE_PATH = OUTPUT_DIR / "pressure_buffer_sensitivity_results.png"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_summary(path: Path) -> dict[str, dict[str, list]]:
    """Return {objective: {column: [values]}}."""
    data: dict[str, dict[str, list]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            obj = row["objective"]
            if obj not in data:
                data[obj] = defaultdict(list)
            for k, v in row.items():
                data[obj][k].append(v)
    return data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _floats(series: list[str]) -> np.ndarray:
    return np.array([float(v) if v != "" else float("nan") for v in series])

def _ints(series: list[str]) -> np.ndarray:
    return np.array([int(v) for v in series])

OBJECTIVE_LABELS = {
    "gravimetric_efficiency":    ("Gravimetric efficiency [−]",  "Maximize",   "#333333"),
    "volumetric_efficiency":     ("Volumetric efficiency [−]",   "Maximize",   "#777777"),
    "vent_time_after_mission_s": ("Time to vent after mission [h]", "Maximize",   "#aaaaaa"),
}

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
def plot(data: dict[str, dict[str, list]]) -> plt.Figure:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    objective_order = [
        "gravimetric_efficiency",
        "vent_time_after_mission_s",
        "volumetric_efficiency",
    ]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.38,
        left=0.07, right=0.97,
        top=0.91, bottom=0.08,
    )

    # ------------------------------------------------------------------
    # Row 0 — objective value vs iteration (one panel per objective)
    # ------------------------------------------------------------------
    ax_obj: list[plt.Axes] = []
    for col_idx, obj_name in enumerate(objective_order):
        ax = fig.add_subplot(gs[0, col_idx])
        ax_obj.append(ax)

        if obj_name not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        d = data[obj_name]
        iters = _ints(d["iteration"])
        obj_vals = _floats(d["objective_value"])
        label_str, sense_str, colour = OBJECTIVE_LABELS[obj_name]

        # Convert vent time to hours for readability
        display_vals = obj_vals / 3600.0 if obj_name == "vent_time_after_mission_s" else obj_vals

        ax.plot(iters, display_vals, marker="o", color=colour, linewidth=2,
                markersize=6, zorder=3)
        ax.fill_between(iters, display_vals, display_vals.min() - 0.001,
                        alpha=0.08, color=colour)

        # Annotate start and end
        ax.annotate(f"{display_vals[0]:.4f}",
                    xy=(iters[0], display_vals[0]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="#555555")
        if not np.all(np.isnan(display_vals)):
            ax.annotate(f"{display_vals[-1]:.4f}",
                        xy=(iters[-1], display_vals[-1]),
                        xytext=(8, -12), textcoords="offset points",
                        fontsize=8, color=colour)

        # Flag flat vent-time
        if obj_name == "vent_time_after_mission_s" and np.all(display_vals == display_vals[0]):
            ax.text(0.5, 0.5, "⚠ Zero sensitivity\n(geometry insensitive\nto vent time here)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#cc4444",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc4444", alpha=0.8))

        ax.set_xlabel("Iteration")
        ax.set_ylabel(label_str)
        ax.set_title(f"{sense_str} {label_str.split('[')[0].strip()}", fontsize=10, fontweight="bold")
        ax.set_xticks(iters)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Delta annotation
        delta = display_vals[-1] - display_vals[0]
        sign = "+" if delta >= 0 else ""
        ax.text(0.97, 0.06, f"Δ = {sign}{delta:.4f}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=8, color=colour,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, alpha=0.7))

    # ------------------------------------------------------------------
    # Row 1, left — design-space trajectory (radius vs length scale)
    # ------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[1, 0])
    markers = {"gravimetric_efficiency": "s", "volumetric_efficiency": "^", "vent_time_after_mission_s": "o"}

    for obj_name in objective_order:
        if obj_name not in data:
            continue
        d = data[obj_name]
        rs = _floats(d["radius_scale"])
        ls = _floats(d["length_scale"])
        iters = _ints(d["iteration"])
        label, _, colour = OBJECTIVE_LABELS[obj_name]

        ax_traj.plot(rs, ls, color=colour, linewidth=1.5, alpha=0.6)
        sc = ax_traj.scatter(rs, ls, c=iters, cmap="gray_r", marker=markers[obj_name],
                             s=60, zorder=4, label=label.split("[")[0].strip(),
                             edgecolors=colour, linewidths=1.2, vmin=0, vmax=4)

    ax_traj.set_xlabel("Radius scale [−]")
    ax_traj.set_ylabel("Length scale [−]")
    ax_traj.set_title("Design-space trajectory", fontsize=10, fontweight="bold")
    ax_traj.legend(fontsize=7, framealpha=0.8)
    ax_traj.axvline(1.0, color="#aaaaaa", linestyle=":", linewidth=1)
    ax_traj.axhline(1.0, color="#aaaaaa", linestyle=":", linewidth=1)
    ax_traj.text(1.002, 1.002, "baseline", fontsize=7, color="#888888")
    ax_traj.grid(linestyle=":", alpha=0.3)

    # Add colourbar for iteration
    cbar = plt.colorbar(sc, ax=ax_traj, pad=0.04)
    cbar.set_label("Iteration", fontsize=8)
    cbar.set_ticks([0, 1, 2, 3, 4])

    # ------------------------------------------------------------------
    # Row 1, middle — geometry at best point vs baseline (bar chart)
    # ------------------------------------------------------------------
    ax_geom = fig.add_subplot(gs[1, 1])

    best_rows: dict[str, dict] = {}
    for obj_name in objective_order:
        if obj_name not in data:
            continue
        d = data[obj_name]
        scores = _floats(d["objective_score"])
        best_idx = int(np.argmax(scores))
        best_rows[obj_name] = {k: v[best_idx] for k, v in d.items()}

    param_labels = ["Tank 1\nradius [m]", "Tank 1\nlength [m]", "Tank 2\nradius [m]", "Tank 2\nlength [m]"]
    param_keys   = ["tank_1_radius_m", "tank_1_length_m", "tank_2_radius_m", "tank_2_length_m"]

    x = np.arange(len(param_labels))
    width = 0.22
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    bar_labels = ["Baseline"] + [OBJECTIVE_LABELS[o][0].split("[")[0].strip() for o in objective_order]
    bar_colours = ["#dddddd"] + [OBJECTIVE_LABELS[o][2] for o in objective_order]

    # Baseline values from first row of any objective
    baseline_vals = []
    any_d = next(iter(data.values()))
    for pk in param_keys:
        baseline_vals.append(float(any_d[pk][0]))

    all_series = [baseline_vals] + [
        [float(best_rows[o][pk]) for pk in param_keys]
        for o in objective_order if o in best_rows
    ]

    for i, (series, lbl, col) in enumerate(zip(all_series, bar_labels, bar_colours)):
        ax_geom.bar(x + offsets[i], series, width, label=lbl, color=col,
                    edgecolor="black", linewidth=0.5)

    ax_geom.set_xticks(x)
    ax_geom.set_xticklabels(param_labels, fontsize=8)
    ax_geom.set_ylabel("Value [m]")
    ax_geom.set_title("Best geometry vs baseline", fontsize=10, fontweight="bold")
    ax_geom.legend(fontsize=7, framealpha=0.8)
    ax_geom.grid(axis="y", linestyle=":", alpha=0.4)

    # ------------------------------------------------------------------
    # Row 1, right — objective improvement summary (horizontal bars)
    # ------------------------------------------------------------------
    ax_sum = fig.add_subplot(gs[1, 2])

    summary_labels, pct_improvements, colours = [], [], []
    for obj_name in objective_order:
        if obj_name not in data:
            continue
        d = data[obj_name]
        obj_vals = _floats(d["objective_value"])
        sense = OBJECTIVE_DEFS_SENSE[obj_name]
        baseline_v = obj_vals[0]
        best_v = obj_vals[np.argmax(_floats(d["objective_score"]))]
        if baseline_v != 0:
            pct = (best_v - baseline_v) / abs(baseline_v) * 100.0
        else:
            pct = 0.0
        if sense == "min":
            pct = -pct  # flip so positive = improvement for min objectives
        label, _, colour = OBJECTIVE_LABELS[obj_name]
        summary_labels.append(label.split("[")[0].strip())
        pct_improvements.append(pct)
        colours.append(colour)

    y_pos = np.arange(len(summary_labels))
    ax_sum.barh(y_pos, pct_improvements, color=colours,
                edgecolor="black", linewidth=0.5)

    for i, (pct, lbl) in enumerate(zip(pct_improvements, summary_labels)):
        sign = "+" if pct >= 0 else ""
        ax_sum.text(pct + 0.005, i, f"{sign}{pct:.3f}%",
                    va="center", ha="left" if pct >= 0 else "right",
                    fontsize=9, color="#333333")

    ax_sum.axvline(0, color="black", linewidth=0.8)
    ax_sum.set_yticks(y_pos)
    ax_sum.set_yticklabels(summary_labels, fontsize=9)
    ax_sum.set_xlabel("Improvement vs baseline [%]")
    ax_sum.set_title("Overall improvement\n(4 steps from baseline)", fontsize=10, fontweight="bold")
    ax_sum.grid(axis="x", linestyle=":", alpha=0.4)

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------
    fig.suptitle(
        "Pressure Buffer System — Sensitivity-Step Optimisation (First Results)\n"
        "Base: CH2 (80 bar) + CCH2 (40 bar) · Valve 16/30 bar · Vent = 1.5 × P_init",
        fontsize=11, fontweight="bold", y=0.98,
    )

    return fig


# ---------------------------------------------------------------------------
# Objective sense table (needed by plot())
# ---------------------------------------------------------------------------
OBJECTIVE_DEFS_SENSE = {
    "gravimetric_efficiency":    "max",
    "volumetric_efficiency":     "max",
    "vent_time_after_mission_s": "max",
}


# ---------------------------------------------------------------------------
# Print quick text summary
# ---------------------------------------------------------------------------
def print_summary(data: dict[str, dict[str, list]]) -> None:
    print("\n" + "=" * 65)
    print("  SENSITIVITY STUDY — QUICK INTERPRETATION")
    print("=" * 65)

    for obj_name in ["gravimetric_efficiency", "vent_time_after_mission_s", "volumetric_efficiency"]:
        if obj_name not in data:
            continue
        d = data[obj_name]
        label, sense, _ = OBJECTIVE_LABELS[obj_name]
        obj_vals = _floats(d["objective_value"])
        scores   = _floats(d["objective_score"])
        best_idx = int(np.argmax(scores))

        baseline_v = obj_vals[0]
        best_v     = obj_vals[best_idx]
        display_fn = (lambda v: v / 3600.0) if obj_name == "vent_time_after_mission_s" else (lambda v: v)
        unit       = "h" if obj_name == "vent_time_after_mission_s" else "−"

        print(f"\n  [{sense.upper()}] {label}")
        print(f"    Baseline  : {display_fn(baseline_v):.5f} {unit}")
        print(f"    Best      : {display_fn(best_v):.5f} {unit}  (iter {best_idx})")
        delta = best_v - baseline_v
        sign  = "+" if delta >= 0 else ""
        print(f"    Change    : {sign}{display_fn(delta):.5f} {unit} "
              f"({sign}{delta/abs(baseline_v)*100:.3f}%)")

        best = {k: v[best_idx] for k, v in d.items()}
        print(f"    Best design: radius_scale={best['radius_scale']}, "
              f"length_scale={best['length_scale']}")
        print(f"      Tank1 r={float(best['tank_1_radius_m']):.4f}m, "
              f"L={float(best['tank_1_length_m']):.4f}m, "
              f"phi={float(best['tank_1_phi']):.4f}")
        print(f"      Tank2 r={float(best['tank_2_radius_m']):.4f}m, "
              f"L={float(best['tank_2_length_m']):.4f}m, "
              f"phi={float(best['tank_2_phi']):.4f}")

        if obj_name == "vent_time_after_mission_s" and np.all(obj_vals == obj_vals[0]):
            print("    *** FLAT: zero sensitivity detected. All perturbations "
                  "produced the same vent time.")
            print("        Vent onset is likely insulation-dominated, not geometry-dominated.")
            print("        To improve: sweep insulation thickness or HTC as design variables.")

    print("\n" + "=" * 65)
    print("  INTERPRETATION NOTES")
    print("=" * 65)
    print("""
  Gravimetric (maximize):
    Best direction → smaller radius + longer tank (↑ phi).
    Each step: r_scale −0.01, L_scale +0.02 → phi increases by ~0.09/step.
    Still completing mission at every iteration — no feasibility risk yet.
    Trend: more elongated = lower structural wall mass fraction.
    Expected end state if continued: approaches a long cylinder (phi → ∞),
    not a sphere.  The sphere result expected for pure structural optimisation
    requires a different objective (minimise wall mass / pressure volume).

  Vent time (minimize):
    Design variables: geometry (radius, length) + insulation thickness.
    Baseline insulation: 50 mm G10 (HTC = 0.01 W/m²K).
    Geometry showed zero sensitivity; insulation thickness is the dominant driver.
    Smaller insulation → faster heat ingress → earlier vent → lower vent time.
    Step direction will reduce insulation scale until the lower bound or a
    competing constraint (mission completion, structural limits) is reached.

  Volumetric (maximize):
    Best direction → larger radius + longer tank.
    Each step: r_scale +0.01, L_scale +0.02.
    Improvement is real but shallow (+0.84% over 4 steps).
    Inner/outer volume ratio improves as the tank grows because wall thickness
    is a smaller fraction of total radius.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not SUMMARY_CSV.exists():
        sys.exit(f"ERROR: {SUMMARY_CSV} not found. Run the sensitivity study first.")

    data = load_summary(SUMMARY_CSV)

    print_summary(data)

    fig = plot(data)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {FIGURE_PATH}")

    try:
        plt.show()
    except Exception:
        pass  # headless environment


if __name__ == "__main__":
    main()
