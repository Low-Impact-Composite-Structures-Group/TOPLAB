"""Plots for optimisation result CSVs (SQP and NSGA-II share the same column layout)."""

from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path
from typing import Sequence

from toplab.optimization.evaluation import OBJECTIVES

_VARIABLE = re.compile(r"(radius|phi|insulation_thickness)_\d+")
_LABELS = {"eta_g": "eta_g [-]", "eta_v": "eta_v [-]", "eta_vent": "eta_vent [-]", "zeta_p": "zeta_p [-]"}


def read_rows(path: Path | str) -> list[dict]:
    """Read a result CSV, converting numeric and boolean columns."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def pareto_from_history(history: Sequence[dict], objectives: Sequence[str]) -> list[dict]:
    """Non-dominated rows (all objectives maximised) of an evaluation history.

    Used when a run was interrupted and no Pareto CSV exists. Duplicate designs (same variable
    values) are collapsed and failed evaluations dropped. As in the NSGA-II result, only feasible
    designs are considered when at least one exists.
    """
    variables = variable_columns(history)
    unique: dict[tuple, dict] = {}
    for row in history:
        if not row.get("error"):
            unique.setdefault(tuple(row[v] for v in variables), row)
    rows = list(unique.values())
    rows = [r for r in rows if r.get("feasible") is True] or rows
    values = [tuple(r[o] for o in objectives) for r in rows]

    def dominated(i: int) -> bool:
        return any(all(a >= b for a, b in zip(values[j], values[i])) and values[j] != values[i]
                   for j in range(len(rows)) if j != i)

    return [r for i, r in enumerate(rows) if not dominated(i)]


def variable_columns(rows: Sequence[dict]) -> list[str]:
    return [key for key in (rows[0] if rows else {}) if _VARIABLE.fullmatch(key)]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_objective_vs_variables(evaluations_csv: Path | str, output_dir: Path | str) -> list[Path]:
    """One scatter per (objective, design variable) over every evaluation, feasible vs infeasible."""
    plt = _plt()
    rows = read_rows(evaluations_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for objective in dict.fromkeys(r["objective"] for r in rows):
        subset = [r for r in rows if r["objective"] == objective and not r["error"]]
        for name in variable_columns(subset):
            fig, ax = plt.subplots(figsize=(7, 5))
            for flag, colour, label in ((False, "tab:red", "infeasible"), (True, "tab:blue", "feasible")):
                points = [r for r in subset if r["feasible"] is flag]
                ax.scatter([r[name] for r in points], [r[objective] for r in points], c=colour,
                           alpha=0.35 if not flag else 0.9, s=22 if not flag else 40, label=label)
            ax.set_xlabel(name)
            ax.set_ylabel(_LABELS[objective])
            ax.set_title(f"{objective} vs {name}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = output_dir / f"{objective}_vs_{name}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)
    return written


def plot_pareto(pareto: Sequence[dict], history: Sequence[dict], objectives: Sequence[str],
                output_dir: Path | str, prefix: str) -> list[Path]:
    """Pairwise objective-space scatter of the Pareto rows over the feasible history rows."""
    plt = _plt()
    objectives = [o for o in objectives if o in OBJECTIVES]
    if len(objectives) < 2:
        return []
    history = [r for r in history if r["feasible"] is True]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, x_name in enumerate(objectives):
        for y_name in objectives[i + 1:]:
            fig, ax = plt.subplots(figsize=(7, 5))
            if history:
                ax.scatter([r[x_name] for r in history], [r[y_name] for r in history],
                           c="lightgray", s=22, label="feasible evaluations")
            ax.scatter([r[x_name] for r in pareto], [r[y_name] for r in pareto],
                       c="tab:red", s=55, edgecolor="black", label="Pareto set")
            ax.set_xlabel(_LABELS[x_name])
            ax.set_ylabel(_LABELS[y_name])
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = output_dir / f"{prefix}_pareto_{x_name}_{y_name}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)
    return written


def plot_pareto_3d(pareto: Sequence[dict], history: Sequence[dict], objectives: Sequence[str],
                   output_dir: Path | str, prefix: str) -> list[Path]:
    """Interactive 3D objective-space HTML for every 3-subset of ``objectives``.

    Feasible history is grey, the Pareto set is coloured by the objective left out of the
    triple (when there is one). Hover shows the design variables and all objectives.
    """
    import plotly.graph_objects as go

    objectives = [o for o in objectives if o in OBJECTIVES]
    if len(objectives) < 3:
        return []
    history = [r for r in history if r["feasible"] is True]
    variables = variable_columns(pareto or history)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def hover(row: dict) -> str:
        parts = [f"{v} = {row[v]:.5g}" for v in variables] + [f"{o} = {row[o]:.5g}" for o in objectives]
        return "<br>".join(parts)

    written = []
    for triple in itertools.combinations(objectives, 3):
        fig = go.Figure()
        if history:
            fig.add_trace(go.Scatter3d(
                x=[r[triple[0]] for r in history], y=[r[triple[1]] for r in history],
                z=[r[triple[2]] for r in history], mode="markers", name="feasible evaluations",
                text=[hover(r) for r in history], hovertemplate="%{text}<extra>feasible</extra>",
                marker=dict(size=3, color="lightgray", opacity=0.5)))
        left_out = [o for o in objectives if o not in triple]
        marker = dict(size=6, line=dict(width=1, color="black"))
        if left_out:
            marker.update(color=[r[left_out[0]] for r in pareto], colorscale="Turbo", showscale=True,
                          colorbar=dict(title=left_out[0]))
        else:
            marker.update(color="crimson")
        fig.add_trace(go.Scatter3d(
            x=[r[triple[0]] for r in pareto], y=[r[triple[1]] for r in pareto],
            z=[r[triple[2]] for r in pareto], mode="markers", name="Pareto set",
            text=[hover(r) for r in pareto], hovertemplate="%{text}<extra>Pareto</extra>",
            marker=marker))
        fig.update_layout(
            title=f"Pareto set: {', '.join(triple)}", template="plotly_white", height=750,
            scene=dict(xaxis_title=_LABELS[triple[0]], yaxis_title=_LABELS[triple[1]],
                       zaxis_title=_LABELS[triple[2]]),
            legend=dict(orientation="h", y=1.02))
        path = output_dir / f"{prefix}_pareto3d_{'_'.join(triple)}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        written.append(path)
    return written
