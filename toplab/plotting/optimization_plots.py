"""Plots for optimisation result CSVs (SQP and NSGA-II share the same column layout)."""

from __future__ import annotations

import csv
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
