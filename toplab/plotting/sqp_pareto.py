"""Interactive visualizations for SQP epsilon-constraint optimization results."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any


OBJECTIVES = ("eta_g", "eta_v", "eta_vent", "zeta_p")
DESIGN_VARIABLES = ("radius", "phi", "insulation_thickness")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read SQP CSV rows and expand the serialized design vector."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric_fields = set(OBJECTIVES) | {
        "epsilon_v", "epsilon_vent", "epsilon_p", "z_shift_star",
        "g_dis", "g_dorm", "g_packaging",
    }
    for row in rows:
        for key in numeric_fields:
            if key in row and row[key] != "":
                row[key] = float(row[key])
        if "x" in row and row["x"] != "":
            values = ast.literal_eval(row["x"])
            row["x"] = [float(value) for value in values]
            for index, name in enumerate(DESIGN_VARIABLES):
                if index < len(row["x"]):
                    row[name] = row["x"][index]
        for key in ("feasible", "success", "packaging_feasible"):
            if key in row:
                row[key] = str(row[key]).lower() == "true"
    return rows


def _hover_text(row: dict[str, Any]) -> str:
    design = "<br>".join(
        f"{name} = {row[name]:.4f}" for name in DESIGN_VARIABLES if name in row
    )
    objectives = "<br>".join(
        f"{name} = {row[name]:.4f}" for name in OBJECTIVES if name in row
    )
    return f"{design}<br>{objectives}"


def _add_objective_points(fig: Any, rows: list[dict[str, Any]], name: str,
                          color: str, size: int, opacity: float) -> None:
    if not rows:
        return
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter3d(
        x=[row["eta_g"] for row in rows],
        y=[row["eta_v"] for row in rows],
        z=[row["zeta_p"] for row in rows],
        mode="markers",
        name=name,
        text=[_hover_text(row) for row in rows],
        hovertemplate="%{text}<extra>" + name + "</extra>",
        marker=dict(color=color, size=size, opacity=opacity),
        customdata=[row["zeta_p"] for row in rows],
    ))


def create_objective_space_figure(history_path: str | Path, pareto_path: str | Path):
    """Create a 3D view of three objectives with the fourth in hover data."""
    import plotly.graph_objects as go

    history = _read_rows(Path(history_path))
    pareto = _read_rows(Path(pareto_path))
    feasible = [row for row in history if row.get("feasible", False)]
    infeasible = [row for row in history if not row.get("feasible", False)]
    fig = go.Figure()
    _add_objective_points(fig, infeasible, "Infeasible evaluations", "#9aa0a6", 5, 0.35)
    _add_objective_points(fig, feasible, "Feasible evaluations", "#0076C2", 6, 0.65)
    _add_objective_points(fig, pareto, "Pareto front", "#E03C31", 10, 1.0)
    fig.update_layout(
        title="SQP Objective Space",
        template="plotly_white",
        height=750,
        margin=dict(l=30, r=30, t=80, b=30),
        legend=dict(orientation="h", y=1.03, x=0.0),
        scene=dict(xaxis_title="eta_g", yaxis_title="eta_v", zaxis_title="zeta_p"),
    )
    return fig


def _create_parallel_coordinates(path: str | Path, fields: tuple[str, ...], title: str):
    import plotly.graph_objects as go

    rows = _read_rows(Path(path))
    dimensions = [{"label": name, "values": [row[name] for row in rows]} for name in fields]
    fig = go.Figure(go.Parcoords(
        line=dict(color=[row["eta_g"] for row in rows], colorscale="Turbo", showscale=True,
                  colorbar=dict(title="eta_g")),
        dimensions=dimensions,
    ))
    fig.update_layout(title=title, template="plotly_white", height=650,
                      margin=dict(l=50, r=50, t=80, b=30))
    return fig


def create_objective_parallel_coordinates(pareto_path: str | Path):
    """Create a parallel-coordinates view containing all four objectives."""
    return _create_parallel_coordinates(pareto_path, OBJECTIVES, "SQP Pareto Objectives")


def create_design_parallel_coordinates(pareto_path: str | Path):
    """Create a parallel-coordinates view of the three design variables."""
    return _create_parallel_coordinates(pareto_path, DESIGN_VARIABLES,
                                        "SQP Pareto Design Variables")


def write_sqp_plots(history_path: str | Path, pareto_path: str | Path,
                    output_directory: str | Path) -> dict[str, Path]:
    """Write independent HTML files for SQP objective and design views."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "objective_space": output_directory / "sqp_objective_space.html",
        "objectives_parallel": output_directory / "sqp_objectives_parallel.html",
        "designs_parallel": output_directory / "sqp_designs_parallel.html",
    }
    create_objective_space_figure(history_path, pareto_path).write_html(
        paths["objective_space"], include_plotlyjs="cdn")
    create_objective_parallel_coordinates(pareto_path).write_html(
        paths["objectives_parallel"], include_plotlyjs="cdn")
    create_design_parallel_coordinates(pareto_path).write_html(
        paths["designs_parallel"], include_plotlyjs="cdn")
    return paths


# Preserve the former figure factory name for existing callers.
create_pareto_front_figure = create_objective_space_figure

# Preserve the old name for callers that have not been updated yet.
write_nsga2_plots = write_sqp_plots
