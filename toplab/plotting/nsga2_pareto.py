"""Interactive visualizations for NSGA-II optimization results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in (
            "radius",
            "phi",
            "insulation_thickness",
            "eta_g",
            "eta_v",
            "f_vent",
        ):
            if key in row and row[key] != "":
                row[key] = float(row[key])
        if "feasible" in row:
            row["feasible"] = str(row["feasible"]).lower() == "true"
    return rows


def _hover_text(row: dict[str, Any]) -> str:
    return (
        f"r = {row['radius']:.4f} m<br>"
        f"phi = {row['phi']:.4f}<br>"
        f"insulation = {row['insulation_thickness']:.4f} m<br>"
        f"eta_g = {row['eta_g']:.4f}<br>"
        f"eta_v = {row['eta_v']:.4f}<br>"
        f"f_vent = {row['f_vent']:.4f}%"
    )


def create_pareto_front_figure(history_path: str | Path, pareto_path: str | Path):
    """Create a coordinated 3D and pairwise Pareto-front figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    history = _read_rows(Path(history_path))
    pareto = _read_rows(Path(pareto_path))
    feasible = [row for row in history if row.get("feasible", False)]
    infeasible = [row for row in history if not row.get("feasible", False)]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "3D objective space",
            "Gravimetric vs volumetric efficiency",
            "Gravimetric efficiency vs venting",
            "Volumetric efficiency vs venting",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    def add_points(rows: list[dict[str, Any]], name: str, color: str, size: int, opacity: float, row: int, col: int) -> None:
        if not rows:
            return
        hover = [_hover_text(item) for item in rows]
        marker = dict(color=color, size=size, opacity=opacity)
        if row == 1 and col == 1:
            trace = go.Scatter3d(
                x=[item["eta_g"] for item in rows],
                y=[item["eta_v"] for item in rows],
                z=[item["f_vent"] for item in rows],
                mode="markers",
                name=name,
                text=hover,
                hovertemplate="%{text}<extra>" + name + "</extra>",
                marker=marker,
            )
        else:
            axes = {
                (1, 2): ("eta_g", "eta_v"),
                (2, 1): ("eta_g", "f_vent"),
                (2, 2): ("eta_v", "f_vent"),
            }
            x_key, y_key = axes[(row, col)]
            trace = go.Scatter(
                x=[item[x_key] for item in rows],
                y=[item[y_key] for item in rows],
                mode="markers",
                name=name,
                text=hover,
                hovertemplate="%{text}<extra>" + name + "</extra>",
                marker=marker,
                showlegend=(row == 1 and col == 2),
            )
        fig.add_trace(trace, row=row, col=col)

    for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
        add_points(infeasible, "Infeasible evaluations", "#9aa0a6", 5, 0.35, row, col)
        add_points(feasible, "Feasible evaluations", "#0076C2", 6, 0.65, row, col)
        add_points(pareto, "Pareto front", "#E03C31", 10, 1.0, row, col)

    fig.update_layout(
        title="NSGA-II Pareto Front",
        template="plotly_white",
        height=900,
        margin=dict(l=30, r=30, t=80, b=30),
        legend=dict(orientation="h", y=1.03, x=0.0),
    )
    fig.update_scenes(
        xaxis_title="Gravimetric efficiency eta_g",
        yaxis_title="Volumetric efficiency eta_v",
        zaxis_title="Vented hydrogen [%]",
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="eta_g", row=1, col=2)
    fig.update_yaxes(title_text="eta_v", row=1, col=2)
    fig.update_xaxes(title_text="eta_g", row=2, col=1)
    fig.update_yaxes(title_text="f_vent [%]", row=2, col=1)
    fig.update_xaxes(title_text="eta_v", row=2, col=2)
    fig.update_yaxes(title_text="f_vent [%]", row=2, col=2)
    return fig


def create_design_parallel_coordinates(pareto_path: str | Path):
    """Create a parallel-coordinates view of Pareto designs and objectives."""
    import plotly.graph_objects as go

    rows = _read_rows(Path(pareto_path))
    dimensions = [
        {"label": "Radius [m]", "values": [row["radius"] for row in rows]},
        {"label": "Phi [-]", "values": [row["phi"] for row in rows]},
        {"label": "Insulation [m]", "values": [row["insulation_thickness"] for row in rows]},
        {"label": "Eta_g", "values": [row["eta_g"] for row in rows]},
        {"label": "Eta_v", "values": [row["eta_v"] for row in rows]},
        {"label": "Vented H2 [%]", "values": [row["f_vent"] for row in rows]},
    ]
    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=[row["eta_g"] for row in rows],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="eta_g"),
            ),
            dimensions=dimensions,
        )
    )
    fig.update_layout(
        title="Pareto Designs and Objectives",
        template="plotly_white",
        height=650,
        margin=dict(l=50, r=50, t=80, b=30),
    )
    return fig


def write_nsga2_plots(history_path: str | Path, pareto_path: str | Path, output_directory: str | Path) -> dict[str, Path]:
    """Write the interactive Pareto and parallel-coordinate HTML plots."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    front_path = output_directory / "pareto_front.html"
    designs_path = output_directory / "pareto_designs_parallel.html"
    create_pareto_front_figure(history_path, pareto_path).write_html(front_path, include_plotlyjs="cdn")
    create_design_parallel_coordinates(pareto_path).write_html(designs_path, include_plotlyjs="cdn")
    return {"pareto_front": front_path, "pareto_designs": designs_path}