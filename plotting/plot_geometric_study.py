from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fluids.hydrogen_retrievers import HydrogenRetriever
from src.mission.mission import Mission


def plot_geometric_study(
    directory: str,
    levels: list[float]
) -> plt.Figure:

    path = Path.cwd() / "data" / "results" / directory

    plot_max_pressures(path)

    fig, ax = plt.subplots()

    plot_gravimetric_data(levels, path, fig, ax)

    volumetric_lines = plot_volumetric_data(path, ax)

    volume_lines = plot_reference_volumes(path, ax)

    ax.set_xlabel("Tank body length [m]")
    ax.set_ylabel("Tank radius [m]")
    lines = [
        volumetric_lines.legend_elements()[0][0],
        volume_lines.legend_elements()[0][0]
    ]
    labels = ["Vol. eff. [-]", "Ref. Vol."]
    plt.legend(
        lines,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        fancybox=True,
        ncol=5
    )
    fig.tight_layout()

    return fig


def load_data(
    path: Path, filename: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(
        path / f"{filename}.csv", header=None
    )
    x = data.iloc[0, 1:]
    y = data.iloc[1:, 0]
    values = data.iloc[1:, 1:]
    return x, y, values


def plot_gravimetric_data(
    levels: list[float],
    path: Path,
    fig: plt.Figure,
    ax: plt.Axes
) -> plt.QuadContourSet:
    x, y, values = load_data(path, "gravimetric_efficiencies")
    graf_effs = ax.contourf(x, y, values, levels=levels)
    cbar = fig.colorbar(graf_effs)
    cbar.set_label("Gravimetric efficiency [-]")
    cbar.set_ticks(levels)

    return graf_effs


def plot_volumetric_data(
    path: Path,
    ax: plt.Axes
) -> plt.QuadContourSet:

    x, y, values = load_data(path, "volumetric_efficiencies")
    volumetric_lines = ax.contour(
        x, y, values, 10,
        colors="black", linestyles="dashed"
    )
    ax.clabel(volumetric_lines, inline=True)
    
    return volumetric_lines


def plot_reference_volumes(
    path: Path,
    ax: plt.Axes
) -> plt.QuadContourSet:

    x, y, values = load_data(path, "tank_volumes")
    
    # Define the mission
    fuel_phase_flow = "liquid"
    missions = [
        Mission.regional(fuel_phase_flow),
        Mission.small_medium_range(fuel_phase_flow),
        Mission.large_passenger_aircraft(fuel_phase_flow)
    ]
    labels = ["REG", "SMR", "LPA"]
    fuel_masses = np.array([
        mission.required_fuel
        for mission in missions
    ])

    # Define the initial conditions of the fuel tank
    pressure = 150e3
    temperature = None
    hydrogen = HydrogenRetriever().get_hydrogen_properties(
        pressure, temperature
    )
    fuel_volumes = fuel_masses / hydrogen.liquid.density
    for fuel_volume, label in zip(fuel_volumes, labels):

        volumes = values / fuel_volume

        volume_lines = ax.contour(
            x, y, volumes, levels=[1], colors="orange"
        )
        try:
            ax.clabel(volume_lines, inline=1, fmt=label)
        except TypeError:
            print(
                f"{label} has no data in the range, thus is not plotted"
            )
    
    return volume_lines


def plot_max_pressures(
    path: Path
) -> plt.Figure:

    fig, ax = plt.subplots()

    x, y, values = load_data(path, "max_pressures")
    values *= 1e-6

    contours = ax.contourf(x, y, values)
    cbar = fig.colorbar(contours)
    cbar.set_label("Max. Pressure [MPa]")
    ax.set_xlabel("Tank body length [m]")
    ax.set_ylabel("Tank radius [m]")

    return fig


def main():
    pass


if __name__ == "__main__":
    main()


# End
