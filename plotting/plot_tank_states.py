

from typing import Protocol

from plotting.figures import Line, SingleFigure, TwinXFigure


import numpy as np

SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5
TO_MEGA = 1e-6


class Performances(Protocol):
    volumetric_efficiency: float
    gravimetric_efficiency: float


class TankStates(Protocol):
    pressures: list[float]
    temperatures: list[float]
    timesteps_in_hours: list[float]
    pressures_in_bar: list[float]
    required_fluxes: list[float]


def plot_tank_loads(
    tank_states: list[TankStates],
    labels: list[str],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            row.timesteps_in_hours,
            row.pressures_in_bar,
            label
        )
        for row, label in zip(tank_states, labels)
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Pressure [bar]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )


def plot_required_flux(
    tank_states: list[TankStates],
    labels: list[str],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            row.timesteps_in_hours[:-1],
            np.array(row.required_fluxes) * (-TO_MEGA),
            label
        )
        for row, label in zip(tank_states, labels)
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Flux [MW]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )

def plot_tank_temperatures(
    tank_states: list[TankStates],
    labels: list[str],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            row.timesteps_in_hours,
            row.temperatures,
            label
        )
        for row, label in zip(tank_states, labels)
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Temperature [K]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )


def plot_tank_efficiencies(
    performances: list[Performances],
    x_data: list[str],
    x_label: str,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            x_data,
            [
                performance.gravimetric_efficiency
                for performance in performances
            ],
            "Gravimetric",
            marker=None
        ),
        Line(
            x_data,
            [
                performance.volumetric_efficiency
                for performance in performances
            ],
            "Volumetric",
            marker=None
        )
    ]
    return SingleFigure(
        data,
        x_label,
        "Efficiency [-]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )

import matplotlib.pyplot as plt

def plot_tank_efficiencies_scatter(
    performances: list[Performances],
    x_data: list[float],
    x_label: str,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    gravimetric_efficiencies = [
        performance.gravimetric_efficiency
        for performance in performances
    ]
    volumetric_efficiencies = [
        performance.volumetric_efficiency
        for performance in performances
    ]

    fig, ax = plt.subplots()
    ax.scatter(x_data, gravimetric_efficiencies, label="Gravimetric", marker='o')
    ax.scatter(x_data, volumetric_efficiencies, label="Volumetric", marker='x')
    ax.set_xlabel(x_label)
    ax.set_ylabel("Efficiency [-]")
    ax.legend()
    ax.grid(True)
    ax.autoscale()  # Automatically adjust axis limits

    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    return fig


def plot_general_properties(
    data: list[list[float]],
    labels: list[float],
    x_data: list[float],
    x_label: str,
    y_label: str,
    xticks: list[float] = None,
    yticks: list[float] = None
):
    data = [
        Line(
            x_data, row, label, marker=None
        )
        for row, label in zip(data, labels)
    ]
    return SingleFigure(
        data, x_label, y_label, x_ticks=xticks, y_ticks=yticks,
    )



def plot_thermo_mechanical_loading(
    tank_states: TankStates,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
):
    times = tank_states.timesteps_in_hours
    data = [
        [Line(
            times,
            tank_states.pressures_in_bar,
            "Pressure"
        )],
        [Line(
            times,
            tank_states.temperatures,
            "Temperatures"
        )]
    ]
    return TwinXFigure(
        data,
        "Time [hour]",
        ["Pressure [bar]", "Temperature [K]"],
        x_ticks=x_ticks,
        y_ticks=[y1ticks, y2ticks],
    )


def plot_tank_fill(
    tank_states: TankStates,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
) -> None:
    times = tank_states.timesteps_in_hours
    data = [
        [
            Line(times, tank_states.liquid_masses, "Liquid"),
            Line(times, tank_states.gas_masses, "Gas"),
            Line(times, tank_states.total_masses, "Total")
        ], [
            Line(times, tank_states.fills, "Fill")
        ]
    ]
    x_label = "Time [hour]"
    y_labels = ["Fuel Mass [kg]", "Fill [%]"]
    fig = TwinXFigure(
        data,
        x_label,
        y_labels,
        x_ticks=x_ticks,
        y_ticks=[y1ticks, y2ticks]
    )
    return fig


def main():
    pass


if __name__ == "__main__":
    main()


# End
