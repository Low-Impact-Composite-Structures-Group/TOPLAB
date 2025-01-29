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
    fills: list[float]
    liquid_masses: list[float]
    gas_masses: list[float]


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
    
def plot_thermo_mechanical_loading_vs_fill(
    tank_states: TankStates,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
):
    fills = tank_states.fills
    data = [
        [Line(
            fills,
            tank_states.pressures_in_bar,
            "Pressure"
        )],
        [Line(
            fills,
            tank_states.temperatures,
            "Temperatures"
        )]
    ]
    return TwinXFigure(
        data,
        "Fill [-]",
        ["Pressure [bar]", "Temperature [K]"],
        x_ticks=x_ticks,
        y_ticks=[y1ticks, y2ticks],
    )
    
def plot_thermo_mechanical_loading_vs_mass(
    tank_states: TankStates,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
):
    # masses = tank_states.liquid_masses + tank_states.gas_masses
    masses = [liquid + gas for liquid, gas in zip(tank_states.liquid_masses, tank_states.gas_masses)]
    data = [
        [Line(
            masses,
            tank_states.pressures_in_bar,
            "Pressure"
        )],
        [Line(
            masses,
            tank_states.temperatures,
            "Temperatures"
        )]
    ]
    return TwinXFigure(
        data,
        "H2 charged [kg]",
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


def plot_single_tank_loads(
    tank_state: TankStates,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            tank_state.timesteps_in_hours,
            tank_state.pressures_in_bar,
            "Pressure"
        )
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Pressure [bar]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )


def plot_single_tank_temperatures(
    tank_state: TankStates,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            tank_state.timesteps_in_hours,
            tank_state.temperatures,
            "Temperature"
        )
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Temperature [K]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )


def plot_single_tank_fill(
    tank_state: TankStates,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
):
    times = tank_state.timesteps_in_hours
    data = [
        [
            Line(times, tank_state.liquid_masses, "Liquid"),
            Line(times, tank_state.gas_masses, "Gas"),
            Line(times, tank_state.total_masses, "Total")
        ], [
            Line(times, tank_state.fills, "Fill")
        ]
    ]
    x_label = "Time [hour]"
    y_labels = ["Fuel Mass [kg]", "Fill [%]"]
    return TwinXFigure(
        data,
        x_label,
        y_labels,
        x_ticks=x_ticks,
        y_ticks=[y1ticks, y2ticks]
    )
    
def plot_single_required_flux(
    tank_state: TankStates,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
     # Ensure both arrays have the same length
    min_length = min(len(tank_state.timesteps_in_hours), len(tank_state.required_fluxes))
    times = tank_state.timesteps_in_hours[:min_length]
    required_fluxes = [flux / -1000 for flux in tank_state.required_fluxes[:min_length]]
    data = [
        Line(
            times,
            required_fluxes,
            "Required flux"
        )
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Heat flux [kW]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )
    
def plot_density_vs_temperature(
    tank_state: TankStates,
    process_label: str,
    hydrogen_density: list[float],
    isobar_label: str,
    isobar_densities: list[float] = None,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    fig, ax = plt.subplots()
    ax.plot(tank_state.temperatures, hydrogen_density, label=process_label, linestyle='solid', marker="")
    
    # Add an arrow to show the direction wrt time
    mid_index = int(len(tank_state.temperatures) / 2)
    ax.annotate(
        '',
        xy=(tank_state.temperatures[mid_index + 1], hydrogen_density[mid_index + 1]),
        xytext=(tank_state.temperatures[mid_index], hydrogen_density[mid_index]),
        arrowprops=dict(arrowstyle="->", color='black')
    )
    
     # Plot the isobar densities if provided
    if isobar_densities is not None:
        ax.plot(tank_state.temperatures, isobar_densities, label=isobar_label, linestyle='dotted', marker="")
    
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Density [kg/m^3]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)
    
    ax.legend()
    return fig
    
    


def main():
    pass


if __name__ == "__main__":
    main()


# End
