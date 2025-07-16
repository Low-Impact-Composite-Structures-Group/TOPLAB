from typing import Protocol, Union

import matplotlib.pyplot as plt
import numpy as np

from plotting.figures import Line, SingleFigure, TwinXFigure
from plotting.plot_style import set_font

SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5
TO_MEGA = 1e-6


def configure_font(font_name: str = "Cambria"):
    """Configure the font for all plotting operations.

    Args:
        font_name: Name of the font family to use (e.g., "Cambria", "Arial", "Times New Roman")

    Example:
        >>> configure_font("Cambria")  # Set to Cambria
        >>> configure_font("Arial")    # Change to Arial
    """
    set_font(font_name)
    print(f"Font configured to: {font_name}")


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
    ), required_fluxes

def plot_heat_flows(
    tank_state: TankStates,
    ohex_heats: list[float],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    # Ensure both arrays have the same length
    min_length = min(len(tank_state.timesteps_in_hours), len(tank_state.required_fluxes), len(ohex_heats))
    times = tank_state.timesteps_in_hours[:min_length]
    required_fluxes = [-flux/1000 for flux in tank_state.required_fluxes[:min_length]]
    ohex_heats = [heats/1000 for heats in ohex_heats[:min_length]]
    #required_fluxes(ihex) + ohex list to get total heat flow
    # Ensure both arrays have the same length
    if len(required_fluxes) != len(ohex_heats):
        min_length = min(len(required_fluxes), len(ohex_heats))
        total_heat = required_fluxes[:min_length]
        ohex_heats = ohex_heats[:min_length]
        times = times[:min_length]
    # Create the data for the plot
    total_heat = [ihex + ohex for ihex, ohex in zip(required_fluxes, ohex_heats)]


    data = [
        Line(
            times,
            required_fluxes,
            "iHEX",
            color='blue'
        ),
        Line(
            times,
            ohex_heats,
            "oHEX",
            color='orange',
            style='-'
        ),
        Line(
            times,
            total_heat,
            "Total Thermal Power Required by System",
            color='green',)
    ]
    return SingleFigure(
        data,
        "Time [hour]",
        "Thermal Power [kW]",
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )

def plot_density_vs_temperature(
    tank_state: TankStates,
    process_label: str,
    hydrogen_density: list[float],
    isobar_labels: list[str],
    isobar_densities: list[list[float]],
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
        for label, densities in zip(isobar_labels, isobar_densities):
            ax.plot(tank_state.temperatures, densities, label=label, linestyle='dotted', marker="")

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Density [kg/m^3]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    ax.legend()
    return fig

def plot_cycle_density_vs_temperature(
    discharge_states: TankStates,
    refuel_states: TankStates,
    dormancy_states: TankStates,
    discharge_densities: list[float],
    refuel_densities: list[float],
    dormancy_densities: list[float],
    process_labels: list[str],
    isobar_densities: list[float] = None,
    isobar_labels: list[str] = None,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    fig, ax = plt.subplots()

    # Plot discharge states
    ax.plot(discharge_states.temperatures, discharge_densities, 'm-', label=process_labels[0])
    mid_index = int(len(discharge_states.temperatures) / 2)
    if mid_index + 1 < len(discharge_states.temperatures) and mid_index + 1 < len(discharge_densities):
        ax.annotate('', xy=(discharge_states.temperatures[mid_index + 1], discharge_densities[mid_index + 1]),
                    xytext=(discharge_states.temperatures[mid_index], discharge_densities[mid_index]),
                    arrowprops=dict(arrowstyle="->", color='m'))

    # Plot refuel states
    ax.plot(refuel_states.temperatures, refuel_densities, 'r-', label=process_labels[1])
    mid_index = int(len(refuel_states.temperatures) / 2)
    if mid_index + 1 < len(refuel_states.temperatures) and mid_index + 1 < len(refuel_densities):
        ax.annotate('', xy=(refuel_states.temperatures[mid_index + 1], refuel_densities[mid_index + 1]),
                    xytext=(refuel_states.temperatures[mid_index], refuel_densities[mid_index]),
                    arrowprops=dict(arrowstyle="->", color='r'))

    # Plot dormancy states
    ax.plot(dormancy_states.temperatures, dormancy_densities, 'b-', label=process_labels[2])
    mid_index = int(len(dormancy_states.temperatures) / 2)
    if mid_index + 1 < len(dormancy_states.temperatures) and mid_index + 1 < len(dormancy_densities):
        ax.annotate('', xy=(dormancy_states.temperatures[mid_index + 1], dormancy_densities[mid_index + 1]),
                    xytext=(dormancy_states.temperatures[mid_index], dormancy_densities[mid_index]),
                    arrowprops=dict(arrowstyle="->", color='b'))

    # Plot the isobar densities if provided
    if isobar_densities is not None:
        ax.plot(discharge_states.temperatures, isobar_densities, label=isobar_labels, linestyle='dotted', marker="")

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Density [kg/m^3]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    ax.legend()
    return fig

def plot_cycle_tank_temperature(
    discharge_states: TankStates,
    refuel_states: TankStates,
    dormancy_states: TankStates,
    process_labels: list[str],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    fig, ax = plt.subplots()

    # Calculate cumulative times
    discharge_end_time = discharge_states.timesteps_in_hours[-1]
    refuel_start_time = discharge_end_time
    refuel_end_time = refuel_start_time + refuel_states.timesteps_in_hours[-1]
    dormancy_start_time = refuel_end_time

    # Plot discharge states
    ax.plot(discharge_states.timesteps_in_hours, discharge_states.temperatures, 'm-', label=process_labels[0])

    # Plot refuel states
    ax.plot([t + refuel_start_time for t in refuel_states.timesteps_in_hours], refuel_states.temperatures, 'r-', label=process_labels[1])

    # Plot dormancy states
    ax.plot([t + dormancy_start_time for t in dormancy_states.timesteps_in_hours], dormancy_states.temperatures, 'b-', label=process_labels[2])

    # Add vertical lines to indicate mode changes
    ax.axvline(x=discharge_end_time, color='k', linestyle='--')
    ax.axvline(x=refuel_end_time, color='k', linestyle='--')

    ax.set_xlabel("Time [hour]")
    ax.set_ylabel("Temperature [K]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    ax.legend()
    return fig

def plot_cycle_tank_pressure(
    discharge_states: TankStates,
    refuel_states: TankStates,
    dormancy_states: TankStates,
    process_labels: list[str],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):

    fig, ax = plt.subplots()

    # Calculate cumulative times
    discharge_end_time = discharge_states.timesteps_in_hours[-1]
    refuel_start_time = discharge_end_time
    refuel_end_time = refuel_start_time + refuel_states.timesteps_in_hours[-1]
    dormancy_start_time = refuel_end_time

    # Plot discharge states
    ax.plot(discharge_states.timesteps_in_hours, [p / 1e5 for p in discharge_states.pressures], 'm-', label=process_labels[0])

    # Plot refuel states
    ax.plot([t + refuel_start_time for t in refuel_states.timesteps_in_hours], [p / 1e5 for p in refuel_states.pressures], 'r-', label=process_labels[1])

    # Plot dormancy states
    ax.plot([t + dormancy_start_time for t in dormancy_states.timesteps_in_hours], [p / 1e5 for p in dormancy_states.pressures], 'b-', label=process_labels[2])

    # Add vertical lines to indicate mode changes
    ax.axvline(x=discharge_end_time, color='k', linestyle='--')
    ax.axvline(x=refuel_end_time, color='k', linestyle='--')

    ax.set_xlabel("Time [hour]")
    ax.set_ylabel("Pressure [bar]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    ax.legend()
    return fig

def plot_cycle_required_flux(
    discharge_states: TankStates,
    refuel_states: TankStates,
    dormancy_states: TankStates,
    process_labels: list[str],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    fig, ax = plt.subplots()

    # Calculate cumulative times
    discharge_end_time = discharge_states.timesteps_in_hours[-1]
    refuel_start_time = discharge_end_time
    refuel_end_time = refuel_start_time + refuel_states.timesteps_in_hours[-1]
    dormancy_start_time = refuel_end_time

    # Ensure both arrays have the same length for discharge
    min_length_discharge = min(len(discharge_states.timesteps_in_hours), len(discharge_states.required_fluxes))
    discharge_times = discharge_states.timesteps_in_hours[:min_length_discharge]
    discharge_fluxes = [flux / -1000 for flux in discharge_states.required_fluxes[:min_length_discharge]]

    # Ensure both arrays have the same length for refuel
    min_length_refuel = min(len(refuel_states.timesteps_in_hours), len(refuel_states.required_fluxes))
    refuel_times = [t + refuel_start_time for t in refuel_states.timesteps_in_hours[:min_length_refuel]]
    refuel_fluxes = [flux / -1000 for flux in refuel_states.required_fluxes[:min_length_refuel]]

    # Ensure both arrays have the same length for dormancy
    min_length_dormancy = min(len(dormancy_states.timesteps_in_hours), len(dormancy_states.required_fluxes))
    dormancy_times = [t + dormancy_start_time for t in dormancy_states.timesteps_in_hours[:min_length_dormancy]]
    dormancy_fluxes = [flux / -1000 for flux in dormancy_states.required_fluxes[:min_length_dormancy]]

    # Plot discharge states
    ax.plot(discharge_times, discharge_fluxes, 'r-', label=process_labels[0])

    # Plot refuel states
    ax.plot(refuel_times, refuel_fluxes, 'g-', label=process_labels[1])

    # Plot dormancy states
    ax.plot(dormancy_times, dormancy_fluxes, 'b-', label=process_labels[2])

    # Add vertical lines to indicate mode changes
    ax.axvline(x=discharge_end_time, color='k', linestyle='--')
    ax.axvline(x=refuel_end_time, color='k', linestyle='--')

    ax.set_xlabel("Time [hour]")
    ax.set_ylabel("Heat flux [kW]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    ax.legend()
    return fig

def plot_cycle_tank_fill(
    discharge_states: TankStates,
    refuel_states: TankStates,
    dormancy_states: TankStates,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
):
    fig, ax = plt.subplots()

    # Calculate cumulative times
    discharge_end_time = discharge_states.timesteps_in_hours[-1]
    refuel_start_time = discharge_end_time
    refuel_end_time = refuel_start_time + refuel_states.timesteps_in_hours[-1]
    dormancy_start_time = refuel_end_time

    # Ensure both arrays have the same length for discharge
    min_length_discharge = min(len(discharge_states.timesteps_in_hours), len(discharge_states.liquid_masses), len(discharge_states.gas_masses), len(discharge_states.total_masses), len(discharge_states.fills))
    discharge_times = discharge_states.timesteps_in_hours[:min_length_discharge]
    discharge_liquid_masses = discharge_states.liquid_masses[:min_length_discharge]
    discharge_gas_masses = discharge_states.gas_masses[:min_length_discharge]
    discharge_total_masses = discharge_states.total_masses[:min_length_discharge]

    # Ensure both arrays have the same length for refuel
    min_length_refuel = min(len(refuel_states.timesteps_in_hours), len(refuel_states.liquid_masses), len(refuel_states.gas_masses), len(refuel_states.total_masses), len(refuel_states.fills))
    refuel_times = [t + refuel_start_time for t in refuel_states.timesteps_in_hours[:min_length_refuel]]
    refuel_liquid_masses = refuel_states.liquid_masses[:min_length_refuel]
    refuel_gas_masses = refuel_states.gas_masses[:min_length_refuel]
    refuel_total_masses = refuel_states.total_masses[:min_length_refuel]

    # Ensure both arrays have the same length for dormancy
    min_length_dormancy = min(len(dormancy_states.timesteps_in_hours), len(dormancy_states.liquid_masses), len(dormancy_states.gas_masses), len(dormancy_states.total_masses), len(dormancy_states.fills))
    dormancy_times = [t + dormancy_start_time for t in dormancy_states.timesteps_in_hours[:min_length_dormancy]]
    dormancy_liquid_masses = dormancy_states.liquid_masses[:min_length_dormancy]
    dormancy_gas_masses = dormancy_states.gas_masses[:min_length_dormancy]
    dormancy_total_masses = dormancy_states.total_masses[:min_length_dormancy]

    # Plot liquid, gas, and total masses
    ax.plot(discharge_times, discharge_liquid_masses, 'b-', label='Liquid')
    ax.plot(discharge_times, discharge_gas_masses, 'r-', label='Gas')
    ax.plot(discharge_times, discharge_total_masses, 'y-.', label='Total')

    ax.plot(refuel_times, refuel_liquid_masses, 'b-')
    ax.plot(refuel_times, refuel_gas_masses, 'r-')
    ax.plot(refuel_times, refuel_total_masses, 'y-.')

    ax.plot(dormancy_times, dormancy_liquid_masses, 'b-')
    ax.plot(dormancy_times, dormancy_gas_masses, 'r-')
    ax.plot(dormancy_times, dormancy_total_masses, 'y-.')

    # Add vertical lines to indicate mode changes
    ax.axvline(x=discharge_end_time, color='k', linestyle='--')
    ax.axvline(x=refuel_end_time, color='k', linestyle='--')

    ax.set_xlabel("Time [hour]")
    ax.set_ylabel("Fuel Mass [kg]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y1ticks:
        ax.set_yticks(y1ticks)

    ax.legend()

    return fig

def plot_mission_mass_flows(
    mass_flows: list[Union[float, list[float]]],
    fuel_flow_keys: list[str],
    durations: list[float],
    total_duration: float,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    fig, ax = plt.subplots()

    # Calculate the cumulative durations for the x-axis
    cumulative_durations = [0] + list(np.cumsum(durations))

    # Plot the mass flows
    for i in range(len(mass_flows)):
        start_time = cumulative_durations[i]
        end_time = cumulative_durations[i + 1]
        color = f'C{i % 10}'  # Ensure unique color for each section

        if isinstance(mass_flows[i], list):
            start_flow, end_flow = mass_flows[i]
            ax.plot([start_time, end_time], [start_flow, end_flow], label=fuel_flow_keys[i], color=color, linestyle='-', marker=None)
        else:
            ax.hlines(mass_flows[i], start_time, end_time, label=fuel_flow_keys[i], color=color, linestyle='-', marker=None)

        # Draw vertical black dotted line for discontinuous transitions
        if i < len(mass_flows) - 1:
            next_flow = mass_flows[i + 1][0] if isinstance(mass_flows[i + 1], list) else mass_flows[i + 1]
            current_end_flow = mass_flows[i][1] if isinstance(mass_flows[i], list) else mass_flows[i]
            if current_end_flow != next_flow:
                ax.vlines(end_time, current_end_flow, next_flow, color='black', linestyle='dotted')
    ax.set_title("Mass Flow vs Time")
    ax.set_xlabel("Mission Duration [s]")
    ax.set_ylabel("Mass Flow [kg/s]")
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    ax.legend()
    ax.grid(True)
    return fig


def plot_tank_mass_flows(
    time_points: list[float],
    tank1_inflow: list[float],
    tank1_outflow: list[float],
    tank2_inflow: list[float],
    tank2_outflow: list[float],
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    """Plot mass flows between tanks and to mission."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ensure all arrays have the same length by truncating to the time array length
    time_len = len(time_points)
    tank1_inflow = tank1_inflow[:time_len]
    tank1_outflow = tank1_outflow[:time_len]
    tank2_inflow = tank2_inflow[:time_len]
    tank2_outflow = tank2_outflow[:time_len]

    # Convert time from seconds to hours for consistent plotting
    time_hours = [t * SECONDS_TO_HOURS for t in time_points]

    # Plot the flow rates
    ax.plot(time_hours, tank2_outflow, 'bo-', label="Tank 2 Outflow (Mission)", linewidth=2)
    ax.plot(time_hours, tank1_inflow, 'k-', label="Tank 1 Inflow", linewidth=2)
    ax.plot(time_hours, tank1_outflow, 'r-', label="Tank 1 Outflow", linewidth=2) # Removed 'o' marker
    ax.plot(time_hours, tank2_inflow, 'g-', label="Tank 2 Inflow", linewidth=2)


    # Add a zero line for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Add annotation to explain the flow relationship
    ax.text(0.02, 0.02, "Tank 1 supplies Tank 2, which supplies the mission",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Customize plot
    ax.set_title("Tank Mass Flow Rates")
    ax.set_xlabel("Time [hour]")
    ax.set_ylabel("Mass Flow Rate [kg/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    return fig
def main():
    pass


if __name__ == "__main__":
    main()


# End