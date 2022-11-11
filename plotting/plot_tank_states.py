

from typing import Protocol

from plotting.figures import Line, SingleFigure, TwinXFigure

SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5 


class TankState(Protocol):
    pressure: float
    temperature: float
    fill: float
    liquid_mass: float
    gas_mass: float
    fuel_mass: float


def plot_tank_loads(
    tank_states: list[list[TankState]],
    labels: list[str],
    timestep: float = 60,
    x_ticks: list[float] = None,
    y_ticks: list[float] = None
):
    data = [
        Line(
            [
                i * timestep * SECONDS_TO_HOURS
                for i, _ in enumerate(row)
            ],
            [
                state.pressure * PASCAL_TO_BAR
                for state in row
            ],
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


def plot_thermo_mechanical_loading(
    tank_states: list[TankState],
    timestep: float,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
):
    # Unpack values for plotting
    pressures = [
        state.pressure * PASCAL_TO_BAR
        for state in tank_states
    ]
    temperatures = [
        state.temperature
        for state in tank_states
    ]
    times = [
        i * timestep * SECONDS_TO_HOURS
        for i, _ in enumerate(pressures)
    ]
    data = [
        [Line(times, pressures, "Pressure")],
        [Line(times, temperatures, "Temperatures")]
    ]
    return TwinXFigure(
        data,
        "Time [hour]",
        ["Pressure [bar]", "Temperature [K]"],
        x_ticks=x_ticks,
        y_ticks=[y1ticks, y2ticks],
    )


def plot_tank_fill(
    tank_states: list[TankState],
    timestep: float,
    x_ticks: list[float] = None,
    y1ticks: list[float] = None,
    y2ticks: list[float] = None
) -> None:

    # Unpack data from tank states
    liquid_mass = [
        state.liquid_mass
        for state in tank_states
    ]
    gas_mass = [
        state.gas_mass
        for state in tank_states
    ]
    total_mass = [
        state.fuel_mass
        for state in tank_states
    ]
    fill = [
        state.fill * 100
        for state in tank_states
    ]
    times = [
        i * timestep * SECONDS_TO_HOURS
        for i, _ in enumerate(tank_states)
    ]

    data = [
        [
            Line(times, liquid_mass, "Liquid"),
            Line(times, gas_mass, "Gas"),
            Line(times, total_mass, "Total")
        ], [
            Line(times, fill, "Fill")
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
