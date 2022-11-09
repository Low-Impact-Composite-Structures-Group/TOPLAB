

from typing import Protocol

import matplotlib.pyplot as plt


SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5 


class DataSchema:
    TANK_STATES = "tank_states"
    LABEL = "label"



class TankState(Protocol):
    pressure: float


def plot_pressure_rise(
    data: list[dict],
    timestep: float,
    xticks: list[float] = None,
    yticks: list[float] = None
):
    for row in data:
        tank_states: list[TankState] = row[DataSchema.TANK_STATES]
        pressures = [
            state.pressure * PASCAL_TO_BAR
            for state in tank_states
        ]
        times = [
            i * timestep * SECONDS_TO_HOURS
            for i, _ in enumerate(pressures)
        ]
        plt.plot(
            times, pressures, label=row[DataSchema.LABEL]
        )
        if xticks is not None:
            plt.xlim((xticks[0], xticks[-1]))
            plt.xticks(xticks)
        plt.xlabel("Time [hours]")
        if yticks is not None:
            plt.ylim((yticks[0], yticks[-1]))
            plt.yticks(yticks)
        plt.ylabel("Pressure [bar]")
        plt.legend()
        plt.grid()
        plt.show()