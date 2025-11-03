import csv
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from facades.analysis_facades import (DrainingAnalysisFacade,
                                      OperationalEnvelope, TankDimensions)
from plotting.plot_geometric_study import plot_geometric_study


class Insulation(Protocol):
    ...


class InitialState(Protocol):
    ...


class Material(Protocol):
    ...


class Tank(Protocol):
    volume: float


class TankStates(Protocol):
    max_pressure: float


class Performance(Protocol):
    gravimetric_efficiency: float
    volumetric_efficiency: float
    tank: Tank
    tank_states: TankStates


def analyse_tank(
    initial_state: InitialState,
    insulation: Insulation,
    tank_material: Material,
    min_pressure: float,
    fuel_mass_flow: float,
    fuel_flow_phase: str,
    levels: list[float],
    directory: str,
    number_of_tanks: int = 1
) -> list[list[Performance]]:

    # Define tank dimensions
    # radii = [0.25, 5.0]
    # body_lengths = [0.0, 5, 10.0]
    body_lengths = [i / 100 for i in range(0, 1001, 20)]
    radii = [i / 100 for i in range(25, 401, 25)]

    body_lengths = [i / 100 for i in range(0, 1001, 50)]
    radii = [i / 100 for i in range(25, 251, 25)]

    minimum_mass = 5
    minimum_fill = 0

    # Perform the analysis
    performances = [
        [
            DrainingAnalysisFacade.analyse(
                TankDimensions(
                    radius, body_length
                ),
                tank_material,
                insulation,
                fuel_mass_flow,
                fuel_flow_phase,
                initial_state,
                OperationalEnvelope(
                    min_pressure=min_pressure,
                    min_mass=minimum_mass,
                    min_fill=minimum_fill,
                )
            )
            for body_length in body_lengths
        ]
        for radius in radii
    ]

    save_results(radii, body_lengths, performances, directory)
    plot_geometric_study(directory, levels)

    plt.show()

    return performances


def save_results(
    radii: list[float],
    body_lengths: list[float],
    performances: list[list[Performance]],
    directory: str
):

    # Directory path
    path = Path.cwd() / "data" / "results" / directory

    # Save gravimetric efficiency
    gravimetric_efficiencies = [
        [
            performance.gravimetric_efficiency
            for performance in row
        ]
        for row in performances
    ]
    save_result(
        radii,
        body_lengths,
        gravimetric_efficiencies,
        path / "gravimetric_efficiencies.csv"
    )

    # Save volumetric efficiency
    volumetric_efficiencies = [
        [
            performance.volumetric_efficiency
            for performance in row
        ]
        for row in performances
    ]
    save_result(
        radii,
        body_lengths,
        volumetric_efficiencies,
        path / "volumetric_efficiencies.csv"
    )

    # Save tank volumes
    tank_volumes = [
        [
            performance.tank.volume
            for performance in row
        ]
        for row in performances
    ]
    save_result(
        radii,
        body_lengths,
        tank_volumes,
        path / "tank_volumes.csv"
    )

    # Save tank pressures
    pressures = [
        [
            performance.tank_states.max_pressure
            for performance in row
        ]
        for row in performances
    ]
    save_result(
        radii,
        body_lengths,
        pressures,
        path / "max_pressures.csv"
    )


def save_result(radii, body_lengths, data, path):

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        body_lengths = np.array(body_lengths)
        radii = np.array([[radius] for radius in radii])
        radii = np.vstack(([0], radii))
        data = np.vstack((body_lengths, data))
        data = np.hstack((radii, data))
        writer.writerows(data)


def main():
    pass


if __name__ == "__main__":
    main()


# End
