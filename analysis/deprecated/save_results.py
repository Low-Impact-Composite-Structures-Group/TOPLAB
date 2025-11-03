import numpy as np

from pathlib import Path

from typing import Protocol

import csv


class Tank(Protocol):
    volume: float


class TankState(Protocol):
    max_pressure: float

class Performance(Protocol):
    gravimetric_efficiency: float
    volumetric_efficiency: float
    tank: Tank
    tank_states: TankState

def save_results(
    radii: list[float],
    body_lengths: list[float],
    performances: list[list[Performance]],
    directory: str
):

    # Directory path
    path = Path.cwd() / "data" / "results" / directory

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


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
