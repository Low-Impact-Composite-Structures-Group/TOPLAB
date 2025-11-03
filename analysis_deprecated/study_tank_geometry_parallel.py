import csv
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster

from facades_deprecated.analysis_facades import (DrainingAnalysisFacade,
                                      OperationalEnvelope, ParallelDrainingAnalysis, TankDimensions)
from plotting.plot_geometric_study import plot_geometric_study
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps, TankFactory, SphericalTank


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



class ParallelizedTankFactory:
    def __init__(self, radii, lengths, material, pressure):
        self.radii = radii
        self.lengths = lengths
        self.material = material
        self.operating_pressure = pressure
        
    def create_tanks(self):
        return [
            CylindricalTankSphericalCaps(
                radius,
                length + 2 * radius,
                self.material,
                self.operating_pressure
            )
            if length != 0 else SphericalTank(
                radius,
                self.material,
                self.operating_pressure
            )
            for i, radius in enumerate(self.radii)
            for j, length in enumerate(self.lengths)
        ]


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
    min_length, max_length, step_length = 0, 10, 0.1
    min_radius, max_radius, step_radius = 0.25, 4, 0.25

    body_lengths = np.arange(
        min_length, max_length + step_length, step_length
    )
    radii = np.arange(
        min_radius, max_radius + step_radius, step_radius
    )

    radius_grid, length_grid = np.meshgrid(
        radii, body_lengths, indexing='ij'
    )

    def compute_volume(radius, length):
        material = tank_material
        operating_pressure = initial_state.pressure
        factory = TankFactory()
        tank = factory.create_tank(
            radius, length, material, operating_pressure
        )
        return tank.volume

    client = Client()

    # Create 1D arrays of radius and length values
    radii = da.from_array(radii, chunks=2)
    lengths = da.from_array(lengths, chunks=2)

    # Create a TankFactory object with arrays of radii and lengths
    factory = TankFactory(radii=radii, lengths=lengths)

    # Use the vectorized create_tanks method to create tank objects for each radius-length combination
    tanks = da.from_array(factory.create_tanks(), chunks=(2,))


    # def compute_pressure(radius, length):

    #     draining = np.empty((len(radius), len(length)), dtype=object)
    #     for i, r in enumerate(radius):
    #         for j, l in enumerate(length):
    #             draining[i, j] = ParallelDrainingAnalysis(
    #                 r,
    #                 l,
    #                 tank_material,
    #                 insulation,
    #                 fuel_mass_flow,
    #                 fuel_flow_phase,
    #                 initial_state,
    #                 OperatingEnvelope(
    #                     None,
    #                     min_pressure,
    #                     None
    #                 )
    #             )

    #     # compute something for each Tank object
    #     result = np.empty((len(radius), len(length)))
    #     for i in range(len(radius)):
    #         for j in range(len(length)):
    #             result[i, j] = draining[i, j].analyse()

    #     return result

    # # Create a Dask client and local cluster
    # cluster = LocalCluster()
    # client = Client(cluster)

    # # Convert the 2D arrays to Dask arrays and distribute the computation
    # radius_dask = da.from_array(radius_grid, chunks=len(radii))
    # length_dask = da.from_array(length_grid, chunks=len(body_lengths))
    # result_dask = da.map_blocks(compute_pressure, radius_dask, length_dask)
    # pressures = result_dask.compute()

    # plt.contourf(body_lengths, radii, pressures)
    # plt.show()



    # Perform the analysis
    # performances = [
    #     [
    #         DrainingAnalysisFacade.analyse(
    #             TankDimensions(
    #                 radius, body_length
    #             ),
    #             tank_material,
    #             insulation,
    #             fuel_mass_flow,
    #             fuel_flow_phase,
    #             initial_state,
    #             OperatingEnvelope(
    #                 None,
    #                 min_pressure,
    #                 None
    #             )
    #         )
    #         for body_length in body_lengths
    #     ]
    #     for radius in radii
    # ]

    # save_results(radii, body_lengths, performances, directory)
    # plot_geometric_study(directory, levels)

    # plt.show()

    # return performances


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
