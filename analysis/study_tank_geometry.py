

from typing import Protocol

from plotting.plot_geometric_study import plot_geometric_study
from src.facades.analysis_facades import (DrainingAnalysisFacade,
                                          OperatingEnvelope, TankDimensions)


class Insulation(Protocol):
    ...


class InitialState(Protocol):
    ...


class Material(Protocol):
    ...


def analyse_tank(
    initial_state: InitialState,
    insulation: Insulation,
    tank_material: Material,
    min_pressure: float,
    fuel_mass_flow: float,
    fuel_flow_phase: str,
    levels: list[float]
):

    # Define tank dimensions
    radii = [i / 100 for i in range(25, 276, 25)]
    radii = [0.5, 1.5, 2.5]
    body_lengths = [0.0, 10.0]

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
                OperatingEnvelope(
                    None,
                    min_pressure,
                    None
                )
            )
            for body_length in body_lengths
        ]
        for radius in radii
    ]

    plot_geometric_study(
        radii, body_lengths, performances, levels
    )


def main():
    pass


if __name__ == "__main__":
    main()


# End
