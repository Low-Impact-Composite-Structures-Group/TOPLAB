

from typing import Protocol
from plotting.plot_geometric_study import plot_geometric_study
from src.dynamics.draining_analysis import AnalyseCylindricalTank
from src.materials.materials import Composite
from src.mission.mission_sections import OutFlow
from src.thermodynamics.tank_states import InitialState


class Insulation(Protocol):
    ...


def analyse_tank(
    initial_state: InitialState,
    insulation: Insulation,
    min_pressure: float,
    fuel_flow_phase,
    levels: list[float]
):

    # Define tank material
    winding_angle = 55
    tank_material = Composite.carbon(winding_angle)

    # Define fuel flow
    fuel_flow = OutFlow.rompokos_cruise(fuel_flow_phase)

    # Define tank dimensions
    radii = [i / 100 for i in range(25, 276, 25)]
    radii = [0.5, 1.5, 2.5]
    body_lengths = [0.0, 10.0]

    # Perform the analysis
    performances = [
        [
            AnalyseCylindricalTank.analyse_tank(
                radius,
                body_length,
                tank_material,
                insulation,
                fuel_flow.mass_flow,
                fuel_flow.phase,
                initial_state,
                min_pressure
            )
            for body_length in body_lengths
        ]
        for radius in radii
    ]

    plot_geometric_study(
        radii, body_lengths, performances, levels
    )