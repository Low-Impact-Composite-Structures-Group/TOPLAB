

from plotting.plot_tank_states import (plot_tank_fill, plot_tank_loads,
                                       plot_tank_temperatures)
from src.dynamics.draining_analysis import AnalyseCylindricalTank
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Metal
from src.mission.mission_sections import OutFlow
from src.thermodynamics.tank_states import InitialState


def perform_analysis():

    # Define the initial state of the tank
    pressure = 140e3
    temperature = None
    fill = 0.97
    initial_state = InitialState(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-2
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    tank_material = Metal.aluminum()

    # Define fuel flow
    fuel_flow = OutFlow.rompokos_cruise("liquid")

    # Define tank dimensions
    body_length = 5
    radii = [1.0, 1.5, 2.0, 2.5]
    labels = [f'{radius} m' for radius in radii]

    # Perform the analysis
    data = [
        AnalyseCylindricalTank.analyse_tank(
            radius,
            body_length,
            tank_material,
            insulation,
            fuel_flow.mass_flow,
            fuel_flow.phase,
            initial_state
        ).tank_states
        for radius in radii
    ]

    fig1 = plot_tank_loads(data, labels)
    fig2 = plot_tank_temperatures(data, labels)
    fig3 = plot_tank_fill(data[-1])

    fig1.show()