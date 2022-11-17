

from plotting.plot_tank_states import (plot_tank_efficiencies, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures)
from src.dynamics.draining_analysis import AnalyseCylindricalTank
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Metal
from src.mission.mission_sections import OutFlow
from src.thermodynamics.tank_states import InitialState


def perform_analysis():

    # Define the initial state of the tank
    pressure = 300e5
    temperature = 70
    fill = 0.0
    initial_state = InitialState(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-13
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    tank_material = Metal.aluminum()

    # Define fuel flow
    fuel_flow = OutFlow.rompokos_cruise("gas")

    # Define the minimum pressure of the tank
    min_pressure = 10e5

    # Define tank dimensions
    body_length = 5
    radii = [i / 100 for i in range(25, 276, 50)]
    labels = [f'{radius} m' for radius in radii]

    # Perform the analysis
    performances = [
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
        for radius in radii
    ]
    data = [performance.tank_states for performance in performances]

    xticks = list(range(0, 61, 10))
    yticks = list(range(0, 401, 50))
    fig1 = plot_tank_loads(data, labels, xticks, yticks)
    yticks = list(range(20, 121, 20))
    fig2 = plot_tank_temperatures(data, labels, xticks, yticks)
    xticks = [i / 100 for i in range(0, 301, 50)]
    yticks = [i / 100 for i in range(0, 101, 20)]
    fig4 = plot_tank_efficiencies(
        performances, radii, "Radius [m]", xticks, yticks
    )
    fig4.show()