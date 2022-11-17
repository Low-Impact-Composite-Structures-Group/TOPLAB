

from plotting.plot_tank_states import (plot_tank_efficiencies, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures)
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

    # Define the minimum pressure of the tank
    min_pressure = 1.3e5

    # Define tank dimensions
    body_length = 5
    radii = [i / 100 for i in range(25, 276, 25)]
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

    xticks = list(range(0, 31, 5))
    yticks = list(range(0, 11, 2))
    fig1 = plot_tank_loads(data, labels, xticks, yticks)
    yticks = list(range(20, 33, 2))
    fig2 = plot_tank_temperatures(data, labels, xticks, yticks)
    y1ticks = list(range(0, 12001, 2000))
    y2ticks = [i / 10 for i in range(0, 11, 2)]
    fig3 = plot_tank_fill(data[-1], xticks, y1ticks, y2ticks)
    xticks = [i / 100 for i in range(0, 301, 50)]
    yticks = [i / 100 for i in range(77, 101, 5)]
    fig4 = plot_tank_efficiencies(
        performances, radii, "Radius [m]", xticks, yticks
    )
    fig1.show()