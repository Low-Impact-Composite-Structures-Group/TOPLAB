

from analysis.study_tank_geometry import analyse_tank
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.thermodynamics.tank_states import InitialState


def perform_analysis():

    # Define the initial state of the tank
    pressure = 1.4e5
    temperature = None
    fill = 0.97
    initial_state = InitialState(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-2
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define fuel flow
    fuel_flow_phase = "gas"

    # Define the minimum pressure of the tank
    min_pressure = 1.4e5

    # Define the levels for the contour plot
    levels = [i / 100 for i in range(82, 101, 2)]

    analyse_tank(
        initial_state,
        insulation,
        min_pressure,
        fuel_flow_phase,
        levels
    )