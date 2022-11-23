

from analysis.study_tank_geometry import analyse_tank
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
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
    thickness = 4e-2
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define the material of the tank
    winding_angle = 55
    material = Composite.carbon(winding_angle)

    # Define fuel flow
    fuel_flow_phase = "gas"
    fuel_flow = OutFlow.rompokos_cruise(fuel_flow_phase)

    # Define the minimum pressure of the tank
    min_pressure = 10e5

    # Define the levels for the contour plot
    levels = [i / 100 for i in range(26, 35, 1)]

    analyse_tank(
        initial_state,
        insulation,
        material,
        min_pressure,
        fuel_flow.mass_flow,
        fuel_flow.phase,
        levels
    )


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
