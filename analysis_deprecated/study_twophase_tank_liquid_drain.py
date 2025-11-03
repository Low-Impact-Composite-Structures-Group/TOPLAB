import numpy

from analysis_deprecated.study_tank_geometry_parallel import analyse_tank
from src.insulation.foam_insulations import VariableFoamInsulation
from src.materials.materials import Composite, Metal
from src.mission.mission_sections import OutFlow
from facades.analysis_facades import InitialConditions
from src.thermodynamics.tank_states import InitialConditions


def perform_analysis():

    # Directory where the results are to be saved
    directory = "twophase_liquid_drain"

    # Define the initial state of the tank
    pressure = 1.5e5
    temperature = None
    fill = 0.97
    initial_state = InitialConditions(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-2
    insulation = VariableFoamInsulation.rohacell(thickness)

    # Define the material of the tank
    winding_angle = numpy.deg2rad(55)
    material = Composite.carbon(winding_angle)
    # material = Metal.aluminum()

    # Define fuel flow
    fuel_flow_phase = "liquid"
    fuel_flow = OutFlow.SMR_cruise(fuel_flow_phase)
    fuel_flow = OutFlow.rompokos_cruise(fuel_flow_phase)

    # Define the minimum pressure of the tank
    # min_pressure = 1.2e5
    min_pressure = None

    # Define the levels for the contour plot
    levels = [i / 100 for i in range(50, 96, 5)]
    levels = [i / 100 for i in range(72, 99, 2)]

    for no_of_tanks in [1]:
        performances = analyse_tank(
            initial_state,
            insulation,
            material,
            min_pressure,
            fuel_flow.mass_flow / no_of_tanks,
            fuel_flow.phase,
            levels,
            directory,
            number_of_tanks=no_of_tanks
        )

def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
