
from plotting.plot_geometric_study import plot_geometric_study
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

    # Define the minimum pressure
    min_pressure = 10e5

    # Define insulation and thermal model
    thickness = 4e-2
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    tank_material = Metal.aluminum()

    # Define fuel flow
    fuel_flow = OutFlow.rompokos_cruise("liquid")

    # Define tank dimensions
    radii = [0.5, 2.0]
    body_lengths = [0.5, 3.0]

    tanks = list()
    for radius in radii:
        row = list()
        for body_length  in body_lengths:
            tank_performance = AnalyseCylindricalTank.analyse_tank(
                radius, body_length,
                tank_material,
                insulation,
                fuel_flow.mass_flow,
                fuel_flow.phase,
                initial_state,
                min_pressure
            )
            row.append(tank_performance)
        tanks.append(row)

    plot_geometric_study(
        radii, body_lengths, tanks
    )


def main():
    pass


if __name__ == "__main__":
    main()


# End