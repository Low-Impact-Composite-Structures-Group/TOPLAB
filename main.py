

from analysis.draining_analysis import perform_draining_analysis
from analysis.recreate_rompokos_mission import perform_analysis
from plotting.plot_tank_states import plot_tank_loads
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.mission.mission_sections import OutFlow
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.tank_states import InitialState


def main():

    # Define tank
    tank = CylindricalTankSphericalCaps.rompokos()
    
    # Define insulation
    thickness = 4e-2
    insulation = ConstantFoamInsulation.polyvinylchloride(thickness)

    # Define initial state of tank
    initial_pressure = 1.4e5
    initial_temperature = None
    initial_fill = 0.95
    initial_state = InitialState(
        initial_pressure, initial_temperature, initial_fill
    )

    # Define fuel flow
    fuel_flow = OutFlow.rompokos_cruise("liquid")

    tank_states = perform_draining_analysis(
        tank,
        initial_state,
        fuel_flow,
        insulation
    )
    fig = plot_tank_loads(
        [tank_states],
        ["Test Analysis"]
    )
    print(max(tank_states, key=lambda x: x.pressure))
    fig.show()



if __name__ == "__main__":
    main()


# End
