

import matplotlib.pyplot as plt

from facades.analysis_facades import (FillingAnalysisFacade, InitialConditions,
                                      OperationalEnvelope, TankDimensions,
                                      OperationalEnvelope)
from plotting.plot_tank_states import (plot_tank_fill,
                                       plot_thermo_mechanical_loading)
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import MissionSection
from src.mission.fuel_flow import InFlow
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps


def perform_analysis():

    # Define the state of the fuel tank
    pressure = 8e5
    temperature = 50
    fill = 0.0
    initial_conditions = InitialConditions(
        pressure, temperature, fill
    )

    # Define the material of the tank
    winding_angle = 55
    material = Composite.carbon(winding_angle)

    # Define the fuel tank
    tank = CylindricalTankSphericalCaps.ahluwalia(material, pressure)
    tank_radius = tank.radius
    tank_body = tank.body_length
    tank_dimensions = TankDimensions(tank_radius, tank_body)


    # Define the operating envelope of the fuel tank
    max_pressure = 10e5
    target_fill = 0.97
    target_mass = 11.0
    operating_envelope = OperationalEnvelope(
        max_pressure=max_pressure,
        target_fill=target_fill,
        target_mass=target_mass
    )

    # Define insulation and thermodynamic model
    insulation_thickens = 4e-2
    insulation = ConstantFoamInsulation.rohacell(
        insulation_thickens
    )

    duration = 1e4
    mass_flow = 0.01
    altitude = 0
    mach_number = 0.01
    mission_section = MissionSection(
            duration,
            [
                InFlow(
                    mass_flow,
                    SinglePhaseRequester().get_hydrogen_properties(
                        16e5, 22
                    )
                )
            ],
            altitude,
            mach_number
        )
    mission = Mission([mission_section])

    tank_performance = FillingAnalysisFacade.analyse(
        tank_dimensions,
        material,
        insulation,
        mission,
        initial_conditions,
        operating_envelope,
    )

    tank_states = tank_performance.tank_states

    y1ticks = None
    y2ticks = None
    xticks = None
    plot_thermo_mechanical_loading(
        tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    y1ticks = None
    y2ticks = None
    plot_tank_fill(
        tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    plt.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
