
from typing import Protocol

import matplotlib.pyplot as plt

from facades.analysis_facades import (InitialConditions, MissionAnalysisFacade,
                                      OperatingEnvelope, TankDimensions)
from plotting.plot_tank_states import (plot_tank_fill,
                                       plot_thermo_mechanical_loading)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps


def perform_analysis():

    # Define the state of the fuel tank
    pressure = 1.4e5
    temperature = None
    fill = 0.95
    initial_conditions = InitialConditions(
        pressure, temperature, fill
    )

    # Define the fuel tank
    tank = CylindricalTankSphericalCaps.rompokos(None, None)
    tank_dimensions = TankDimensions(
        tank.radius, tank.body_length
    )

    # Define the tank material
    winding_angle = 55
    material = Composite.carbon(winding_angle)

    # Define the target conditions
    operating_envelope = OperatingEnvelope(None, None, None)

    # Define insulation and thermodynamic model
    insulation_thickens = 8e-2
    insulation = ConstantFoamInsulation.polyvinylchloride(
        insulation_thickens
    )

    # Define the mission
    mission = Mission.rompokos()

    tank_performance = MissionAnalysisFacade.analyse(
        tank_dimensions,
        material,
        insulation,
        mission,
        initial_conditions,
        operating_envelope
    )

    y1ticks = [i / 10 for i in range(14, 23)]
    y2ticks = [i / 10 for i in range(210, 251, 5)]
    xticks = [i for i in range(0, 25, 4)]

    plot_thermo_mechanical_loading(
        tank_performance.tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    y1ticks = [i for i in range(0, int(10001), int(2e3))]
    y2ticks = [i for i in range(0, 101, 20)]
    plot_tank_fill(
        tank_performance.tank_states,
        xticks,
        y1ticks,
    )
    plt.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
