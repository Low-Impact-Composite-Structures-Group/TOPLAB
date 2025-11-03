

import numpy
from facades.analysis_facades import (DrainingAnalysisFacade, InitialConditions, MissionAnalysisFacade, OperatingEnvelope,
                                      SwitchPhaseDrainingAnalysis,
                                      TankDimensions)
from plotting.plot_tank_states import (plot_tank_efficiencies, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission_sections import OutFlow

from plotting.figures import Line, SingleFigure


def perform_analysis():

    # Define the initial state of the tank
    pressure = 140e3
    temperature = None
    fill = 0.97
    initial_state = InitialConditions(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 1e-2
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    # raise ValueError("Metal material")
    tank_material = Composite.carbon(numpy.deg2rad(55))

    # Define fuel flow 
    fuel_flow = OutFlow.SMR_cruise("liquid")

    # Define the minimum pressure of the tank
    max_pressure = 3e5
    min_pressure = 1.3e5

    # Define tank dimensions
    body_length = 1
    min_radius = 25
    max_radius = 575
    radius_step = 5
    radii = [i / 100 for i in range(min_radius, max_radius+radius_step, radius_step)]
    print(radii)

    # Perform the analysis
    performances_switch = [
        SwitchPhaseDrainingAnalysis.analyse(
            TankDimensions(
                radius, body_length
            ),
            tank_material,
            insulation,
            fuel_flow.mass_flow,
            initial_state,
            OperatingEnvelope(
                max_pressure,
                min_pressure,
                None
            )
        )
        for radius in radii
    ]

    performances_normal = [
        DrainingAnalysisFacade.analyse(
            TankDimensions(
                radius, body_length
            ),
            tank_material,
            insulation,
            fuel_flow.mass_flow,
            fuel_flow.phase,
            initial_state,
            OperatingEnvelope(
                1e10,
                min_pressure,
                None
            )
        )
        for radius in radii
    ]

    normal_line = Line(
        radii,
        [perf.gravimetric_efficiency for perf in performances_normal],
        "Regular"
    )
    switch_line = Line(
        radii,
        [perf.gravimetric_efficiency for perf in performances_switch],
        "Switch"
    )

    xticks = [i / 100 for i in range(0, 601, 100)]
    yticks = [i / 100 for i in range(90, 101, 2)]
    fig = SingleFigure(
        [normal_line, switch_line],
        "Radius [m]",
        "Gravimetric Efficiency [-]",
        xticks,
        yticks
    )

    fig.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
