from plotting.plot_tank_states import (plot_tank_fill, plot_tank_loads,
                                       plot_tank_temperatures)
from src.facades.analysis_facades import (MissionAnalysisFacade,
                                          OperatingEnvelope, TankDimensions)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.tank_states import InitialState


def perform_analysis():

    # Define type of fuel flow
    fuel_phase_flow = "liquid"

    # Define the mission
    missions = [
        Mission.regional(fuel_phase_flow),
        Mission.small_medium_range(fuel_phase_flow),
        Mission.large_passenger_aircraft(fuel_phase_flow)
    ]
    labels = ["REG", "SMR", "LPA"]

    # Define the initial state of the tank
    pressure = 1.4e5
    temperature = None
    fill = 0.97
    initial_state = InitialState(pressure, temperature, fill)

    # Define operating window of the pressure vessel
    min_pressure = 1.1e5
    operating_window = OperatingEnvelope(
        max_pressure=None,
        min_pressure=min_pressure,
        min_temperature=None
    )

    # Define required fuel
    fuel_masses = [mission.required_fuel for mission in missions]
    initial_fuel = initial_state.get_hydrogen_properties()
    fuel_volumes = [
        fuel_mass / initial_fuel.liquid.density
        for fuel_mass in fuel_masses
    ]

    # Define tank dimensions, based on the required fuel volume
    VOLUME_MARGIN = 1.15
    tank_volumes = [
        VOLUME_MARGIN * fuel_volume for fuel_volume in fuel_volumes
    ]
    tank_radius = 1.0
    tank_lengths = [
        CylindricalTankSphericalCaps.length_from_radius_and_volume(
            tank_radius, tank_volume
        )
        for tank_volume in tank_volumes
    ]
    tanks_dimensions = [
        TankDimensions(tank_radius, body_length)
        for body_length in tank_lengths
    ]

    # Things required for the analysis
    insulation_thickness = 8e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Define the material of the tank
    winding_angle = 55
    material = Composite.carbon(winding_angle)

    # Analyse mission
    data = [
        MissionAnalysisFacade.analyse(
            tank_dimension,
            material,
            insulation,
            mission,
            initial_state,
            operating_window
        )
        for tank_dimension, mission in zip(tanks_dimensions, missions)
    ]

    time_ticks = list(range(0, 13, 2))
    pressure_ticks = [i/10 for i in range(14, 33, 2)]
    mass_ticks = list(range(0, 50001, 10000))
    fill_ticks = [i/10 for i in range(0, 11, 2)]
    temp_ticks = [i/10 for i in range(215, 251, 5)]
    fig = plot_tank_loads(
        [row.tank_states for row in data],
        labels,
        time_ticks,
        pressure_ticks
    )
    plot_tank_temperatures(
        [row.tank_states for row in data],
        labels,
        time_ticks,
        temp_ticks
    )
    plot_tank_fill(
        data[-1].tank_states, time_ticks, mass_ticks, fill_ticks
    )
    fig.show()



def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
