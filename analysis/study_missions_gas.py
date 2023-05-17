import math
from plotting.plot_tank_states import (plot_required_flux, plot_tank_fill, plot_tank_loads,
                                       plot_tank_temperatures)
from facades.analysis_facades import (MissionAnalysisFacade,
                                          OperatingEnvelope, TankDimensions)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.tank_states import InitialState

def radius_from_volume_sphere(volume: float) -> float:
    return (3 * volume / (4 * math.pi)) ** (1/3)


def perform_analysis():

    # Define type of fuel flow
    fuel_phase_flow = "gas"

    # Define the mission
    missions = [
        Mission.regional(fuel_phase_flow),
        Mission.small_medium_range(fuel_phase_flow),
        Mission.large_passenger_aircraft(fuel_phase_flow)
    ]
    labels = ["REG", "SMR", "LPA"]

    # Define the initial state of the tank
    pressure = 300e5
    temperature = 70
    fill = 0.0
    initial_state = InitialState(pressure, temperature, fill)

    # Define operating window of the pressure vessel
    min_temperature = 40
    min_pressure = 15e5
    min_temperature = None
    operating_window = OperatingEnvelope(
        max_pressure=None,
        min_pressure=min_pressure,
        min_temperature=min_temperature
    )

    # Define required fuel
    fuel_masses = [mission.required_fuel for mission in missions]
    initial_fuel = initial_state.get_hydrogen_properties()
    VOLUME_MARGIN = 1.15
    fuel_volumes = [
        fuel_mass / initial_fuel.density * VOLUME_MARGIN
        for fuel_mass in fuel_masses
    ]

    # Define tank dimensions, based on the required fuel volume
    fuselage_radii = [
        2.8, 4.1, 5.6
    ]
    tank_radii = [
        radius_from_volume_sphere(fuel_volume)
        if radius_from_volume_sphere(fuel_volume) <= fuselage_radius
        else fuselage_radius
        for fuselage_radius, fuel_volume
        in zip(fuselage_radii, fuel_volumes)
    ]
    tanks_dimensions = [
        TankDimensions(
            tank_radius,
            CylindricalTankSphericalCaps.length_from_radius_and_volume(
                tank_radius, fuel_volume
            )
        )
        for fuel_volume, tank_radius in zip(fuel_volumes, tank_radii)
    ]

    # Things required for the analysis
    insulation_thickness = 4e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Define the material of the tank
    winding_angle = 55
    material = Composite.carbon(math.radians(winding_angle))

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
    for mission, label in zip(data, labels):
        print("Mission: ", label)
        print("Gravimetric Efficiency\t:", mission.gravimetric_efficiency)
        print("Volumetric Efficiency\t:", mission.volumetric_efficiency)

    time_ticks = list(range(0, 13, 2))
    pressure_ticks = [i for i in range(0, 401, 50)]
    temp_ticks = [i for i in range(0, 101, 20)]
    flux_ticks = [i/100 for i in range(0, 101, 10)]
    mass_ticks = list(range(0, 50001, 10000))
    fill_ticks = [i/10 for i in range(0, 11, 2)]
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
    plot_required_flux(
        [row.tank_states for row in data],
        labels,
        time_ticks,
        flux_ticks
    )
    fig.show()



def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
