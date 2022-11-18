from plotting.plot_tank_states import plot_tank_fill, plot_tank_loads, plot_tank_temperatures
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.dynamics.dynamic_models import DynamicModelFactory
from src.mission.mission import Mission
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.materials.materials import Metal
from src.thermodynamics.tank_states import InitialState, TargetState
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.insulation.foam_insulations import ConstantFoamInsulation

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

    # Define minimum pressure in the case og gas phase draining
    min_pressure = 1.1e5

    # Define required fuel
    fuel_masses = [mission.required_fuel for mission in missions]
    initial_fuel = initial_state.get_hydrogen_properties()
    fuel_volumes = [
        fuel_mass / initial_fuel.liquid.density
        for fuel_mass in fuel_masses
    ]

    # Define tank
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
    total_tank_lengths = [
        tank_length + 2 * tank_radius
        for tank_length in tank_lengths
    ]
    material = Metal.aluminum()
    tanks = [
        CylindricalTankSphericalCaps(
            tank_radius, total_tank_length, material, pressure
        )
        for total_tank_length in total_tank_lengths
    ]

    # Things required for the analysis
    insulation_thickness = 8e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)
    stopping_criteria = []
    target_conditions = TargetState(
        None, min_pressure, None, None, None
    )
    timestep = 60
    multistep_method = EulerMethod(timestep)
    dynamic_model_factory = DynamicModelFactory()
    thermal_model = ThermodynamicModel(
        SingleZoneModel(), ForcedConvectionModel(), insulation
    )
    heat_flux_factor = 1

    # Analyse mission
    data = [
        MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            stopping_criteria,
            target_conditions,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor
        )
        for tank, mission in zip(tanks, missions)
    ]

    time_ticks = list(range(0, 13, 2))
    pressure_ticks = [i/10 for i in range(14, 33, 2)]
    mass_ticks = list(range(0, 50001, 10000))
    fill_ticks = [i/10 for i in range(0, 11, 2)]
    temp_ticks = [i/10 for i in range(215, 251, 5)]
    fig = plot_tank_loads(
        data, labels, time_ticks, pressure_ticks
    )
    plot_tank_temperatures(
        data, labels, time_ticks, temp_ticks
    )
    plot_tank_fill(data[-1], time_ticks, mass_ticks, fill_ticks)
    fig.show()



def main():
    pass


if __name__ == "__main__":
    main()


# End
