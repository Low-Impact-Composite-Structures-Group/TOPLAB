
import matplotlib.pyplot as plt
from src.efficiencies.efficiency_computers import GravimetricEfficiency, VolumetricEfficiency
from plotting.plot_tank_states import plot_general_properties
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
import copy

def perform_analysis():

    # Define the number of tanks
    number_of_tanks = [1, 3, 5, 7, 9]

    # Define type of fuel flow
    fuel_phase_flow = "liquid"

    # Define the mission
    mission =  Mission.large_passenger_aircraft(fuel_phase_flow)
    missions = list()
    for no_of_tanks in number_of_tanks:
        sections = list()
        for section in mission.sections:
            section = copy.deepcopy(section)
            section.fuel_flows[0].mass_flow = (
                section.fuel_flows[0].mass_flow / no_of_tanks
            )
            sections.append(section)
        missions.append(Mission(sections))

    # Define the initial state of the tank
    pressure = 1.4e5
    temperature = None
    fill = 0.97
    initial_state = InitialState(pressure, temperature, fill)

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()
    fuel_volume = fuel_mass / initial_fuel.liquid.density

    # Define tank
    total_tank_volume = 1.2 * fuel_volume
    tank_volumes = [
        total_tank_volume / no_tanks
        for no_tanks in number_of_tanks
    ]
    tank_radii = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    tank_lengths = [
        [
            CylindricalTankSphericalCaps.length_from_radius_and_volume(
                tank_radius, tank_volume
            )
            for tank_radius in tank_radii
        ]
        for tank_volume in tank_volumes
        
    ]
    total_tank_lengths = [
        [
            tank_length + 2 * tank_radius
            for tank_length, tank_radius in zip(lengths_row, tank_radii)
        ]
        for lengths_row in tank_lengths
    ]
    material = Metal.aluminum()
    tanks = [
        [
            CylindricalTankSphericalCaps(
                tank_radius, tank_length, material, pressure
            )
            for tank_length, tank_radius in zip(lengths_row, tank_radii)
        ]
        for lengths_row in total_tank_lengths
    ]

    # Things required for the analysis
    insulation_thickness = 8e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)
    stopping_criteria = []
    target_conditions = TargetState(None, None, None, None)
    timestep = 60
    multistep_method = EulerMethod(timestep)
    dynamic_model_factory = DynamicModelFactory()
    thermal_model = ThermodynamicModel(
        SingleZoneModel(), ForcedConvectionModel(), insulation
    )
    heat_flux_factor = 1

    # Analyse mission
    data = [
        [
            MissionAnalysis.perform_analysis(
                tank,
                initial_state,
                miss,
                stopping_criteria,
                target_conditions,
                multistep_method,
                dynamic_model_factory,
                thermal_model,
                heat_flux_factor
            )
            for tank in row
        ]
        for row, miss in zip(tanks, missions)
    ]
    grav_effs = [
        [
            GravimetricEfficiency(
                tank, insulation, tank_states.first_state, no_tanks
            ).efficiency
            for tank_states, tank in zip(data_row, tank_row)
        ]
        for data_row, tank_row, no_tanks in
        zip(data, tanks, number_of_tanks)
    ]

    vol_effs = [
        [
            VolumetricEfficiency(tank, insulation,no_tanks).efficiency
            for tank in row
        ]
        for row, no_tanks in zip(tanks, number_of_tanks)
    ]

    xlabel = "Tank Radius [m]"
    xticks = [i / 10 for i in range(0, 26, 5)]
    yticks = [i / 10 for i in range(0, 11, 2)]

    fig1 = plot_general_properties(
        grav_effs,
        number_of_tanks,
        tank_radii,
        xlabel,
        "Gravimetric Efficiency [-]",
        xticks,
        yticks
    )
    fig2 = plot_general_properties(
        vol_effs,
        number_of_tanks,
        tank_radii,
        xlabel,
        "Volumetric Efficiency [-]",
        xticks,
        yticks
    )
    fig1.show()


def main():
    pass


if __name__ == "__main__":
    main()


# End
