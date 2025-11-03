
import numpy
from facades.analysis_facades import InitialConditions, MissionAnalysisFacade, OperatingEnvelope, TankDimensions
from src.efficiencies.efficiency_computers import HexagonVolumetricEfficiency
from plotting.plot_tank_states import plot_general_properties
from src.mission.mission import Mission
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.materials.materials import Composite
from src.insulation.foam_insulations import ConstantFoamInsulation
import copy
from src.thermodynamics.tank_states import InitialState


def perform_analysis():

    # Define the number of tanks
    number_of_tanks = [1, 3, 5, 7, 9]

    # Define type of fuel flow
    fuel_phase_flow = "liquid"

    # Define the mission
    mission =  Mission.small_medium_range(fuel_phase_flow)
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
    initial_conditions = InitialConditions.two_phase_initial()
    initial_state = InitialState(**initial_conditions.__dict__)


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
    tank_radii = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
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
            if tank_length is not None else None
            for tank_length, tank_radius in zip(lengths_row, tank_radii)
        ]
        for lengths_row in tank_lengths
    ]
    material = Composite.carbon(numpy.radians(55))
    tanks = [
        [
            CylindricalTankSphericalCaps(
                tank_radius, tank_length, material, initial_conditions.pressure
            )
            if tank_length is not None else None
            for tank_length, tank_radius in zip(lengths_row, tank_radii)
        ]
        for lengths_row in total_tank_lengths
    ]

    # Things required for the analysis
    insulation_thickness = 8e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Analyse mission
    analysis_facade = MissionAnalysisFacade(
        volumetric_efficiency_computer=HexagonVolumetricEfficiency()
    )

    data = [
        [
            analysis_facade.analyse(
                TankDimensions(tank.radius, tank.body_length),
                material,
                insulation,
                miss,
                InitialConditions.two_phase_initial(),
                OperatingEnvelope.none()
            )
            if tank is not None else None
            for tank in row
        ]
        for row, miss in zip(tanks, missions)
    ]


    grav_effs = [
        [
            data_point.gravimetric_efficiency
            if data_point is not None else None
            for data_point in data_row
        ]
        for data_row in data
    ]

    vol_effs = [
        [
            data_point.volumetric_efficiency
            if data_point is not None else None
            for data_point in data_row
        ]
        for data_row in data
    ]

    xlabel = "Tank Radius [m]"
    xticks = [i / 10 for i in range(0, 31, 5)]
    yticks = [i / 10 for i in range(0, 11, 2)]
    labels = [f"{tank} Tank{'s' if tank != 1 else ''}" for tank in number_of_tanks]

    fig1 = plot_general_properties(
        grav_effs,
        labels,
        tank_radii,
        xlabel,
        "Gravimetric Efficiency [-]",
        xticks,
        yticks
    )
    fig2 = plot_general_properties(
        vol_effs,
        labels,
        tank_radii,
        xlabel,
        "Volumetric Efficiency [-]",
        xticks,
        yticks
    )
    fig1.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
