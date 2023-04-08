from facades.analysis_facades import InitialConditions, MissionAnalysisFacade, OperatingEnvelope
from plotting.plot_tank_states import plot_general_properties
from src.mission.mission import Mission
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.materials.materials import Composite, Metal
from src.thermodynamics.tank_states import InitialState
from src.insulation.foam_insulations import ConstantFoamInsulation
import copy

import numpy

def perform_analysis():

    # Define the number of tanks
    number_of_tanks = numpy.arange(1, 10, 2)

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
    initial_conditions = InitialConditions(
        pressure, temperature, fill
    )
    initial_state = InitialState(pressure, temperature, fill)
    operating_envelope = OperatingEnvelope(None, None, None)

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()
    fuel_volume = fuel_mass / initial_fuel.liquid.density

    # Define tank
    total_tank_volume = 1.1 * fuel_volume
    tank_volumes = [
        total_tank_volume / no_tanks
        for no_tanks in number_of_tanks
    ]
    tank_radii = numpy.arange(0.25, 4.0, 0.25)
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
    material = Composite.carbon(numpy.deg2rad(55))
    tanks = [
        [
            
            CylindricalTankSphericalCaps(
                tank_radius, tank_length, material, pressure
            ) if tank_length - 2 * tank_radius > 0 else None
            for tank_length, tank_radius in zip(lengths_row, tank_radii)
            
        ]
        for lengths_row in total_tank_lengths
    ]

    # Things required for the analysis
    insulation_thickness = 8e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Analyse mission
    data = [
        [
            MissionAnalysisFacade.analyse(
                tank,
                material,
                insulation,
                miss,
                initial_conditions,
                operating_envelope
            ) if tank is not None else None
            for tank in row
        ]
        for row, miss in zip(tanks, missions)
    ]

    # fig = plot_general_properties(
    #     [
    #         row
    #         for row in tank_lengths
    #     ],
    #     [
    #         f"{n} tank" if n == 1 else f"{n} Tanks"
    #         for n in number_of_tanks
            
    #     ],
    #     tank_radii,
    #     "Tank Radius [m]",
    #     "Tank Body Length [m]"
    # )
    # fig = plot_general_properties(
    #     [
    #         [
    #             point.tank.operating_pressure * 1e-5
    #             if point is not None else None
    #             for point in row
    #         ]
    #         for row in data
    #     ],
    #     [
    #         f"{n} tank" if n == 1 else f"{n} Tanks"
    #         for n in number_of_tanks
            
    #     ],
    #     tank_radii,
    #     "Tank Radius [m]",
    #     "Max Pressure [kPa]"
    # )
    fig = plot_general_properties(
        [
            [
                point.gravimetric_efficiency
                if point is not None else None
                for point in row
            ]
            for row in data
        ],
        [
            f"{n} Tank" if n == 1 else f"{n} Tanks"
            for n in number_of_tanks
            
        ],
        tank_radii,
        "Tank Radius [m]",
        "Gravimetric Efficiency [-]",
        numpy.arange(0, 4.01, 0.5),
        numpy.arange(0.5, 1.01, 0.1)
    )
    fig = plot_general_properties(
        [
            [
                point.volumetric_efficiency
                if point is not None else None
                for point in row
            ]
            for row in data
        ],
        [
            f"{n} Tank" if n == 1 else f"{n} Tanks"
            for n in number_of_tanks
            
        ],
        tank_radii,
        "Tank Radius [m]",
        "Volumetric Efficiency [-]",
        numpy.arange(0, 4.01, 0.5),
        numpy.arange(0.5, 1.01, 0.1)
    )
    # fig = plot_general_properties(
    #     [
    #         [
    #             point.tank.structural_mass
    #             if point is not None else None
    #             for point in data[0]
    #         ],
    #         [
    #             point.tank.surface_area * insulation_thickness * insulation.density
    #             if point is not None else None
    #             for point in data[0]
    #         ],[
    #             point.tank_states.total_masses[0]
    #             if point is not None else None
    #             for point in data[0]
    #         ]
    #     ],
    #     [
    #         "Structural", "Insulation", "Fuel"
    #     ],
    #     tank_radii,
    #     "Tank Radius [m]",
    #     "Masses"
    # )
    # fig = plot_general_properties(
    #     [
    #         [
    #             point.tank.structural_mass
    #             if point is not None else None
    #             for point in data[2]
    #         ],
    #         [
    #             point.tank.surface_area * insulation_thickness * insulation.density
    #             if point is not None else None
    #             for point in data[2]
    #         ],[
    #             point.tank_states.total_masses[0]
    #             if point is not None else None
    #             for point in data[2]
    #         ]
    #     ],
    #     [
    #         "Structural", "Insulation", "Fuel"
    #     ],
    #     tank_radii,
    #     "Tank Radius [m]",
    #     "Masses"
    # )
    fig.show()