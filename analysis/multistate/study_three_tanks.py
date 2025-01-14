

import matplotlib.pyplot as plt

from facades.analysis_facades import (FillingAnalysisFacade, InitialConditions,
                                      OperatingEnvelope, TankDimensions, GenericTankDimensions,
                                      TargetConditions)
from plotting.plot_tank_states import (plot_tank_fill,
                                       plot_thermo_mechanical_loading, plot_thermo_mechanical_loading_vs_fill, plot_thermo_mechanical_loading_vs_mass)
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, MissionSection
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps, WinnefeldTank
import numpy as np
import time
import os
import yaml


def perform_analysis():

    # Record the start time
    start_time = time.time()

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Construct the path to the YAML file
    yaml_path = os.path.join(script_dir, 'input.yaml')

    # Load configuration from YAML file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters from the YAML file
    radius = config.get('tank radius', None)
    body_length = config.get('tank body length', None)
    insulation_thickness = config.get('insulation thickness', None)
    winding_angle = config.get('winding angle', None)
    initial_pressure = config.get('initial pressure', None)
    initial_temperature = config.get('initial temperature', None)
    initial_fill = config.get('start fill', None)
    head_type = config.get('head type', None)
    psi = config.get('psi', None)
    target_fill = config.get('target fill', None)
    target_mass = config.get('target mass', None)
    min_pressure = config.get('min pressure', None)
    max_pressure = config.get('max pressure', None)
    min_temperature = config.get('min temperature', None)
    max_temperature = config.get('max temperature', None)
    refuel_duration = config.get('refuel duration', None)
    refuel_rate = config.get('refuel rate', None)
    refuel_altitude = config.get('refuel altitude', None)
    refuel_mach = config.get('refuel mach', None)

    # Assemble initial conditions   
    initial_conditions = InitialConditions(
        pressure=initial_pressure, 
        temperature=initial_temperature, 
        fill=initial_fill
    )
    
    # Define the material of the tank
    material = Composite.carbon(np.radians(winding_angle))
    
    # Define the first tank (the one which is refuelled)
    if (head_type == 'se'): 
        b = radius/psi
        total_length = body_length + 2*b
        tank = WinnefeldTank(radius, total_length, radius, b, material, initial_pressure)
        tank_dimensions = GenericTankDimensions(tank.radius, tank.body_length, tank.a , tank.b)
 
    elif (head_type == 'hemi'):
        total_length = body_length + 2*radius
        tank = CylindricalTankSphericalCaps(radius, total_length, material, initial_pressure)
        tank_dimensions = TankDimensions(tank.radius, tank.body_length)
    else: 
        raise ValueError('Head type not recognised')
        

    # Assemble the target conditions
    target_conditions = TargetConditions(fuel_mass=target_mass, fill=target_fill)

    # Define the operating envelope of the fuel tank
    operating_envelope = OperatingEnvelope(
        max_pressure, min_pressure, min_temperature
    )

    # Define insulation and thermodynamic model
    insulation = ConstantFoamInsulation.rohacell(
        insulation_thickness
    )


    mission_section = MissionSection(
            refuel_duration,
            [
                InFlow(
                    refuel_rate,
                    SinglePhaseRequester().get_hydrogen_properties(
                        16e5, 22
                    )
                )
            ],
            refuel_altitude,
            refuel_mach,
            "Refuelling"
        )
    mission = Mission([mission_section])

    tank_performance = FillingAnalysisFacade.analyse(
        tank_dimensions,
        material,
        insulation,
        mission,
        initial_conditions,
        operating_envelope,
        target_conditions
    )

    tank_states = tank_performance.tank_states

    plot_thermo_mechanical_loading(tank_states)
    plot_tank_fill(tank_states)
    # plot_thermo_mechanical_loading_vs_fill(tank_states)
    plot_thermo_mechanical_loading_vs_mass(tank_states)
    
    plt.show()



def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
