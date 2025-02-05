from plotting.plot_tank_states import plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux, plot_single_tank_loads, plot_density_vs_temperature
from facades.analysis_facades import DormancyAnalysisFacade, InitialConditions, OperatingEnvelope, TankDimensions, TargetConditions
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.mission.mission_sections import MissionSection, OutFlow
import matplotlib.pyplot as plt
import numpy as np
import os
import time
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
    radius = config.get('tank properties', {}).get('tank radius', None)
    length = config.get('tank properties', {}).get('tank length', None)
    insulation_thickness =config.get('tank properties', {}).get('insulation thickness', None)
    tank_material_type = config.get('tank properties', {}).get('tank material type', None)
    tank_material_name = config.get('tank properties', {}).get('tank material', None)
    winding_angle_degrees = config.get('tank properties', {}).get('winding angle', None)
    head_type = config.get('tank properties', {}).get('head type', None)
    min_pressure = config.get('dormancy', {}).get('min pressure', None)
    min_temperature = config.get('dormancy', {}).get('min temperature', None)
    max_pressure = config.get('dormancy', {}).get('max pressure', None)
    initial_pressure = config.get('dormancy', {}).get('initial pressure', None)
    initial_temperature = config.get('dormancy', {}).get('initial temperature', None)
    initial_fill = config.get('dormancy', {}).get('initial fill', None)
    mass_flow = config.get('dormancy', {}).get('venting flow rate', None)
    duration = config.get('dormancy', {}).get('duration', None)
    altitude = config.get('dormancy', {}).get('altitude', None)
    mach_number = config.get('dormancy', {}).get('mach number', None)
    target_fill = config.get('dormancy', {}).get('target fill', None)
    target_mass = config.get('dormancy', {}).get('target mass', None) 
    
    # Check for None values and print them
    parameters = {
        'tank radius': radius,
        'insulation thickness': insulation_thickness,
        'tank material type': tank_material_type,
        'tank material': tank_material_name,
        'winding angle': winding_angle_degrees,
        'head type': head_type,
        'min pressure': min_pressure,
        'min temperature': min_temperature,
        'max pressure': max_pressure,
        'initial pressure': initial_pressure,
        'initial temperature': initial_temperature,
        'initial fill': initial_fill,
        'mass flow rate': mass_flow,
        'duration': duration,
        'altitude': altitude,
        'mach number': mach_number,
        'target fill': target_fill,
        'target mass': target_mass
    }

    for param, value in parameters.items():
        if value is None:
            print(f"Warning: {param} is None")
    
    initial_conditions = InitialConditions(
        initial_pressure, initial_temperature, initial_fill
    )

    winding_angle = np.radians(winding_angle_degrees)
    # Instantiate the tank material
    tank_material_class = globals()[tank_material_type]
    if tank_material_name == 'carbon':
        tank_material = getattr(tank_material_class, tank_material_name)(winding_angle)
    else:
        tank_material = getattr(tank_material_class, tank_material_name)()
    
    # Define the fuel tank
    tank = CylindricalTankSphericalCaps(radius, length, tank_material, initial_pressure)
    tank_body = tank.body_length
    tank_dimensions = TankDimensions(radius, tank_body)

    # Define the target conditions
    target_conditions = TargetConditions(target_mass, target_fill)
    
    # Define operating window
    operating_window = OperatingEnvelope(max_pressure, min_pressure, min_temperature)
    
    # Instantiate the insulation
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    mission_section = MissionSection(
            duration,
            [
                OutFlow(
                    mass_flow,
                    SinglePhaseRequester().get_hydrogen_properties(
                        initial_pressure, initial_temperature
                    )
                )
            ],
            altitude,
            mach_number
        )
    mission = Mission([mission_section])
    
    tank_performance = DormancyAnalysisFacade.analyse(
        tank_dimensions,
        tank_material,
        insulation,
        mission,
        initial_conditions,
        operating_window,
        target_conditions
    )

    # Print some parameters
    print(f"Tank length: {length}")
    print(f"Tank radius: {radius}")
    print(f"Insulation thickness: {insulation_thickness}")

    # Listify the densities for plotting
    densities = []
    phases = []
    density_at_min_p_isobar = []
    for state in tank_performance.tank_states.states:
        phase = state.hydrogen.phase
        if phase == 'twophase':
            density = state.hydrogen.two_phase.density
        if phase in ["gas", "supercritical"]:
            density = state.hydrogen.gas.density
        elif phase in ["liquid", "supercritical_liquid"]:
            density = state.hydrogen.liquid.density
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        
        densities.append(density)
        phases.append(phase)
        temperature = state.temperature
        density_at_isobar = SinglePhaseRequester().get_property(min_pressure, temperature, "D")
        density_at_min_p_isobar.append(density_at_isobar)
      
    # Print the time taken
    print(f"Time taken: {time.time() - start_time}")
    
    # Plotting
    fig_tank_temperatures = plot_single_tank_temperatures(tank_performance.tank_states)
    fig_tanK_pressures = plot_single_tank_loads(tank_performance.tank_states)
    fig_req_flux = plot_single_required_flux(tank_performance.tank_states)
    fig_tank_fill = plot_single_tank_fill(tank_performance.tank_states)
    fig_density_vs_temperature_gas = plot_density_vs_temperature(tank_performance.tank_states, "Dormancy", densities, "15 bar isobar", density_at_min_p_isobar)
    plt.show()

def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
