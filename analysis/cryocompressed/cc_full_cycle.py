import os
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_cycle_tank_fill, plot_cycle_tank_temperature, plot_cycle_tank_pressure, plot_cycle_required_flux, plot_cycle_density_vs_temperature
from facades.analysis_facades import OperatingEnvelope, TankDimensions, GenericTankDimensions, MissionAnalysisFacade, FillingAnalysisFacade,DormancyAnalysisFacade, InitialConditions, TargetConditions
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps, WinnefeldTank
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.mission.mission_sections import InFlow, OutFlow, MissionSection

def radius_from_volume_sphere(volume: float) -> float:
    return (3 * volume / (4 * np.pi)) ** (1/3)

def load_config():
    script_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(script_dir, 'input.yaml')
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def perform_discharge_analysis(config):
    
    start_time = time.time()
    
    print('\n*********************************************')
    print("======== Starting discharge analysis ========")
    print('*********************************************\n\n')

     # Extract parameters from the YAML file
    radius = config.get('tank properties', {}).get('tank radius', None)
    insulation_thickness =config.get('tank properties', {}).get('insulation thickness', None)
    VOLUME_MARGIN = config.get('tank properties', {}).get('volume margin', None)
    tank_material_type = config.get('tank properties', {}).get('tank material type', None)
    tank_material_name = config.get('tank properties', {}).get('tank material', None)
    winding_angle_degrees = config.get('tank properties', {}).get('winding angle', None)
    head_type = config.get('tank properties', {}).get('head type', None)
    mission_name = config.get('discharge', {}).get('mission', None)
    fuel_flow = config.get('discharge', {}).get('fuel flow', None)
    min_pressure = config.get('discharge', {}).get('min pressure', None)
    min_temperature = config.get('discharge', {}).get('min temperature', None)
    max_pressure = config.get('discharge', {}).get('max pressure', None)
    initial_pressure = config.get('discharge', {}).get('initial pressure', None)
    initial_temperature = config.get('discharge', {}).get('initial temperature', None)
    fill = config.get('discharge', {}).get('initial fill', None)
    

 # Check for None values and print them
    parameters = {
        'tank radius': radius,
        'insulation thickness': insulation_thickness,
        'volume margin': VOLUME_MARGIN,
        'tank material type': tank_material_type,
        'tank material': tank_material_name,
        'winding angle': winding_angle_degrees,
        'head type': head_type,
        'mission': mission_name,
        'fuel flow': fuel_flow,
        'min pressure': min_pressure,
        'min temperature': min_temperature,
        'max pressure': max_pressure,
        'initial pressure': initial_pressure,
        'initial temperature': initial_temperature,
        'initial fill': fill
    }

    for param, value in parameters.items():
        if value is None:
            print(f"Warning: {param} is None")

    # Populate initial state
    initial_state = InitialState(initial_pressure, initial_temperature, fill)

    winding_angle = np.radians(winding_angle_degrees)
    # Instantiate the tank material
    tank_material_class = globals()[tank_material_type]
    if tank_material_name == 'carbon':
        tank_material = getattr(tank_material_class, tank_material_name)(winding_angle)
    else:
        tank_material = getattr(tank_material_class, tank_material_name)()

    # Define the mission using the string from the input file
    mission = getattr(Mission, mission_name)(fuel_flow)

    # Define operating window
    operating_window = OperatingEnvelope(max_pressure, min_pressure, min_temperature)

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()
    print(f"fuel mass: {fuel_mass}")

    # Get fuel volume
    fuel_volume = fuel_mass / initial_fuel.density * VOLUME_MARGIN
    
    # Take the radius of the limiting sphere if the volume is too small for a cylinder + hemi head
    if radius_from_volume_sphere(fuel_volume) <= radius:
        radius = radius_from_volume_sphere(fuel_volume)
    
    # Instantiate the insulation
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Calculate the tank length based on SE or Hemi head configuration
    # Instantiate the tank dimensions
    if head_type == 'hemi':
        length = CylindricalTankSphericalCaps.length_from_radius_and_volume(radius, fuel_volume)
        tank_dimensions = TankDimensions(radius, length)
    elif head_type == 'se':
        length = WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, 0.5*radius)
        tank_dimensions = GenericTankDimensions(radius,length , radius, 0.5*radius)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")
    
    # Print some parameters
    print(f"Tank length: {length}")
    print(f"Tank radius: {radius}")
    print(f"Fuel volume: {fuel_volume}")
    print(f"Insulation thickness: {insulation_thickness}")
    print(f"Initial state: {initial_state}")
    
    # perform the mission analysis
    performance = MissionAnalysisFacade.analyse(
        tank_dimensions,
        tank_material,
        insulation,
        mission,
        initial_state,
        operating_window
    )
    
    discharge_simulation_time = time.time() - start_time

    return performance, discharge_simulation_time


def perform_refuel_analysis(config):
    
    start_time = time.time()
    print('\n********************************************')
    print("========= Starting refuel analysis =========")
    print('********************************************\n\n')

    # Extract parameters from the YAML file
    radius = config.get('tank properties', {}).get('tank radius', None)
    length = config.get('tank properties', {}).get('tank length', None)
    insulation_thickness =config.get('tank properties', {}).get('insulation thickness', None)
    tank_material_type = config.get('tank properties', {}).get('tank material type', None)
    tank_material_name = config.get('tank properties', {}).get('tank material', None)
    winding_angle_degrees = config.get('tank properties', {}).get('winding angle', None)
    head_type = config.get('tank properties', {}).get('head type', None)
    min_pressure = config.get('refuelling', {}).get('min pressure', None)
    min_temperature = config.get('refuelling', {}).get('min temperature', None)
    max_pressure = config.get('refuelling', {}).get('max pressure', None)
    initial_pressure = config.get('refuelling', {}).get('initial pressure', None)
    initial_temperature = config.get('refuelling', {}).get('initial temperature', None)
    initial_fill = config.get('refuelling', {}).get('initial fill', None)
    mass_flow = config.get('refuelling', {}).get('mass flow rate', None)
    duration = config.get('refuelling', {}).get('duration', None)
    altitude = config.get('refuelling', {}).get('altitude', None)
    mach_number = config.get('refuelling', {}).get('mach number', None)
    target_fill = config.get('refuelling', {}).get('target fill', None)
    target_mass = config.get('refuelling', {}).get('target mass', None) 
    
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
                InFlow(
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
    
    tank_performance = FillingAnalysisFacade.analyse(
        tank_dimensions,
        tank_material,
        insulation,
        mission,
        initial_conditions,
        operating_window,
        target_conditions
    )

    refuelling_simulation_time = time.time() - start_time

    return tank_performance, refuelling_simulation_time


def perform_dormancy_analysis(config):
    start_time = time.time()
    print('\n**********************************************')
    print("========= Starting dormancy analysis =========")
    print('**********************************************\n\n')

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
    
    dormancy_simulation_time = time.time() - start_time

    return tank_performance, dormancy_simulation_time

def extract_densities(tank_performance):
    densities = []
    for state in tank_performance.tank_states.states:
        phase = state.hydrogen.phase
        if phase == 'twophase':
            density = state.hydrogen.two_phase.density
        elif phase in ["gas", "supercritical"]:
            density = state.hydrogen.gas.density
        elif phase in ["liquid", "supercritical_liquid"]:
            density = state.hydrogen.liquid.density
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        densities.append(density)
    return densities

def perform_analysis():
    config = load_config()
    discharge_performance, discharge_simulation_time = perform_discharge_analysis(config)
    refuel_performance, refuelling_simulation_time = perform_refuel_analysis(config)
    dormancy_performance, dormancy_simulation_time = perform_dormancy_analysis(config)
    
    total_simulation_time = discharge_simulation_time + refuelling_simulation_time + dormancy_simulation_time
    
    print('\n********************************************')
    print("======= FULL CYCLE ANALYSIS COMPLETE =======")
    print('********************************************\n\n')
    print(f"Total simulation time: {total_simulation_time:.3f} seconds")
    print(f"Discharge simulation time: {discharge_simulation_time:.3f} seconds")
    print(f"Refuelling simulation time: {refuelling_simulation_time:.3f} seconds")
    print(f"Dormancy simulation time: {dormancy_simulation_time:.3f} seconds")
    
    # Plotting
    discharge_densities = extract_densities(discharge_performance)
    refuel_densities = extract_densities(refuel_performance)
    dormancy_densities = extract_densities(dormancy_performance)
    process_labels = ['Discharge', 'Refuel', 'Dormancy']
    plot_cycle_density_vs_temperature(discharge_performance.tank_states, refuel_performance.tank_states, dormancy_performance.tank_states, discharge_densities, refuel_densities, dormancy_densities, process_labels)
    plot_cycle_tank_temperature(discharge_performance.tank_states, refuel_performance.tank_states, dormancy_performance.tank_states, process_labels)
    plot_cycle_tank_pressure(discharge_performance.tank_states, refuel_performance.tank_states, dormancy_performance.tank_states, process_labels)
    plot_cycle_required_flux(discharge_performance.tank_states, refuel_performance.tank_states, dormancy_performance.tank_states, process_labels)
    plot_cycle_tank_fill(discharge_performance.tank_states, refuel_performance.tank_states, dormancy_performance.tank_states) 
    plt.show()
    
def main():
    perform_analysis()

if __name__ == "__main__":
    main()