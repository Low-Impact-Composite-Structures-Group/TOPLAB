

from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux, plot_single_tank_loads, plot_density_vs_temperature, plot_required_flux)
from plotting.tank_render import plot_tank
from facades.analysis_facades import (OperatingEnvelope, TankDimensions, GenericTankDimensions, MissionAnalysisFacade)
from src.insulation.foam_insulations import ConstantFoamInsulation, VariableFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import WinnefeldTank, CylindricalTankSphericalCaps
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import yaml

# helper function to ensure computed length is not negative
def radius_from_volume_sphere(volume: float) -> float:
    return (3 * volume / (4 * np.pi)) ** (1/3)

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
    insulation_thickness = config.get('insulation thickness', None)
    VOLUME_MARGIN = config.get('volume margin', None)
    tank_material_type = config.get('tank material type', None)
    tank_material_name = config.get('tank material', None)
    winding_angle_degrees = config.get('winding angle', None)
    mission_name = config.get('mission', None)
    fuel_flow = config.get('fuel flow', None)
    min_pressure = config.get('min pressure', None)
    min_temperature = config.get('min temperature', None)
    max_pressure = config.get('max pressure', None)
    initial_pressure = config.get('initial pressure', None)
    initial_temperature = config.get('initial temperature', None)
    fill = config.get('initial fill', None)
    head_type = config.get('head type', None)

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
    
    # Instantiate the SinglePhaseRequester
    requester = SinglePhaseRequester()

    # Listify the densities for plotting
    densities = []
    phases = []
    density_at_min_p_isobar = []
    for state in performance.tank_states.states:
        phase = state.hydrogen.phase
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
    fig_tank_temperatures = plot_single_tank_temperatures(performance.tank_states)
    fig_tanK_pressures = plot_single_tank_loads(performance.tank_states)
    fig_req_flux = plot_single_required_flux(performance.tank_states)
    fig_tank_fill = plot_single_tank_fill(performance.tank_states)
    fig_density_vs_temperature_gas = plot_density_vs_temperature(performance.tank_states, "Discharge", densities, "15 bar isobar", density_at_min_p_isobar)
    plt.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
