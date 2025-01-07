from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_single_tank_fill, plot_tank_loads,
                                       plot_single_tank_loads, plot_single_tank_temperatures, plot_single_required_flux)
from plotting.tank_render import plot_tank
from facades.analysis_facades import (DrainingAnalysisFacade, InitialConditions,
                                      OperatingEnvelope, TankDimensions, GenericTankDimensions, MissionAnalysisFacade)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission_sections import OutFlow
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import WinnefeldTank
import numpy as np
import time
import os
import json
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def perform_analysis():
    # Record the start time
    start_time = time.time()

    # Manually enter the values for radius and insulation thickness
    radius = 1.03  # [m]
    thickness = 0.08  # [m]

    # Other parameters (these can also be manually entered or read from a configuration file)
    VOLUME_MARGIN = 1.9  # [m^3/m^3]
    psi = 2.0  # [m/m]
    tank_material_type = 'Composite'
    tank_material_name = 'carbon'
    winding_angle_degrees = 55.0  # [degrees]
    mission_name = 'fly_eco_mission'
    fuel_flow = 'gas'
    SAVE_RESULTS = True
    RENDER_3D = True
    min_pressure = 130000  # [Pa]
    min_temperature = None  # [K]
    max_pressure = None  # [Pa]
    initial_pressure = 140000  # [Pa]
    initial_temperature = None  # [K]
    fill = 0.97

    initial_state = InitialState(
        initial_pressure, initial_temperature, fill
    )

    winding_angle = np.radians(winding_angle_degrees)
    # Instantiate the tank material
    tank_material_class = globals()[tank_material_type]
    if tank_material_name == 'carbon':
        tank_material = getattr(tank_material_class, tank_material_name)(winding_angle)
    else:
        tank_material = getattr(tank_material_class, tank_material_name)()

    # Define the mission using the string from the input file
    mission = getattr(Mission, mission_name)(fuel_flow)

    operating_window = OperatingEnvelope(
        max_pressure=max_pressure,
        min_pressure=min_pressure,
        min_temperature=min_temperature
    )

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()

    # Get fuel volume
    fuel_volume = fuel_mass / initial_fuel.liquid.density

    length = WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, radius/psi)
    insulation = ConstantFoamInsulation.rohacell(thickness)
    print(f"Analyzing radius: {radius:.4f}, thickness: {thickness:.4f}")
    performance = MissionAnalysisFacade.analyse(
        GenericTankDimensions(
            radius, length, radius, radius/psi),
        tank_material,
        insulation,
        mission,
        initial_state,
        operating_window
    )
    print(f"Length = {length:.4f} m")
    print(f"Gravimetric efficiency = {performance.gravimetric_efficiency:.4f}")
    print(f"Volumetric efficiency = {performance.volumetric_efficiency:.4f}")

    # Listify optimal tank performance
    optimum = [performance.tank_states]

    # Compute zeta (zeta = r/t)
    zeta_value = radius / thickness
    label = f'{zeta_value} [m/m]'

    # Create plots for tank loads and efficiencies
    tank_loads_fig = plot_tank_loads(optimum, [label], None, None)

    # Optional 3D render
    if RENDER_3D:
        tank_render = plot_tank(radius, radius/psi, length)


    fig_tank_temperatures = plot_single_tank_temperatures(
        performance.tank_states
    )
    fig_tank_fills = plot_single_tank_fill(
        performance.tank_states
    )
    
    fig_req_flux = plot_single_required_flux(performance.tank_states)

    # Save results if the flag is enabled
    if SAVE_RESULTS:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'check_results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'results.json')

        results_data = {
            'radius': radius,
            'thickness': thickness,
            'length': length,
            'gravimetric_efficiency': performance.gravimetric_efficiency,
            'volumetric_efficiency': performance.volumetric_efficiency,
            'zeta_value': zeta_value
        }

        tank_loads_fig.savefig(os.path.join(results_dir, "tank_loads_plot.png"), dpi=300, bbox_inches='tight')

        with open(results_file, 'w') as file:
            json.dump(results_data, file, indent=4)
        if RENDER_3D:
            with open(os.path.join(results_dir, 'tank_render.pkl'), 'wb') as file:
                pickle.dump(tank_render, file)
 
        fig_tank_temperatures.savefig(os.path.join(results_dir, "tank_temperatures_plot.png"), dpi=300, bbox_inches='tight')
        fig_tank_fills.savefig(os.path.join(results_dir, "tank_fills_plot.png"), dpi=300, bbox_inches='tight')
        fig_req_flux.savefig(os.path.join(results_dir, "required_flux_plot.png"), dpi=300, bbox_inches='tight')

    # Record the end time
    end_time = time.time()
    print(f"Check completed in {end_time - start_time:.2f} seconds")

    # Show all plots at once
    plt.show()

def main():
    perform_analysis()

if __name__ == "__main__":
    main()