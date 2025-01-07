from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux)
from plotting.tank_render import plot_tank
from facades.analysis_facades import (DrainingAnalysisFacade, InitialConditions,
                                      OperatingEnvelope, TankDimensions, GenericTankDimensions, MissionAnalysisFacade)
from src.insulation.foam_insulations import ConstantFoamInsulation, VariableFoamInsulation
from src.materials.materials import Composite
from src.mission.mission_sections import OutFlow
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import WinnefeldTank
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
import json
import pickle


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
    radius_min = config.get('minimum radius', None)
    radius_max = config.get('maximum radius', None)
    thickness_min = config.get('minimum insulation thickness', None)
    thickness_max = config.get('maximum insulation thickness', None)
    VOLUME_MARGIN = config.get('volume margin', None)
    psi = config.get('psi', None)
    tank_material_type = config.get('tank material type', None)
    tank_material_name = config.get('tank material', None)
    winding_angle_degrees = config.get('winding angle', None)
    mission_name = config.get('mission', None)
    fuel_flow = config.get('fuel flow', None)
    SAVE_RESULTS = config.get('save results', None)
    PLOT_EXTRA = config.get('plot extra', None)
    RESPONSE_SURFACE = config.get('response surface', None)
    grid_size = config.get('response surface grid size', None)
    RENDER_3D = config.get('render 3D', None)
    min_pressure = config.get('operating floor pressure', None)
    min_temperature = config.get('operating floor temperature', None)
    max_pressure = config.get('operating ceiling pressure', None)
    initial_pressure = config.get('initial pressure', None)
    initial_temperature = config.get('initial temperature', None)
    fill = config.get('fill', None)

    initial_state = InitialState(
        initial_pressure, initial_temperature, fill
    )


    # Calculate derived parameters
    initial_guess = [
        (radius_min + radius_max) / 2,
        (thickness_min + thickness_max) / 2
    ]

    bounds_radius_thickness = [
        (radius_min, radius_max),
        (thickness_min, thickness_max)
    ]

    winding_angle = np.radians(winding_angle_degrees)
        # Instantiate the tank material
    tank_material_class = globals()[tank_material_type]
    if tank_material_name == 'carbon':
        tank_material = getattr(tank_material_class, tank_material_name)(winding_angle)
    else:
        tank_material = getattr(tank_material_class, tank_material_name)()

    # Define the mission using the string from the input file
    mission = getattr(Mission, mission_name)(fuel_flow)
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

    # List to store the results of each iteration
    all_performances = []
    all_tank_states = []
    all_thicknesses = []
    all_radii = []
    all_graveffs = []
    all_vol_effs = []
    optimization_path = []

    # Define the objective function
    def objective_function(params):
        radius, thickness = params
        length = WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, radius/psi)
        insulation = ConstantFoamInsulation.rohacell(thickness)
        print(f"Analyzing radius: {radius:.4f}, thickness: {thickness:.4f}")
        performance = MissionAnalysisFacade.analyse(
            GenericTankDimensions(
                radius, WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, radius/psi), radius, radius/psi),
            tank_material,
            insulation,
            mission,
            initial_state,
            operating_window
        )
        all_performances.append(performance)
        all_tank_states.append(performance.tank_states)
        all_thicknesses.append(thickness)
        all_radii.append(radius)
        all_graveffs.append(performance.gravimetric_efficiency)
        all_vol_effs.append(performance.volumetric_efficiency)
        optimization_path.append((radius, thickness, performance.gravimetric_efficiency))
        print(f"Length = {length:.4f} m")
        print(f"Gravimetric efficiency = {performance.gravimetric_efficiency:.4f}")
        return -performance.gravimetric_efficiency

    # Perform the optimization
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds_radius_thickness)
    optimal_radius, optimal_thickness = result.x
    optimal_length = WinnefeldTank.length_from_radius_b_and_volume(optimal_radius, VOLUME_MARGIN * fuel_volume, optimal_radius/psi)
    print(f'Optimal radius: {optimal_radius:.4f} m')
    print(f'Optimal b: {optimal_radius/psi:.4f} m')
    print(f'Optimal thickness: {optimal_thickness:.4f} m')
    print(f'Optimal length: {optimal_length:.4f} m')
    print(f'Maximized gravimetric efficiency: {-result.fun:.4f}')

    optimal_volumetric_efficiency = [performance.volumetric_efficiency for performance in all_performances]
    print(f"Max volumetric efficiency = {max(optimal_volumetric_efficiency):.4f}")

    # Perform the analysis with the optimized parameters
    optimal_performance = MissionAnalysisFacade.analyse(GenericTankDimensions(
            optimal_radius, WinnefeldTank.length_from_radius_b_and_volume(
            optimal_radius, VOLUME_MARGIN * fuel_volume, optimal_radius/psi), optimal_radius, optimal_radius/psi),
        tank_material,
        ConstantFoamInsulation.rohacell(optimal_thickness),
        mission,
        initial_state,
        operating_window)

    # Listify optimal tank performance
    optimum = [optimal_performance.tank_states]

    # Compute zeta (zeta = r/t) for each radius and thickness
    zeta_values = [all_radii / all_thicknesses for all_radii, all_thicknesses in zip(all_radii, all_thicknesses)]

     # Create plots for tank loads and efficiencies searched in the optimization
    tank_loads_fig = plot_tank_loads(optimum, ["Internal pressure"], None, None)
    etas_fig = plot_tank_efficiencies_scatter(all_performances, zeta_values, "zeta (r/t) [m/m]", None, None)


    if RESPONSE_SURFACE:
        # Create a grid of radius and b values
        radius_values = np.linspace(radius_min, radius_max, grid_size)
        thickness_values = np.linspace(thickness_min, thickness_max, grid_size)
        radius_grid, thickness_grid = np.meshgrid(radius_values, thickness_values)

        # Compute gravimetric efficiency for each combination of radius and b
        gravimetric_efficiency_grid = np.zeros_like(radius_grid)
        volumetric_efficiency_grid = np.zeros_like(radius_grid)
        for i in range(radius_grid.shape[0]):
            for j in range(radius_grid.shape[1]):
                radius = radius_grid[i, j]
                thickness = thickness_grid[i, j]
                print(f"Analyzing radius: {radius:.4f}, thickness: {thickness:.4f}")
                print(f"Case {i * grid_size + j + 1} of {grid_size ** 2}")
                performance = MissionAnalysisFacade.analyse(GenericTankDimensions(
                    radius, WinnefeldTank.length_from_radius_b_and_volume(
                    radius, VOLUME_MARGIN * fuel_volume, radius/psi), radius, radius/psi),
                    tank_material,
                    ConstantFoamInsulation.rohacell(thickness),
                    mission,
                    initial_state,
                    operating_window
                )
                gravimetric_efficiency_grid[i, j] = performance.gravimetric_efficiency
                volumetric_efficiency_grid[i, j] = performance.volumetric_efficiency

        # Plot the surfaces
        fig_ge_surface = plt.figure()
        ax_ge = fig_ge_surface.add_subplot(111, projection='3d')
        ax_ge.plot_surface(radius_grid, thickness_grid, gravimetric_efficiency_grid, cmap='viridis', alpha=0.6)
        
        # Plot the optimization path as blue dots and connect them with arrows
        for k in range(len(optimization_path) - 1):
            r1, t1, ge1 = optimization_path[k]
            r2, t2, ge2 = optimization_path[k + 1]
            ax_ge.scatter(r1, t1, ge1, color='blue') 
            ax_ge.quiver(r1, t1, ge1, r2 - r1, t2 - t1, ge2 - ge1, color='blue', arrow_length_ratio=0.1)  # Arrow
            
        ax_ge.plot([optimal_radius], [optimal_thickness], [-result.fun], marker='x', markersize=6, color='r')

        ax_ge.set_xlabel('Internal tank radius [m]')
        ax_ge.set_ylabel('Insulation thickness [m]')
        ax_ge.set_zlabel('Gravimetric Efficiency')
        ax_ge.set_title('Gravimetric Efficiency Response Surface')

        ax_ge.text2D(0.05, 0.95, f"Optimal Radius = {optimal_radius:.2f} m", transform=ax_ge.transAxes)
        ax_ge.text2D(0.05, 0.90, f"Optimal thickness = {optimal_thickness:.2f} m", transform=ax_ge.transAxes)
        ax_ge.text2D(0.05, 0.85, f"Optimal Gravimetric Efficiency = {-result.fun:.2f}", transform=ax_ge.transAxes)

        fig_ve_surface = plt.figure()
        ax_ve = fig_ve_surface.add_subplot(111, projection='3d')
        ax_ve.plot_surface(radius_grid, thickness_grid, volumetric_efficiency_grid, cmap='viridis')

        ax_ve.set_xlabel('Internal tank radius [m]')
        ax_ve.set_ylabel('Insulation thickness [m]')
        ax_ve.set_zlabel('Volumetric Efficiency')
        ax_ve.set_title('Volumetric Efficiency Response Surface')

        ax_ve.text2D(0.05, 0.95, f"Optimal Radius = {optimal_radius:.2f} m", transform=ax_ve.transAxes)
        ax_ve.text2D(0.05, 0.90, f"Optimal thickness = {optimal_thickness:.2f} m", transform=ax_ve.transAxes)
        ax_ve.text2D(0.05, 0.85, f"Optimal Volumetric Efficiency = {max(optimal_volumetric_efficiency):.2f}", transform=ax_ve.transAxes)

    # Optional 3D render
    if RENDER_3D:
        tank_render = plot_tank(optimal_radius, optimal_radius/psi, optimal_length)

    # Optional plotting
    if PLOT_EXTRA:
        fig_tank_temperatures = plot_single_tank_temperatures(
            optimal_performance[0].tank_states
        )
        fig_tank_fills = plot_single_tank_fill(
            optimal_performance[0].tank_states
        )
       
        fig_req_flux = plot_single_required_flux(optimal_performance.tank_states)

    # Save results if the flag is enabled
    if SAVE_RESULTS:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'results.json')

        results_data = {
            'optimal_radius': optimal_radius,
            'optimal_thickness': optimal_thickness,
            'optimal_length': optimal_length,
            'maximized_gravimetric_efficiency': -result.fun,
            'max_volumetric_efficiency': max(optimal_volumetric_efficiency),
            'zeta_values': zeta_values,
            'volumetric_efficiencies': all_vol_effs,
            'gravimetric_efficiencies': all_graveffs
        }

        tank_loads_fig.savefig(os.path.join(results_dir, "tank_loads_plot.png"), dpi=300, bbox_inches='tight')
        etas_fig.savefig(os.path.join(results_dir, "etas_plot.png"), dpi=300, bbox_inches='tight')

        with open(results_file, 'w') as file:
            json.dump(results_data, file, indent=4)
        if RESPONSE_SURFACE: 
            with open(os.path.join(results_dir, 'fig_ge_surface.pkl'), 'wb') as file:
                pickle.dump(fig_ge_surface, file)
            with open(os.path.join(results_dir, 'fig_ve_surface.pkl'), 'wb') as file:
                pickle.dump(fig_ve_surface, file)
        if RENDER_3D:
            with open(os.path.join(results_dir, 'tank_render.pkl'), 'wb') as file:
                pickle.dump(tank_render, file)

        if PLOT_EXTRA:
            fig_tank_temperatures.savefig(os.path.join(results_dir, "tank_temperatures_plot.png"), dpi=300, bbox_inches='tight')
            fig_tank_fills.savefig(os.path.join(results_dir, "tank_fills_plot.png"), dpi=300, bbox_inches='tight')
            fig_req_flux.savefig(os.path.join(results_dir, "required_flux_plot.png"), dpi=300, bbox_inches='tight')


    # Record the end time
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")



    # Show all plots at once
    plt.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
