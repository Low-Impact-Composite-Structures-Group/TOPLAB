from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures, plot_required_flux)
from plotting.tank_render import (plot_tank)
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


def perform_analysis():

    # Record the start time
    start_time = time.time()
    # Plotting flag
    PLOT_EXTRA = False
    # Create response surface
    RESPONSE_SURFACE = False
    # create 3D render
    RENDER_3D = True

    # Define the initial state of the tank
    pressure = 140e3 # [Pa]
    temperature = None # [K]
    fill = 0.97
    initial_state = InitialState(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-2 # [m]
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    winding_angle = 55.0 # [degrees] placeholder value
    tank_material = Composite.carbon(winding_angle)

    # Define fuel flow
    # fuel_flow = OutFlow.fly_eco_cruise("liquid")
    fuel_flow = "liquid"

    mission = Mission.fly_eco_mission(fuel_flow)

    # Define the minimum pressure of the tank
    min_pressure = 1.3e5 # [Pa]
    min_temperature = None # [K]
    operating_window = OperatingEnvelope(
        max_pressure=None,
        min_pressure=min_pressure,
        min_temperature=min_temperature
    )

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()

    # Get fuel volume
    fuel_volume = fuel_mass / initial_fuel.liquid.density
    VOLUME_MARGIN = 1.15


    # List to store the results of each iteration
    all_performances = []
    all_tank_states = []
    all_b = []
    all_radii = []

    # Define the objective function
    def objective_function(params):
        radius, b = params
        length = WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, b)
        performance = MissionAnalysisFacade.analyse(
            GenericTankDimensions(
                radius, WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, b), radius, radius),
            tank_material,
            insulation,
            mission,
            initial_state,
            operating_window
        )
        all_performances.append(performance)
        all_tank_states.append(performance.tank_states)
        all_b.append(b)
        all_radii.append(radius)
        print(f"Length = {length} m")
        return -performance.gravimetric_efficiency

    # Initial guess for radius and b
    initial_guess = [1.5, 0.5]

    # bounds of values taken by radius and b
    radius_min = 1.0
    radius_max = 2.0
    b_min = 0.5
    b_max = 1.0
    bounds_radius_b=[(radius_min, radius_max), (b_min, b_max)]


    # Perform the optimization
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds_radius_b)
    optimal_radius, optimal_b = result.x
    print(f'Optimal radius: {optimal_radius:.4f} m')
    print(f'Optimal b: {optimal_b:.4f} m')
    print(f'Minimized gravimetric efficiency: {-result.fun:.4f}')

    optimal_volumetric_efficiency = [performance.volumetric_efficiency for performance in all_performances]
    print(f"Max volumetric efficiency = ", max(optimal_volumetric_efficiency))

    # Perform the analysis with the optimized parameters
    optimal_performance =[ MissionAnalysisFacade.analyse(GenericTankDimensions(
            optimal_radius, WinnefeldTank.length_from_radius_b_and_volume(
            optimal_radius, VOLUME_MARGIN * fuel_volume, optimal_b), optimal_radius, optimal_radius),
        tank_material,
        insulation,
        mission,
        initial_state,
        operating_window)]

    # Listify optimal tank performance
    optimum = [performance.tank_states for performance in optimal_performance]

    # compute psi
    # See Winnefeld paper for the definition of psi
    # "Modelling and Designing Cryogenic Hydrogen Tanks for Future Aircraft Applications (2018)"
    psi_values = [all_radii / all_b for all_radii, all_b in zip(all_radii, all_b)]
    labels = [f'{psi} [-]' for psi in psi_values]

    # Create plots for tank loads and efficiencies searched in the optimization
    tank_loads_fig = plot_tank_loads(optimum, labels, None, None)
    etas_fig = plot_tank_efficiencies_scatter(all_performances, psi_values, "psi (c/b) [m/m]", None, None)


    if RESPONSE_SURFACE:
        grid_size = 10
        # Create a grid of radius and b values
        radius_values = np.linspace(radius_min, radius_max, grid_size)
        b_values = np.linspace(b_min, b_max, grid_size)
        radius_grid, b_grid = np.meshgrid(radius_values, b_values)

        # Compute gravimetric efficiency for each combination of radius and b
        gravimetric_efficiency_grid = np.zeros_like(radius_grid)
        volumetric_efficiency_grid = np.zeros_like(radius_grid)
        for i in range(radius_grid.shape[0]):
            for j in range(radius_grid.shape[1]):
                radius = radius_grid[i, j]
                b = b_grid[i, j]
                print(f"Analyzing radius: {radius:.4f}, b: {b:.4f}")
                print(f"Case {i * grid_size + j + 1} of {grid_size ** 2}")
                performance = MissionAnalysisFacade.analyse(GenericTankDimensions(
                    radius, WinnefeldTank.length_from_radius_b_and_volume(
                    radius, VOLUME_MARGIN * fuel_volume, b), radius, radius),
                    tank_material,
                    insulation,
                    mission,
                    initial_state,
                    operating_window
                )
                gravimetric_efficiency_grid[i, j] = performance.gravimetric_efficiency
                volumetric_efficiency_grid[i, j] = performance.volumetric_efficiency

        # Plot the surfaces
        fig_ge_surface = plt.figure()
        ax_ge = fig_ge_surface.add_subplot(111, projection='3d')
        ax_ge.plot_surface(radius_grid, b_grid, gravimetric_efficiency_grid, cmap='viridis')
        ax_ge.plot([optimal_radius], [optimal_b], [-result.fun], marker='o', markersize=5, color='r')

        ax_ge.set_xlabel('Radius')
        ax_ge.set_ylabel('b')
        ax_ge.set_zlabel('Gravimetric Efficiency')
        ax_ge.set_title('Gravimetric Efficiency Response Surface')

        ax_ge.text2D(0.05, 0.95, f"Optimal Radius = {optimal_radius:.2f}", transform=ax_ge.transAxes)
        ax_ge.text2D(0.05, 0.90, f"Optimal b = {optimal_b:.2f}", transform=ax_ge.transAxes)
        ax_ge.text2D(0.05, 0.85, f"Optimal Gravimetric Efficiency = {-result.fun:.2f}", transform=ax_ge.transAxes)

        fig_ve_surface = plt.figure()
        ax_ve = fig_ve_surface.add_subplot(111, projection='3d')
        ax_ve.plot_surface(radius_grid, b_grid, volumetric_efficiency_grid, cmap='viridis')

        ax_ve.set_xlabel('Radius')
        ax_ve.set_ylabel('b')
        ax_ve.set_zlabel('Volumetric Efficiency')
        ax_ve.set_title('Volumetric Efficiency Response Surface')

        ax_ve.text2D(0.05, 0.95, f"Optimal Radius = {optimal_radius:.2f}", transform=ax_ve.transAxes)
        ax_ve.text2D(0.05, 0.90, f"Optimal b = {optimal_b:.2f}", transform=ax_ve.transAxes)
        ax_ve.text2D(0.05, 0.85, f"Optimal Volumetric Efficiency = {max(optimal_volumetric_efficiency):.2f}", transform=ax_ve.transAxes)

     # Show the figures
    tank_loads_fig.show()
    etas_fig.show()





    # Record the end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Optional 3D render
    if RENDER_3D:
        plot_tank(optimal_radius, optimal_b, WinnefeldTank.length_from_radius_b_and_volume(optimal_radius, VOLUME_MARGIN * fuel_volume, optimal_b))

    # Optional plotting
    if PLOT_EXTRA:
        fig_extra = plot_tank_loads(
            all_tank_states,
            labels
        )
        plot_tank_temperatures(
            all_tank_states,
            labels
        )
        plot_tank_fill(
            all_tank_states[-1]
        )

    # Show all plots at once
    plt.show()



def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
