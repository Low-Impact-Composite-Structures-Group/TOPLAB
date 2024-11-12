from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures, plot_required_flux)
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
from scipy.optimize import minimize


def perform_analysis():

    # Record the start time
    start_time = time.time()
    # Plotting flag
    PLOT_ALL = False

    # Define the initial state of the tank
    pressure = 140e3 # [Pa]
    temperature = None
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
    min_temperature = None
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
        return -performance.gravimetric_efficiency

    # Initial guess for radius and b
    initial_guess = [0.1, 0.3]

    # bounds of values taken by radius and b
    bounds_radius_b=[(0.1, 0.5), (0.1, 1.0)]

    # Perform the optimization
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds_radius_b)
    optimal_radius, optimal_b = result.x
    print(f'Optimal radius: {optimal_radius}')
    print(f'Optimal b: {optimal_b}')
    print(f'Minimized gravimetric efficiency: {-result.fun}')

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

    # Show the figures
    tank_loads_fig.show()
    etas_fig.show()

    # Record the end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


    # Optional plotting
    if (PLOT_ALL):
        fig = plot_tank_loads(
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

        # Show the figure
        fig.show()




def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
