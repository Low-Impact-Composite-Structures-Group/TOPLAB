from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_single_tank_fill, plot_tank_loads,
                                       plot_single_tank_loads, plot_single_tank_temperatures, plot_required_flux)
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



def perform_analysis():
    # Record the start time
    start_time = time.time()
    
    
    optimal_radius = 1.0
    optimal_thickness = 0.08
    psi = 2.0
    VOLUME_MARGIN = 1.5
    min_pressure = 1.3e5
    # Define the initial state of the tank
    pressure = 140e3
    temperature = None
    fill = 0.97
    initial_state = InitialState(
        pressure, temperature, fill
    )
    
    winding_angle_degrees = 55
    winding_angle = np.radians(winding_angle_degrees)
    
    # Define fuel flow
    mission = Mission.fly_eco_mission("liquid")
    tank_material = Composite.carbon(winding_angle)


    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()

    # Get fuel volume
    fuel_volume = fuel_mass / initial_fuel.liquid.density
    
    operating_window = OperatingEnvelope(
            None,
            min_pressure,
            None
        )

    optimal_length = WinnefeldTank.length_from_radius_b_and_volume(optimal_radius, VOLUME_MARGIN * fuel_volume, optimal_radius/psi)

    # Perform the analysis with the optimized parameters
    optimal_performance = [MissionAnalysisFacade.analyse(GenericTankDimensions(
            optimal_radius, optimal_length , optimal_radius, optimal_radius/psi),
        tank_material,
        ConstantFoamInsulation.rohacell(optimal_thickness),
        mission,
        initial_state,
        operating_window)]

    plot_tank(optimal_radius, optimal_radius/psi, optimal_length)


    fig_tank_temperatures = plot_single_tank_temperatures(
        optimal_performance[0].tank_states
    )
    fig_tank_fills = plot_single_tank_fill(
        optimal_performance[0].tank_states
    )
    
    # Record the end time
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
    print(f'Optimal radius: {optimal_radius:.4f} m')
    print(f'Optimal b: {optimal_radius/psi:.4f} m')
    print(f'Optimal thickness: {optimal_thickness:.4f} m')
    print(f'Optimal length: {optimal_length:.4f} m')
    print(f'Maximized gravimetric efficiency: {optimal_performance[0].gravimetric_efficiency:.4f} [-]')

    # Show all plots at once
    plt.show()

def main():
    perform_analysis()

if __name__ == "__main__":
    main()

# End
