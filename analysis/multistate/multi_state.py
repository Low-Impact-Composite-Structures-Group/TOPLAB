import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux, plot_single_tank_loads, plot_density_vs_temperature
from facades.analysis_facades import MissionAnalysisFacade, OperatingEnvelope, TankDimensions, FillingAnalysisFacade, TargetConditions, InitialConditions, DualTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, MissionSection
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import SphericalTank
from src.fluids.hydrogen_retrievers import SinglePhaseRequester, TwoPhaseRequester
import numpy as np
import os
import yaml

def perform_analysis():
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(script_dir, 'input.yaml')

    # Load configuration from YAML file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters from the YAML file
    tank_name_1 = config.get('tank 1', {}).get('tank name', None)
    radius_1 = config.get('tank 1', {}).get('tank radius', None)
    p_init_1 = config.get('tank 1', {}).get('initial pressure', None)
    t_init_1 = config.get('tank 1', {}).get('initial temperature', None)
    fill_1 = config.get('tank 1', {}).get('fill', None)
    p_max_1 = config.get('tank 1', {}).get('max pressure', None)
    p_min_1 = config.get('tank 1', {}).get('min pressure', None)
    t_min_1 = config.get('tank 1', {}).get('min temperature', None)
    ambient_heat_load_1 = config.get('tank 1', {}).get('ambient heat load', None)

    tank_name_2 = config.get('tank 2', {}).get('tank name', None)
    radius_2 = config.get('tank 2', {}).get('tank radius', None)
    p_init_2 = config.get('tank 2', {}).get('initial pressure', None)
    t_init_2 = config.get('tank 2', {}).get('initial temperature', None)
    fill_2 = config.get('tank 2', {}).get('fill', None)
    p_max_2 = config.get('tank 2', {}).get('max pressure', None)
    p_min_2 = config.get('tank 2', {}).get('min pressure', None)
    t_min_2 = config.get('tank 2', {}).get('min temperature', None)
    ambient_heat_load_2 = config.get('tank 2', {}).get('ambient heat load', None)

    # instantiate the Tank and TankDimensions object
    tank_material = Composite.carbon(np.radians(55))
    tank_1 = SphericalTank(radius_1, tank_material, p_init_1)
    tank_2 = SphericalTank(radius_2, tank_material, p_init_2)
    tank_dimensions_1 = TankDimensions(radius_1, 0.0)
    tank_dimensions_2 = TankDimensions(radius_2, 0.0)

    # Mission and material
    mission_1 = Mission.triathlon()
    mission_2 = Mission.triathlon()
    tank_material = Composite.carbon(np.radians(55))
    # TODO: remove dummy insulation; not used since we are applying constant heat load
    insulation_thickness = 0.001  # [m]
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)


    initial_state_1 = InitialState(p_init_1, t_init_1, fill_1)
    initial_state_2 = InitialState(p_init_2, t_init_2, fill_2)

    operating_window_1 = OperatingEnvelope(p_max_1, p_min_1, t_min_1)
    operating_window_2 = OperatingEnvelope(p_max_2, p_min_2, t_min_2)
    # print(f"Initial State 1: {initial_state_1}")
    # print(f"Initial State 2: {initial_state_2}")
    # print(f"Operating Window 1: {operating_window_1}")
    # print(f"Operating Window 2: {operating_window_2}")

    # Calculate required fuel and split between tanks
    fuel_mass_1 = mission_1.required_fuel
    fuel_mass_2 = mission_2.required_fuel
    print(f"Required fuel mass for tank 1: {fuel_mass_1} kg")
    print(f"Required fuel mass for tank 2: {fuel_mass_2} kg")

    tank_performance = DualTankAnalysisFacade.analyse(
        tank_dimensions_1,
        tank_dimensions_2,
        tank_material,
        insulation,
        mission_1,
        ambient_heat_load_1,
        InitialConditions(p_init_1, t_init_1, fill_1),
        InitialConditions(p_init_2, t_init_2, fill_2),
        operating_window_1,
        operating_window_2,
        TargetConditions(fuel_mass_2, 0.0)
    )
    tank_performance_1 = tank_performance.tank_states[0]
    tank_performance_2 = tank_performance.tank_states[1]
    fig_tank_fill_1 = plot_single_tank_fill(tank_performance_1)
    fig_tank_fill_1.set_title("Tank 1 Fill Level")
    fig_tank_fill_2 = plot_single_tank_fill(tank_performance_2)
    fig_tank_fill_2.set_title("Tank 2 Fill Level")
    fig_tank_temperatures_1 = plot_single_tank_temperatures(tank_performance_1)
    fig_tank_temperatures_1.set_title("Tank 1 Internal Temperature")
    fig_tank_temperatures_2 = plot_single_tank_temperatures(tank_performance_2)
    fig_tank_temperatures_2.set_title("Tank 2 Internal Temperature")
    fig_tank_pressures_1 = plot_single_tank_loads(tank_performance_1)
    fig_tank_pressures_1.set_title("Tank 1 Internal Pressure")
    fig_tank_pressures_2 = plot_single_tank_loads(tank_performance_2)
    fig_tank_pressures_2.set_title("Tank 2 Internal Pressure")

    plt.show()


