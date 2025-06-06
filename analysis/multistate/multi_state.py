import matplotlib.pyplot as plt
from facades.analysis_facades import MissionAnalysisFacade, OperatingEnvelope, TankDimensions
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import SphericalTank
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

    # instantiate the tanks
    tank_material = Composite.carbon(np.radians(55))

    tank_1 = SphericalTank(radius_1, tank_material, p_init_1)
    tank_2 = SphericalTank(radius_2, tank_material, p_init_2)

    # Mission and material
    fuel_flow = "gas"
    mission = Mission.regional(fuel_flow)
    tank_material = Composite.carbon(np.radians(55))
    # TODO: remove dummy insulation; not used since we are applying constant heat load
    insulation_thickness = 0.001  # [m]
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)


    initial_state_1 = InitialState(p_init_1, t_init_1, fill_1)
    initial_state_2 = InitialState(p_init_2, t_init_2, fill_2)

    operating_window_1 = OperatingEnvelope(p_max_1, p_min_1, t_min_1)
    operating_window_2 = OperatingEnvelope(p_max_2, p_min_2, t_min_2)

    # print(f"Tank 1: {tank_1}")
    # print(f"Tank 2: {tank_2}")

    print(f"Initial State 1: {initial_state_1}")
    print(f"Initial State 2: {initial_state_2}")
    print(f"Operating Window 1: {operating_window_1}")
    print(f"Operating Window 2: {operating_window_2}")
    # Calculate required fuel and split between tanks
    # fuel_mass = mission.required_fuel
    # initial_fuel = initial_state_1.get_hydrogen_properties()
    # fuel_volume = fuel_mass / initial_fuel.liquid.density
    # tank_volume = 1.2 * fuel_volume / 2  # Split between 2 tanks
