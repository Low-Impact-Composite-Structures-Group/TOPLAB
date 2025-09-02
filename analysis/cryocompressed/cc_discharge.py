from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux, plot_single_tank_loads, plot_density_vs_temperature, plot_required_flux, plot_mission_mass_flows, plot_heat_flows)
from plotting.tank_render import plot_tank
from facades.analysis_facades import (OperatingEnvelope, TankDimensions, GenericTankDimensions, MissionAnalysisFacade)
from src.insulation.foam_insulations import ConstantFoamInsulation, VariableFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import WinnefeldTank, CylindricalTankSphericalCaps
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_model_factories import DynamicModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.external_models import NaturalConvectionModel
from src.dynamics.stopping_criteria import NoFuelMass, TankIsEmpty
from CoolProp.CoolProp import PropsSI, PhaseSI # type: ignore
import CoolProp.CoolProp as CP # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os
import time
import yaml # type: ignore
import csv
from plotting.sb_plotting import SeabornPlotter

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
    altitude = config.get('discharge', {}).get('altitude', None)
    mach_number = config.get('discharge', {}).get('mach number', None)
    duration = config.get('discharge', {}).get('duration', None)
    phase = config.get('discharge', {}).get('phase', None)
    throttle = config.get('discharge', {}).get('throttle', None)
    mission_type = config.get('discharge', {}).get('mission type', None)
    outlet_pressure = config.get('discharge', {}).get('outlet pressure', None)
    outlet_temperature = config.get('discharge', {}).get('outlet temperature', None)
    constant_heat_flux = config.get('discharge', {}).get('constant heat flux', None)
    timestep = config.get('discharge', {}).get('timestep', 60)  # Default to 60 seconds if not specified

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
        'initial fill': fill,
        'altitude': altitude,
        'mach number': mach_number,
        'duration': duration,
        'phase': phase,
        'mission type': mission_type,
        'outlet pressure': outlet_pressure,
        'outlet temperature': outlet_temperature,
        'constant heat flux': constant_heat_flux,
        'timestep': timestep
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

    # Define the mission
    if mission_type == 0: # if mission is a single section
        mission_section = [Mission.discharge_section(duration, altitude,fuel_flow, throttle, phase, mach_number)]
        mission = Mission(mission_section)
    elif mission_type == 1: # if mission is a mission comprised of multiple sections
        mission_class = getattr(Mission, mission_name)
        mission = mission_class()
    else:
        raise ValueError(f"Unsupported mission type: {mission_type}")

    # Define operating window
    operating_window = OperatingEnvelope(max_pressure, min_pressure, min_temperature)

    # Define required fuel
    fuel_mass = mission.required_fuel*VOLUME_MARGIN
    initial_fuel = initial_state.get_hydrogen_properties()
    print(f"fuel mass: {fuel_mass}")

    # Get fuel volume
    fuel_volume = fuel_mass / initial_fuel.density
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
        total_length = length + 2 * radius
        tank = CylindricalTankSphericalCaps(radius, total_length, tank_material, max_pressure)
    elif head_type == 'se':
        length = WinnefeldTank.length_from_radius_b_and_volume(radius, VOLUME_MARGIN * fuel_volume, 0.5*radius)
        tank_dimensions = GenericTankDimensions(radius,length , radius, 0.5*radius)
        tank = WinnefeldTank(radius, length, radius, 0.5 * radius, tank_material, max_pressure)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")

    # Print some parameters
    print(f"Tank length: {total_length}")
    print(f"Tank radius: {radius}")
    print(f"Fuel volume: {fuel_volume}")
    print(f"Insulation thickness: {insulation_thickness}")
    print(f"Initial state: {initial_state}")

    # Access the surface area of individual sections
    for section in tank.sections:
        print(f"Section type: {type(section).__name__}, Surface area: {section.surface_area} m^2")

    # Access the total surface area of the tank
    total_surface_area = tank.surface_area
    print(f"Total surface area of the tank: {total_surface_area} m^2")
    # Get heat load from area and constant heat flux
    constant_heat_flux = constant_heat_flux * total_surface_area

    # Print timestep being used
    print(f"Using timestep: {timestep} seconds")

    # Create analysis components with custom timestep
    multistep_method = EulerMethod(timestep)
    dynamic_model_factory = DynamicModelFactory()
    thermal_model = ThermodynamicModel(
        SingleZoneModel(),
        NaturalConvectionModel(),
        insulation,
        constant_heat_flux=constant_heat_flux
    )

    # Define stopping criteria and target conditions
    stopping_criteria = [NoFuelMass(), TankIsEmpty()]
    target_conditions = operating_window._define_target_conditions() if hasattr(operating_window, '_define_target_conditions') else None

    # Create target state manually if the method doesn't exist
    if target_conditions is None:
        from src.thermodynamics.tank_states import TargetState
        target_conditions = TargetState(
            max_pressure=max_pressure,
            min_pressure=min_pressure,
            min_temperature=min_temperature,
            fill=None,
            mass=1.0  # Minimum mass limit
        )

    # perform the mission analysis using direct MissionAnalysis call
    tank_states = MissionAnalysis.perform_analysis(
        tank,
        initial_state,
        mission,
        stopping_criteria,
        target_conditions,
        multistep_method,
        dynamic_model_factory,
        thermal_model,
        heat_flux_factor=1.0
    )

    # Create performance object for compatibility with existing code
    from src.efficiencies.tank_performance import TankPerformance
    performance = TankPerformance(tank, insulation, tank_states)

    # set target enthalpy value from desired output conditions
    outlet_pressure_kpa = outlet_pressure / 1000
    h_out = CP.PropsSI('H', 'P', outlet_pressure_kpa, 'T', outlet_temperature, 'PARAHYDROGEN')
    print(f"Enthalpy at {outlet_pressure_kpa} kPa and {outlet_temperature} K: {h_out}")

    # Listify the densities for plotting
    densities = []
    phases = []
    enthalpies = []
    masses = []
    density_at_15_bar = []
    density_at_20_bar = []
    density_at_400_bar = []
    density_at_500_bar = []
    density_lists = [density_at_15_bar, density_at_20_bar, density_at_400_bar, density_at_500_bar]
    pressure_values = [15e5, 20e5, 400e5, 500e5]
    isobar_labels = ["15 bar isobar", "20 bar isobar", "400 bar isobar", "500 bar isobar"]

    for state in performance.tank_states.states:
        phase = state.hydrogen.phase
        if phase in ["gas", "supercritical"]:
            density = state.hydrogen.gas.density
            enthalpy = state.hydrogen.gas.enthalpy
            mass = state.gas_mass
        elif phase in ["liquid", "supercritical_liquid"]:
            density = state.hydrogen.liquid.density
            enthalpy = state.hydrogen.liquid.enthalpy
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        densities.append(density)
        phases.append(phase)
        enthalpies.append(enthalpy)
        masses.append(mass)
        temperature = state.temperature
        for pressure, density_list in zip(pressure_values, density_lists):
            density_at_isobar = SinglePhaseRequester().get_property(pressure, temperature, "D")
            density_list.append(density_at_isobar)

    # Extract mass_flow, fuel_flow_key, and duration from each MissionSection
    mass_flows = []
    for section in mission.sections:
        if isinstance(section.fuel_flows[0].mass_flow, list):
            mass_flows.append([abs(flow) for flow in section.fuel_flows[0].mass_flow])
        else:
            mass_flows.append([abs(section.fuel_flows[0].mass_flow), abs(section.fuel_flows[0].mass_flow)])
    fuel_flow_keys = [section.fuel_flow_key for section in mission.sections]
    durations = [section.duration for section in mission.sections]
    durations_hrs = [duration / 3600 for duration in durations]

    # Determine the length of enthalpies
    enthalpies_length = len(enthalpies)

    # Flatten the list of mass_flows to one dimension
    flattened_mass_flows = [flow for sublist in mass_flows for flow in sublist]

    # Interpolate the mass_flows to match the length of enthalpies
    total_duration = sum(durations)
    interpolated_mass_flows = []
    for i, duration in enumerate(durations):
        num_points = int(enthalpies_length * (duration / total_duration))
        start_flow = flattened_mass_flows[2 * i]
        stop_flow = flattened_mass_flows[2 * i + 1]
        interpolated_mass_flows.extend(np.linspace(start_flow, stop_flow, num_points))

    # Ensure the interpolated_mass_flows has the same length as enthalpies
    if len(interpolated_mass_flows) < enthalpies_length:
        interpolated_mass_flows.extend([interpolated_mass_flows[-1]] * (enthalpies_length - len(interpolated_mass_flows)))
    elif len(interpolated_mass_flows) > enthalpies_length:
        interpolated_mass_flows = interpolated_mass_flows[:enthalpies_length]

    # Create a new list called ohex_heat
    ohex_heat = [(h_out - enthalpies[i])*interpolated_mass_flows[i] for i in range(enthalpies_length)]

    # Sum up the durations to get the total mission duration
    total_duration = sum(durations_hrs)

    # Print the time taken
    print(f"Time taken: {time.time() - start_time}")

    # Plotting
    # Create a SeabornPlotter instance for consistent styling
    plotter = SeabornPlotter(font="Cambria", palette="deep")

    # Create single tank state plots with Seaborn styling
    fig_tank_states = plotter.plot_single_tank_states(performance.tank_states)

    # Plot mission mass flows with Seaborn styling
    fig_mass_flows = plotter.plot_single_mission_flows(mass_flows, fuel_flow_keys, durations, total_duration,
                                               interpolated_mass_flows)

    # Keep your other specialized plots
    fig_req_flux, ihex_heat = plot_single_required_flux(performance.tank_states)
    fig_density_vs_temperature_gas = plot_density_vs_temperature(
        performance.tank_states,
        "Discharge",
        densities,
        isobar_labels,
        density_lists
    )

    # Generate saturation line data
    sat_temps, sat_liquid_densities, sat_vapor_densities = get_hydrogen_saturation_line()

    # Add saturation lines to the existing plot
    ax = fig_density_vs_temperature_gas.axes[0]  # Get the main axes from the figure
    ax.plot(sat_temps, sat_liquid_densities, 'k--', label='Saturated Liquid')
    ax.plot(sat_temps, sat_vapor_densities, 'k--', label='Saturated Vapor')

    # Fill between the saturation lines to indicate two-phase region
    ax.fill_between(sat_temps, sat_liquid_densities, sat_vapor_densities, alpha=0.1, color='gray')

    # Refresh the legend to include new entries
    ax.legend()

    plot_heat_flows(performance.tank_states, ohex_heat)

        # Save some lists to a CSV file
    output_csv_path = os.path.join(script_dir, '../../data/results/cc_results/lists.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the original data
        writer.writerow(['Enthalpies', 'oHEX Heat', 'Mass Flows', 'Masses', 'iHEX Heat', 'Densities'])
        for enthalpy, ohex, mdot, mass, ihex, density in zip(enthalpies, ohex_heat, interpolated_mass_flows, masses, ihex_heat, densities):
            writer.writerow([enthalpy, ohex, mdot, mass, ihex, density])

    # Write saturation data to a separate file
    saturation_csv_path = os.path.join(script_dir, '../../data/results/cc_results/saturation_line.csv')
    with open(saturation_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write saturation line data
        writer.writerow(['Temperature (K)', 'Saturated Liquid Density', 'Saturated Vapor Density'])
        for temp, liq_density, vap_density in zip(sat_temps, sat_liquid_densities, sat_vapor_densities):
            writer.writerow([temp, liq_density, vap_density])

    plt.close(fig_req_flux.fig)

    # save mission mass flow and tank states as png with higher dpi
    fig_mass_flows.savefig(os.path.join(script_dir, '../../data/results/cc_results/mission_mass_flows.png'), dpi=1000)
    fig_tank_states.savefig(os.path.join(script_dir, '../../data/results/cc_results/tank_states.png'), dpi=1000)

    plt.show()

def get_hydrogen_saturation_line(min_temp=14.0, max_temp=33.0, num_points=100):
    """Generate saturation line data for hydrogen."""
    temperatures = np.linspace(min_temp, max_temp, num_points)
    liquid_densities = []
    vapor_densities = []

    requester = SinglePhaseRequester()

    for temp in temperatures:
        try:
            # Get saturated pressure at this temperature
            pressure = PropsSI('P', 'T', temp, 'Q', 0, 'PARAHYDROGEN')

            # Get saturated liquid density
            liquid_density = PropsSI('D', 'T', temp, 'Q', 0, 'PARAHYDROGEN')
            liquid_densities.append(liquid_density)

            # Get saturated vapor density
            vapor_density = PropsSI('D', 'T', temp, 'Q', 1, 'PARAHYDROGEN')
            vapor_densities.append(vapor_density)
        except Exception as e:
            print(f"Error at temperature {temp}K: {e}")
            # Handle errors by continuing with last valid value
            if liquid_densities:
                liquid_densities.append(liquid_densities[-1])
            else:
                liquid_densities.append(None)

            if vapor_densities:
                vapor_densities.append(vapor_densities[-1])
            else:
                vapor_densities.append(None)

    return temperatures, liquid_densities, vapor_densities


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
