from plotting.plot_tank_states import plot_single_required_flux
from facades.analysis_facades import (OperatingEnvelope, TankDimensions, MissionAnalysisFacade)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import cycler
import numpy as np # type: ignore


def radius_from_volume_sphere(volume: float) -> float:
    return (3 * volume / (4 * np.pi)) ** (1/3)

def perform_analysis(constant_heat_flux, results):

    # Define parameters
    radius = 0.714
    insulation_thickness = 0.04
    VOLUME_MARGIN = 1.3
    tank_material_type = 'Composite'
    tank_material_name = 'carbon'
    winding_angle_degrees = 55.0
    mission_name = 'triathlon'
    min_pressure = 2000000
    min_temperature = None
    max_pressure = 45000000
    initial_pressure = 40000000
    initial_temperature = 70.0
    fill = 0.0
    duration = 10 * 3600  # 10 hours in seconds
    phase = 'gas'
    outlet_pressure = 2000000
    outlet_temperature = 200

    # Populate initial state
    initial_state = InitialState(initial_pressure, initial_temperature, fill)

    winding_angle = np.radians(winding_angle_degrees)
    # Instantiate the tank material
    tank_material_class = globals()[tank_material_type]
    tank_material = getattr(tank_material_class, tank_material_name)(winding_angle)

    # Define the mission
    mission_class = getattr(Mission, mission_name)
    mission = mission_class()

    # Define operating window
    operating_window = OperatingEnvelope(max_pressure, min_pressure, min_temperature)

    # Define required fuel
    fuel_mass = mission.required_fuel * VOLUME_MARGIN
    initial_fuel = initial_state.get_hydrogen_properties()

    # Get fuel volume
    fuel_volume = fuel_mass / initial_fuel.density
    # Take the radius of the limiting sphere if the volume is too small for a cylinder + hemi head
    if radius_from_volume_sphere(fuel_volume) <= radius:
        radius = radius_from_volume_sphere(fuel_volume)

    # Instantiate dummy insulation (NOT USED)
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Calculate the tank length based on SE or Hemi head configuration
    # Instantiate the tank dimensions
    length = CylindricalTankSphericalCaps.length_from_radius_and_volume(radius, fuel_volume)
    tank_dimensions = TankDimensions(radius, length)
    total_length = length + 2 * radius
    tank = CylindricalTankSphericalCaps(radius, total_length, tank_material, max_pressure)

    # Access the total surface area of the tank
    total_surface_area = tank.surface_area

    # Perform the mission analysis
    performance = MissionAnalysisFacade.analyse(
        tank_dimensions,
        tank_material,
        insulation, # dummy arg
        mission,
        initial_state,
        operating_window,
        constant_heat_flux
    )

    # Set target enthalpy value from desired output conditions
    outlet_pressure_kpa = outlet_pressure / 1000
    h_out = CP.PropsSI('H', 'P', outlet_pressure_kpa, 'T', outlet_temperature, 'PARAHYDROGEN')

    # extract enthalpies from tank states
    enthalpies = []

    for state in performance.tank_states.states:
        phase = state.hydrogen.phase
        if phase in ["gas", "supercritical"]:
            enthalpy = state.hydrogen.gas.enthalpy
        elif phase in ["liquid", "supercritical_liquid"]:
            enthalpy = state.hydrogen.liquid.enthalpy
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        enthalpies.append(enthalpy)

    # Extract mass_flow and duration from each MissionSection
    mass_flows = []
    for section in mission.sections:
        if isinstance(section.fuel_flows[0].mass_flow, list):
            mass_flows.append([abs(flow) for flow in section.fuel_flows[0].mass_flow])
        else:
            mass_flows.append([abs(section.fuel_flows[0].mass_flow), abs(section.fuel_flows[0].mass_flow)])

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
    ohex_heat = [(h_out - enthalpies[i])/100 for i in range(enthalpies_length)]

    # get ihex heat list and close the intermediate plot
    _, ihex_heat = plot_single_required_flux(performance.tank_states)
    ihex_heat = [ihex * 1000 for ihex in ihex_heat]
    plt.close()

    # Sum up the durations to get the total mission duration
    total_duration = sum(durations_hrs)

    # Collect results
    results.append({
        'constant_heat_flux': constant_heat_flux,
        'combined_heat': [abs(ohex) + abs(ihex) for ohex, ihex in zip(ohex_heat, ihex_heat)],
        'total_duration': total_duration,
        'tank_length': total_length,
        'tank_radius': radius,
        'fuel_volume': fuel_volume,
        'fuel_mass': fuel_mass,
        'initial_state': initial_state,
        'total_surface_area': total_surface_area,
        'min_tank_pressure': min_pressure,
        'outlet_pressure': outlet_pressure,
        'outlet_temperature': outlet_temperature
    })

def main():
    start = 0
    stop = 10000
    num = 10
    heat_flux_values = np.linspace(start, stop, num)
    results = []
    timestep = 10
    for heat_flux in heat_flux_values:
        print(f"Running analysis with constant_heat_flux = {heat_flux} W/m²")
        perform_analysis(heat_flux, results)

    # Print the parameters once before plotting
    if results:
        print(f"Tank length: {results[0]['tank_length']}")
        print(f"Tank radius: {results[0]['tank_radius']}")
        print(f"Fuel volume: {results[0]['fuel_volume']}")
        print(f"Fuel mass: {results[0]['fuel_mass']}")
        print(f"Initial state: {results[0]['initial_state']}")
        print(f"Total surface area of the tank: {results[0]['total_surface_area']} m^2")
        print(f"Minimum tank pressure: {results[0]['min_tank_pressure']} Pa")
        print(f"Outlet pressure: {results[0]['outlet_pressure']} Pa")

    # Set color cycle
    plt.rc('axes', prop_cycle=(cycler('color', plt.cm.viridis(np.linspace(0, 1, len(heat_flux_values))))))

    # Plot results
    plt.figure()
    for result in results:
        time_values = np.arange(len(result['combined_heat'])) * timestep
        plt.plot(time_values, result['combined_heat'], label=f"Heat Flux: {result['constant_heat_flux']:.1f} W/m²")
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.xlabel('Time [s]')
    plt.ylabel('Total required heat by storage system (oHEX + iHEX) [W]')
    plt.title('Sensitivity to ambient heat load')
    plt.show()

if __name__ == "__main__":
    main()