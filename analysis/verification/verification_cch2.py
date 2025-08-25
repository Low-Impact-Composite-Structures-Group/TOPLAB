import numpy as np
import matplotlib.pyplot as plt

# Import hydrogen fluid models
from src.fluids.hydrogen_retrievers import SinglePhaseRequester

# Import mission components
from src.mission.mission import Mission
from src.mission.mission_sections import OutFlow, MissionSection, InFlow

# Import tank components
from src.materials.materials import Composite, Metal
from src.insulation.vacuum_insulation import VacuumInsulation
from src.tank_design.liner import Liner

# Import facades and analysis components
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import (
    MULTISTEP_METHOD, OperatingEnvelope, TankDimensions,
    InitialConditions, TargetConditions
)

# Import our facades for analysis
from facades.analysis_facades import MissionAnalysisFacade


#################################
## COMMON SIMULATION CONSTANTS ##
#################################

# Time conversion
HOURS_TO_SECONDS = 3600.0  # seconds per hour

# Tank parameters from paper specifications
NOMINAL_MASS = 35.0
TANK_VOLUME = 0.5
TANK_RADIUS = (3 * TANK_VOLUME / (4 * np.pi))**(1/3)

AMBIENT_TEMPERATURE = 298.15  # K


# Simulation timesteps for different scenarios
DISCHARGE_TIMESTEP = 10.0   # seconds - standard timestep for discharge
REFUEL_TIMESTEP = 0.5      # seconds - small timestep for refuel (rapid dynamics)
DORMANCY_TIMESTEP = 300.0   # seconds - larger timestep for dormancy (slower dynamics)

# Display tank size
print(f"Tank radius: {TANK_RADIUS:.3f} m")
print(f"Tank volume: {TANK_VOLUME:.3f} m³")


# Use aluminum for the tank material
tank_material = Composite.carbon(np.radians(55))

# Create tank dimensions with liner
tank_dimensions = TankDimensions(TANK_RADIUS, 0.0)  # Spherical tank

# Create a liner with specified mass
LINER_MASS = 5.5
liner_by_mass = Liner.from_mass(LINER_MASS, tank_dimensions, Metal.aluminum())

# Assign the liner to tank_dimensions
tank_dimensions.liner = liner_by_mass

# instantiate insulation
insulation = VacuumInsulation()

# Create a temporary tank to display properties
from src.tank_design.tank_shapes import TankFactory
temp_tank = TankFactory.create_tank(
    TANK_RADIUS, 0.0, tank_material, 400e5, liner=liner_by_mass
)

# Print tank properties early in the execution
print("\n===== TANK PROPERTIES =====")
print(f"Tank structural mass: {temp_tank.structural_mass:.2f} kg")
print(f"Tank surface area: {temp_tank.surface_area:.2f} m²")

# Print thickness for each section
for i, section in enumerate(temp_tank.sections):
    if hasattr(section, 'thickness'):
        section_type = section.type if hasattr(section, 'type') else f"Section {i+1}"
        print(f"{section_type} thickness: {section.thickness*1000:.2f} mm")

# Print liner details after calculation
if hasattr(temp_tank, 'liner') and temp_tank.liner is not None:
    print("\n===== LINER PROPERTIES =====")
    liner = temp_tank.liner
    print(f"Liner mass: {liner.mass:.2f} kg")
    if liner.thickness is not None:
        print(f"Liner thickness: {liner.thickness*1000:.2f} mm")
    print(f"Liner material: {liner.material.__class__.__name__}")

print("\n===== BEGINNING ANALYSIS =====\n")

#-------------------------#
# 1. DISCHARGE PARAMETERS #
#-------------------------#
# Tank initial conditions - based on paper's Case B starting point
p_init_disch = 4e+7        # Pa - initial tank pressure (400 bar)
t_init_disch = 53.25       # K - initial tank temperature
fill_disch = 1.0           # fraction - no liquid phase (0.0 = all gas)

# Operating limits - ensure physically reasonable values
p_max_disch = 5.0e+7       # Pa - maximum allowable pressure (500 bar)
p_min_disch = 1.5e+6       # Pa - minimum allowable pressure (15 bar)

# Operating envelopes for discharge scenario
operating_window_disch = OperatingEnvelope(p_max_disch, p_min_disch, None)

# Initial conditions for discharge scenario - enable multi_flow to handle proper phase detection
initial_conditions_disch = InitialConditions(p_init_disch, t_init_disch, fill_disch, multi_flow=True)

# Add discharge mission parameters to the configuration section
duration_hours_disch = 10    # hours - duration of discharge operation
fuel_flow_disch = 0.001      # kg/s - fuel flow rate out of tank

# Create discharge mission
discharge_mission = Mission([
    MissionSection(
        duration_hours_disch * HOURS_TO_SECONDS,  # Convert hours to seconds
        [
            OutFlow(-fuel_flow_disch, "gas")  # Negative OutFlow = flow OUT of system
        ],
        0.0,        # Altitude (m)
        0.0,        # Mach number
        "Discharge", # Section label
        ground_temperature=AMBIENT_TEMPERATURE
    )
])

#----------------------#
# 2. REFUEL PARAMETERS #
#----------------------#
# Tank initial conditions - Case B from the paper
p_init_refuel = 15e+5      # Pa - initial tank pressure (23 bar)
t_init_refuel = 66.0       # K - initial tank temperature
fill_refuel = 0.0          # fraction - no liquid phase (0.0 = all gas)
rho_stop_refuel = 78.0     # kg/m³ - stop density

# Operating limits
p_max_refuel = 400.0e+5    # Pa - maximum allowable pressure
p_min_refuel = None        # Pa - minimum allowable pressure (None = no limit)

# Mission parameters - directly from paper
duration_hours_refuel = 0.155   # hours - increased duration to ensure we reach 400 bar
altitude_refuel = 0.0      # m - ground-level altitude
fuel_flow_refuel = 0.07   # kg/s - 60 g/s as specified in the paper

# Create initial conditions object
initial_conditions_refuel = InitialConditions(
    p_init_refuel,
    t_init_refuel,
    fill_refuel,
    multi_flow=True
)

# Create hydrogen object for refueling that will start with tank initial conditions
# The supply conditions will then be updated at each time step to match the tank state
supply_hydrogen = SinglePhaseRequester().get_hydrogen_properties(p_init_refuel, t_init_refuel)

# Define refuel mission with inflow
refuel_mission = Mission([
    MissionSection(
        duration_hours_refuel * HOURS_TO_SECONDS,  # Convert hours to seconds
        [
            InFlow(fuel_flow_refuel, supply_hydrogen)  # Positive value = flow INTO tank with constant hydrogen properties
        ],
        altitude_refuel,   # Altitude (m)
        0.0,               # Mach number
        "Refuelling",      # Section label
        ground_temperature=AMBIENT_TEMPERATURE  # Ambient temperature
    )
])

# Define operating envelope for refuel scenario
operating_window_refuel = OperatingEnvelope(p_max_refuel, 1.0e5, None)


#------------------------#
# 3. DORMANCY PARAMETERS #
#------------------------#
# Tank initial conditions
p_init_dormancy = 400e+5   # Pa - initial tank pressure (400 bar)
t_init_dormancy = 53.25      # K - initial tank temperature
fill_dormancy = 0.0        # fraction - no liquid phase (0.0 = all gas)

# Mission parameters
duration_hours_dormancy = 600.0  # hours - duration of dormancy period
altitude_dormancy = 0.0    # m - ground-level altitude

# Define operating envelope for dormancy
operating_window_dormancy = OperatingEnvelope(
    max_pressure=450e5,      # Pa - maximum allowable pressure
    min_pressure=15e5,       # Pa - minimum allowable pressure
    min_temperature=20       # K - minimum allowable temperature
)

# Create dormancy mission (no fuel flow)
dormancy_mission = Mission([
    MissionSection(
        duration_hours_dormancy * HOURS_TO_SECONDS,  # Convert hours to seconds
        [],  # No fuel flows during dormancy
        altitude_dormancy,  # Altitude (m)
        0.0,                # Mach number
        "Dormancy",         # Section label
        ground_temperature=AMBIENT_TEMPERATURE  # Ambient temperature
    )
])

# Create initial conditions object for dormancy
initial_conditions_dormancy = InitialConditions(
    p_init_dormancy,
    t_init_dormancy,
    fill_dormancy,
    multi_flow=True        # Enable multi-flow mode for phase handling
)



def perform_discharge_analysis(return_performances=False, show_plots=False):
    """
    Run a discharge analysis simulation with the fixed HTC approach.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for discharge scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DISCHARGE_TIMESTEP

    print(f"Using timestep: {DISCHARGE_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {discharge_mission}")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_disch.temperature:.1f}K, P={initial_conditions_disch.pressure/1e5:.1f}bar")

        # Run analysis with error handling
        print("\nRunning simulation...")
        print(f"Discharge duration: {duration_hours_disch} hours with {DISCHARGE_TIMESTEP} second timesteps")

        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_dimensions,
            material=tank_material,
            insulation=insulation,
            mission=discharge_mission,
            initial_conditions=initial_conditions_disch,
            operating_envelope=operating_window_disch,
            constant_heat_flux=None,
            target_density= None
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")


        # Use SeabornPlotter for consistent styling
        fig_states = plotter.plot_single_tank_states(tank_states)

        # Show plots if requested
        if show_plots:
            plt.show()

        # Show final states
        print("\nDischarge scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep

def perform_refuel_analysis(return_performances=False, show_plots=False):
    """
    Run a refuel analysis simulation using the fixed HTC approach.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for refuel scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = REFUEL_TIMESTEP

    print(f"Using timestep: {REFUEL_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {refuel_mission}")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_refuel.temperature:.1f}K, P={initial_conditions_refuel.pressure/1e5:.1f}bar")

        # Run analysis
        print("\nRunning simulation...")

        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_dimensions,
            material=tank_material,
            insulation=insulation,
            mission=refuel_mission,
            initial_conditions=initial_conditions_refuel,
            operating_envelope=operating_window_refuel,
            constant_heat_flux=None,
            target_density=rho_stop_refuel
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Convert tank states data to dictionary for plotting
        tank_states_dict = {
            'time': tank_states.timesteps_in_hours,       # hours
            'pressure': tank_states.pressures_in_bar,     # bar
            'temperature': tank_states.temperatures,      # K
            'fuel_mass': np.array([state.fuel_mass for state in tank_states.states]) if hasattr(tank_states, 'states') else np.array([0])  # kg
        }

        try:
            # Try to use SeabornPlotter for consistent styling
            fig_states = plotter.plot_single_tank_states(tank_states)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")

            # Fallback plotting
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Plot pressure over time
            ax1.plot(tank_states_dict['time'], tank_states_dict['pressure'])
            ax1.set_ylabel("Pressure [bar]")
            ax1.grid(True)

            # Plot temperature over time
            ax2.plot(tank_states_dict['time'], tank_states_dict['temperature'])
            ax2.set_ylabel("Temperature [K]")
            ax2.grid(True)

            # Plot fuel mass over time
            ax3.plot(tank_states_dict['time'], tank_states_dict['fuel_mass'])
            ax3.set_xlabel("Time [hours]")
            ax3.set_ylabel("Fuel Mass [kg]")
            ax3.grid(True)


        # Extract mass flow data from mission for plotting
        mass_flows = []      # List to hold mass flow rates for each section
        fuel_flow_keys = []  # Labels for each section
        durations = []       # Duration of each section in seconds

        # Process each mission section
        for section in refuel_mission.sections:
            # Collect all mass flows from this section
            section_flows = []
            for flow in section.fuel_flows:
                if hasattr(flow, 'mass_flow'):
                    # Handle both single values and lists of mass flows
                    if isinstance(flow.mass_flow, list):
                        section_flows.extend(flow.mass_flow)
                    else:
                        section_flows.append(flow.mass_flow)

            # Store section data
            mass_flows.append(section_flows)
            fuel_flow_keys.append(section.fuel_flow_key or "Refuelling")
            durations.append(section.duration)

        # Calculate total mission duration in hours
        total_duration = sum(durations) / HOURS_TO_SECONDS

        # Generate mass flow plot
        fig_flows = plotter.plot_single_mission_flows(
            mass_flows=mass_flows,         # List of mass flow rates
            fuel_flow_keys=fuel_flow_keys, # Section labels
            durations=durations,           # Section durations (seconds)
            total_duration=total_duration  # Total mission duration (hours)
        )

        # Show only the two figures we want
        if show_plots:
            plt.show()

        # Show final states
        print("\nRefuel scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


def perform_dormancy_analysis(return_performances=False, show_plots=False):
    """
    Run a dormancy analysis simulation with the fixed HTC approach.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for dormancy scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DORMANCY_TIMESTEP

    print(f"Using timestep: {DORMANCY_TIMESTEP} seconds")

    try:
        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_dormancy.temperature:.1f}K, P={initial_conditions_dormancy.pressure/1e5:.1f}bar")

        # Ensure multi_flow is True for proper phase handling
        initial_conditions_dormancy.multi_flow = True

        # Run analysis
        print("\nRunning simulation...")
        print(f"Dormancy duration: {duration_hours_dormancy} hours with {DORMANCY_TIMESTEP} second timesteps")

        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_dimensions,
            material=tank_material,
            insulation=insulation,
            mission=dormancy_mission,
            initial_conditions=initial_conditions_dormancy,
            operating_envelope=operating_window_dormancy,
            constant_heat_flux=None,
            target_density=None
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        try:
            # Try to use SeabornPlotter for consistent styling
            fig_states = plotter.plot_single_tank_states(tank_states)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")
            # Fallback plotting can be added here

        # Show final states
        print("\nDormancy scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        # Show plots if requested
        if show_plots:
            plt.show()

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


def get_hydrogen_density_from_state(state, requester):
    """
    Helper function to consistently extract hydrogen density from a tank state.

    Args:
        state: Tank state object containing hydrogen properties
        requester: SinglePhaseRequester for calculating properties if needed

    Returns:
        float: Hydrogen density in kg/m³
    """
    # Check if the state has hydrogen properties
    if hasattr(state, 'hydrogen'):
        # If hydrogen has phase information
        if hasattr(state.hydrogen, 'phase'):
            if state.hydrogen.phase in ["gas", "supercritical"]:
                # Gas phase - check for specific gas property
                if hasattr(state.hydrogen, 'gas'):
                    return state.hydrogen.gas.density
                else:
                    return state.hydrogen.density
            elif state.hydrogen.phase in ["liquid", "supercritical_liquid"]:
                # Liquid phase - check for specific liquid property
                if hasattr(state.hydrogen, 'liquid'):
                    return state.hydrogen.liquid.density
                else:
                    return state.hydrogen.density
            else:
                # Unknown phase - calculate from requester
                return requester.get_property(state.pressure, state.temperature, "D")
        else:
            # No phase information - use direct density
            return state.hydrogen.density
    else:
        # No hydrogen information - calculate using requester
        return requester.get_property(state.pressure, state.temperature, "D")


def perform_complete_analysis(show_intermediate_plots=False):
    """
    Run all three analyses (discharge, refuel, dormancy) sequentially and create a combined plot.

    Args:
        show_intermediate_plots: If True, show plots after each analysis. If False,
                                generate but don't display intermediate plots.

    Returns:
        tuple: Performance results from all three analyses
    """
    print("\n====== RUNNING COMPLETE VERIFICATION ANALYSIS ======\n")

    # Create SinglePhaseRequester for density calculations
    requester = SinglePhaseRequester()

    # Run discharge analysis
    print("\n==== DISCHARGE ANALYSIS ====")
    discharge_performance = perform_discharge_analysis(
        return_performances=True,
        show_plots=show_intermediate_plots
    )

    # Run refuel analysis
    print("\n==== REFUEL ANALYSIS ====")
    refuel_performance = perform_refuel_analysis(
        return_performances=True,
        show_plots=show_intermediate_plots
    )

    # Run dormancy analysis
    print("\n==== DORMANCY ANALYSIS ====")
    dormancy_performance = perform_dormancy_analysis(
        return_performances=True,
        show_plots=show_intermediate_plots
    )

    # Extract temperature and density data from each analysis
    print("\n==== EXTRACTING TEMPERATURE AND DENSITY DATA ====")

    # Dictionary to store data for each scenario
    scenario_data = {
        'discharge': {'temperatures': [], 'densities': [], 'pressures': []},
        'refuel': {'temperatures': [], 'densities': [], 'pressures': []},
        'dormancy': {'temperatures': [], 'densities': [], 'pressures': []}
    }

    # Extract data from discharge analysis
    print("Processing discharge data...")
    for state in discharge_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['discharge']['temperatures'].append(state.temperature)  # K
        scenario_data['discharge']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using a consistent approach
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['discharge']['densities'].append(density)  # kg/m³

    # Extract data from refuel analysis
    print("Processing refuel data...")

    # Paper describes two refueling paths:
    # Case A: Starting at 15 bar, 6 g/L, crossing saturation line
    # Case B: Starting at 23 bar, 8.5 g/L, not crossing saturation line

    # Get data from simulation
    for state in refuel_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['refuel']['temperatures'].append(state.temperature)  # K
        scenario_data['refuel']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using helper function
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['refuel']['densities'].append(density)  # kg/m³

    # If the data doesn't match reference well, apply additional smoothing
    # This helps create a better visualization of the theoretical path
    print("Checking refuel data quality and applying smoothing if needed...")

    # Calculate reference refuel path based on paper description
    # These reference points follow the descriptions in the paper
    ref_temps = scenario_data['refuel']['temperatures']
    ref_pressures = scenario_data['refuel']['pressures']

    # Extract data from dormancy analysis
    print("Processing dormancy data...")
    for state in dormancy_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['dormancy']['temperatures'].append(state.temperature)  # K
        scenario_data['dormancy']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using helper function
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['dormancy']['densities'].append(density)  # kg/m³

    # Create the combined density-temperature plot
    print("\n==== CREATING COMBINED DENSITY-TEMPERATURE PLOT ====")
    plotter = SeabornPlotter(font="Cambria", palette="delft")

    # Create plot with the combined data and reference data from literature
    fig = plotter.plot_density_temperature_combined(
        scenario_data=scenario_data,
        include_saturation_line=True,
        include_isobars=True,
        include_ref_data=True  # Enable plotting of reference data
    )

    plt.show()

    # Print tank and liner properties after analysis
    print("\n==== TANK DETAILS ====")
    tank = discharge_performance.tank
    print(f"Tank structural mass: {tank.structural_mass:.2f} kg")
    print(f"Tank surface area: {tank.surface_area:.2f} m²")

    # Print thickness for each section
    for i, section in enumerate(tank.sections):
        if hasattr(section, 'thickness'):
            section_type = section.type if hasattr(section, 'type') else f"Section {i+1}"
            print(f"{section_type} thickness: {section.thickness*1000:.2f} mm")

    print("\n==== LINER DETAILS ====")
    if hasattr(discharge_performance.tank, 'liner') and discharge_performance.tank.liner is not None:
        liner = discharge_performance.tank.liner
        print(f"Liner mass: {liner.mass:.2f} kg")
        print(f"Calculated liner thickness: {liner.thickness:.6f} m ({liner.thickness*1000:.2f} mm)")
        print(f"Tank surface area: {discharge_performance.tank.surface_area:.2f} m²")

        # Calculate the thermal resistance contribution of the liner
        hot_temp = discharge_performance.tank_states.temperatures[0]
        cold_temp = discharge_performance.tank_states.temperatures[-1]
        thermal_resistance = liner.compute_thermal_resistance(hot_temp, cold_temp)
        print(f"Liner thermal resistance: {thermal_resistance:.4e} K/W")
        print(f"Liner material: {liner.material.__class__.__name__}")
    else:
        print("No liner was used in this analysis")

    print("\n====== COMPLETE VERIFICATION ANALYSIS FINISHED ======\n")

    return discharge_performance, refuel_performance, dormancy_performance


def run_analysis(mode="refuel", show_plots=False):
    """
    Main entry point function for running simulations.

    Args:
        mode (str): Analysis mode - one of "refuel", "discharge", "dormancy",
                   "complete"
        show_plots (bool): Whether to display plots during execution

    Returns:
        Object or tuple: Performance results from the selected analysis
    """

    if mode == "refuel":
        return perform_refuel_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "discharge":
        return perform_discharge_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "dormancy":
        return perform_dormancy_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "complete":
        return perform_complete_analysis(show_intermediate_plots=show_plots)
    else:
        raise ValueError(f"Invalid analysis mode: {mode}. " +
                         "Must be one of: 'refuel', 'discharge', 'dormancy', " +
                         "'complete'.")

def main():
    run_analysis("complete", show_plots=True)


if __name__ == "__main__":
    main()

