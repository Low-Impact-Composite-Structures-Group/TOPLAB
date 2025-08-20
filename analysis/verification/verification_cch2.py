"""
Verification scenarios for cryo-compressed hydrogen (CCH2) tank.

This module provides simulation capabilities for three standard scenarios:
1. Discharge - Fuel flow out of the tank
2. Refuel - Filling the tank with hydrogen
3. Dormancy - Tank sitting with no fuel flow in/out (heat soak)

Each scenario can be run individually or in sequence using the complete analysis function.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import hydrogen fluid models
from src.fluids.hydrogen_retrievers import SinglePhaseRequester

# Import mission components
from src.mission.mission import Mission
from src.mission.mission_sections import OutFlow, MissionSection, InFlow

# Import tank components
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite

# Import facades and analysis components
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import (
    MULTISTEP_METHOD, OperatingEnvelope, TankDimensions,
    InitialConditions, TargetConditions, Insulation,
    MissionAnalysisFacade
)

# Import thermodynamic model
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


#################################
## COMMON SIMULATION CONSTANTS ##
#################################

# Time conversion
HOURS_TO_SECONDS = 3600.0  # seconds per hour

# Simulation timesteps for different scenarios
DISCHARGE_TIMESTEP = 5.0   # seconds - standard timestep for discharge
REFUEL_TIMESTEP = 1.0      # seconds - small timestep for refuel (rapid dynamics)
DORMANCY_TIMESTEP = 60.0  # seconds - larger timestep for dormancy (slower dynamics)

# Tank parameters
NOMINAL_MASS = 35.0        # kg - target hydrogen mass
TANK_VOLUME = 0.5          # m³ - internal volume of the tank
TANK_RADIUS = (3 * TANK_VOLUME / (4 * np.pi))**(1/3)  # m - radius assuming spherical tank

# Heat transfer coefficient from reference paper
# Adjust this value to match reference paper data
HEAT_TRANSFER_COEFFICIENT = 0.025  # W/m²K

# Display tank size
print(f"Calculated tank radius: {TANK_RADIUS:.3f} m")
print(f"Using heat transfer coefficient: {HEAT_TRANSFER_COEFFICIENT} W/m²K")


#######################################
## SCENARIO CONFIGURATIONS ##
#######################################

#-------------------------#
# 1. DISCHARGE PARAMETERS #
#-------------------------#
# Tank initial conditions
p_init_disch = 4e+7        # Pa - initial tank pressure
t_init_disch = 51.8       # K - initial tank temperature
fill_disch = 0.0           # fraction - no liquid phase (0.0 = all gas)

# Operating limits
p_max_disch = 5.0e+8       # Pa - maximum allowable pressure
p_min_disch = 1500000      # Pa - minimum allowable pressure
ambient_heat_load_disch = None  # Set to None to use thermal resistance model

# Instantiate common tank objects
tank_material = Composite.carbon(np.radians(55))
tank_dimensions = TankDimensions(TANK_RADIUS, 0.0)  # Spherical tank

# Insulation configuration
insulation_thickness = 0.05  # m - thermal insulation thickness
insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

# Operating envelopes for discharge scenario
operating_window_disch = OperatingEnvelope(p_max_disch, p_min_disch, None)

# Initial conditions for discharge scenario
initial_conditions_disch = InitialConditions(p_init_disch, t_init_disch, fill_disch, multi_flow=False)

# Define tank configuration for all analysis scenarios
tank_config = [
    {
        "dimensions": tank_dimensions,
        "material": tank_material,
        "insulation": insulation,
        "heat_flux": ambient_heat_load_disch
    }
]

# Add discharge mission parameters to the configuration section
duration_hours_disch = 10    # hours - duration of discharge operation
fuel_flow_disch = 0.001      # kg/s - fuel flow rate out of tank

# Create discharge mission
discharge_mission = Mission([
    MissionSection(
        duration_hours_disch * HOURS_TO_SECONDS,  # Convert hours to seconds
        [
            OutFlow(-fuel_flow_disch, "gas")  # NEGATIVE OutFlow = flow INTO system
        ],
        0.0,        # Altitude (m)
        0.0,        # Mach number
        "Discharge" # Section label
    )
])

# Custom MissionAnalysisFacade that uses our FixedHTCThermodynamicModel
class FixedHTCMissionAnalysisFacade(MissionAnalysisFacade):
    @classmethod
    def _define_thermal_model(
        cls,
        insulation: Insulation,
        constant_heat_flux: float = None
    ) -> ThermodynamicModel:
        # Create a thermal model with fixed heat transfer coefficient
        from facades.analysis_facades import INTERNAL_MODEL, EXTERNAL_MODEL

        return ThermodynamicModel(
            INTERNAL_MODEL,
            EXTERNAL_MODEL,
            insulation,
            constant_heat_flux=constant_heat_flux,
            heat_transfer_coefficient=HEAT_TRANSFER_COEFFICIENT
        )

def perform_discharge_analysis(return_performances=False, show_plots=False):
    """
    Run a discharge analysis simulation with the configured parameters.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for discharge scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DISCHARGE_TIMESTEP

    print(f"\n=== DISCHARGE ANALYSIS ===")
    print(f"Using timestep: {DISCHARGE_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {discharge_mission}")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_disch.temperature:.1f}K, P={initial_conditions_disch.pressure/1e5:.1f}bar")

        # Run analysis
        print("\nRunning simulation...")
        print(f"Discharge duration: {duration_hours_disch} hours with {DISCHARGE_TIMESTEP} second timesteps")

        # Use our custom facade with fixed HTC model
        tank_performance = FixedHTCMissionAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=discharge_mission,
            initial_conditions=initial_conditions_disch,
            operating_envelope=operating_window_disch,
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
            fig_combined = plotter.plot_single_tank_states(tank_states)
        except ValueError:
            # Fallback plotting if the plotter fails
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

            # Add title
            fig.suptitle("Tank States During Discharge")
            fig_combined = fig

        # Extract mass flow data from mission for plotting
        mass_flows = []      # List to hold mass flow rates for each section
        fuel_flow_keys = []  # Labels for each section
        durations = []       # Duration of each section in seconds

        # Process each mission section
        for section in discharge_mission.sections:
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
            fuel_flow_keys.append(section.fuel_flow_key if section.fuel_flow_key else "Section")
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
        print("\nDischarge scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


#----------------------#
# 2. REFUEL PARAMETERS #
#----------------------#
# Tank initial conditions
p_init_refuel = 23e+5      # Pa - initial tank pressure
t_init_refuel = 70        # K - initial tank temperature
fill_refuel = 0.0          # fraction - no liquid phase (0.0 = all gas)

# Operating limits
p_max_refuel = 400.0e+5    # Pa - maximum allowable pressure
p_min_refuel = None        # Pa - minimum allowable pressure (None = no limit)
# Using the thermal resistance model with adjusted insulation thickness
# to match the reference paper's heat transfer coefficient of ~0.025 W/m²K
ambient_heat_load_refuel = None  # Set to None to use thermal resistance model

# Supply hydrogen conditions
supply_temp = 24            # K - hydrogen supply temperature
supply_pressure = 2.0e+5   # Pa - hydrogen supply pressure

# Mission parameters
duration_hours_refuel = 0.155  # hours - duration of refueling operation
altitude_refuel = 0.0      # m - ground-level altitude
fuel_flow_refuel = 0.06    # kg/s - fuel flow rate into tank

# Create initial conditions object
initial_conditions_refuel = InitialConditions(
    p_init_refuel,
    t_init_refuel,
    fill_refuel,
    multi_flow=True        # Enable multi-flow mode for phase handling
)

# Create dummy hydrogen object for mission definition
# This will be dynamically updated during simulation by the cryopump model
dummy_hydrogen = SinglePhaseRequester().get_hydrogen_properties(supply_pressure, supply_temp)

# Define refuel mission with inflow
refuel_mission = Mission([
    MissionSection(
        duration_hours_refuel * HOURS_TO_SECONDS,  # Convert hours to seconds
        [
            InFlow(fuel_flow_refuel, dummy_hydrogen)  # Positive value = flow INTO tank
        ],
        altitude_refuel,  # Altitude (m)
        0.0,              # Mach number
        "Refuelling"      # Section label
    )
])

# Define operating envelope for refuel scenario
operating_window_refuel = OperatingEnvelope(p_max_refuel, p_min_refuel, None)

def perform_refuel_analysis(return_performances=False, show_plots=False):
    """
    Run a refuel analysis simulation with the configured parameters.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for refuel scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = REFUEL_TIMESTEP

    print(f"\n=== REFUEL ANALYSIS ===")
    print(f"Using timestep: {REFUEL_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {refuel_mission}")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_refuel.temperature:.1f}K, P={initial_conditions_refuel.pressure/1e5:.1f}bar")

        # Ensure multi_flow is True (redundant but safer)
        initial_conditions_refuel.multi_flow = True

        # Run analysis
        print("\nRunning simulation...")
        print(f"Refuel duration: {duration_hours_refuel} hours with {REFUEL_TIMESTEP} second timesteps")

        target_conditions = TargetConditions(
            fuel_mass=35,
            fill=1.0
        )

        tank_performance = FixedHTCMissionAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=refuel_mission,
            initial_conditions=initial_conditions_refuel,
            operating_envelope=operating_window_refuel,
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
            plt.savefig("refuel_tank_states.png", dpi=300)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")

            # Could add fallback plotting here as in dormancy function

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
        plt.savefig("refuel_mass_flows.png", dpi=300)

        # Show only the two figures we want
        if show_plots:
            plt.show()

        # Show final states
        print("\Refuel scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


#------------------------#
# 3. DORMANCY PARAMETERS #
#------------------------#
# Tank initial conditions
p_init_dormancy = 400e+5   # Pa - initial tank pressure (400 bar)
t_init_dormancy = 51.8    # K - initial tank temperature
fill_dormancy = 0.0        # fraction - no liquid phase (0.0 = all gas)
ambient_heat_load_dormancy = None  # Set to None to use thermal resistance model

# Mission parameters
duration_hours_dormancy = 60.0  # hours - duration of dormancy period (reduced from 60.0 to improve simulation speed)
altitude_dormancy = 0.0    # m - ground-level altitude

# Create dormancy mission (no fuel flow)
dormancy_mission = Mission([
    Mission.dormancy_section(
        duration=duration_hours_dormancy,
        altitude=altitude_dormancy,
        fuel_flow=0.0,     # No fuel flow during dormancy
        throttle=0.0,      # No engine throttle
        phase="gas",       # Dummy value (not used in dormancy)
        mach_number=0.0    # No movement
    )
])

# Create initial conditions object for dormancy
initial_conditions_dormancy = InitialConditions(
    p_init_dormancy,
    t_init_dormancy,
    fill_dormancy,
    multi_flow=True        # Enable multi-flow mode for phase handling
)

def perform_dormancy_analysis(return_performances=False, show_plots=False):
    """
    Run a dormancy analysis simulation with the configured parameters.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for dormancy scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DORMANCY_TIMESTEP

    print(f"\n=== DORMANCY ANALYSIS ===")
    print(f"Using timestep: {DORMANCY_TIMESTEP} seconds")

    try:
        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_dormancy.temperature:.1f}K, P={initial_conditions_dormancy.pressure/1e5:.1f}bar")

        # Ensure multi_flow is True for proper phase handling
        initial_conditions_dormancy.multi_flow = True

        # Define operating envelope for dormancy
        operating_window_dormancy = OperatingEnvelope(
            max_pressure=450e5,      # Pa - maximum allowable pressure
            min_pressure=15e5,       # Pa - minimum allowable pressure
            min_temperature=20       # K - minimum allowable temperature
        )

        # Run analysis
        print("\nRunning simulation...")
        print(f"Dormancy duration: {duration_hours_dormancy} hours with {DORMANCY_TIMESTEP} second timesteps")

        tank_performance = FixedHTCMissionAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=dormancy_mission,
            initial_conditions=initial_conditions_dormancy,
            operating_envelope=operating_window_dormancy,
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
            plt.savefig("dormancy_tank_states.png", dpi=300)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")

            # Fallback plotting if the plotter fails
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

            # Add title and save
            fig.suptitle("Tank States During Dormancy")
            plt.savefig("dormancy_tank_states.png", dpi=300)

        # Show final states
        print("\nDormancy scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        # Show only the figures we want
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
    for state in refuel_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['refuel']['temperatures'].append(state.temperature)  # K
        scenario_data['refuel']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using helper function
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['refuel']['densities'].append(density)  # kg/m³

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

    plt.savefig("combined_density_temperature.png", dpi=300)
    plt.show()

    print("\n====== COMPLETE VERIFICATION ANALYSIS FINISHED ======\n")

    return discharge_performance, refuel_performance, dormancy_performance


def run_analysis(mode="complete", show_plots=False):
    """
    Main entry point function for running simulations.

    Args:
        mode (str): Analysis mode - one of "complete", "discharge", "refuel", "dormancy"
        show_plots (bool): Whether to display plots during execution

    Returns:
        Object or tuple: Performance results from the selected analysis
    """
    print(f"\n====== RUNNING {mode.upper()} ANALYSIS ======\n")

    if mode == "complete":
        return perform_complete_analysis(show_plots)
    elif mode == "discharge":
        return perform_discharge_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "refuel":
        return perform_refuel_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "dormancy":
        return perform_dormancy_analysis(return_performances=True, show_plots=show_plots)
    else:
        raise ValueError(f"Invalid analysis mode: {mode}. " +
                         "Must be one of: 'complete', 'discharge', 'refuel', 'dormancy'.")



