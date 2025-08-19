from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.mission.mission import Mission
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.mission.mission_sections import OutFlow, MissionSection, InFlow
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import MULTISTEP_METHOD, OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, FillingAnalysisFacade, RefuellingAnalysisFacade, InOutTankAnalysisFacade, MissionAnalysisFacade
import numpy as np
import matplotlib.pyplot as plt


##########################
## COMMON CONFIGURATION ##
##########################

HOURS_TO_SECONDS = 3600.0
# Add scenario-specific timesteps
REFUEL_TIMESTEP = 0.1    # Smaller timestep for refuel (faster dynamics)
DISCHARGE_TIMESTEP = 5.0  # Standard timestep for discharge
DORMANCY_TIMESTEP = 60.0  # Larger timestep for dormancy (slower dynamics)
NOMINAL_MASS = 35.0  # kg

# geometry
tank_volume = 0.5
radius = (3 * tank_volume / (4 * np.pi))**(1/3)
print(f"Calculated tank radius: {radius:.3f} m")


#########################
## DISCHARGE ANALYSIS  ##
#########################

# Define Tank 1 parameters (reservoir) - 100kg, 300K, 500 bar
p_init_disch = 4e+7  # Pa
t_init_disch = 53.15  # K
fill_disch = 0.0  # no liquid
p_max_disch = 5.0e+8  # Pa
p_min_disch = 1500000  # Pa
ambient_heat_load_disch = 2.0  # W/m²

# Instantiate tank objects
tank_material = Composite.carbon(np.radians(55))
tank_dimensions = TankDimensions(radius, 0.0)  # Spherical tank

# Insulation for both tanks
insulation_thickness = 0.05  # m
insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

# Operating envelopes
operating_window_disch = OperatingEnvelope(p_max_disch, p_min_disch, None)

# Initial conditions
initial_conditions_disch = InitialConditions(p_init_disch, t_init_disch, fill_disch, multi_flow=False)

# Define tank configurations for MultiTankAnalysisFacade
tank_config = [
    {
        "dimensions": tank_dimensions,
        "material": tank_material,
        "insulation": insulation,
        "heat_flux": ambient_heat_load_disch
    }
]

def perform_discharge_analysis(return_performances=False, show_plots=False):

    # Set timestep
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DISCHARGE_TIMESTEP
    print(f"\n=== DISCHARGE ANALYSIS ===")
    print(f"Using timestep: {DISCHARGE_TIMESTEP} seconds")

    duration_hours_disch = 10
    fuel_flow_disch = 0.001  # [kg/s] example value
    discharge_mission = Mission([
        MissionSection(
            duration_hours_disch * HOURS_TO_SECONDS,
            [
                OutFlow(-fuel_flow_disch, "gas")  # NEGATIVE OutFlow = INFLOW to system
            ],
            0.0,
            0.0,
            "Discharge"
        )
    ])

    try:
        # Print initial info
        print(f"Mission details: {discharge_mission}")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_disch.temperature:.1f}K, P={initial_conditions_disch.pressure/1e5:.1f}bar")

        # Run analysis
        print("\nRunning simulation...")
        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=discharge_mission,
            initial_conditions=initial_conditions_disch,
            operating_envelope=operating_window_disch,
            constant_heat_flux=tank_config[0]['heat_flux']
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Create plots for pressure and temperature
        # Convert pandas Series to numpy arrays before plotting

        # Convert tank_states data to numpy arrays before plotting
        tank_states_dict = {
            'time': tank_states.timesteps_in_hours,  # Use timesteps_in_hours instead of time
            'pressure': tank_states.pressures_in_bar,  # Use pressures_in_bar instead of pressure
            'temperature': tank_states.temperatures,
            'fuel_mass': np.array([state.fuel_mass for state in tank_states.states]) if hasattr(tank_states, 'states') else np.array([0])
        }

        try:
            fig_combined = plotter.plot_single_tank_states(tank_states)
        except ValueError:
            # Create a simple combined plot with numpy arrays
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            ax1.plot(tank_states_dict['time'], tank_states_dict['pressure'])
            ax1.set_ylabel("Pressure [bar]")
            ax1.grid(True)

            ax2.plot(tank_states_dict['time'], tank_states_dict['temperature'])
            ax2.set_ylabel("Temperature [K]")
            ax2.grid(True)

            ax3.plot(tank_states_dict['time'], tank_states_dict['fuel_mass'])
            ax3.set_xlabel("Time [hours]")
            ax3.set_ylabel("Fuel Mass [kg]")
            ax3.grid(True)

            fig.suptitle("Tank States During Discharge")
            fig_combined = fig

        # Extract mission data for mass flow plotting
        mass_flows = []
        fuel_flow_keys = []
        durations = []

        # Get data from mission sections
        for section in discharge_mission.sections:
            # Extract flows keeping original signs to indicate direction
            section_flows = []
            for flow in section.fuel_flows:
                if hasattr(flow, 'mass_flow'):
                    if isinstance(flow.mass_flow, list):
                        section_flows.extend(flow.mass_flow)
                    else:
                        section_flows.append(flow.mass_flow)

            mass_flows.append(section_flows)
            fuel_flow_keys.append(section.fuel_flow_key if section.fuel_flow_key else "Section")
            durations.append(section.duration)

        # Calculate total duration in hours
        total_duration = sum(durations) / 3600.0

        # Plot the mission mass flows
        fig_flows = plotter.plot_single_mission_flows(
            mass_flows=mass_flows,
            fuel_flow_keys=fuel_flow_keys,
            durations=durations,
            total_duration=total_duration
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


#####################
## REFUEL ANALYSIS ##
#####################

# Define Tank 1 parameters (reservoir)
p_init_refuel = 15e+5  # Pa
t_init_refuel = 65  # K
fill_refuel = 0.0 # no liquid
p_max_refuel = 5.0e+8  # Pa
p_min_refuel = None  # Pa
ambient_heat_load_refuel = 0.0  # W/m²


# mission details for refuel
duration_hours_refuel = 0.12  # Duration of refuel in hours
altitude_refuel = 0.0  # Altitude in meters
fuel_flow_refuel = 0.07  # Fuel flow rate in kg/s

# get refuel hydrogen properties
refuel_hydrogen = SinglePhaseRequester().get_hydrogen_properties(p_init_refuel, t_init_refuel)

# Set multi_flow flag to True in initial conditions
initial_conditions_refuel = InitialConditions(p_init_refuel, t_init_refuel, fill_refuel,
                                             multi_flow=True)

# Get refuel hydrogen properties
refuel_hydrogen = SinglePhaseRequester().get_hydrogen_properties(p_init_refuel, t_init_refuel)

# Make sure the mission uses an InFlow (positive value = flow INTO the tank)
refuel_mission = Mission([
    MissionSection(
        duration_hours_refuel * HOURS_TO_SECONDS,
        [
            InFlow(fuel_flow_refuel, refuel_hydrogen)  # Positive InFlow = INFLOW to system
        ],
        altitude_refuel,
        0.0,
        "Refuelling"
    )
])

# Initial conditions
operating_window_refuel = OperatingEnvelope(p_max_refuel, p_min_refuel, None)

def perform_refuel_analysis(return_performances=False, show_plots=False):

    # Set timestep
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
        target_conditions = TargetConditions(
            fuel_mass=35,
            fill=1.0
        )

        tank_performance = RefuellingAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=refuel_mission,
            constant_heat_flux=tank_config[0]['heat_flux'],
            initial_conditions=initial_conditions_refuel,
            operating_envelope=operating_window_refuel,
            target_conditions=target_conditions
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Convert tank_states data to numpy arrays before plotting
        tank_states_dict = {
            'time': tank_states.timesteps_in_hours,
            'pressure': tank_states.pressures_in_bar,
            'temperature': tank_states.temperatures,
            'fuel_mass': np.array([state.fuel_mass for state in tank_states.states]) if hasattr(tank_states, 'states') else np.array([0])
        }

        try:
            # Plot tank states
            fig_states = plotter.plot_single_tank_states(tank_states)
            plt.savefig("refuel_tank_states.png", dpi=300)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")

        # Extract mission data for mass flow plotting
        mass_flows = []
        fuel_flow_keys = []
        durations = []

        # Get flow data from the mission
        for section in refuel_mission.sections:
            section_flows = []
            for flow in section.fuel_flows:
                # Get the flow value
                if hasattr(flow, 'mass_flow'):
                    if isinstance(flow.mass_flow, list):
                        section_flows.extend(flow.mass_flow)
                    else:
                        section_flows.append(flow.mass_flow)

            # Store the section data
            mass_flows.append(section_flows)
            fuel_flow_keys.append(section.fuel_flow_key or "Refuelling")
            durations.append(section.duration)

        # Calculate total duration in hours
        total_duration = sum(durations) / 3600.0

        # Plot mass flows with invert_flow=True to show refueling as positive
        fig_flows = plotter.plot_single_mission_flows(
            mass_flows=mass_flows,
            fuel_flow_keys=fuel_flow_keys,
            durations=durations,
            total_duration=total_duration,
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


#######################
## DORMANCY ANALYSIS ##
#######################

# DORMANCY ANALYSIS CONFIGURATION
duration_hours = 60.0  # Duration of dormancy in hours
altitude = 0.0  # Altitude in meters

# Create a dormancy mission with a single section
dormancy_mission = Mission([
    Mission.dormancy_section(
        duration=duration_hours,
        altitude=altitude,
        fuel_flow=0.0,  # Will be forced to zero anyway
        throttle=0.0,   # Will be forced to zero anyway
        phase="gas",    # Dummy value, not used
        mach_number=0.0
    )
])

# Define Tank 1 parameters for dormancy - 100kg, 300K, 200 bar
p_init_dormancy = 400e+5  # Pa (200 bar)
t_init_dormancy = 53.15   # K
fill_dormancy = 0.0     # no liquid
ambient_heat_load_dormancy = 20.0  # W/m²

# Initial conditions for dormancy
initial_conditions_dormancy = InitialConditions(p_init_dormancy, t_init_dormancy, fill_dormancy,
                                                 multi_flow=True)

def perform_dormancy_analysis(return_performances=False, show_plots=False):
    # Set timestep
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

        # Define operating envelope. ensure that max_pressure is defined (and larger than initial pressure)

        operating_window_dormancy = OperatingEnvelope(
            max_pressure=450e5,  # Pa
            min_pressure=15e5,   # Pa
            min_temperature=20   # K
        )

        # Run analysis
        print("\nRunning simulation...")
        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=dormancy_mission,
            initial_conditions=initial_conditions_dormancy,
            operating_envelope=operating_window_dormancy,
            constant_heat_flux=ambient_heat_load_dormancy,
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Convert tank_states data to numpy arrays before plotting
        tank_states_dict = {
            'time': tank_states.timesteps_in_hours,
            'pressure': tank_states.pressures_in_bar,
            'temperature': tank_states.temperatures,
            'fuel_mass': np.array([state.fuel_mass for state in tank_states.states]) if hasattr(tank_states, 'states') else np.array([0])
        }

        try:
            # Plot tank states
            fig_states = plotter.plot_single_tank_states(tank_states)
            plt.savefig("dormancy_tank_states.png", dpi=300)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")

            # Fallback plotting
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            ax1.plot(tank_states_dict['time'], tank_states_dict['pressure'])
            ax1.set_ylabel("Pressure [bar]")
            ax1.grid(True)

            ax2.plot(tank_states_dict['time'], tank_states_dict['temperature'])
            ax2.set_ylabel("Temperature [K]")
            ax2.grid(True)

            ax3.plot(tank_states_dict['time'], tank_states_dict['fuel_mass'])
            ax3.set_xlabel("Time [hours]")
            ax3.set_ylabel("Fuel Mass [kg]")
            ax3.grid(True)

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
        scenario_data['discharge']['temperatures'].append(state.temperature)
        scenario_data['discharge']['pressures'].append(state.pressure)

        # Get density based on phase
        if hasattr(state, 'hydrogen'):
            if hasattr(state.hydrogen, 'phase'):
                if state.hydrogen.phase in ["gas", "supercritical"]:
                    if hasattr(state.hydrogen, 'gas'):
                        density = state.hydrogen.gas.density
                    else:
                        density = state.hydrogen.density
                elif state.hydrogen.phase in ["liquid", "supercritical_liquid"]:
                    if hasattr(state.hydrogen, 'liquid'):
                        density = state.hydrogen.liquid.density
                    else:
                        density = state.hydrogen.density
                else:
                    # Default to getting density from requester
                    density = requester.get_property(state.pressure, state.temperature, "D")
            else:
                # If no phase attribute, use the direct density
                density = state.hydrogen.density
        else:
            # If no hydrogen attribute, calculate from requester
            density = requester.get_property(state.pressure, state.temperature, "D")

        scenario_data['discharge']['densities'].append(density)

    # Extract data from refuel analysis
    print("Processing refuel data...")
    for state in refuel_performance.tank_states.states:
        scenario_data['refuel']['temperatures'].append(state.temperature)
        scenario_data['refuel']['pressures'].append(state.pressure)

        # Get density based on phase (same logic as above)
        if hasattr(state, 'hydrogen'):
            if hasattr(state.hydrogen, 'phase'):
                if state.hydrogen.phase in ["gas", "supercritical"]:
                    if hasattr(state.hydrogen, 'gas'):
                        density = state.hydrogen.gas.density
                    else:
                        density = state.hydrogen.density
                elif state.hydrogen.phase in ["liquid", "supercritical_liquid"]:
                    if hasattr(state.hydrogen, 'liquid'):
                        density = state.hydrogen.liquid.density
                    else:
                        density = state.hydrogen.density
                else:
                    density = requester.get_property(state.pressure, state.temperature, "D")
            else:
                density = state.hydrogen.density
        else:
            density = requester.get_property(state.pressure, state.temperature, "D")

        scenario_data['refuel']['densities'].append(density)

    # Extract data from dormancy analysis
    print("Processing dormancy data...")
    for state in dormancy_performance.tank_states.states:
        scenario_data['dormancy']['temperatures'].append(state.temperature)
        scenario_data['dormancy']['pressures'].append(state.pressure)

        # Get density based on phase (same logic as above)
        if hasattr(state, 'hydrogen'):
            if hasattr(state.hydrogen, 'phase'):
                if state.hydrogen.phase in ["gas", "supercritical"]:
                    if hasattr(state.hydrogen, 'gas'):
                        density = state.hydrogen.gas.density
                    else:
                        density = state.hydrogen.density
                elif state.hydrogen.phase in ["liquid", "supercritical_liquid"]:
                    if hasattr(state.hydrogen, 'liquid'):
                        density = state.hydrogen.liquid.density
                    else:
                        density = state.hydrogen.density
                else:
                    density = requester.get_property(state.pressure, state.temperature, "D")
            else:
                density = state.hydrogen.density
        else:
            density = requester.get_property(state.pressure, state.temperature, "D")

        scenario_data['dormancy']['densities'].append(density)

    # Create the combined density-temperature plot
    print("\n==== CREATING COMBINED DENSITY-TEMPERATURE PLOT ====")
    plotter = SeabornPlotter(font="Cambria", palette="delft")

    # We'll add a new method to SeabornPlotter to handle this plot
    # For now, we'll call a placeholder that we'll implement next
    fig = plotter.plot_density_temperature_combined(
        scenario_data=scenario_data,
        include_saturation_line=True,
        include_isobars=True
    )

    plt.savefig("combined_density_temperature.png", dpi=300)
    plt.show()

    print("\n====== COMPLETE VERIFICATION ANALYSIS FINISHED ======\n")

    return discharge_performance, refuel_performance, dormancy_performance



