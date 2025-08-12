import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_single_tank_temperatures, plot_single_tank_loads
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import MULTISTEP_METHOD, OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt


def perform_refuel_analysis(return_performances=False, target_mass=126.76):
    """
    Perform refuel analysis with configuration from config file.

    Args:
        return_performances: Whether to return performance data
        target_mass: Target mass for Tank 1 (defaults to mission requirement)
    """
    # Import configuration from the config file
    from analysis.multistate.multi_state_config import (
        # Mission details
        refuel_mission,
        # Tank configurations
        tank_configs,
        # Initial conditions
        initial_conditions_1_refuel, initial_conditions_2_refuel,
        # Operating envelopes
        operating_window_1, operating_window_2,
        # Interaction rules
        refuel_interaction_rules,
        # Timestep
        REFUEL_TIMESTEP
    )

    # Store original timestep
    original_timestep = MULTISTEP_METHOD.timestep

    # Set scenario-specific timestep
    MULTISTEP_METHOD.timestep = REFUEL_TIMESTEP
    print(f"Using timestep: {REFUEL_TIMESTEP} seconds for refuel scenario")

    # Print the mission requirement for clarity
    print(f"Target fuel mass for mission: {target_mass:.2f} kg")

    try:
        # Directly use the initial conditions from config file
        ic_list = [initial_conditions_1_refuel, initial_conditions_2_refuel]

        # Print the initial conditions for
        for i, ic in enumerate(ic_list):
            print(f"Tank {i+1} initial conditions: T={ic.temperature:.1f}K, P={ic.pressure/1e5:.1f}bar, mass_fraction={ic.mass_fraction:.6f}")

        # Run multi-tank analysis
        print("Analyzing multi-tank system for refuel scenario...")
        tank_performances = MultiTankAnalysisFacade.analyse(
            tank_configurations=tank_configs,
            mission=refuel_mission,
            interaction_rules=refuel_interaction_rules,  # Use imported rules instead of local definition
            initial_conditions=ic_list,
            operating_envelopes=[operating_window_1, operating_window_2],
            target_conditions=[
                TargetConditions(target_mass, 0.0),  # Stop when Tank 1 reaches target mass
                TargetConditions(999999.0, 0.0)      # Very high value so Tank 2 doesn't trigger stopping
            ]
        )

        # Extract results
        tank_states_1 = tank_performances[0].tank_states
        tank_states_2 = tank_performances[1].tank_states

        # Access flow data for plotting
        flow_data = MultiTankAnalysisFacade.flow_data

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="deep")

        # Plot tank mass flows
        flow_fig = plotter.plot_tank_mass_flows(
            flow_data['time'],
            flow_data['tank1_inflow'],
            flow_data['tank1_outflow'],
            flow_data['tank2_inflow'],
            flow_data['tank2_outflow']
        )

        # Create comparative plots
        comparative_fig = plotter.plot_comparative_tank_states(
            tank_states_1,
            tank_states_2,
            figsize=(15, 5),
            titles=[
                "Fuel Masses - Reservoir vs Consumer (Refuel)",
                "Tank Temperatures - Reservoir vs Consumer (Refuel)",
                "Tank Pressures - Reservoir vs Consumer (Refuel)"
            ]
        )

        plt.show()

        # Print final states
        print("\nRefuel scenario complete. Final states:")
        for i, state in enumerate(tank_performances):
            last_state = state.tank_states.last_state
            print(f"Tank {i+1}: T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")
            if i == 0:
                mission_percent = (last_state.fuel_mass / target_mass) * 100
                print(f"Tank 1 filled to {mission_percent:.1f}% of mission requirement")
        print()

        if return_performances:
            return tank_performances
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep