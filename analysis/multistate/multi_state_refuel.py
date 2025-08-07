import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_single_tank_temperatures, plot_single_tank_loads
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt


def perform_refuel_analysis(initial_states=None, return_performances=False):
    """
    Perform refuel analysis with optional initial states from previous scenario.
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

    # Import the multistep method directly
    from facades.analysis_facades import MULTISTEP_METHOD

    # Store original timestep
    original_timestep = MULTISTEP_METHOD.timestep

    # Set scenario-specific timestep
    MULTISTEP_METHOD.timestep = REFUEL_TIMESTEP
    print(f"Using timestep: {REFUEL_TIMESTEP} seconds for refuel scenario")

    try:
        # Use provided initial states if available
        ic_list = [initial_conditions_1_refuel, initial_conditions_2_refuel]
        if initial_states:
            ic_list = []
            for i, state in enumerate(initial_states):
                # Convert TankState to InitialConditions
                ic = InitialConditions(
                    pressure=state.pressure,
                    temperature=state.temperature,
                    fill=0.0,  # Assuming no liquid for refuel initial state
                    multi_flow=True,
                    mass_fraction=state.fuel_mass / state.tank.compute_max_fuel_mass(state.temperature, state.pressure)
                )
                ic_list.append(ic)
                print(f"Using initial state for Tank {i+1} from previous scenario: T={state.temperature:.1f}K, P={state.pressure/1e5:.1f}bar")

        # Run multi-tank analysis
        print("Analyzing multi-tank system for refuel scenario...")
        tank_performances = MultiTankAnalysisFacade.analyse(
            tank_configurations=tank_configs,
            mission=refuel_mission,
            interaction_rules=refuel_interaction_rules,
            initial_conditions=ic_list,
            operating_envelopes=[operating_window_1, operating_window_2],
            target_conditions=[
                TargetConditions(600, 0.0),
                TargetConditions(5.0, 0.0)
            ]
        )

        # Extract results
        tank_states_1 = tank_performances[0].tank_states
        tank_states_2 = tank_performances[1].tank_states

        # Access flow data for plotting
        flow_data = MultiTankAnalysisFacade.flow_data

        # Initialize the plotter at the beginning of the analysis
        plotter = SeabornPlotter(font="Cambria", palette="deep")

        flow_fig = plotter.plot_tank_mass_flows(
            flow_data['time'],
            flow_data['tank1_inflow'],
            flow_data['tank1_outflow'],
            flow_data['tank2_inflow'],
            flow_data['tank2_outflow']
        )

        # Create comparative plots (all in one figure)
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

        # Add this before the finally block
        print("\nRefuel scenario complete. Final states:")
        for i, state in enumerate(tank_performances):
            last_state = state.tank_states.last_state
            print(f"Tank {i+1}: T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")
        print()

        # Then in the if return_performances block, just do the return
        if return_performances:
            return tank_performances
    finally:
        # Restore original timestep - critical to avoid side effects
        MULTISTEP_METHOD.timestep = original_timestep