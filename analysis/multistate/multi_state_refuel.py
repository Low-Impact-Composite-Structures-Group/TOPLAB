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


def perform_refuel_analysis():

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
        refuel_interaction_rules
    )

    # Print mission information
    print(f"Mission created: {refuel_mission.__class__.__name__}")

    # Run multi-tank analysis
    print("Analyzing multi-tank system for refuel scenario...")
    tank_performances = MultiTankAnalysisFacade.analyse(
        tank_configurations=tank_configs,
        mission=refuel_mission,
        interaction_rules=refuel_interaction_rules,
        initial_conditions=[initial_conditions_1_refuel, initial_conditions_2_refuel],
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

    # Return the performance data for potential use

def main():
    perform_refuel_analysis()

if __name__ == "__main__":
    main()

