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

# Import configuration from the config file
from analysis.multistate.multi_state_config import (
    # Mission details
    mission, total_fuel_mass,
    # Tank configurations
    tank_configs,
    # Initial conditions
    initial_conditions_1, initial_conditions_2,
    # Operating envelopes
    operating_window_1, operating_window_2,
    # Interaction rules
    interaction_rules
)

def perform_discharge_analysis():
    # All the configuration is now imported from multi_state_config.py

    # Print mission information
    print(f"Mission created: {mission.__class__.__name__}")
    print(f"Number of sections: {len(mission.sections)}")
    for i, section in enumerate(mission.sections):
        print(f"Section {i+1}: {section.duration/3600:.4f}h")
    print(f"Total duration: {sum(s.duration for s in mission.sections)/3600:.4f}h")
    print(f"Total fuel mass required for mission: {total_fuel_mass:.2f} kg")


    # Run multi-tank analysis
    print("Analyzing multi-tank system for discharge scenario...")
    tank_performances = MultiTankAnalysisFacade.analyse(
        tank_configurations=tank_configs,
        mission=mission,
        interaction_rules=interaction_rules,
        initial_conditions=[initial_conditions_1, initial_conditions_2],
        operating_envelopes=[operating_window_1, operating_window_2],
        target_conditions=[
            TargetConditions(0.10 * total_fuel_mass, 0.0),
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

    # Plot the mass flows
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
            "Fuel Masses - Reservoir vs Consumer (Discharge)",
            "Tank Temperatures - Reservoir vs Consumer (Discharge)",
            "Tank Pressures - Reservoir vs Consumer (Discharge)"
        ]
    )

    plt.show()

    # Return the performance data for potential use
def main():
    perform_discharge_analysis()

if __name__ == "__main__":
    main()
