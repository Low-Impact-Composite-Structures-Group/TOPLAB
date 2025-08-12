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


def perform_dormancy_analysis(return_performances=False):
    """
    Perform dormancy analysis with configuration from config file.

    Args:
        return_performances: Whether to return the tank performances

    Returns:
        List of TankPerformance objects if return_performances is True
    """
    # Import configuration from the config file
    from analysis.multistate.multi_state_config import (
        # Mission details
        dormancy_mission,
        # Tank configurations
        tank_configs,
        # Initial conditions
        initial_conditions_1_dormancy, initial_conditions_2_dormancy,
        # Operating envelopes
        operating_window_1, operating_window_2,
        # Interaction rules
        dormancy_interaction_rules,
        # Timestep
        DORMANCY_TIMESTEP
    )

    # Store original timestep
    original_timestep = MULTISTEP_METHOD.timestep

    # Set scenario-specific timestep
    MULTISTEP_METHOD.timestep = DORMANCY_TIMESTEP
    print(f"\n=== DORMANCY ANALYSIS ===")
    print(f"Using timestep: {DORMANCY_TIMESTEP} seconds")
    print(f"Duration: {dormancy_mission.sections[0].duration/3600:.1f} hours")

    try:
        # Use initial conditions directly from config
        ic_list = [initial_conditions_1_dormancy, initial_conditions_2_dormancy]

        # Print the initial conditions for clarity
        print("\nInitial tank states:")
        for i, ic in enumerate(ic_list):
            print(f"Tank {i+1}: T={ic.temperature:.1f}K, P={ic.pressure/1e5:.1f}bar, mass_fraction={ic.mass_fraction:.4f}")

        # Run multi-tank analysis
        print("\nRunning dormancy simulation...")
        tank_performances = MultiTankAnalysisFacade.analyse(
            tank_configurations=tank_configs,
            mission=dormancy_mission,
            interaction_rules=dormancy_interaction_rules,
            initial_conditions=ic_list,
            operating_envelopes=[operating_window_1, operating_window_2],
            target_conditions=[
                TargetConditions(5.0, 0.0),  # Minimal targets since we want to run the full dormancy period
                TargetConditions(5.0, 0.0)
            ]
        )

        # Extract results
        tank_states_1 = tank_performances[0].tank_states
        tank_states_2 = tank_performances[1].tank_states

        # Access flow data for plotting
        flow_data = MultiTankAnalysisFacade.flow_data

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="deep")

        # Create comparative plots
        comparative_fig = plotter.plot_comparative_tank_states(
            tank_states_1,
            tank_states_2,
            figsize=(15, 5),
            titles=[
                "Fuel Masses - Reservoir vs Consumer (Dormancy)",
                "Tank Temperatures - Reservoir vs Consumer (Dormancy)",
                "Tank Pressures - Reservoir vs Consumer (Dormancy)"
            ]
        )

        plt.show()

        # Show final states
        print("\nDormancy scenario complete. Final states:")
        for i, state in enumerate(tank_performances):
            last_state = state.tank_states.last_state
            print(f"Tank {i+1}: T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performances
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


def main():
    perform_dormancy_analysis()


if __name__ == "__main__":
    main()
