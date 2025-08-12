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


def perform_discharge_analysis(return_performances=False):
    """Perform discharge analysis with cleaner output"""
    from analysis.multistate.multi_state_config import (
        mission, total_fuel_mass, tank_configs,
        initial_conditions_1, initial_conditions_2,
        operating_window_1, operating_window_2, interaction_rules, DISCHARGE_TIMESTEP
    )

    # Set timestep
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DISCHARGE_TIMESTEP
    print(f"\n=== DISCHARGE ANALYSIS ===")
    print(f"Using timestep: {DISCHARGE_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission type: {mission.__class__.__name__}")
        print(f"Required mission fuel: {total_fuel_mass:.2f} kg")

        # Create and adjust initial conditions
        ic_list = [initial_conditions_1, initial_conditions_2]
        print("\nInitial tank states:")
        for i, ic in enumerate(ic_list):
            print(f"Tank {i+1}: T={ic.temperature:.1f}K, P={ic.pressure/1e5:.1f}bar, mass_fraction={ic.mass_fraction:.4f}")

        # Run analysis
        print("\nRunning simulation...")
        tank_performances = MultiTankAnalysisFacade.analyse(
            tank_configurations=tank_configs,
            mission=mission,
            interaction_rules=interaction_rules,
            initial_conditions=ic_list,
            operating_envelopes=[operating_window_1, operating_window_2],
            target_conditions=[
                TargetConditions(0.10 * total_fuel_mass, 0.0),
                TargetConditions(5.0, 0.0)
            ]
        )

        # Extract results and plot
        tank_states_1 = tank_performances[0].tank_states
        tank_states_2 = tank_performances[1].tank_states

        # Access flow data for plotting
        flow_data = MultiTankAnalysisFacade.flow_data

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="deep")

        # Plot the mass flows
        flow_fig = plotter.plot_tank_mass_flows(
            flow_data['time'],
            flow_data['tank1_inflow'],
            flow_data['tank1_outflow'],
            flow_data['tank2_inflow'],
            flow_data['tank2_outflow']
        )

        # Plot the tank states
        comparative_fig = plotter.plot_comparative_tank_states(
            tank_states_1, tank_states_2, figsize=(15, 5),
            titles=["Fuel Masses - Reservoir vs Consumer (Discharge)",
                   "Tank Temperatures - Reservoir vs Consumer (Discharge)",
                   "Tank Pressures - Reservoir vs Consumer (Discharge)"]
        )

        plt.show()

        # Show final states
        print("\nDischarge scenario complete. Final states:")
        for i, state in enumerate(tank_performances):
            last_state = state.tank_states.last_state
            print(f"Tank {i+1}: T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performances
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


def main():
    perform_discharge_analysis()


if __name__ == "__main__":
    main()