import matplotlib.pyplot as plt
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import MULTISTEP_METHOD, OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, FillingAnalysisFacade, RefuellingAnalysisFacade, InOutTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt


def perform_refuel_analysis(return_performances=False):
    """Perform refuel analysis with cleaner output"""
    from analysis.singletank.cch2.cch2_config import (
        refuel_mission, total_fuel_mass, tank_config,
        initial_conditions_refuel,
        operating_window_refuel, REFUEL_TIMESTEP
    )

    # Set timestep
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = REFUEL_TIMESTEP
    print(f"\n=== REFUEL ANALYSIS ===")
    print(f"Using timestep: {REFUEL_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {refuel_mission}")
        print(f"Required mission fuel: {total_fuel_mass:.2f} kg")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_refuel.temperature:.1f}K, P={initial_conditions_refuel.pressure/1e5:.1f}bar, mass={initial_conditions_refuel.mass_fraction*total_fuel_mass:.1f}kg")

        # Ensure multi_flow is True (redundant but safer)
        initial_conditions_refuel.multi_flow = True

        # Run analysis
        print("\nRunning simulation...")
        target_conditions = TargetConditions(
            fuel_mass=total_fuel_mass,
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
            invert_flow=True  # This is the key change!
        )
        plt.savefig("refuel_mass_flows.png", dpi=300)

        # Show only the two figures we want
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


def main():
    perform_refuel_analysis()


if __name__ == "__main__":
    main()