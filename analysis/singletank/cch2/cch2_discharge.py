import matplotlib.pyplot as plt
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import MULTISTEP_METHOD, OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MissionAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt


def perform_discharge_analysis(return_performances=False):
    """Perform discharge analysis with cleaner output"""
    from analysis.singletank.cch2.cch2_config import (
        mission, total_fuel_mass, tank_config,
        initial_conditions_disch,
        operating_window_disch, DISCHARGE_TIMESTEP
    )

    # Set timestep
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DISCHARGE_TIMESTEP
    print(f"\n=== DISCHARGE ANALYSIS ===")
    print(f"Using timestep: {DISCHARGE_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {mission}")
        print(f"Required mission fuel: {total_fuel_mass:.2f} kg")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_disch.temperature:.1f}K, P={initial_conditions_disch.pressure/1e5:.1f}bar, mass={initial_conditions_disch.mass_fraction*total_fuel_mass:.1f}kg")

        # Run analysis
        print("\nRunning simulation...")
        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_config[0]['dimensions'],
            material=tank_config[0]['material'],
            insulation=tank_config[0]['insulation'],
            mission=mission,
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
        for section in mission.sections:
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


def main():
    perform_discharge_analysis()


if __name__ == "__main__":
    main()