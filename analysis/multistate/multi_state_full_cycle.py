import argparse
import sys
from analysis.multistate.multi_state_discharge import perform_discharge_analysis
from analysis.multistate.multi_state_refuel import perform_refuel_analysis
from analysis.multistate.multi_state_dormancy import perform_dormancy_analysis
from facades.analysis_facades import InitialConditions

def perform_analysis(mode=None, scenarios=None):
    """
    Perform hydrogen tank analysis in different modes.

    Args:
        mode (str): "individual" to run separate scenarios or "cycle" to run sequential scenarios
        scenarios (list): Scenarios to run ["refuel", "discharge", "dormancy"] or subset
    """
    # Check if called directly or via main.py
    # If mode is None, we were called from main.py and should check for command line args
    if mode is None:
        # Parse command line args only if this is being run directly
        # (not when imported from main.py)
        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description="Run hydrogen tank analysis")
            parser.add_argument("--mode", choices=["individual", "cycle"], default="individual",
                              help="Run individual scenarios or full cycle analysis")
            parser.add_argument("--scenarios", nargs="+", choices=["refuel", "discharge", "dormancy"],
                              help="Scenarios to run (default: all)")

            args = parser.parse_args()
            mode = args.mode
            scenarios = args.scenarios
        else:
            # Default behavior when called from main.py without args
            mode = "individual"
            scenarios = None

    if mode == "individual":
        # Run the selected individual scenario (existing behavior)
        if scenarios is None or "refuel" in scenarios:
            perform_refuel_analysis()
        if scenarios is None or "discharge" in scenarios:
            perform_discharge_analysis()
        if scenarios is None or "dormancy" in scenarios:
            perform_dormancy_analysis()

    elif mode == "cycle":
        # Run scenarios sequentially with state transfer between them
        if scenarios is None:
            scenarios = ["refuel", "discharge", "dormancy"]

        prev_tank_states = None

        # Run each scenario in sequence, passing final state to next scenario
        for scenario in scenarios:
            if scenario == "refuel":
                tank_performances = perform_refuel_analysis(return_performances=True)
                prev_tank_states = [perf.tank_states.last_state for perf in tank_performances]
                print(f"\nRefuel scenario complete. Final states:")
                for i, state in enumerate(prev_tank_states):
                    print(f"Tank {i+1}: T={state.temperature:.1f}K, P={state.pressure/1e5:.1f}bar, mass={state.fuel_mass:.1f}kg")

            elif scenario == "discharge":
                if prev_tank_states:
                    # Transfer states from previous scenario
                    tank_performances = perform_discharge_analysis(
                        initial_states=prev_tank_states,
                        return_performances=True
                    )
                    prev_tank_states = [perf.tank_states.last_state for perf in tank_performances]
                    print(f"\nDischarge scenario complete. Final states:")
                    for i, state in enumerate(prev_tank_states):
                        print(f"Tank {i+1}: T={state.temperature:.1f}K, P={state.pressure/1e5:.1f}bar, mass={state.fuel_mass:.1f}kg")
                else:
                    tank_performances = perform_discharge_analysis(return_performances=True)
                    prev_tank_states = [perf.tank_states.last_state for perf in tank_performances]

            elif scenario == "dormancy":
                if prev_tank_states:
                    # Transfer states from previous scenario
                    tank_performances = perform_dormancy_analysis(
                        initial_states=prev_tank_states,
                        return_performances=True
                    )
                    prev_tank_states = [perf.tank_states.last_state for perf in tank_performances]
                    print(f"\nDormancy scenario complete. Final states:")
                    for i, state in enumerate(prev_tank_states):
                        print(f"Tank {i+1}: T={state.temperature:.1f}K, P={state.pressure/1e5:.1f}bar, mass={state.fuel_mass:.1f}kg")
                else:
                    perform_dormancy_analysis()

if __name__ == "__main__":
    perform_analysis()