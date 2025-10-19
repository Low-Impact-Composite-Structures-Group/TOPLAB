#!/usr/bin/env python3
"""
Direct debug of coupling flows from real multi-tank simulation
"""

import sys
sys.path.append('.')

# Run the coupled analysis and examine the coupling flow data directly
def examine_coupling_data():
    """Run simulation and directly examine coupling flow data."""
    print("🔍 Running coupled simulation and examining coupling flow data...")

    # Import and run the driver similar to how it works
    from src.orchestration.system_orchestrator import SystemOrchestrator
    from src.configuration.enhanced_scenario_configuration import ConfigurationLoader

    # Load the same configuration used in the driver
    config_path = "analysis/multi_tank_systems/coupled_ch2_cch2/coupled_ch2_cch2_config_new_format.yaml"

    print(f"📄 Loading configuration: {config_path}")
    config = ConfigurationLoader.load_enhanced_config(config_path)

    print("🔧 Creating orchestrator...")
    orchestrator = SystemOrchestrator(config)

    print("🚀 Running simulation...")
    results = orchestrator.run_simulation()

    print(f"📊 Simulation completed: {len(results.times)} timesteps, {results.n_tanks} tanks")

    # Now examine the coupling flow data
    print("\n🔍 EXAMINING COUPLING FLOW DATA:")

    for tank_idx in range(results.n_tanks):
        print(f"\n--- Tank {tank_idx + 1} ---")
        tank_data = results._extract_tank_arrays(tank_idx)

        # Get coupling flow statistics
        coupling_inflows = tank_data['coupling_inflow_rates']
        coupling_outflows = tank_data['coupling_outflow_rates']
        mission_inflows = tank_data['inflow_rates']
        mission_outflows = tank_data['outflow_rates']

        print(f"Coupling inflows - Max: {max(coupling_inflows):.3f} g/s, Mean: {sum(coupling_inflows)/len(coupling_inflows):.3f} g/s")
        print(f"Coupling outflows - Max: {max(coupling_outflows):.3f} g/s, Mean: {sum(coupling_outflows)/len(coupling_outflows):.3f} g/s")
        print(f"Mission inflows - Max: {max(mission_inflows):.3f} g/s, Mean: {sum(mission_inflows)/len(mission_inflows):.3f} g/s")
        print(f"Mission outflows - Max: {max(mission_outflows):.3f} g/s, Mean: {sum(mission_outflows)/len(mission_outflows):.3f} g/s")

        # Find specific time periods with coupling flows
        nonzero_coupling_inflow_indices = [i for i, x in enumerate(coupling_inflows) if x > 0.1]
        nonzero_coupling_outflow_indices = [i for i, x in enumerate(coupling_outflows) if x > 0.1]

        if nonzero_coupling_inflow_indices:
            print(f"Non-zero coupling inflows at {len(nonzero_coupling_inflow_indices)} time points")
            print(f"First few values: {[coupling_inflows[i] for i in nonzero_coupling_inflow_indices[:5]]}")
        else:
            print("No significant coupling inflows detected")

        if nonzero_coupling_outflow_indices:
            print(f"Non-zero coupling outflows at {len(nonzero_coupling_outflow_indices)} time points")
            print(f"First few values: {[coupling_outflows[i] for i in nonzero_coupling_outflow_indices[:5]]}")
        else:
            print("No significant coupling outflows detected")

    # Now let's check what the plotting function would display
    print("\n🎨 EXAMINING WHAT PLOTTING FUNCTION DISPLAYS:")

    from src.plotting.multi_tank_plotting import DelftColourPlotter
    import matplotlib.pyplot as plt

    plotter = DelftColourPlotter(analysis_name="Debug Coupling", use_greyscale=False)

    # Extract the exact same data that the plotting function uses
    for tank_idx in range(results.n_tanks):
        print(f"\n--- Plotting Data for Tank {tank_idx + 1} ---")
        tank_data = results._extract_tank_arrays(tank_idx)

        # This is the exact same logic from the plotting function
        coupling_inflow_rates = tank_data['coupling_inflow_rates']
        coupling_outflow_rates = tank_data['coupling_outflow_rates']

        # Total inflow = mission inflow + coupling inflow
        total_inflow = []
        for i in range(len(tank_data['inflow_rates'])):
            mission_inflow = tank_data['inflow_rates'][i] if i < len(tank_data['inflow_rates']) else 0
            coupling_inflow = coupling_inflow_rates[i] if i < len(coupling_inflow_rates) else 0
            total_inflow.append(mission_inflow + coupling_inflow)

        # Total outflow = mission outflow + coupling outflow (make negative for display)
        total_outflow = []
        for i in range(len(tank_data['outflow_rates'])):
            mission_outflow = tank_data['outflow_rates'][i] if i < len(tank_data['outflow_rates']) else 0
            coupling_outflow = coupling_outflow_rates[i] if i < len(coupling_outflow_rates) else 0
            total_outflow.append(-(mission_outflow + coupling_outflow))  # Negative for display

        print(f"Total inflow (mission + coupling) - Max: {max(total_inflow):.3f} g/s")
        print(f"Total outflow (mission + coupling) - Min: {min(total_outflow):.3f} g/s (negative for display)")

        # Check if coupling is being included properly
        max_mission_inflow = max(tank_data['inflow_rates'])
        max_coupling_inflow = max(coupling_inflow_rates)
        max_total_inflow = max(total_inflow)

        print(f"Verification: Max mission inflow = {max_mission_inflow:.3f}, Max coupling inflow = {max_coupling_inflow:.3f}")
        print(f"Expected total = {max_mission_inflow + max_coupling_inflow:.3f}, Actual total = {max_total_inflow:.3f}")

        if abs((max_mission_inflow + max_coupling_inflow) - max_total_inflow) < 0.001:
            print("✅ Coupling flows are correctly included in plot data")
        else:
            print("❌ Coupling flows may not be correctly included in plot data")

    return results

if __name__ == "__main__":
    results = examine_coupling_data()
    print("✅ Coupling flow examination complete!")