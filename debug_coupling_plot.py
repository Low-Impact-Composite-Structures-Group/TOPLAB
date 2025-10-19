#!/usr/bin/env python3
"""
Debug script to test coupling flow plotting - create scenario with actual coupling flows.
"""

import sys
sys.path.append('.')

from src.multi_tank.orchestration.system_orchestrator import SystemOrchestrator
from analysis.multi_tank_systems.shared_configs.ch2_cch2_coupled_config import create_ch2_cch2_coupled_config
from src.plotting.multi_tank_plotting import DelftColourPlotter
import matplotlib.pyplot as plt

def create_coupling_scenario():
    """Create a scenario designed to trigger coupling flows."""
    print("🔧 Creating coupling scenario...")

    # Create configuration with coupling
    config = create_ch2_cch2_coupled_config()

    # Modify mission duration to be longer to trigger coupling flows
    config.mission.sections[0].duration = 7200  # 2 hours instead of short duration

    # Lower the coupling valve activation pressure to ensure it triggers
    config.coupling_rules[0].activation_conditions['pressure_threshold'] = 10.0  # 10 bar instead of 16
    config.coupling_rules[0].hysteresis['deactivation_pressure'] = 15.0  # 15 bar instead of 30

    # Increase max flow rate to see more significant coupling
    config.coupling_rules[0].flow_parameters['max_flow_rate'] = 200.0  # 200 g/s instead of 100

    print(f"   Modified coupling: opens at {config.coupling_rules[0].activation_conditions['pressure_threshold']} bar")
    print(f"   Modified coupling: closes at {config.coupling_rules[0].hysteresis['deactivation_pressure']} bar")
    print(f"   Modified coupling: max flow {config.coupling_rules[0].flow_parameters['max_flow_rate']} g/s")

    return config

def debug_coupling_flows():
    """Run simulation with coupling flows and debug the plotting data."""

    # Create scenario designed to produce coupling flows
    config = create_coupling_scenario()

    # Create and run system
    orchestrator = SystemOrchestrator(config)
    print("🚀 Running simulation with coupling flows...")
    results = orchestrator.run_simulation()

    print(f"📊 Simulation completed: {len(results.times)} timesteps, {results.n_tanks} tanks")

    # Extract coupling flow data for both tanks
    print("\n🔍 Analyzing coupling flow data:")

    for tank_idx in range(results.n_tanks):
        tank_data = results._extract_tank_arrays(tank_idx)

        max_coupling_inflow = max(tank_data['coupling_inflow_rates'])
        max_coupling_outflow = max(tank_data['coupling_outflow_rates'])
        mean_coupling_inflow = sum(tank_data['coupling_inflow_rates'])/len(tank_data['coupling_inflow_rates'])
        mean_coupling_outflow = sum(tank_data['coupling_outflow_rates'])/len(tank_data['coupling_outflow_rates'])

        print(f"\nTank {tank_idx + 1}:")
        print(f"   Max coupling inflow: {max_coupling_inflow:.3f} g/s")
        print(f"   Max coupling outflow: {max_coupling_outflow:.3f} g/s")
        print(f"   Mean coupling inflow: {mean_coupling_inflow:.3f} g/s")
        print(f"   Mean coupling outflow: {mean_coupling_outflow:.3f} g/s")

        # Show first few non-zero values if any
        nonzero_inflow = [x for x in tank_data['coupling_inflow_rates'] if x > 0.1]
        nonzero_outflow = [x for x in tank_data['coupling_outflow_rates'] if x > 0.1]

        if nonzero_inflow:
            print(f"   First 5 non-zero inflow values: {nonzero_inflow[:5]}")
        if nonzero_outflow:
            print(f"   First 5 non-zero outflow values: {nonzero_outflow[:5]}")

    # Create and test the plotting
    print("\n🎨 Testing mass flow plotting...")
    plotter = DelftColourPlotter(analysis_name="Coupling Debug", use_greyscale=False)

    # Plot mass flows for Tank 2 (should show coupling inflows)
    fig = plotter.plot_mass_flows(results, tank_index=1, include_coupling_flows=True)
    plt.savefig("debug_coupling_flows_tank2.png", dpi=150, bbox_inches='tight')
    print("   Saved: debug_coupling_flows_tank2.png")

    # Plot mass flows for Tank 1 (should show coupling outflows)
    fig = plotter.plot_mass_flows(results, tank_index=0, include_coupling_flows=True)
    plt.savefig("debug_coupling_flows_tank1.png", dpi=150, bbox_inches='tight')
    print("   Saved: debug_coupling_flows_tank1.png")

    plt.close('all')

    return results

if __name__ == "__main__":
    print("🔧 Testing coupling flow plotting...")
    results = debug_coupling_flows()
    print("✅ Coupling flow debug complete!")