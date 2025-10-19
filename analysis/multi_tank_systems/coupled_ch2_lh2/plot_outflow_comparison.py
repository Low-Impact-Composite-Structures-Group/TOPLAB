#!/usr/bin/env python3
"""
Plot Target vs Actual Outflow Comparison

This script creates a comparison plot showing:
1. Target outflow (mission demand)
2. Actual outflow (pressure-based calculation)
3. Coupling inflow from CH2 tank

This reveals how well the flow-matching controller performs.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from src.multi_tank.system.multi_tank_system import MultiTankSystem
from src.configuration.enhanced_scenario_configuration import EnhancedScenarioConfig

def load_system_with_history():
    """Load the multi-tank system and run to collect outflow history."""

    # Load configuration
    config_path = "coupled_ch2_lh2_config_new_format.yaml"
    config = EnhancedScenarioConfig.from_yaml(config_path)

    # Create system
    system = MultiTankSystem(config)

    # Run simulation
    print("🚀 Running simulation to collect outflow history...")

    # Set up initial conditions
    tank1_config = config.tank_configs[0]
    tank2_config = config.tank_configs[1]

    # Calculate initial state
    T1 = tank1_config.initial_temperature_K
    P1 = tank1_config.initial_pressure_bar * 1e5  # Convert to Pa
    rho1 = system.tanks[0].fluid.density(T1, P1)
    m1 = rho1 * system.tanks[0].geometry.volume

    T2 = tank2_config.initial_temperature_K
    P2 = tank2_config.initial_pressure_bar * 1e5  # Convert to Pa
    rho2 = system.tanks[1].fluid.density(T2, P2)
    m2 = rho2 * system.tanks[1].geometry.volume

    # Initial state vector: [m1, T1, m2, T2]
    y0 = [m1, T1, m2, T2]

    # Mission duration (1.02 hours = 3688.8 seconds)
    mission_duration = 3688.8
    t_span = (0.0, mission_duration)

    # Run simulation
    from scipy.integrate import solve_ivp

    def system_ode(t, y):
        return system.system_derivatives(t, y)

    sol = solve_ivp(
        system_ode,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-6,
        atol=1e-8,
        max_step=20.0,
        dense_output=True
    )

    print(f"✅ Simulation completed. Final time: {sol.t[-1]:.1f}s")

    # Get coupling history from the flow-matching valve
    coupling_valve = None
    for valve in system.coupling_valves:
        if hasattr(valve, 'outflow_history'):
            coupling_valve = valve
            break

    if coupling_valve is None:
        print("❌ No coupling valve with outflow history found!")
        return None

    history = coupling_valve.outflow_history

    print(f"📊 Collected {len(history['time'])} data points")
    return history

def create_outflow_comparison_plot(history):
    """Create target vs actual outflow comparison plot."""

    if not history or len(history['time']) == 0:
        print("❌ No history data available for plotting!")
        return

    # Convert to numpy arrays
    times = np.array(history['time'])
    target_outflow = np.array(history['target_outflow']) * 1000  # Convert to g/s
    actual_outflow = np.array(history['actual_outflow']) * 1000  # Convert to g/s
    coupling_inflow = np.array(history['coupling_inflow']) * 1000  # Convert to g/s

    # Convert time to minutes
    times_min = times / 60.0

    print(f"📊 Plotting {len(times)} data points")
    print(f"   Time range: {times_min[0]:.1f} - {times_min[-1]:.1f} minutes")
    print(f"   Target outflow range: {target_outflow.min():.1f} - {target_outflow.max():.1f} g/s")
    print(f"   Actual outflow range: {actual_outflow.min():.1f} - {actual_outflow.max():.1f} g/s")
    print(f"   Coupling inflow range: {coupling_inflow.min():.1f} - {coupling_inflow.max():.1f} g/s")

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Main comparison plot
    plt.subplot(2, 1, 1)
    plt.plot(times_min, target_outflow, 'b-', linewidth=2, label='Target Outflow (Mission Demand)', alpha=0.8)
    plt.plot(times_min, actual_outflow, 'r-', linewidth=2, label='Actual Outflow (Pressure-Based)', alpha=0.8)
    plt.plot(times_min, coupling_inflow, 'g--', linewidth=2, label='Coupling Inflow (CH2→LH2)', alpha=0.8)

    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Flow Rate (g/s)', fontsize=14)
    plt.title('Target vs Actual Outflow Comparison\nFlow-Matching PID Controller Performance', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, times_min[-1])

    # Calculate and show key metrics
    avg_target = np.mean(target_outflow)
    avg_actual = np.mean(actual_outflow)
    avg_coupling = np.mean(coupling_inflow)

    # Error analysis
    outflow_error = target_outflow - actual_outflow
    max_error = np.max(np.abs(outflow_error))
    rmse_error = np.sqrt(np.mean(outflow_error**2))

    plt.text(0.02, 0.98, f'Average Target: {avg_target:.1f} g/s\n'
                          f'Average Actual: {avg_actual:.1f} g/s\n'
                          f'Average Coupling: {avg_coupling:.1f} g/s\n'
                          f'Max Error: {max_error:.1f} g/s\n'
                          f'RMSE Error: {rmse_error:.1f} g/s',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Error plot
    plt.subplot(2, 1, 2)
    plt.plot(times_min, outflow_error, 'purple', linewidth=2, label='Outflow Error (Target - Actual)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Error (g/s)', fontsize=14)
    plt.title('Outflow Error: Target - Actual', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, times_min[-1])

    plt.tight_layout()

    # Save plot
    output_path = "output/plots/outflow_comparison_flow_matching_controller.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Plot saved to: {output_path}")

    # Summary analysis
    print("\n" + "="*80)
    print("📊 OUTFLOW COMPARISON ANALYSIS")
    print("="*80)
    print(f"Target Outflow Statistics:")
    print(f"  Average: {avg_target:.2f} g/s")
    print(f"  Range: {target_outflow.min():.1f} - {target_outflow.max():.1f} g/s")

    print(f"\nActual Outflow Statistics:")
    print(f"  Average: {avg_actual:.2f} g/s")
    print(f"  Range: {actual_outflow.min():.1f} - {actual_outflow.max():.1f} g/s")

    print(f"\nCoupling Inflow Statistics:")
    print(f"  Average: {avg_coupling:.2f} g/s")
    print(f"  Range: {coupling_inflow.min():.1f} - {coupling_inflow.max():.1f} g/s")
    print(f"  Max coupling flow (still hitting limit): {coupling_inflow.max():.1f} g/s")

    print(f"\nError Analysis:")
    print(f"  Maximum absolute error: {max_error:.2f} g/s")
    print(f"  RMSE: {rmse_error:.2f} g/s")
    print(f"  Mean error: {np.mean(outflow_error):.2f} g/s")

    # Check if coupling is saturated
    coupling_limit = 110.0  # g/s
    saturated_points = np.sum(coupling_inflow >= 0.95 * coupling_limit)
    saturation_percent = 100 * saturated_points / len(coupling_inflow)

    print(f"\nCoupling Saturation Analysis:")
    print(f"  Coupling limit: {coupling_limit} g/s")
    print(f"  Points at >95% of limit: {saturated_points} ({saturation_percent:.1f}%)")

    if saturation_percent > 10:
        print(f"  ❌ HIGH SATURATION: Coupling frequently hits flow limit!")
        print(f"     This indicates the pressure-based approach is struggling.")
    else:
        print(f"  ✅ LOW SATURATION: Coupling rarely hits flow limit.")

if __name__ == "__main__":
    print("📊 Outflow Comparison Analysis for Flow-Matching Controller")
    print("="*80)

    # Load system and run simulation
    history = load_system_with_history()

    if history:
        # Create comparison plot
        create_outflow_comparison_plot(history)
    else:
        print("❌ Failed to collect outflow history data!")