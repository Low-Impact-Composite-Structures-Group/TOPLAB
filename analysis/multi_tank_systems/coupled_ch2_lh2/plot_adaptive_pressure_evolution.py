#!/usr/bin/env python3
"""
Plot Mission-Adaptive Pressure Evolution Analysis
================================================================================
Visualizes the evolving pressure requirements in Tank 2 to enable flow over time
for the CH2→LH2 coupled system with mission-adaptive pressurization.

This script extracts diagnostic data from the MissionAdaptivePressureValve to show:
- Real-time mission flow requirements
- Dynamic pressure thresholds (activation/deactivation)
- Required discharge pressure based on piping physics
- Actual tank pressure vs. required pressure
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.orchestration.system_orchestrator import SystemOrchestrator
from src.configuration.scenario_configuration import ScenarioConfig


def plot_adaptive_pressure_evolution():
    """Plot the evolution of adaptive pressure requirements during mission."""

    print("🔍 MISSION-ADAPTIVE PRESSURE EVOLUTION ANALYSIS")
    print("="*80)
    print("Analyzing dynamic pressure requirements for CH2→LH2 coupling system")
    print("Extracting valve diagnostic data for real-time pressure calculations")
    print("="*80)

    # Load configuration and create orchestrator
    config_path = "coupled_ch2_lh2_config.yaml"
    config = ScenarioConfig.from_yaml(config_path)
    orchestrator = SystemOrchestrator(config)

    # Run simulation to collect dynamic data
    print("🚀 Running simulation to collect valve diagnostic data...")
    solver_method = "LSODA"
    solver_config = {
        'rtol': 1e-4,
        'atol': 1e-7,
        'max_step': 10.0
    }

    results = orchestrator.run_simulation(solver_method=solver_method, solver_config=solver_config)

    # Extract valve diagnostic data
    print("📊 Extracting valve diagnostic data...")
    valve = orchestrator.tank_system.coupling_valves[0]  # Mission adaptive valve
    diagnostic_data = valve.get_diagnostic_data()

    # Get simulation time and tank pressure data
    time_data = results.times / 3600  # Convert to hours
    combined_data = results.get_combined_data()
    tank2_pressure = combined_data['pressures'][1]  # Already in bar from get_combined_data    # Extract valve diagnostic arrays
    valve_time = np.array(diagnostic_data['time_history']) / 3600  # Convert to hours
    required_pressure = np.array(diagnostic_data['required_pressure_history']) / 1e5  # Convert to bar
    activation_threshold = np.array(diagnostic_data['activation_threshold_history']) / 1e5  # Convert to bar
    mission_flow = np.array(diagnostic_data['mission_flow_history']) * 1000  # Convert to g/s

    # Create the comprehensive pressure evolution plot
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mission-Adaptive Pressure Evolution: CH2→LH2 Coupling System',
                 fontsize=16, fontweight='bold')

    # Plot 1: Mission Flow Profile
    ax1.plot(valve_time, mission_flow, 'b-', linewidth=2, label='Mission Flow Rate')
    ax1.fill_between(valve_time, 0, mission_flow, alpha=0.3, color='blue')
    ax1.set_xlabel('Mission Time (hours)')
    ax1.set_ylabel('Flow Rate (g/s)')
    ax1.set_title('ATR72 Mission Flow Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add flow rate statistics
    max_flow = np.max(mission_flow)
    avg_flow = np.mean(mission_flow)
    ax1.text(0.02, 0.98, f'Max: {max_flow:.1f} g/s\nAvg: {avg_flow:.1f} g/s',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Required vs. Actual Pressure
    ax2.plot(time_data, tank2_pressure, 'r-', linewidth=2, label='Actual Tank 2 Pressure')
    ax2.plot(valve_time, required_pressure, 'g--', linewidth=2, label='Required Discharge Pressure')
    ax2.plot(valve_time, activation_threshold, 'orange', linewidth=1.5, label='Activation Threshold')

    # Highlight pressure deficit regions
    pressure_deficit = tank2_pressure < np.interp(time_data, valve_time, required_pressure)
    ax2.fill_between(time_data, 0, tank2_pressure, where=pressure_deficit,
                     alpha=0.2, color='red', label='Pressure Shortfall')

    ax2.set_xlabel('Mission Time (hours)')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Dynamic Pressure Requirements vs. Actual Tank Pressure')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Pressure Margins and Safety
    pressure_margin = tank2_pressure - np.interp(time_data, valve_time, required_pressure)
    ax3.plot(time_data, pressure_margin, 'purple', linewidth=2, label='Pressure Margin')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Margin (Critical)')
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Design Margin (1 bar)')

    # Color-code safe vs. critical regions
    ax3.fill_between(time_data, pressure_margin, 0, where=(pressure_margin >= 0),
                     alpha=0.3, color='green', label='Safe Operation')
    ax3.fill_between(time_data, pressure_margin, 0, where=(pressure_margin < 0),
                     alpha=0.3, color='red', label='Critical Operation')

    ax3.set_xlabel('Mission Time (hours)')
    ax3.set_ylabel('Pressure Margin (bar)')
    ax3.set_title('Safety Margin: Available vs. Required Pressure')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Dynamic Threshold Evolution
    ax4.plot(valve_time, activation_threshold, 'orange', linewidth=2, label='Activation Threshold')
    ax4.plot(valve_time, required_pressure, 'green', linewidth=2, label='Required Pressure')
    ax4.fill_between(valve_time, required_pressure, activation_threshold,
                     alpha=0.3, color='orange', label='Control Margin')

    ax4.set_xlabel('Mission Time (hours)')
    ax4.set_ylabel('Pressure (bar)')
    ax4.set_title('Evolution of Dynamic Pressure Thresholds')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add comprehensive statistics text box
    stats_text = f"""MISSION STATISTICS:
Duration: {valve_time[-1]:.2f} hours
Total Fuel: {np.trapz(mission_flow, valve_time*3600)/1000:.1f} kg
Peak Flow: {max_flow:.1f} g/s
Min Required P: {np.min(required_pressure):.1f} bar
Max Required P: {np.max(required_pressure):.1f} bar
Avg Margin: {np.mean(pressure_margin):.1f} bar"""

    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    output_dir = "output/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{output_dir}/Mission_Adaptive_Pressure_Evolution.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    print(f"✅ Pressure evolution plot saved: {plot_filename}")
    print("\n📊 ANALYSIS SUMMARY:")
    print("="*50)
    print(f"Mission Duration: {valve_time[-1]:.2f} hours")
    print(f"Total Fuel Consumed: {np.trapz(mission_flow, valve_time*3600)/1000:.1f} kg")
    print(f"Peak Flow Rate: {max_flow:.1f} g/s")
    print(f"Average Flow Rate: {avg_flow:.1f} g/s")
    print(f"Minimum Required Pressure: {np.min(required_pressure):.1f} bar")
    print(f"Maximum Required Pressure: {np.max(required_pressure):.1f} bar")
    print(f"Average Pressure Margin: {np.mean(pressure_margin):.1f} bar")
    print(f"Minimum Pressure Margin: {np.min(pressure_margin):.1f} bar")

    if np.min(pressure_margin) < 0:
        print("⚠️  WARNING: Negative pressure margins detected - system may require pressurization")
        critical_times = time_data[pressure_margin < 0]
        print(f"   Critical periods: {len(critical_times)} time points")
        print(f"   First critical time: {critical_times[0]:.3f} hours")
    else:
        print("✅ All pressure margins positive - system operated safely")

    plt.show()

    return fig, diagnostic_data


if __name__ == "__main__":
    plot_adaptive_pressure_evolution()