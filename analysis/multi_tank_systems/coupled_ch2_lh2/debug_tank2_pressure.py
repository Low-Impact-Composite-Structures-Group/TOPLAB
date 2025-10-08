#!/usr/bin/env python3
"""
Debug script to examine Tank 2 pressure evolution in detail
"""

import sys
import os
import numpy as np

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.orchestration.system_orchestrator import SystemOrchestrator
from src.configuration.scenario_configuration import ScenarioConfig

def debug_tank2_pressure():
    """Debug Tank 2 pressure evolution."""

    print("🔍 TANK 2 PRESSURE EVOLUTION DEBUG")
    print("="*60)

    # Load configuration and create orchestrator
    config_path = "coupled_ch2_lh2_config.yaml"
    config = ScenarioConfig.from_yaml(config_path)
    orchestrator = SystemOrchestrator(config)

    print("🚀 Running simulation...")
    solver_method = "LSODA"
    solver_config = {
        'rtol': 1e-4,
        'atol': 1e-7,
        'max_step': 10.0
    }

    results = orchestrator.run_simulation(solver_method=solver_method, solver_config=solver_config)

    # Get simulation data
    time_data = results.times / 3600  # Convert to hours
    combined_data = results.get_combined_data()

    # Tank pressures and masses (already converted to bar in get_combined_data)
    tank1_pressure = combined_data['pressures'][0]  # Already in bar
    tank2_pressure = combined_data['pressures'][1]  # Already in bar
    tank1_mass = combined_data['masses'][0]
    tank2_mass = combined_data['masses'][1]

    print(f"\n📊 SIMULATION RESULTS:")
    print(f"Duration: {time_data[-1]:.3f} hours")
    print(f"Data points: {len(time_data)}")

    print(f"\n🔋 TANK 1 (CH2) EVOLUTION:")
    print(f"Initial: P={tank1_pressure[0]:.1f} bar, m={tank1_mass[0]:.1f} kg")
    print(f"Final:   P={tank1_pressure[-1]:.1f} bar, m={tank1_mass[-1]:.1f} kg")
    print(f"Pressure range: {np.min(tank1_pressure):.1f} - {np.max(tank1_pressure):.1f} bar")

    print(f"\n🔋 TANK 2 (LH2) EVOLUTION:")
    print(f"Initial: P={tank2_pressure[0]:.3f} bar, m={tank2_mass[0]:.1f} kg")
    print(f"Final:   P={tank2_pressure[-1]:.3f} bar, m={tank2_mass[-1]:.1f} kg")
    print(f"Pressure range: {np.min(tank2_pressure):.3f} - {np.max(tank2_pressure):.3f} bar")
    print(f"Pressure variation: {np.max(tank2_pressure) - np.min(tank2_pressure):.3f} bar")

    # Check if Tank 2 pressure is essentially zero
    if np.max(tank2_pressure) < 0.1:
        print("⚠️  WARNING: Tank 2 pressure appears to be near zero throughout mission!")
        print("   This suggests a problem with pressure maintenance or calculation.")

    # Sample pressure values at different times
    print(f"\n📈 TANK 2 PRESSURE SAMPLES:")
    sample_indices = [0, len(time_data)//4, len(time_data)//2, 3*len(time_data)//4, -1]
    for i in sample_indices:
        print(f"   t={time_data[i]:.3f}h: P={tank2_pressure[i]:.6f} bar, m={tank2_mass[i]:.1f} kg")

    # Get valve diagnostic data
    valve = orchestrator.tank_system.coupling_valves[0]
    diagnostic_data = valve.get_diagnostic_data()

    print(f"\n🔗 VALVE DIAGNOSTIC DATA:")
    print(f"Diagnostic points collected: {len(diagnostic_data['time_history'])}")
    if len(diagnostic_data['time_history']) > 0:
        valve_times = np.array(diagnostic_data['time_history']) / 3600
        required_pressures = np.array(diagnostic_data['required_pressure_history']) / 1e5
        print(f"Valve time range: {valve_times[0]:.3f} - {valve_times[-1]:.3f} hours")
        print(f"Required pressure range: {np.min(required_pressures):.3f} - {np.max(required_pressures):.3f} bar")

        # Sample valve data
        print(f"\n📈 REQUIRED PRESSURE SAMPLES:")
        sample_indices = [0, len(valve_times)//4, len(valve_times)//2, 3*len(valve_times)//4, -1]
        for i in sample_indices:
            print(f"   t={valve_times[i]:.3f}h: Required P={required_pressures[i]:.3f} bar")

    return time_data, tank2_pressure, combined_data

if __name__ == "__main__":
    debug_tank2_pressure()