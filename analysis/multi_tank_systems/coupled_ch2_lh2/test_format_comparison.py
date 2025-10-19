#!/usr/bin/env python3
"""
Test Format Comparison

Compare results between old and new configuration formats to ensure identical behavior.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import enhanced framework
from src.configuration.enhanced_scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator


def compare_tank_results(old_results, new_results, tolerance=1e-6):
    """Compare tank results between two configurations."""

    comparison = {
        'success': True,
        'time_match': False,
        'final_states_match': False,
        'differences': []
    }

    # Compare time arrays
    time_diff = np.abs(old_results.times[-1] - new_results.times[-1])
    if time_diff < tolerance:
        comparison['time_match'] = True
    else:
        comparison['differences'].append(f"Final time difference: {time_diff:.8f} s")

    # Compare final tank states
    if len(old_results.multi_tank_states) > 0 and len(new_results.multi_tank_states) > 0:
        old_final = old_results.multi_tank_states[-1]
        new_final = new_results.multi_tank_states[-1]

        states_match = True
        for tank_idx in range(min(old_final.n_tanks, new_final.n_tanks)):
            old_tank = old_final.get_tank_state(tank_idx)
            new_tank = new_final.get_tank_state(tank_idx)

            # Compare key properties
            mass_diff = abs(old_tank.fuel_mass - new_tank.fuel_mass)
            temp_diff = abs(old_tank.temperature - new_tank.temperature)
            press_diff = abs(old_tank.pressure - new_tank.pressure)

            if mass_diff > tolerance:
                comparison['differences'].append(f"Tank {tank_idx+1} mass diff: {mass_diff:.8f} kg")
                states_match = False

            if temp_diff > tolerance:
                comparison['differences'].append(f"Tank {tank_idx+1} temp diff: {temp_diff:.8f} K")
                states_match = False

            if press_diff > tolerance:
                comparison['differences'].append(f"Tank {tank_idx+1} pressure diff: {press_diff:.8f} Pa")
                states_match = False

        comparison['final_states_match'] = states_match

    comparison['success'] = comparison['time_match'] and comparison['final_states_match']
    return comparison


def test_format_comparison():
    """Test that old and new formats produce identical results."""

    old_config_path = current_dir / "coupled_ch2_lh2_config.yaml"
    new_config_path = current_dir / "coupled_ch2_lh2_config_new_format.yaml"

    print("=" * 80)
    print("TESTING FORMAT COMPARISON: OLD vs NEW")
    print("=" * 80)

    if not old_config_path.exists():
        print(f"❌ Old format config not found: {old_config_path}")
        return False

    if not new_config_path.exists():
        print(f"❌ New format config not found: {new_config_path}")
        return False

    try:
        # Load old format
        print(f"📄 Loading old format configuration...")
        old_config = ScenarioConfig.from_yaml(str(old_config_path))
        old_orchestrator = SystemOrchestrator(old_config)

        # Run old format simulation
        print(f"🚀 Running old format simulation...")
        start_time = time.time()
        old_results = old_orchestrator.run_simulation()
        old_duration = time.time() - start_time

        print(f"✅ Old format completed in {old_duration:.2f}s")
        print(f"   Final time: {old_results.times[-1]:.1f}s, Points: {len(old_results.times)}")

        # Load new format
        print(f"\n📄 Loading new format configuration...")
        new_config = ScenarioConfig.from_yaml(str(new_config_path))
        new_orchestrator = SystemOrchestrator(new_config)

        # Run new format simulation
        print(f"🚀 Running new format simulation...")
        start_time = time.time()
        new_results = new_orchestrator.run_simulation()
        new_duration = time.time() - start_time

        print(f"✅ New format completed in {new_duration:.2f}s")
        print(f"   Final time: {new_results.times[-1]:.1f}s, Points: {len(new_results.times)}")

        # Compare results
        print(f"\n🔍 Comparing results...")
        comparison = compare_tank_results(old_results, new_results)

        if comparison['success']:
            print(f"✅ RESULTS MATCH! Formats produce identical results")
            print(f"   Time arrays match: {comparison['time_match']}")
            print(f"   Final states match: {comparison['final_states_match']}")
        else:
            print(f"❌ RESULTS DIFFER!")
            for diff in comparison['differences']:
                print(f"   - {diff}")

        return comparison['success']

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_format_comparison()

    if success:
        print(f"\n🎉 Format Comparison: SUCCESS!")
        print(f"✅ Old and new formats produce identical results")
        print(f"✅ Migration is mathematically validated")
    else:
        print(f"\n❌ Format Comparison: FAILED")
        print(f"❌ Results differ between old and new formats")
        print(f"❌ Investigation needed before migration")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)