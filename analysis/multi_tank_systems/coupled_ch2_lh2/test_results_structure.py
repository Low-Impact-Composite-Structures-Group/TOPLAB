#!/usr/bin/env python3
"""
Test Analysis Results Structure

Check what the results object actually contains.
"""

import sys
import time
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import enhanced framework
from src.configuration.enhanced_scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator


def test_results_structure():
    """Test what the results object contains."""

    config_path = current_dir / "coupled_ch2_lh2_config_new_format.yaml"

    print("Testing results structure...")

    try:
        # Load and run
        config = ScenarioConfig.from_yaml(str(config_path))
        orchestrator = SystemOrchestrator(config)

        print("Running simulation...")
        results = orchestrator.run_simulation()

        print(f"✅ Simulation completed!")
        print(f"Results type: {type(results)}")
        print(f"Results attributes: {dir(results)}")

        # Check what's actually in the results
        if hasattr(results, '__dict__'):
            print(f"Results dict keys: {list(results.__dict__.keys())}")

        # Investigate multi_tank_states structure
        if hasattr(results, 'multi_tank_states') and results.multi_tank_states:
            print(f"\nMultiTankState investigation:")
            print(f"Number of states: {len(results.multi_tank_states)}")
            first_state = results.multi_tank_states[0]
            print(f"First state type: {type(first_state)}")
            print(f"First state attributes: {dir(first_state)}")

            # Check if it has tank_states inside
            if hasattr(first_state, '__dict__'):
                print(f"First state dict: {list(first_state.__dict__.keys())}")

            # Try accessing tank data
            if hasattr(first_state, 'tank_states'):
                print(f"Tank states inside: {type(first_state.tank_states)}")
                if first_state.tank_states:
                    tank_0 = first_state.tank_states[0]
                    print(f"Tank 0 type: {type(tank_0)}")
                    print(f"Tank 0 attributes: {dir(tank_0)}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_results_structure()