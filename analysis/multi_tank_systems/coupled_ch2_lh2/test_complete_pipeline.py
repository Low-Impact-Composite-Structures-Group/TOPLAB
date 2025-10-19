#!/usr/bin/env python3
"""
Test Complete Analysis Pipeline with New Format

This script tests the complete analysis pipeline with the new format configuration.
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


def test_complete_pipeline():
    """Test complete analysis pipeline with new format."""

    config_path = current_dir / "coupled_ch2_lh2_config_new_format.yaml"

    print("=" * 80)
    print("TESTING COMPLETE ANALYSIS PIPELINE WITH NEW FORMAT")
    print("=" * 80)

    try:
        # Load configuration
        print(f"📄 Loading configuration...")
        config = ScenarioConfig.from_yaml(str(config_path))

        # Create orchestrator
        print(f"🔧 Creating orchestrator...")
        orchestrator = SystemOrchestrator(config)

        # Run simulation with normal parameters
        print(f"🚀 Running simulation...")
        start_time = time.time()
        results = orchestrator.run_simulation()
        end_time = time.time()

        print(f"✅ Simulation completed successfully!")
        print(f"   Duration: {end_time - start_time:.2f} s")
        print(f"   Final time: {results.times[-1]:.2f} s")
        print(f"   Tank states: {len(results.multi_tank_states)} tanks")
        print(f"   Time points: {len(results.times)}")

        # Check tank final states
        if results.multi_tank_states:
            final_multi_state = results.multi_tank_states[-1]
            for i in range(len(orchestrator.tank_geometries)):
                final_tank_state = final_multi_state.get_tank_state(i)
                final_pressure = final_tank_state.pressure / 1e5  # Convert to bar
                final_temp = final_tank_state.temperature        # K
                final_density = final_tank_state.density         # kg/m³
                print(f"   Tank {i+1} final: P={final_pressure:.1f} bar, T={final_temp:.1f} K, ρ={final_density:.1f} kg/m³")

        print(f"\n✅ Complete pipeline working with new format!")
        print(f"✅ Ready for plotting and full analysis!")

        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_complete_pipeline()

    if success:
        print(f"\n🎉 Phase 2 Pipeline Test: SUCCESS!")
        print(f"✅ New format works with complete analysis pipeline")
        print(f"✅ Ready for comparison with old format results")
    else:
        print(f"\n❌ Phase 2 Pipeline Test: FAILED")
        print(f"❌ Pipeline needs debugging")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)