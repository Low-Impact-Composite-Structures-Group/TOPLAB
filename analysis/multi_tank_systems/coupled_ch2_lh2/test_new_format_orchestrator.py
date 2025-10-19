#!/usr/bin/env python3
"""
Test SystemOrchestrator with New Format Configuration

This script tests whether the SystemOrchestrator can work with the new
network-based configuration format without any issues.
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


def test_new_format_orchestrator():
    """Test SystemOrchestrator with new format configuration."""

    config_path = current_dir / "coupled_ch2_lh2_config_new_format.yaml"

    print("=" * 80)
    print("TESTING SYSTEM ORCHESTRATOR WITH NEW FORMAT")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print("Testing new network-based configuration format")
    print("=" * 80)

    try:
        # Load configuration
        print(f"📄 Loading new format configuration...")
        config = ScenarioConfig.from_yaml(str(config_path))
        print(f"✅ Configuration loaded successfully!")
        print(f"   Format: {config.config_format}")
        print(f"   Tanks: {config.get_tank_count()}")
        print(f"   Nodes: {len(config.get_network_nodes())}")
        print(f"   Edges: {len(config.get_network_edges())}")

        # Test SystemOrchestrator creation
        print(f"\n🔧 Creating SystemOrchestrator...")
        orchestrator = SystemOrchestrator(config)

        print(f"✅ SystemOrchestrator created successfully!")
        print(f"   Scenario: {orchestrator.scenario_config.analysis_name}")

        # Test if we can access key properties
        print(f"\n📊 Testing orchestrator properties...")
        print(f"   Tank geometries: {len(orchestrator.scenario_config.tank_geometries)}")
        print(f"   Tank materials: {len(orchestrator.scenario_config.tank_materials)}")
        print(f"   Mission count: {orchestrator.scenario_config.get_mission_count()}")

        # Test mission access
        missions = orchestrator.scenario_config.mission_sequence.missions
        if missions:
            mission = missions[0]
            print(f"   Mission type: {mission.type}")
            print(f"   Mission profile: {mission.profile}")
            print(f"   Assigned to: Tank {mission.assigned_to}")

        print(f"\n✅ All basic tests passed!")
        print(f"✅ SystemOrchestrator is compatible with new format!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_new_format_orchestrator()

    if success:
        print(f"\n🎉 Phase 2 Initial Test: SUCCESS!")
        print(f"✅ New format configuration works with SystemOrchestrator")
        print(f"✅ Ready for full analysis pipeline test")
    else:
        print(f"\n❌ Phase 2 Initial Test: FAILED")
        print(f"❌ SystemOrchestrator needs updates for new format")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)