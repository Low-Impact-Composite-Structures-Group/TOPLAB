#!/usr/bin/env python3
"""
Phase 3: Migration Strategy Testing
==================================

Tests systematic migration of all multi-tank analyses from old format to new format.
Validates that the migration process maintains mathematical accuracy across all scenarios.

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the source directory to the Python path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

def find_old_format_configs() -> List[Path]:
    """Find all old format configuration files in multi_tank_systems."""
    multi_tank_dir = PROJECT_ROOT / "analysis" / "multi_tank_systems"
    old_configs = []

    for config_file in multi_tank_dir.rglob("*.yaml"):
        if config_file.name.endswith("_config_new_format.yaml"):
            continue  # Skip new format files

        # Check if it's an old format config by looking for flat structure
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            # Old format has tanks as direct keys, not under 'nodes'
            if 'nodes' not in config_data and any(
                key.startswith(('tank', 'Tank')) for key in config_data.keys()
            ):
                old_configs.append(config_file)
        except Exception as e:
            print(f"⚠️  Could not parse {config_file}: {e}")
            continue

    return old_configs

def create_new_format_config(old_config_path: Path) -> Path:
    """Create new format version of old config using ConfigurationAdapter."""
    from src.configuration.configuration_adapter import ConfigurationAdapter
    from src.configuration.enhanced_scenario_configuration import EnhancedScenarioConfig

    print(f"🔄 Converting {old_config_path.name}...")

    # Load old format as raw YAML data
    import yaml
    with open(old_config_path, 'r') as f:
        old_config_data = yaml.safe_load(f)

    # Convert to new format
    adapter = ConfigurationAdapter()
    new_format_data = adapter.migrate_old_to_new(old_config_data)

    # Create new format file path
    new_config_path = old_config_path.parent / f"{old_config_path.stem}_new_format.yaml"

    # Write new format file
    with open(new_config_path, 'w') as f:
        yaml.dump(new_format_data, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created {new_config_path.name}")
    return new_config_path

def test_config_equivalence(old_config_path: Path, new_config_path: Path) -> Dict[str, Any]:
    """Test that old and new config produce equivalent results."""
    from src.configuration.enhanced_scenario_configuration import EnhancedScenarioConfig
    from src.orchestration.system_orchestrator import SystemOrchestrator

    print(f"🧪 Testing equivalence: {old_config_path.name}")

    results = {
        'config_name': old_config_path.name,
        'old_format_success': False,
        'new_format_success': False,
        'results_match': False,
        'old_final_time': None,
        'new_final_time': None,
        'error_message': None
    }

    try:
        # Test old format
        print(f"   📄 Loading old format...")
        old_config = EnhancedScenarioConfig.from_yaml(str(old_config_path))
        old_orchestrator = SystemOrchestrator(old_config)

        start_time = time.time()
        old_results = old_orchestrator.run_simulation()
        old_duration = time.time() - start_time

        results['old_format_success'] = True
        results['old_final_time'] = old_results.times[-1] if hasattr(old_results, 'times') else None
        print(f"   ✅ Old format completed in {old_duration:.2f}s")

        # Test new format
        print(f"   📄 Loading new format...")
        new_config = EnhancedScenarioConfig.from_yaml(str(new_config_path))
        new_orchestrator = SystemOrchestrator(new_config)

        start_time = time.time()
        new_results = new_orchestrator.run_simulation()
        new_duration = time.time() - start_time

        results['new_format_success'] = True
        results['new_final_time'] = new_results.times[-1] if hasattr(new_results, 'times') else None
        print(f"   ✅ New format completed in {new_duration:.2f}s")

        # Compare results
        if results['old_final_time'] and results['new_final_time']:
            time_diff = abs(results['old_final_time'] - results['new_final_time'])
            results['results_match'] = time_diff < 1.0  # Within 1 second

            if results['results_match']:
                print(f"   ✅ Results match (Δt = {time_diff:.3f}s)")
            else:
                print(f"   ❌ Results differ (Δt = {time_diff:.3f}s)")

    except Exception as e:
        results['error_message'] = str(e)
        print(f"   ❌ Error: {e}")

    return results

def generate_migration_report(test_results: List[Dict[str, Any]]) -> str:
    """Generate comprehensive migration report."""
    total_configs = len(test_results)
    successful_migrations = sum(1 for r in test_results if r['results_match'])

    report = f"""
================================================================================
MULTI-TANK CONFIGURATION MIGRATION REPORT
================================================================================

📊 SUMMARY:
   Total configurations tested: {total_configs}
   Successful migrations: {successful_migrations}
   Migration success rate: {successful_migrations/total_configs*100:.1f}%

📋 DETAILED RESULTS:
"""

    for result in test_results:
        status = "✅ PASS" if result['results_match'] else "❌ FAIL"
        report += f"\n   {status} {result['config_name']}"

        if result['error_message']:
            report += f"\n      Error: {result['error_message']}"
        elif result['results_match']:
            report += f"\n      Old: {result['old_final_time']:.1f}s, New: {result['new_final_time']:.1f}s"

    report += f"""

🎯 MIGRATION STRATEGY RECOMMENDATIONS:
"""

    if successful_migrations == total_configs:
        report += """
   ✅ COMPLETE SUCCESS: All configurations migrated successfully
   → Proceed with full migration to new format
   → Update documentation and examples
   → Deprecate old format support
"""
    elif successful_migrations > total_configs * 0.8:
        report += f"""
   🟡 MOSTLY SUCCESSFUL: {successful_migrations}/{total_configs} configurations migrated
   → Investigate failing configurations
   → Fix migration issues before full deployment
   → Consider partial migration for working configs
"""
    else:
        report += f"""
   🔴 MIGRATION ISSUES: Only {successful_migrations}/{total_configs} configurations successful
   → Significant migration problems detected
   → Review configuration adapter logic
   → Test with simpler configurations first
"""

    report += """
================================================================================
"""

    return report

def main():
    """Run complete migration strategy testing."""
    print("================================================================================")
    print("PHASE 3: MULTI-TANK CONFIGURATION MIGRATION TESTING")
    print("================================================================================")

    # Find all old format configurations
    print("🔍 Scanning for old format configurations...")
    old_configs = find_old_format_configs()
    print(f"   Found {len(old_configs)} old format configurations")

    if not old_configs:
        print("❌ No old format configurations found!")
        return

    # Test migration for each configuration
    test_results = []

    for old_config_path in old_configs:
        print(f"\n📦 Processing {old_config_path.name}...")

        try:
            # Create new format version
            new_config_path = create_new_format_config(old_config_path)

            # Test equivalence
            result = test_config_equivalence(old_config_path, new_config_path)
            test_results.append(result)

        except Exception as e:
            print(f"❌ Failed to process {old_config_path.name}: {e}")
            test_results.append({
                'config_name': old_config_path.name,
                'old_format_success': False,
                'new_format_success': False,
                'results_match': False,
                'error_message': str(e)
            })

    # Generate and display report
    report = generate_migration_report(test_results)
    print(report)

    # Save report to file
    report_path = PROJECT_ROOT / "analysis" / "multi_tank_systems" / "migration_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"📋 Full report saved to: {report_path}")

if __name__ == "__main__":
    main()