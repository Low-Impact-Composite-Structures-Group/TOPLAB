"""
Test Suite for ScenarioConfig - Unified Configuration Parser

Tests the ScenarioConfig class which integrates:
- NIST materials framework
- Mission sequence configuration
- Tank geometry parsing
- YAML file validation

Usage:
    pytest test/multi_tank_tests/test_scenario_config.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "src"))

from src.multi_tank.configuration.scenario_configuration import ScenarioConfig


class TestScenarioConfig:
    """Test suite for ScenarioConfig unified parser."""

    def test_scenario_config_from_yaml(self):
        """Test ScenarioConfig can parse single_tank_cch2_config.yaml"""
        config_path = project_root / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

        assert config_path.exists(), f"Test config file not found: {config_path}"

        scenario = ScenarioConfig.from_yaml(config_path)

        # Verify basic properties
        assert scenario.analysis_name == "Single Tank CCH2 Benchmark"
        assert scenario.version == "1.0"
        assert "orchestrated framework" in scenario.description

        # Verify tank parsing
        assert scenario.get_tank_count() == 1
        assert 1 in scenario.tank_geometries
        assert scenario.tank_geometries[1]['phi'] == 3.0

        # Verify mission parsing
        assert scenario.get_mission_count() == 1
        assert len(scenario.mission_sequence.missions) == 1

        mission = scenario.mission_sequence.missions[0]
        assert mission.type == "discharge"
        assert mission.profile == "csv"

        # Verify materials parsing (new format is per-tank materials)
        tank_materials = scenario.get_tank_materials(1)
        assert "liner" in tank_materials
        assert "composite" in tank_materials

        # Test material properties
        liner = tank_materials["liner"]
        composite = tank_materials["composite"]


        # Test temperature-dependent properties
        cp_liner_300k = liner.get_specific_heat(300.0)
        cp_composite_300k = composite.get_specific_heat(300.0)

        assert 900 < cp_liner_300k < 1000  # Expected range for aluminum
        assert 8 < cp_composite_300k < 8200  # Expected range for composite

    def test_scenario_config_validation(self):
        """Test ScenarioConfig validation catches errors"""
        config_path = project_root / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

        # Should not raise for valid config
        scenario = ScenarioConfig.from_yaml(config_path)
        scenario.validate()  # Should pass

    def test_scenario_config_summary(self):
        """Test ScenarioConfig summary generation"""
        config_path = project_root / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

        scenario = ScenarioConfig.from_yaml(config_path)
        summary = scenario.summary()

        assert "Single Tank CCH2 Benchmark" in summary
        assert "Tanks: 1" in summary
        assert "Missions: 1" in summary
        assert "Nodes: 1" in summary


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])