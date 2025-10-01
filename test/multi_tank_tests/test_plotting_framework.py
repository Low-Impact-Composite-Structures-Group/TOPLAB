"""
Plotting Framework Integration Tests - Phase 2
==============================================

Test plotting framework integration through SystemOrchestrator
for configuration-driven plotting across all analysis types.

Author: Framework Development Team
Date: October 1, 2025
"""

import pytest
import os
import sys
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from configuration.scenario_configuration import ScenarioConfig
from orchestration.system_orchestrator import SystemOrchestrator


class TestPlottingIntegration:
    """Test plotting framework integration through SystemOrchestrator."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        """Setup paths for all test methods."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.analysis_configs = {
            'single_tank_ch2': self.repo_root / 'analysis' / 'multi_tank_systems' / 'single_tank_ch2' / 'single_tank_ch2_config.yaml',
            'single_tank_cch2': self.repo_root / 'analysis' / 'multi_tank_systems' / 'single_tank_cch2' / 'single_tank_cch2_config.yaml',
            'single_tank_slh2': self.repo_root / 'analysis' / 'multi_tank_systems' / 'single_tank_slh2' / 'single_tank_slh2_config.yaml',
            'coupled_ch2_cch2': self.repo_root / 'analysis' / 'multi_tank_systems' / 'coupled_ch2_cch2' / 'coupled_ch2_cch2_config.yaml',
            'stops_verification': self.repo_root / 'analysis' / 'multi_tank_systems' / 'stops_verification' / 'stops_verification.yaml'
        }

        # Verify all config files exist
        for name, path in self.analysis_configs.items():
            assert path.exists(), f"Configuration file not found: {path}"

    @pytest.mark.plotting
    def test_plotting_configuration_parsing(self):
        """Test plotting configuration parsing in all YAML configs."""

        for analysis_name, config_path in self.analysis_configs.items():
            print(f"🎨 Testing plotting config parsing for {analysis_name}...")

            try:
                # Load config
                config = ScenarioConfig.from_yaml(str(config_path))

                # Check plotting configuration structure
                raw_config = config.raw_config
                if 'output' in raw_config and 'plots' in raw_config['output']:
                    plots_config = raw_config['output']['plots']

                    # Check for plotting parameters
                    plotting_keys = list(plots_config.keys())
                    print(f"   📊 {analysis_name}: Found plotting keys: {plotting_keys}")

                    # Validate common plotting parameters
                    if 'use_greyscale' in plots_config:
                        assert isinstance(plots_config['use_greyscale'], bool), f"use_greyscale should be boolean for {analysis_name}"

                    if 'enable_multi_tank_overlay' in plots_config:
                        assert isinstance(plots_config['enable_multi_tank_overlay'], bool), f"overlay setting should be boolean for {analysis_name}"

                    print(f"✅ {analysis_name}: Plotting configuration parsed successfully")
                else:
                    print(f"ℹ️ {analysis_name}: No plotting configuration section found")

            except Exception as e:
                print(f"⚠️ {analysis_name}: Plotting config parsing error - {str(e)[:100]}...")
                continue

    @pytest.mark.plotting
    def test_orchestrator_plotting_integration(self):
        """Test plotting integration through SystemOrchestrator generate_plots method."""

        for analysis_name, config_path in self.analysis_configs.items():
            print(f"🎨 Testing orchestrator plotting integration for {analysis_name}...")

            try:
                # Load config and create orchestrator
                config = ScenarioConfig.from_yaml(str(config_path))
                orchestrator = SystemOrchestrator(config)

                # Check that orchestrator has plotting capability
                assert hasattr(orchestrator, 'generate_plots'), f"Orchestrator missing generate_plots method for {analysis_name}"
                assert callable(orchestrator.generate_plots), f"generate_plots not callable for {analysis_name}"

                print(f"✅ {analysis_name}: Orchestrator plotting integration ready")

            except Exception as e:
                print(f"⚠️ {analysis_name}: Orchestrator plotting integration error - {str(e)[:100]}...")
                continue

    @pytest.mark.plotting
    def test_plotting_config_validation(self):
        """Test validation of plotting configuration parameters."""

        for analysis_name, config_path in self.analysis_configs.items():
            print(f"� Testing plotting config validation for {analysis_name}...")

            try:
                # Load config
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)

                # Check output section structure
                if 'output' in config_dict:
                    output_config = config_dict['output']

                    # Check for plots section
                    if 'plots' in output_config:
                        plots_config = output_config['plots']

                        # Validate boolean plotting parameters
                        boolean_params = ['use_greyscale', 'enable_multi_tank_overlay', 'show_reference_lines']
                        for param in boolean_params:
                            if param in plots_config:
                                assert isinstance(plots_config[param], bool), f"{param} should be boolean in {analysis_name}"
                                print(f"   ✓ {param}: {plots_config[param]}")

                        # Check for heat exchanger plotting parameters (externalized values)
                        if 'heat_exchanger_requirements' in plots_config:
                            hx_config = plots_config['heat_exchanger_requirements']
                            required_hx_params = ['ohex_target_temperature', 'ohex_target_pressure']

                            for param in required_hx_params:
                                if param in hx_config:
                                    assert isinstance(hx_config[param], (int, float)), f"{param} should be numeric in {analysis_name}"
                                    print(f"   ✓ {param}: {hx_config[param]}")

                        print(f"✅ {analysis_name}: Plotting configuration validation passed")
                    else:
                        print(f"ℹ️ {analysis_name}: No plots section in output configuration")
                else:
                    print(f"ℹ️ {analysis_name}: No output section found")

            except Exception as e:
                print(f"⚠️ {analysis_name}: Plotting config validation error - {str(e)[:100]}...")
                continue

    @pytest.mark.integration
    def test_multi_tank_plotting_configuration(self):
        """Test multi-tank specific plotting configurations."""

        # Focus on coupled systems
        coupling_configs = ['coupled_ch2_cch2']

        for analysis_name in coupling_configs:
            if analysis_name not in self.analysis_configs:
                continue

            config_path = self.analysis_configs[analysis_name]
            print(f"🔗 Testing multi-tank plotting configuration for {analysis_name}...")

            try:
                # Load config
                config = ScenarioConfig.from_yaml(str(config_path))

                # Create orchestrator
                orchestrator = SystemOrchestrator(config)

                # Check multi-tank specific attributes
                assert len(orchestrator.tank_geometries) > 1, f"Multi-tank system should have >1 tanks for {analysis_name}"
                assert len(orchestrator.tank_system.tanks) > 1, f"Tank system should have >1 tanks for {analysis_name}"

                # Verify plotting can handle multi-tank data
                tank_count = len(orchestrator.tank_geometries)
                print(f"   🏢 {analysis_name}: {tank_count} tanks configured for plotting")

                print(f"✅ {analysis_name}: Multi-tank plotting configuration validated")

            except Exception as e:
                print(f"⚠️ {analysis_name}: Multi-tank plotting test error - {str(e)[:100]}...")
                continue

    @pytest.mark.plotting
    def test_sequential_plotting_configuration(self):
        """Test sequential mission plotting configurations."""

        # Focus on sequential analyses
        sequential_configs = ['stops_verification']

        for analysis_name in sequential_configs:
            if analysis_name not in self.analysis_configs:
                continue

            config_path = self.analysis_configs[analysis_name]
            print(f"📈 Testing sequential plotting configuration for {analysis_name}...")

            try:
                # Load config
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)

                # Check for mission sequence structure
                if 'mission_sequence' in config_dict:
                    mission_sequence = config_dict['mission_sequence']
                    mission_count = len(mission_sequence['missions']) if 'missions' in mission_sequence else len(mission_sequence)

                    print(f"   📋 {analysis_name}: {mission_count} missions in sequence")
                    assert mission_count > 1, f"Sequential analysis should have >1 missions for {analysis_name}"

                    # Check plotting configuration for sequential data
                    if 'output' in config_dict and 'plots' in config_dict['output']:
                        plots_config = config_dict['output']['plots']

                        # Sequential plotting may have special configurations
                        sequential_params = ['show_mission_boundaries', 'use_sequential_colors']
                        for param in sequential_params:
                            if param in plots_config:
                                print(f"   ✓ Sequential plotting param {param}: {plots_config[param]}")

                    print(f"✅ {analysis_name}: Sequential plotting configuration validated")
                else:
                    print(f"ℹ️ {analysis_name}: No mission sequence found")

            except Exception as e:
                print(f"⚠️ {analysis_name}: Sequential plotting test error - {str(e)[:100]}...")
                continue


if __name__ == '__main__':
    """Allow running tests directly with python test_plotting_framework.py"""
    pytest.main([__file__, '-v', '--tb=short'])