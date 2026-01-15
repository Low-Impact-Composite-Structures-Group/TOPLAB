"""
Test Suite for Mission Configuration Parsing

Tests the mission configuration parsing logic for the multi-tank framework.
Validates that different mission types and profiles are correctly parsed
and that appropriate mission objects are created.

Test Coverage:
- Mission configuration parsing from YAML
- Mission type and profile validation
- Parameter validation for different profiles
- Mission object creation
- Error handling for invalid configurations

Usage:
    pytest test/multi_tank_tests/test_mission_config.py -v
    python test/multi_tank_tests/run_tests.py --module mission_config
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Mock mission configuration parser (will be implemented in ScenarioConfig)
@dataclass
class MissionConfig:
    """Mock mission configuration class for testing."""
    type: str
    profile: str
    ambient_temperature: float
    parameters: Dict[str, Any]
    name: Optional[str] = None  # For mission sequences

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create MissionConfig from dictionary."""
        mission_data = config_dict.get('mission', {})

        # Extract type and profile
        mission_type = mission_data.get('type')
        profile = mission_data.get('profile')
        ambient_temp = mission_data.get('ambient_temperature', 288.15)

        # Extract additional parameters
        # Current configs use nested mission.parameters for profile-specific args.
        parameters = dict(mission_data.get('parameters', {}) or {})
        for key, value in mission_data.items():
            if key in ['type', 'profile', 'ambient_temperature', 'parameters']:
                continue
            parameters[key] = value

        return cls(
            type=mission_type,
            profile=profile,
            ambient_temperature=ambient_temp,
            parameters=parameters
        )

    @classmethod
    def from_sequence_item(cls, mission_item: dict):
        """Create MissionConfig from a mission sequence item."""
        mission_type = mission_item.get('type')
        profile = mission_item.get('profile')
        ambient_temp = mission_item.get('ambient_temperature', 288.15)
        name = mission_item.get('name')

        # Extract additional parameters
        parameters = {}
        for key, value in mission_item.items():
            if key not in ['type', 'profile', 'ambient_temperature', 'name']:
                parameters[key] = value

        return cls(
            type=mission_type,
            profile=profile,
            ambient_temperature=ambient_temp,
            parameters=parameters,
            name=name
        )

    def validate(self):
        """Validate mission configuration."""
        # Check required fields
        if not self.type:
            raise ValueError("Mission type is required")

        if not self.profile:
            raise ValueError("Mission profile is required")

        # Check valid types
        valid_types = ['discharge', 'refuel', 'dormancy']
        if self.type not in valid_types:
            raise ValueError(f"Invalid mission type '{self.type}'. Must be one of: {valid_types}")

        # Check valid profiles
        valid_profiles = ['atr72', 'csv', 'constant_flow', 'custom', 'cryogenic', 'ambient', 'storage']
        if self.profile not in valid_profiles:
            raise ValueError(f"Invalid profile '{self.profile}'. Must be one of: {valid_profiles}")

        if self.profile == 'csv':
            required_params = ['csv_file']
            missing_params = [p for p in required_params if p not in self.parameters]
            if missing_params:
                raise ValueError(f"Profile '{self.profile}' requires parameters: {missing_params}")

        # Profile-specific parameter validation
        if self.profile == 'constant_flow':
            required_params = ['flow_rate', 'duration']
            missing_params = [p for p in required_params if p not in self.parameters]
            if missing_params:
                raise ValueError(f"Profile '{self.profile}' requires parameters: {missing_params}")

        elif self.profile == 'custom':
            required_params = ['sections']
            missing_params = [p for p in required_params if p not in self.parameters]
            if missing_params:
                raise ValueError(f"Profile '{self.profile}' requires parameters: {missing_params}")

        elif self.type == 'refuel':
            required_params = ['target_mass']
            missing_params = [p for p in required_params if p not in self.parameters]
            if missing_params:
                raise ValueError(f"Refuel missions require parameters: {missing_params}")

        elif self.type == 'dormancy' and self.profile == 'storage':
            required_params = ['duration']
            missing_params = [p for p in required_params if p not in self.parameters]
            if missing_params:
                raise ValueError(f"Storage dormancy requires parameters: {missing_params}")

        # Validate stopping criteria if provided
        if 'stopping_criteria' in self.parameters:
            self._validate_stopping_criteria()

    def _validate_stopping_criteria(self):
        """Validate stopping criteria configuration."""
        stopping_criteria = self.parameters.get('stopping_criteria', {})

        if not isinstance(stopping_criteria, dict):
            raise ValueError("stopping_criteria must be a dictionary")

        # Validate use_duration_stopping if provided
        if 'use_duration_stopping' in stopping_criteria:
            if not isinstance(stopping_criteria['use_duration_stopping'], bool):
                raise ValueError("use_duration_stopping must be a boolean")

        # Validate use_density_stopping_events if provided
        if 'use_density_stopping_events' in stopping_criteria:
            if not isinstance(stopping_criteria['use_density_stopping_events'], bool):
                raise ValueError("use_density_stopping_events must be a boolean")

        # Validate minimum_density if provided
        if 'minimum_density' in stopping_criteria:
            density = stopping_criteria['minimum_density']
            if not isinstance(density, (int, float)) or density <= 0:
                raise ValueError("minimum_density must be a positive number")

        # Check consistency: if use_duration_stopping is true, mission must have some form of duration
        if stopping_criteria.get('use_duration_stopping', False):
            has_explicit_duration = 'duration' in self.parameters
            has_custom_sections = 'sections' in self.parameters and isinstance(self.parameters['sections'], list)
            has_profile_duration = self.profile in ['atr72', 'csv']  # Profile-driven missions have implicit duration

            if not (has_explicit_duration or has_custom_sections or has_profile_duration):
                raise ValueError("use_duration_stopping=true requires mission to have 'duration' parameter, 'sections' with durations, or a profile with predefined duration (e.g., atr72)")
@dataclass
class MissionSequenceConfig:
    """Mock mission sequence configuration class for testing."""
    missions: list[MissionConfig]

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create MissionSequenceConfig from dictionary."""
        # Check if this is a single mission or sequence
        if 'mission' in config_dict:
            # Single mission - convert to sequence with one item
            single_mission = MissionConfig.from_dict(config_dict)
            return cls(missions=[single_mission])

        elif 'mission_sequence' in config_dict:
            # Mission sequence
            sequence_data = config_dict['mission_sequence']
            missions = []

            for mission_item in sequence_data:
                mission_config = MissionConfig.from_sequence_item(mission_item)
                missions.append(mission_config)

            return cls(missions=missions)

        else:
            # No mission configuration found
            return cls(missions=[])

    def validate(self):
        """Validate mission sequence configuration."""
        if not self.missions:
            raise ValueError("At least one mission is required")

        # Validate each mission in the sequence
        for i, mission in enumerate(self.missions):
            try:
                mission.validate()
            except ValueError as e:
                raise ValueError(f"Mission {i+1} ('{mission.name or 'unnamed'}'): {e}")


class TestMissionConfigBasics:
    """Test basic mission configuration parsing."""

    def test_atr72_mission_parsing(self):
        """Test parsing ATR72 mission configuration."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',
                'ambient_temperature': 288.15
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.type == 'discharge'
        assert mission_config.profile == 'atr72'
        assert mission_config.ambient_temperature == 288.15
        assert mission_config.parameters == {}

    def test_constant_flow_mission_parsing(self):
        """Test parsing constant flow mission configuration."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.05,
                'duration': 3600,
                'phase': 'gas'
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.type == 'discharge'
        assert mission_config.profile == 'constant_flow'
        assert mission_config.ambient_temperature == 288.15
        assert mission_config.parameters['flow_rate'] == 0.05
        assert mission_config.parameters['duration'] == 3600
        assert mission_config.parameters['phase'] == 'gas'

    def test_custom_mission_parsing(self):
        """Test parsing custom multi-section mission configuration."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'custom',
                'ambient_temperature': 288.15,
                'sections': [
                    {'duration': 300, 'flow_rate': 0.02, 'phase': 'gas'},
                    {'duration': 1800, 'flow_rate': 0.08, 'phase': 'gas'}
                ]
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.type == 'discharge'
        assert mission_config.profile == 'custom'
        assert len(mission_config.parameters['sections']) == 2
        assert mission_config.parameters['sections'][0]['duration'] == 300
        assert mission_config.parameters['sections'][1]['flow_rate'] == 0.08


class TestMissionValidation:
    """Test mission configuration validation."""

    def test_valid_atr72_validation(self):
        """Test validation of valid ATR72 configuration."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',
                'ambient_temperature': 288.15
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        # Should not raise any exceptions
        mission_config.validate()

    def test_valid_constant_flow_validation(self):
        """Test validation of valid constant flow configuration."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.05,
                'duration': 3600
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        # Should not raise any exceptions
        mission_config.validate()

    def test_missing_type_validation(self):
        """Test validation fails for missing mission type."""
        config_dict = {
            'mission': {
                'profile': 'atr72',
                'ambient_temperature': 288.15
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="Mission type is required"):
            mission_config.validate()

    def test_invalid_type_validation(self):
        """Test validation fails for invalid mission type."""
        config_dict = {
            'mission': {
                'type': 'invalid_type',
                'profile': 'atr72',
                'ambient_temperature': 288.15
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="Invalid mission type 'invalid_type'"):
            mission_config.validate()

    def test_invalid_profile_validation(self):
        """Test validation fails for invalid profile."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'invalid_profile',
                'ambient_temperature': 288.15
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="Invalid profile 'invalid_profile'"):
            mission_config.validate()

    def test_missing_constant_flow_parameters(self):
        """Test validation fails for missing constant flow parameters."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.05
                # Missing duration parameter
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="requires parameters: \\['duration'\\]"):
            mission_config.validate()

    def test_missing_custom_sections(self):
        """Test validation fails for missing custom sections."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'custom',
                'ambient_temperature': 288.15
                # Missing sections parameter
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="requires parameters: \\['sections'\\]"):
            mission_config.validate()


class TestRefuelMissionParsing:
    """Test refuel mission parsing and validation."""

    def test_cryogenic_refuel_parsing(self):
        """Test parsing cryogenic refuel mission."""
        config_dict = {
            'mission': {
                'type': 'refuel',
                'profile': 'cryogenic',
                'ambient_temperature': 288.15,
                'refuel_rate': 0.1,
                'target_mass': 25.0,
                'inlet_temperature': 20.0,
                'inlet_pressure': 100000
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()

        assert mission_config.type == 'refuel'
        assert mission_config.profile == 'cryogenic'
        assert mission_config.parameters['refuel_rate'] == 0.1
        assert mission_config.parameters['target_mass'] == 25.0
        assert mission_config.parameters['inlet_temperature'] == 20.0

    def test_ambient_refuel_parsing(self):
        """Test parsing ambient refuel mission."""
        config_dict = {
            'mission': {
                'type': 'refuel',
                'profile': 'ambient',
                'ambient_temperature': 288.15,
                'refuel_rate': 0.05,
                'target_mass': 20.0
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()

        assert mission_config.type == 'refuel'
        assert mission_config.profile == 'ambient'
        assert mission_config.parameters['refuel_rate'] == 0.05
        assert mission_config.parameters['target_mass'] == 20.0

    def test_missing_refuel_parameters(self):
        """Test validation fails for missing refuel parameters."""
        config_dict = {
            'mission': {
                'type': 'refuel',
                'profile': 'cryogenic',
                'ambient_temperature': 288.15,
                'refuel_rate': 0.1
                # Missing target_mass parameter
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="Refuel missions require parameters: \\['target_mass'\\]"):
            mission_config.validate()


class TestDormancyMissionParsing:
    """Test dormancy mission parsing and validation."""

    def test_storage_mission_parsing(self):
        """Test parsing storage dormancy mission."""
        config_dict = {
            'mission': {
                'type': 'dormancy',
                'profile': 'storage',
                'ambient_temperature': 288.15,
                'duration': 86400,
                'vent_pressure': 450e5
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()

        assert mission_config.type == 'dormancy'
        assert mission_config.profile == 'storage'
        assert mission_config.parameters['duration'] == 86400
        assert mission_config.parameters['vent_pressure'] == 450e5

    def test_missing_storage_duration(self):
        """Test validation fails for missing storage duration."""
        config_dict = {
            'mission': {
                'type': 'dormancy',
                'profile': 'storage',
                'ambient_temperature': 288.15
                # Missing duration parameter
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="requires parameters: \\['duration'\\]"):
            mission_config.validate()


class TestYAMLConfigurationFiles:
    """Test parsing actual YAML configuration files."""

    def test_atr72_config_file(self):
        """Test parsing the actual ATR72 configuration file."""
        config_file = project_root / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

        if not config_file.exists():
            pytest.skip(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()

        assert mission_config.type == 'discharge'
        assert mission_config.profile == 'csv'
        assert mission_config.ambient_temperature == 288.15

    def test_coupled_ch2_cch2_config_file(self):
        """Test parsing the coupled CH2-CCH2 configuration file."""
        config_file = project_root / "analysis" / "multi_tank_systems" / "coupled_ch2_cch2" / "coupled_ch2_cch2_config.yaml"

        if not config_file.exists():
            pytest.skip(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()

        assert mission_config.type == 'discharge'
        assert mission_config.profile == 'csv'
        assert mission_config.ambient_temperature == 288.15

    def test_stops_verification_basic_config(self):
        """Test parsing basic configuration without running complex sequence validation."""
        config_file = project_root / "analysis" / "multi_tank_systems" / "stops_verification" / "stops_verification.yaml"

        if not config_file.exists():
            pytest.skip(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Check basic structure without full validation
        assert config_dict.get('analysis_name') is not None
        assert 'mission_sequence' in config_dict
        assert 'missions' in config_dict['mission_sequence']
        assert len(config_dict['mission_sequence']['missions']) >= 2


class TestMissionParameterExtraction:
    """Test parameter extraction for different mission profiles."""

    def test_extract_constant_flow_parameters(self):
        """Test extracting parameters for constant flow missions."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.08,
                'duration': 2400,
                'phase': 'liquid'
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        # Test parameter extraction
        assert mission_config.parameters['flow_rate'] == 0.08
        assert mission_config.parameters['duration'] == 2400
        assert mission_config.parameters['phase'] == 'liquid'

    def test_extract_custom_sections_parameters(self):
        """Test extracting parameters for custom section missions."""
        sections = [
            {'duration': 600, 'flow_rate': 0.03, 'phase': 'gas'},
            {'duration': 3600, 'flow_rate': 0.07, 'phase': 'twophase'},
            {'duration': 900, 'flow_rate': 0.02, 'phase': 'gas'}
        ]

        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'custom',
                'ambient_temperature': 288.15,
                'sections': sections
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        # Test section extraction
        extracted_sections = mission_config.parameters['sections']
        assert len(extracted_sections) == 3

        for i, section in enumerate(extracted_sections):
            assert section['duration'] == sections[i]['duration']
            assert section['flow_rate'] == sections[i]['flow_rate']
            assert section['phase'] == sections[i]['phase']


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_mission_config(self):
        """Test handling of empty mission configuration."""
        config_dict = {}

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.type is None
        assert mission_config.profile is None
        assert mission_config.ambient_temperature == 288.15  # Default value
        assert mission_config.parameters == {}

    def test_partial_mission_config(self):
        """Test handling of partial mission configuration."""
        config_dict = {
            'mission': {
                'type': 'discharge'
                # Missing profile and other parameters
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.type == 'discharge'
        assert mission_config.profile is None
        assert mission_config.parameters == {}

    def test_default_ambient_temperature(self):
        """Test default ambient temperature when not specified."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72'
                # No ambient_temperature specified
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.ambient_temperature == 288.15

    def test_custom_ambient_temperature(self):
        """Test custom ambient temperature override."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',
                'ambient_temperature': 300.0
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        assert mission_config.ambient_temperature == 300.0


class TestMissionSequences:
    """Test mission sequence configuration and chaining."""

    def test_single_mission_as_sequence(self):
        """Test that single missions are converted to sequences."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',
                'ambient_temperature': 288.15
            }
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        assert len(sequence_config.missions) == 1
        mission = sequence_config.missions[0]
        assert mission.type == 'discharge'
        assert mission.profile == 'atr72'
        assert mission.name is None

    def test_discharge_refuel_dormancy_sequence(self):
        """Test discharge -> refuel -> dormancy sequence parsing."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'discharge_phase',
                    'type': 'discharge',
                    'profile': 'atr72',
                    'ambient_temperature': 288.15
                },
                {
                    'name': 'refuel_phase',
                    'type': 'refuel',
                    'profile': 'cryogenic',
                    'ambient_temperature': 288.15,
                    'refuel_rate': 0.1,
                    'target_mass': 25.0,
                    'inlet_temperature': 20.0
                },
                {
                    'name': 'dormancy_phase',
                    'type': 'dormancy',
                    'profile': 'storage',
                    'ambient_temperature': 288.15,
                    'duration': 86400
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        assert len(sequence_config.missions) == 3

        # Check discharge phase
        discharge = sequence_config.missions[0]
        assert discharge.name == 'discharge_phase'
        assert discharge.type == 'discharge'
        assert discharge.profile == 'atr72'
        assert discharge.parameters == {}

        # Check refuel phase
        refuel = sequence_config.missions[1]
        assert refuel.name == 'refuel_phase'
        assert refuel.type == 'refuel'
        assert refuel.profile == 'cryogenic'
        assert refuel.parameters['refuel_rate'] == 0.1
        assert refuel.parameters['target_mass'] == 25.0
        assert refuel.parameters['inlet_temperature'] == 20.0

        # Check dormancy phase
        dormancy = sequence_config.missions[2]
        assert dormancy.name == 'dormancy_phase'
        assert dormancy.type == 'dormancy'
        assert dormancy.profile == 'storage'
        assert dormancy.parameters['duration'] == 86400

    def test_custom_discharge_sequence(self):
        """Test sequence with custom discharge followed by storage."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'custom_discharge',
                    'type': 'discharge',
                    'profile': 'custom',
                    'ambient_temperature': 288.15,
                    'sections': [
                        {'duration': 600, 'flow_rate': 0.03, 'phase': 'gas'},
                        {'duration': 3600, 'flow_rate': 0.07, 'phase': 'gas'}
                    ]
                },
                {
                    'name': 'storage',
                    'type': 'dormancy',
                    'profile': 'storage',
                    'ambient_temperature': 288.15,
                    'duration': 43200  # 12 hours
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        assert len(sequence_config.missions) == 2

        # Check custom discharge
        custom_discharge = sequence_config.missions[0]
        assert custom_discharge.name == 'custom_discharge'
        assert custom_discharge.type == 'discharge'
        assert custom_discharge.profile == 'custom'
        assert len(custom_discharge.parameters['sections']) == 2

        # Check storage
        storage = sequence_config.missions[1]
        assert storage.name == 'storage'
        assert storage.type == 'dormancy'
        assert storage.parameters['duration'] == 43200

    def test_sequence_validation(self):
        """Test validation of mission sequences."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'discharge_phase',
                    'type': 'discharge',
                    'profile': 'atr72',
                    'ambient_temperature': 288.15
                },
                {
                    'name': 'refuel_phase',
                    'type': 'refuel',
                    'profile': 'cryogenic',
                    'ambient_temperature': 288.15,
                    'refuel_rate': 0.1,
                    'target_mass': 25.0
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        # Should not raise any exceptions
        sequence_config.validate()

    def test_sequence_validation_with_invalid_mission(self):
        """Test that sequence validation catches invalid missions."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'valid_discharge',
                    'type': 'discharge',
                    'profile': 'atr72',
                    'ambient_temperature': 288.15
                },
                {
                    'name': 'invalid_refuel',
                    'type': 'refuel',
                    'profile': 'cryogenic',
                    'ambient_temperature': 288.15,
                    'refuel_rate': 0.1
                    # Missing target_mass parameter
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="Mission 2 \\('invalid_refuel'\\): Refuel missions require parameters: \\['target_mass'\\]"):
            sequence_config.validate()

    def test_empty_sequence_validation(self):
        """Test that empty sequences are invalid."""
        config_dict = {
            'mission_sequence': []
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="At least one mission is required"):
            sequence_config.validate()

    def test_sequence_with_different_ambient_temperatures(self):
        """Test sequence where each mission has different ambient temperature."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'hot_discharge',
                    'type': 'discharge',
                    'profile': 'constant_flow',
                    'ambient_temperature': 300.0,  # Hot conditions
                    'flow_rate': 0.05,
                    'duration': 1800
                },
                {
                    'name': 'cold_refuel',
                    'type': 'refuel',
                    'profile': 'cryogenic',
                    'ambient_temperature': 250.0,  # Cold conditions
                    'refuel_rate': 0.1,
                    'target_mass': 20.0,
                    'inlet_temperature': 15.0
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        assert len(sequence_config.missions) == 2
        assert sequence_config.missions[0].ambient_temperature == 300.0
        assert sequence_config.missions[1].ambient_temperature == 250.0

    def test_no_mission_configuration(self):
        """Test handling when no mission configuration is provided."""
        config_dict = {
            'other_config': 'value'
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)

        assert len(sequence_config.missions) == 0


class TestStoppingCriteriaValidation:
    """Test stopping criteria configuration and validation."""

    def test_basic_stopping_criteria_validation(self):
        """Test basic stopping criteria validation."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.05,
                'duration': 1800,
                'stopping_criteria': {
                    'minimum_density': 5.8,
                    'use_density_stopping_events': False,
                    'use_duration_stopping': True
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()  # Should not raise

        # Check that stopping criteria are properly stored
        stopping_criteria = mission_config.parameters['stopping_criteria']
        assert stopping_criteria['minimum_density'] == 5.8
        assert stopping_criteria['use_density_stopping_events'] == False
        assert stopping_criteria['use_duration_stopping'] == True

    def test_use_duration_stopping_validation(self):
        """Test use_duration_stopping boolean validation."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.05,
                'duration': 1800,
                'stopping_criteria': {
                    'use_duration_stopping': 'invalid'  # Should be boolean
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="use_duration_stopping must be a boolean"):
            mission_config.validate()

    def test_atr72_with_duration_stopping_works(self):
        """Test that use_duration_stopping=true works with ATR72 profile (has implicit duration)."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',  # ATR72 has implicit duration from predefined sections
                'ambient_temperature': 288.15,
                'stopping_criteria': {
                    'use_duration_stopping': True
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()  # Should not raise - ATR72 has implicit duration

        stopping_criteria = mission_config.parameters['stopping_criteria']
        assert stopping_criteria['use_duration_stopping'] == True

    def test_use_duration_stopping_with_custom_sections(self):
        """Test use_duration_stopping=true works with custom sections."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'custom',
                'ambient_temperature': 288.15,
                'sections': [
                    {'duration': 600, 'flow_rate': 0.03, 'phase': 'gas'},
                    {'duration': 1200, 'flow_rate': 0.05, 'phase': 'gas'}
                ],
                'stopping_criteria': {
                    'use_duration_stopping': True
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()  # Should not raise

    def test_use_duration_stopping_fails_for_refuel_without_duration(self):
        """Test that use_duration_stopping=true fails for profiles without duration support."""
        config_dict = {
            'mission': {
                'type': 'refuel',
                'profile': 'cryogenic',  # Refuel profiles don't have duration
                'ambient_temperature': 288.15,
                'refuel_rate': 0.1,
                'target_mass': 25.0,
                'stopping_criteria': {
                    'use_duration_stopping': True
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="use_duration_stopping=true requires mission to have.*duration"):
            mission_config.validate()

    def test_minimum_density_validation(self):
        """Test minimum_density parameter validation."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',
                'ambient_temperature': 288.15,
                'stopping_criteria': {
                    'minimum_density': -5.8  # Should be positive
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="minimum_density must be a positive number"):
            mission_config.validate()

    def test_stopping_criteria_not_dict_fails(self):
        """Test that stopping_criteria must be a dictionary."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'atr72',
                'ambient_temperature': 288.15,
                'stopping_criteria': "invalid"  # Should be dict
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        with pytest.raises(ValueError, match="stopping_criteria must be a dictionary"):
            mission_config.validate()

    def test_combined_stopping_criteria(self):
        """Test mission with both density and duration stopping criteria."""
        config_dict = {
            'mission': {
                'type': 'discharge',
                'profile': 'constant_flow',
                'ambient_temperature': 288.15,
                'flow_rate': 0.05,
                'duration': 1800,
                'stopping_criteria': {
                    'minimum_density': 5.8,
                    'use_density_stopping_events': True,
                    'use_duration_stopping': True
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()  # Should not raise

        # Both stopping criteria should be enabled
        stopping_criteria = mission_config.parameters['stopping_criteria']
        assert stopping_criteria['use_density_stopping_events'] == True
        assert stopping_criteria['use_duration_stopping'] == True

    def test_refuel_mission_with_duration_stopping(self):
        """Test refuel mission with duration-based stopping."""
        config_dict = {
            'mission': {
                'type': 'refuel',
                'profile': 'cryogenic',
                'ambient_temperature': 288.15,
                'refuel_rate': 0.1,
                'target_mass': 25.0,
                'inlet_temperature': 20.0,
                'max_duration': 1200,  # Maximum refuel time
                'stopping_criteria': {
                    'use_duration_stopping': True
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)

        # Should fail because refuel doesn't have 'duration' parameter or profile-based duration
        with pytest.raises(ValueError, match="use_duration_stopping=true requires mission to have.*duration"):
            mission_config.validate()

    def test_dormancy_with_duration_stopping_compatible(self):
        """Test that dormancy missions are compatible with duration stopping."""
        config_dict = {
            'mission': {
                'type': 'dormancy',
                'profile': 'storage',
                'ambient_temperature': 288.15,
                'duration': 86400,  # 24 hours
                'stopping_criteria': {
                    'use_duration_stopping': True,
                    'minimum_density': 40.0  # Keep minimum density during storage
                }
            }
        }

        mission_config = MissionConfig.from_dict(config_dict)
        mission_config.validate()  # Should not raise


class TestMissionSequenceIntegration:
    """Test integration scenarios for mission sequences."""

    def test_verification_cch2_style_sequence(self):
        """Test a verification_cch2 style mission sequence."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'discharge_to_minimum',
                    'type': 'discharge',
                    'profile': 'constant_flow',
                    'ambient_temperature': 288.15,
                    'flow_rate': 0.05,
                    'duration': 2400  # 40 minutes discharge
                },
                {
                    'name': 'fast_refuel',
                    'type': 'refuel',
                    'profile': 'cryogenic',
                    'ambient_temperature': 288.15,
                    'refuel_rate': 0.15,  # Fast refuel
                    'target_mass': 25.0,
                    'inlet_temperature': 20.0,
                    'inlet_pressure': 100000
                },
                {
                    'name': 'long_term_storage',
                    'type': 'dormancy',
                    'profile': 'storage',
                    'ambient_temperature': 288.15,
                    'duration': 172800,  # 48 hours storage
                    'vent_pressure': 450e5
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)
        sequence_config.validate()

        # Verify all phases are configured correctly
        assert len(sequence_config.missions) == 3

        names = [mission.name for mission in sequence_config.missions]
        types = [mission.type for mission in sequence_config.missions]

        assert names == ['discharge_to_minimum', 'fast_refuel', 'long_term_storage']
        assert types == ['discharge', 'refuel', 'dormancy']

    def test_multi_discharge_sequence(self):
        """Test sequence with multiple discharge phases."""
        config_dict = {
            'mission_sequence': [
                {
                    'name': 'first_flight',
                    'type': 'discharge',
                    'profile': 'atr72',
                    'ambient_temperature': 288.15
                },
                {
                    'name': 'partial_refuel',
                    'type': 'refuel',
                    'profile': 'ambient',
                    'ambient_temperature': 288.15,
                    'refuel_rate': 0.08,
                    'target_mass': 15.0  # Partial refuel only
                },
                {
                    'name': 'second_flight',
                    'type': 'discharge',
                    'profile': 'constant_flow',
                    'ambient_temperature': 288.15,
                    'flow_rate': 0.04,
                    'duration': 1200  # Shorter flight
                },
                {
                    'name': 'overnight_storage',
                    'type': 'dormancy',
                    'profile': 'storage',
                    'ambient_temperature': 285.0,  # Slightly cooler overnight
                    'duration': 28800  # 8 hours
                }
            ]
        }

        sequence_config = MissionSequenceConfig.from_dict(config_dict)
        sequence_config.validate()

        assert len(sequence_config.missions) == 4

        # Check that we have multiple discharge phases
        discharge_missions = [m for m in sequence_config.missions if m.type == 'discharge']
        assert len(discharge_missions) == 2
        assert discharge_missions[0].profile == 'atr72'
        assert discharge_missions[1].profile == 'constant_flow'


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])