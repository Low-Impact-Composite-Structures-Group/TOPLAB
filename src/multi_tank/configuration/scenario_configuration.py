"""
Scenario Configuration for Multi-Tank Systems

This module provides the ScenarioConfig class that loads YAML configurations
directly for multi-tank hydrogen storage system analysis.

Key Features:
- Direct YAML loading without format migrations
- Network-based configuration format support
- Tank geometry and mission configuration parsing
- Coupling rules extraction from network edges

Author: Multi-Tank Framework
Date: October 2025
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "src"))

from materials.materials_for_multi_tank.nist_material import NISTMaterial, get_material_by_nist_path


@dataclass
class MissionConfig:
    type: str
    profile: str
    ambient_temperature: float
    parameters: Dict[str, Any]
    name: Optional[str] = None
    assigned_to: Optional[int] = None  # Tank ID that executes this mission

    @classmethod
    def from_dict(cls, config_dict: dict, config_format: str = "old"):
        """Create MissionConfig from dictionary, handling both old and new formats."""
        mission_data = config_dict.get('mission', {})

        # Handle new format
        if config_format == "new":
            mission_type = mission_data.get('type')
            profile = mission_data.get('profile')
            ambient_temp = mission_data.get('ambient_temperature', 288.15)
            parameters = mission_data.get('parameters', {})
            assigned_to = mission_data.get('assigned_to_node')  # New format uses assigned_to_node
        else:
            # Handle old format
            mission_type = mission_data.get('type')
            profile = mission_data.get('profile')
            ambient_temp = mission_data.get('ambient_temperature', 288.15)
            parameters = mission_data.get('parameters', {})
            assigned_to = mission_data.get('assigned_to')  # Old format uses assigned_to

        return cls(
            type=mission_type,
            profile=profile,
            ambient_temperature=ambient_temp,
            parameters=parameters,
            assigned_to=assigned_to
        )


@dataclass
class MissionSequenceConfig:
    missions: List[MissionConfig]

    @classmethod
    def from_dict(cls, config_dict: dict, config_format: str = "old"):
        """Create MissionSequenceConfig from dictionary, handling both formats."""
        mission_data = config_dict.get('mission', {})

        if 'sequence' in mission_data:
            missions = []
            for i, seq_mission in enumerate(mission_data['sequence']):
                mission_config = {
                    'mission': seq_mission
                }
                missions.append(MissionConfig.from_dict(mission_config, config_format))
            return cls(missions=missions)
        else:
            single_mission = MissionConfig.from_dict(config_dict, config_format)
            return cls(missions=[single_mission])

    def validate(self):
        if not self.missions:
            raise ValueError("At least one mission is required")
        for mission in self.missions:
            if not mission.type or not mission.profile:
                raise ValueError("Mission type and profile are required")


class ScenarioConfig:
    """
    Simplified ScenarioConfig that loads YAML configurations directly.
    Supports network-based configuration format for multi-tank systems.
    """

    def __init__(self, config_dict: Dict[str, Any], config_format: str, config_path: Optional[str] = None):
        self.config_dict = config_dict
        self.config_format = config_format
        self._config_path = config_path

        # Parse analysis metadata (always new format now)
        analysis = config_dict.get('analysis', {})
        self.analysis_name = analysis.get('name', 'Unnamed Analysis')
        self.description = analysis.get('description', '')
        self.version = analysis.get('version', '1.0')

        # Parse mission configuration
        self.mission_sequence = MissionSequenceConfig.from_dict(config_dict, "new")

        # Parse materials and geometry (always new format)
        self.materials = self._parse_materials_new_format()
        self.tank_materials = self._parse_tank_materials_new_format()
        self.tank_geometries = self._parse_tank_geometries_new_format()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        """
        Load configuration directly from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            EnhancedScenarioConfig instance
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        # Load YAML directly
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert network edges to coupling_rules for SystemOrchestrator
        if 'network' in config_dict and 'edges' in config_dict['network']:
            edges = config_dict['network']['edges']
            coupling_rules = []

            for edge in edges:
                connection_type = edge.get('connection_type', 'pressure_compensation')
                from_node = edge.get('from_node', 1)
                to_node = edge.get('to_node', 2)
                edge_id = edge.get('edge_id', f'coupling_{from_node}_{to_node}')

                rule = {
                    'coupling_id': edge_id,
                    'coupling_type': connection_type,
                    'description': edge.get('description', ''),
                    'participants': {
                        'source': from_node,
                        'target': to_node,
                    },
                    'activation_conditions': edge.get('activation_conditions', {}),
                    'control_parameters': edge.get('control_parameters', {}),
                    'flow_parameters': edge.get('flow_parameters', {}),
                    'flow_physics': edge.get('flow_physics', {}),
                    'discharge_piping': edge.get('discharge_piping', {}),
                }
                coupling_rules.append(rule)

            config_dict['coupling_rules'] = coupling_rules
            print(f"   ✓ Converted {len(coupling_rules)} edges to coupling_rules for legacy compatibility")

        # Always treat as new format (network-based)
        scenario = cls(config_dict, "new", str(yaml_path))
        return scenario

    def _parse_materials_new_format(self) -> Dict[str, NISTMaterial]:
        """Parse materials from new format (from physics section)."""
        materials = {}

        # In new format, global materials would be in physics section
        # For now, return empty dict as materials are per-tank in new format
        return materials

    def _parse_tank_materials_new_format(self) -> Dict[int, Dict[str, NISTMaterial]]:
        """Parse per-tank materials from new format."""
        tank_materials = {}
        network = self.config_dict.get('network', {})
        nodes = network.get('nodes', [])

        for node in nodes:
            if node.get('type') == 'tank':
                tank_id = node.get('node_id')
                if tank_id is not None:
                    tank_materials[tank_id] = {}

                    node_materials = node.get('materials', {})
                    for material_name, material_data in node_materials.items():
                        if material_name == 'safety_margin':
                            continue

                        nist_path = material_data.get('nist_path')
                        if nist_path:
                            try:
                                tank_materials[tank_id][material_name] = get_material_by_nist_path(nist_path)
                            except ValueError as e:
                                print(f"Warning: Could not load NIST material {nist_path} for tank {tank_id}: {e}")

        return tank_materials

    def _parse_tank_geometries_new_format(self) -> Dict[int, Dict[str, Any]]:
        """Parse tank geometries from new format."""
        geometries = {}
        network = self.config_dict.get('network', {})
        nodes = network.get('nodes', [])

        for node in nodes:
            if node.get('type') == 'tank':
                tank_id = node.get('node_id')
                if tank_id is not None:
                    # Combine geometry, initial conditions, and operating limits
                    geometry = {}

                    # Add geometry parameters
                    node_geometry = node.get('geometry', {})
                    geometry.update(node_geometry)

                    # Add initial conditions with 'initial_' prefix for compatibility
                    initial_conditions = node.get('initial_conditions', {})
                    for param, value in initial_conditions.items():
                        geometry[f'initial_{param}'] = value

                    # Add operating limits
                    operating_limits = node.get('operating_limits', {})
                    geometry.update(operating_limits)

                    # Include stopping criteria that affect system-wide stopping logic
                    # e.g. minimum usable density per tank
                    stopping_criteria = node.get('stopping_criteria', {})
                    if 'minimum_density' in stopping_criteria:
                        try:
                            geometry['minimum_density'] = float(stopping_criteria['minimum_density'])
                        except Exception:
                            # Keep raw value if it cannot be converted cleanly
                            geometry['minimum_density'] = stopping_criteria['minimum_density']

                    geometries[tank_id] = geometry

        return geometries

    def _parse_materials_old_format(self) -> Dict[str, NISTMaterial]:
        """Parse global materials from old format (backward compatibility)."""
        materials = {}
        materials_data = self.config_dict.get('materials', {})

        for material_name, material_data in materials_data.items():
            if material_name == 'safety_margin':
                continue

            nist_path = material_data.get('nist_path')
            if nist_path:
                try:
                    materials[material_name] = get_material_by_nist_path(nist_path)
                except ValueError as e:
                    print(f"Warning: Could not load NIST material {nist_path}: {e}")

        return materials

    def _parse_tank_materials_old_format(self) -> Dict[int, Dict[str, NISTMaterial]]:
        """Parse per-tank materials from old format."""
        tank_materials = {}
        tank_materials_data = self.config_dict.get('tank_materials', {})

        for tank_id, tank_material_data in tank_materials_data.items():
            tank_id_int = int(tank_id)
            tank_materials[tank_id_int] = {}

            for material_name, material_data in tank_material_data.items():
                if material_name == 'safety_margin':
                    continue

                nist_path = material_data.get('nist_path')
                if nist_path:
                    try:
                        tank_materials[tank_id_int][material_name] = get_material_by_nist_path(nist_path)
                    except ValueError as e:
                        print(f"Warning: Could not load NIST material {nist_path} for tank {tank_id}: {e}")

        return tank_materials

    def _parse_tank_geometries_old_format(self) -> Dict[int, Dict[str, Any]]:
        """Parse tank geometries from old format."""
        geometries = {}
        geometry_data = self.config_dict.get('geometry', {})

        for tank_id, geom_data in geometry_data.items():
            geometries[int(tank_id)] = geom_data

        return geometries

    def validate(self):
        """Validate the scenario configuration."""
        self.mission_sequence.validate()

        if not self.tank_geometries:
            raise ValueError("At least one tank geometry is required")

    def get_tank_count(self) -> int:
        """Get the number of tanks in the configuration."""
        return len(self.tank_geometries)

    def get_mission_count(self) -> int:
        """Get the number of missions in the configuration."""
        return len(self.mission_sequence.missions)

    def get_tank_materials(self, tank_id: int) -> Dict[str, NISTMaterial]:
        """Get materials for specific tank - per-tank materials are mandatory."""
        if tank_id in self.tank_materials:
            return self.tank_materials[tank_id]
        else:
            raise RuntimeError(f"No materials configuration found for tank {tank_id}. "
                             f"Per-tank materials must be specified.")

    def get_tank_material_config(self, tank_id: int) -> Dict[str, Any]:
        """Get raw material configuration for specific tank."""
        if self.config_format == "new":
            # Find tank node in network
            network = self.config_dict.get('network', {})
            nodes = network.get('nodes', [])

            for node in nodes:
                if node.get('node_id') == tank_id and node.get('type') == 'tank':
                    return node.get('materials', {})

            raise RuntimeError(f"No material configuration found for tank {tank_id}")
        else:
            # Old format
            tank_materials_data = self.config_dict.get('tank_materials', {})

            # Try both string and int keys
            for key in [str(tank_id), tank_id]:
                if key in tank_materials_data:
                    return tank_materials_data[key]

            raise RuntimeError(f"No material configuration found for tank {tank_id}")

    def get_network_nodes(self) -> List[Dict[str, Any]]:
        """Get all network nodes (new format only)."""
        if self.config_format != "new":
            raise RuntimeError("Network nodes are only available in new format")

        network = self.config_dict.get('network', {})
        return network.get('nodes', [])

    def get_network_edges(self) -> List[Dict[str, Any]]:
        """Get all network edges (new format only)."""
        if self.config_format != "new":
            raise RuntimeError("Network edges are only available in new format")

        network = self.config_dict.get('network', {})
        return network.get('edges', [])

    def summary(self) -> str:
        """Get a summary of the scenario configuration."""
        lines = []
        lines.append(f"Scenario: {self.analysis_name}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Version: {self.version}")
        lines.append(f"Format: {self.config_format}")
        lines.append(f"Tanks: {self.get_tank_count()}")
        lines.append(f"Missions: {self.get_mission_count()}")

        if self.config_format == "new":
            lines.append(f"Nodes: {len(self.get_network_nodes())}")
            lines.append(f"Edges: {len(self.get_network_edges())}")

        return "\n".join(lines)


def main():
    """Test the enhanced scenario configuration."""
    test_config_path = Path(__file__).parent.parent.parent / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

    if not test_config_path.exists():
        print(f"Test configuration file not found: {test_config_path}")
        return False

    try:
        print("Testing EnhancedScenarioConfig with single_tank_cch2_config.yaml...")

        scenario = ScenarioConfig.from_yaml(test_config_path)

        print("Configuration parsed successfully!")
        print("=" * 60)
        print(scenario.summary())
        print("=" * 60)

        print("\nTank Materials:")
        for tank_id in scenario.tank_materials:
            materials = scenario.get_tank_materials(tank_id)
            print(f"  Tank {tank_id}:")
            for name, material in materials.items():
                try:
                    specific_heat = material.get_specific_heat(300.0)
                    print(f"    - {name}: {material.__class__.__name__}, Cp(300K) = {specific_heat:.2f} J/kg/K")
                except:
                    print(f"    - {name}: {material.__class__.__name__}")

        print("\nMission Details:")
        for i, mission in enumerate(scenario.mission_sequence.missions, 1):
            print(f"  Mission {i}: {mission.type} - {mission.profile}")
            if mission.assigned_to:
                print(f"    Assigned to: Tank {mission.assigned_to}")

        print("\nTank Geometry:")
        for tank_id, geometry in scenario.tank_geometries.items():
            initial_pressure = geometry.get('initial_pressure', 0)
            if isinstance(initial_pressure, str):
                initial_pressure = float(initial_pressure)
            print(f"  Tank {tank_id}: phi={geometry.get('phi', 'N/A')}, P_init={initial_pressure/1e5:.0f} bar")

        return True

    except Exception as e:
        print(f"Enhanced ScenarioConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)