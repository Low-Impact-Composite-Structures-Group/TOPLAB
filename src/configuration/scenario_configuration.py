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

from src.materials.nist_materials import NISTMaterial, get_material_by_nist_path


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
        # Legacy format: top-level mission_sequence.missions
        if config_format == "old":
            mission_sequence = config_dict.get('mission_sequence', {})
            missions_list = mission_sequence.get('missions')
            if isinstance(missions_list, list) and missions_list:
                missions: List[MissionConfig] = []
                for seq_mission in missions_list:
                    missions.append(
                        MissionConfig(
                            type=seq_mission.get('type'),
                            profile=seq_mission.get('profile'),
                            ambient_temperature=mission_sequence.get('ambient_temperature', 288.15),
                            parameters={k: v for k, v in seq_mission.items() if k not in {'type', 'profile', 'ambient_temperature', 'name'}},
                            name=seq_mission.get('name')
                        )
                    )
                return cls(missions=missions)

        # New format: mission.sequence
        mission_data = config_dict.get('mission', {})
        if 'sequence' in mission_data:
            missions: List[MissionConfig] = []
            for seq_mission in mission_data['sequence']:
                mission_config = {
                    'mission': seq_mission
                }
                missions.append(MissionConfig.from_dict(mission_config, config_format))
            return cls(missions=missions)

        # Single mission
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

        if self.config_format == "new":
            # Parse analysis metadata (new format)
            analysis = config_dict.get('analysis', {})
            self.analysis_name = analysis.get('name', 'Unnamed Analysis')
            self.description = analysis.get('description', '')
            self.version = analysis.get('version', '1.0')

            # Parse mission configuration
            self.mission_sequence = MissionSequenceConfig.from_dict(config_dict, "new")

            # Parse materials and geometry
            self.materials = self._parse_materials_new_format()
            self.tank_materials = self._parse_tank_materials_new_format()
            self.tank_geometries = self._parse_tank_geometries_new_format()
        elif self.config_format == "old":
            # Parse analysis metadata (legacy format)
            self.analysis_name = config_dict.get('analysis_name', config_dict.get('analysis', {}).get('name', 'Unnamed Analysis'))
            self.description = config_dict.get('description', config_dict.get('analysis', {}).get('description', ''))
            self.version = str(config_dict.get('version', config_dict.get('analysis', {}).get('version', '1.0')))

            # Parse mission configuration (supports legacy mission_sequence)
            self.mission_sequence = MissionSequenceConfig.from_dict(config_dict, "old")

            # Parse legacy materials/geometry
            self.materials = self._parse_materials_old_format()
            self.tank_materials = self._parse_tank_materials_old_format()
            self.tank_geometries = self._parse_tank_geometries_old_format()
        else:
            raise ValueError(f"Unknown config_format '{self.config_format}'. Expected 'new' or 'old'.")

    @staticmethod
    def _edge_participant(edge: Dict[str, Any], key: str) -> Any:
        participants = edge.get('participants', {}) or {}
        if key == 'source':
            return participants.get('source', edge.get('from_node'))
        if key == 'target':
            return participants.get('target', edge.get('to_node'))
        raise ValueError(f"Unknown participant key '{key}'")

    @classmethod
    def _compile_network_coupling_rules(cls, config_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile the declarative network graph into legacy coupling-rule records."""
        network = config_dict.get('network', {}) or {}
        nodes = network.get('nodes', []) or []
        edges = network.get('edges', []) or []

        if not edges:
            return []

        node_by_id: Dict[Any, Dict[str, Any]] = {}
        tank_node_ids: List[Any] = []
        for node in nodes:
            node_id = node.get('node_id')
            if node_id is None:
                continue
            node_by_id[node_id] = node
            if node.get('type') == 'tank':
                tank_node_ids.append(node_id)

        outgoing_edges: Dict[Any, List[Dict[str, Any]]] = {}
        incoming_edges: Dict[Any, List[Dict[str, Any]]] = {}
        for edge in edges:
            source_node = cls._edge_participant(edge, 'source')
            target_node = cls._edge_participant(edge, 'target')
            outgoing_edges.setdefault(source_node, []).append(edge)
            incoming_edges.setdefault(target_node, []).append(edge)

        handled_edges = set()
        coupling_rules: List[Dict[str, Any]] = []

        def _build_rule(edge: Dict[str, Any], coupling_type: str, source_node: Any, target_node: Any) -> Dict[str, Any]:
            return {
                'coupling_id': edge.get('edge_id', f'coupling_{source_node}_{target_node}'),
                'coupling_type': coupling_type,
                'description': edge.get('description', ''),
                'participants': {
                    'source': source_node,
                    'target': target_node,
                },
                'activation_conditions': edge.get('activation_conditions', {}),
                'control_parameters': edge.get('control_parameters', {}),
                'flow_parameters': edge.get('flow_parameters', {}),
                'flow_physics': edge.get('flow_physics', {}),
                'discharge_piping': edge.get('discharge_piping', {}),
                'peripheral_components': edge.get('peripheral_components', edge.get('components', [])),
                'main_conditioning_components': edge.get('main_conditioning_components', []),
                'discharge_conditioning': edge.get('discharge_conditioning', []),
                'split_fraction': edge.get('split_fraction'),
            }

        def _is_tank(node_id: Any) -> bool:
            node = node_by_id.get(node_id)
            return bool(node and node.get('type') == 'tank')

        for node in nodes:
            if node.get('type') != 'junction':
                continue

            junction_id = node.get('node_id')
            if junction_id is None:
                continue

            junction_incoming = incoming_edges.get(junction_id, [])
            junction_outgoing = outgoing_edges.get(junction_id, [])
            if not junction_incoming or not junction_outgoing:
                continue

            junction_role = node.get('junction_role') or node.get('role') or ''
            role_normalized = str(junction_role).lower()

            source_edge = next((edge for edge in junction_incoming if _is_tank(cls._edge_participant(edge, 'source'))), None)
            if source_edge is None:
                continue

            source_node = cls._edge_participant(source_edge, 'source')

            if role_normalized == 'mixer' or any(edge.get('connection_type') == 'pressure_triggered_discharge' for edge in junction_incoming):
                discharge_edge = next((edge for edge in junction_incoming if edge.get('connection_type') == 'pressure_triggered_discharge'), source_edge)
                outlet_edge = next((edge for edge in junction_outgoing if _is_tank(cls._edge_participant(edge, 'target'))), junction_outgoing[0])

                target_node = cls._edge_participant(outlet_edge, 'target')
                rule = _build_rule(discharge_edge, 'pressure_triggered_discharge', source_node, -1)
                discharge_conditioning = outlet_edge.get('discharge_conditioning', outlet_edge.get('components', []))
                if discharge_conditioning:
                    rule['discharge_conditioning'] = discharge_conditioning
                coupling_rules.append(rule)
                handled_edges.update({edge.get('edge_id') for edge in junction_incoming})
                handled_edges.update({edge.get('edge_id') for edge in junction_outgoing})
                continue

            split_edges = [edge for edge in junction_outgoing if edge.get('split_fraction') is not None]
            if not split_edges:
                continue

            peripheral_edge = min(split_edges, key=lambda edge: float(edge.get('split_fraction', 1.0)))
            main_edge = next((edge for edge in junction_outgoing if edge is not peripheral_edge), peripheral_edge)
            target_node = cls._edge_participant(peripheral_edge, 'target')
            if not _is_tank(target_node):
                continue

            rule = _build_rule(peripheral_edge, 'proportional_split', source_node, target_node)
            rule['split_fraction'] = peripheral_edge.get('split_fraction')
            main_components = main_edge.get('main_conditioning_components', main_edge.get('components', []))
            peripheral_components = peripheral_edge.get('peripheral_components', peripheral_edge.get('components', []))
            if main_components:
                rule['main_conditioning_components'] = main_components
            if peripheral_components:
                rule['peripheral_components'] = peripheral_components
            coupling_rules.append(rule)
            handled_edges.update({edge.get('edge_id') for edge in junction_incoming})
            handled_edges.update({edge.get('edge_id') for edge in junction_outgoing})

        for edge in edges:
            edge_id = edge.get('edge_id')
            if edge_id in handled_edges:
                continue

            connection_type = edge.get('connection_type', 'pressure_compensation')
            source_node = cls._edge_participant(edge, 'source')
            target_node = cls._edge_participant(edge, 'target')

            if connection_type == 'proportional_split' and _is_tank(source_node) and _is_tank(target_node):
                coupling_rules.append(_build_rule(edge, connection_type, source_node, target_node))
                continue

            if connection_type == 'pressure_triggered_discharge' and _is_tank(source_node):
                coupling_rules.append(_build_rule(edge, connection_type, source_node, -1))
                continue

            if _is_tank(source_node) and _is_tank(target_node):
                coupling_rules.append(_build_rule(edge, connection_type, source_node, target_node))

        return coupling_rules

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

        # Detect format
        is_new_format = isinstance(config_dict, dict) and 'network' in config_dict
        is_old_format = isinstance(config_dict, dict) and (
            'mission_sequence' in config_dict or 'analysis_name' in config_dict or 'geometry' in config_dict
        )

        if is_new_format:
            config_dict['coupling_rules'] = cls._compile_network_coupling_rules(config_dict)

            return cls(config_dict, "new", str(yaml_path))

        if is_old_format:
            return cls(config_dict, "old", str(yaml_path))

        raise ValueError(f"Unrecognized configuration format in {yaml_path}")

    # TODO(DR): HDF5 config support is currently on hold; production analyses
    # should continue to use YAML via `ScenarioConfig.from_yaml()` until the full
    # orchestration stack is made file-format-agnostic.

    @classmethod
    def from_hdf5(cls, hdf5_path: Union[str, Path]):
        """
        Load configuration from an HDF5 file written by
        ``src.configuration.hdf5_io.write_config_to_hdf5``.

        The HDF5 file must encode a *new-format* config dict (i.e. it must
        contain a ``network`` group).  Use
        ``ScenarioConfig.to_hdf5`` / ``write_config_to_hdf5`` to create one.

        Args:
            hdf5_path: Path to the ``.h5`` configuration file.

        Returns:
            ScenarioConfig instance (identical to what ``from_yaml`` would
            produce for the equivalent YAML config).
        """
        from src.configuration.hdf5_io import read_config_from_hdf5

        hdf5_path = Path(hdf5_path)
        config_dict = read_config_from_hdf5(hdf5_path)

        if 'network' not in config_dict:
            raise ValueError(
                f"HDF5 config at {hdf5_path} does not contain a 'network' group. "
                "Only new-format configs are supported via from_hdf5()."
            )

        config_dict['coupling_rules'] = cls._compile_network_coupling_rules(config_dict)
        return cls(config_dict, "new", str(hdf5_path))

    def to_hdf5(self, hdf5_path: Union[str, Path]) -> None:
        """Serialise this config's ``config_dict`` to an HDF5 file.

        This is the inverse of :meth:`from_hdf5`.  The ``coupling_rules``
        injected by :meth:`from_yaml` are *not* persisted — they are always
        recomputed on load from the ``network`` section.

        Args:
            hdf5_path: Destination path for the ``.h5`` file.
        """
        from src.configuration.hdf5_io import write_config_to_hdf5

        # Omit the compiled coupling_rules; they are derived from network
        storable = {k: v for k, v in self.config_dict.items() if k != 'coupling_rules'}
        write_config_to_hdf5(storable, hdf5_path)

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