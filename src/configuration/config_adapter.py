#!/usr/bin/env python3
"""
Configuration Adapter for Multi-Tank Analysis Framework

This module provides bidirectional conversion between the old flat YAML format
and the new intuitive network-based format. Enables gradual migration without
breaking existing analyses.

Author: Analysis Framework Team
Date: October 13, 2025
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
import copy

# Import ConfigurationError - handle both relative and absolute imports
try:
    from .config_validator import ConfigurationError
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config_validator import ConfigurationError


class ConfigurationAdapter:
    """
    Adapter for converting between old flat format and new network format.

    Supports:
    - Automatic format detection
    - Old-to-new format conversion
    - New-to-old format conversion (for legacy compatibility)
    - Validation of conversion correctness
    """

    def __init__(self):
        """Initialize the configuration adapter."""
        self.conversion_warnings = []

    def detect_format(self, config: Dict[str, Any]) -> str:
        """
        Detect whether config uses old flat format or new network format.

        Args:
            config: Loaded YAML configuration dictionary

        Returns:
            'new' if network-based format, 'old' if flat format

        Raises:
            ConfigurationError: If format cannot be determined
        """
        # New format indicators
        if 'network' in config and isinstance(config['network'], dict):
            if 'nodes' in config['network'] and isinstance(config['network']['nodes'], list):
                return 'new'

        # Old format indicators
        old_format_keys = ['geometry', 'tank_materials', 'coupling_rules']
        if any(key in config for key in old_format_keys):
            return 'old'

        raise ConfigurationError(
            "Cannot determine configuration format. "
            "Configuration must use either old format (with 'geometry', 'tank_materials') "
            "or new format (with 'network.nodes')."
        )

    def migrate_old_to_new(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert old flat format to new network-based format.

        Args:
            old_config: Configuration in old flat format

        Returns:
            Configuration in new network format

        Raises:
            ConfigurationError: If conversion fails
        """
        self.conversion_warnings = []
        new_config = {}

        try:
            # Convert analysis metadata
            self._convert_analysis_metadata(old_config, new_config)

            # Convert network topology
            self._convert_network_topology(old_config, new_config)

            # Convert mission configuration
            self._convert_mission_config(old_config, new_config)

            # Convert physics configuration
            self._convert_physics_config(old_config, new_config)

            # Convert solver configuration
            self._convert_solver_config(old_config, new_config)

            # Convert output configuration
            self._convert_output_config(old_config, new_config)

        except Exception as e:
            raise ConfigurationError(f"Failed to convert old format to new format: {e}")

        return new_config

    def migrate_new_to_old(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert new network format to old flat format.

        Args:
            new_config: Configuration in new network format

        Returns:
            Configuration in old flat format

        Note:
            This is primarily for backward compatibility during migration.
            Some new format features may not convert perfectly to old format.
        """
        self.conversion_warnings = []
        old_config = {}

        try:
            # Convert analysis metadata
            self._convert_analysis_metadata_reverse(new_config, old_config)

            # Convert network back to flat structure
            self._convert_network_topology_reverse(new_config, old_config)

            # Convert other sections
            self._convert_mission_config_reverse(new_config, old_config)
            self._convert_physics_config_reverse(new_config, old_config)
            self._convert_solver_config_reverse(new_config, old_config)
            self._convert_output_config_reverse(new_config, old_config)

        except Exception as e:
            raise ConfigurationError(f"Failed to convert new format to old format: {e}")

        return old_config

    def _convert_analysis_metadata(self, old_config: Dict, new_config: Dict):
        """Convert analysis metadata section."""
        # Extract analysis info from old format
        analysis_name = old_config.get('analysis_name', 'Unnamed Analysis')
        description = old_config.get('description', 'No description provided')
        version = old_config.get('version', '1.0')

        new_config['analysis'] = {
            'name': analysis_name,
            'description': description,
            'version': version
        }

    def _convert_network_topology(self, old_config: Dict, new_config: Dict):
        """Convert network topology from old flat format to new structure."""
        network = old_config.get('network', {})
        geometry = old_config.get('geometry', {})
        tank_materials = old_config.get('tank_materials', {})
        stopping_criteria = old_config.get('stopping_criteria', {})

        # Initialize network structure
        new_config['network'] = {
            'nodes': [],
            'edges': []
        }

        # Convert tanks (nodes)
        if 'tanks' in network:
            tank_list = network['tanks']
        else:
            # Infer tanks from geometry keys
            tank_list = [{'tank_id': int(k), 'tank_type': 'unknown'}
                        for k in geometry.keys() if str(k).isdigit()]

        for tank_info in tank_list:
            tank_id = tank_info['tank_id']
            node = self._convert_tank_to_node(
                tank_id, tank_info, geometry, tank_materials, stopping_criteria, old_config
            )
            new_config['network']['nodes'].append(node)

        # Convert coupling rules to edges
        coupling_rules = old_config.get('coupling_rules', [])
        for coupling in coupling_rules:
            edge = self._convert_coupling_to_edge(coupling, old_config)
            new_config['network']['edges'].append(edge)

    def _convert_tank_to_node(self, tank_id: int, tank_info: Dict, geometry: Dict,
                             tank_materials: Dict, stopping_criteria: Dict, old_config: Dict) -> Dict:
        """Convert individual tank to node format."""
        node = {
            'node_id': tank_id,
            'type': 'tank',
            'fluid': tank_info.get('tank_type', 'H2'),  # Default to H2 if not specified
            'description': f"Tank {tank_id}"
        }

        # Convert geometry
        tank_geometry = geometry.get(tank_id, {})
        if tank_geometry:
            node['geometry'] = {}
            if 'phi' in tank_geometry:
                node['geometry']['phi'] = tank_geometry['phi']
            if 'radius' in tank_geometry:
                node['geometry']['radius'] = tank_geometry['radius']
            if 'mission_based_sizing' in tank_geometry:
                node['geometry']['mission_based_sizing'] = tank_geometry['mission_based_sizing']

            # Convert initial conditions from geometry section
            initial_conditions = {}
            if 'initial_pressure' in tank_geometry:
                initial_conditions['pressure'] = tank_geometry['initial_pressure']
            if 'initial_temperature' in tank_geometry:
                initial_conditions['temperature'] = tank_geometry['initial_temperature']
            if 'initial_density' in tank_geometry:
                initial_conditions['density'] = tank_geometry['initial_density']

            if initial_conditions:
                node['initial_conditions'] = initial_conditions

            # Convert operating limits from geometry section
            operating_limits = {}
            if 'minimum_pressure' in tank_geometry:
                operating_limits['minimum_pressure'] = tank_geometry['minimum_pressure']
            if 'venting_pressure' in tank_geometry:
                operating_limits['venting_pressure'] = tank_geometry['venting_pressure']

            if operating_limits:
                node['operating_limits'] = operating_limits

        # Convert materials
        tank_material = tank_materials.get(tank_id, {})
        if tank_material:
            node['materials'] = copy.deepcopy(tank_material)

        # Convert stopping criteria (if tank-specific)
        if stopping_criteria:
            node['stopping_criteria'] = copy.deepcopy(stopping_criteria)

        # Add default plotting section
        node['plotting'] = {
            'show_reference_pressures': False,
            'show_operating_limits': True,
            'color_scheme': 'auto'
        }

        return node

    def _convert_coupling_to_edge(self, coupling: Dict, old_config: Dict) -> Dict:
        """Convert coupling rule to edge format."""
        edge = {
            'edge_id': coupling.get('coupling_id', 'unnamed_coupling'),
            'connection_type': coupling.get('coupling_type', 'simple_coupling'),
            'description': coupling.get('description', 'No description')
        }

        # Extract participants
        participants = coupling.get('participants', {})
        if 'source' in participants:
            edge['from_node'] = participants['source']
        if 'target' in participants:
            edge['to_node'] = participants['target']

        # Convert piping information
        if 'discharge_piping' in coupling:
            piping = coupling['discharge_piping']
            edge['piping'] = {}
            if 'diameter_m' in piping:
                edge['piping']['diameter_m'] = piping['diameter_m']
            if 'length_m' in piping:
                edge['piping']['length_m'] = piping['length_m']
            if 'roughness_m' in piping:
                edge['piping']['roughness_m'] = piping['roughness_m']
            if 'loss_coefficient' in piping:
                edge['piping']['loss_coefficient'] = piping['loss_coefficient']

        # Convert flow parameters to flow_physics
        if 'flow_parameters' in coupling:
            flow_params = coupling['flow_parameters']
            edge['flow_physics'] = {
                'orifice_flow': {},
                'safety_limits': {}
            }

            if 'max_flow_rate_kg_s' in flow_params:
                edge['flow_physics']['safety_limits']['max_flow_rate_kg_s'] = flow_params['max_flow_rate_kg_s']
            if 'orifice_diameter_m' in flow_params:
                edge['flow_physics']['orifice_flow']['orifice_diameter_m'] = flow_params['orifice_diameter_m']

        # Convert control parameters
        if 'control_parameters' in coupling:
            edge['control_parameters'] = copy.deepcopy(coupling['control_parameters'])

        return edge

    def _convert_mission_config(self, old_config: Dict, new_config: Dict):
        """Convert mission configuration."""
        old_mission = old_config.get('mission', {})

        new_config['mission'] = {}
        if 'type' in old_mission:
            # 'type' becomes 'mode' or can be inferred
            pass
        if 'profile' in old_mission:
            new_config['mission']['profile'] = old_mission['profile']
        if 'ambient_temperature' in old_mission:
            new_config['mission']['ambient_temperature'] = old_mission['ambient_temperature']
        if 'assigned_to' in old_mission:
            new_config['mission']['assigned_to_node'] = old_mission['assigned_to']

    def _convert_physics_config(self, old_config: Dict, new_config: Dict):
        """Convert physics configuration."""
        old_physics = old_config.get('flow_physics', {})

        if old_physics:
            new_config['physics'] = {
                'fluid_properties': old_physics.get('fluid_properties', {}),
                'numerical': old_physics.get('numerical', {})
            }

    def _convert_solver_config(self, old_config: Dict, new_config: Dict):
        """Convert solver configuration."""
        old_solver = old_config.get('solver', {})

        if old_solver:
            new_config['solver'] = copy.deepcopy(old_solver)

    def _convert_output_config(self, old_config: Dict, new_config: Dict):
        """Convert output configuration."""
        old_output = old_config.get('output', {})

        if old_output:
            new_config['output'] = copy.deepcopy(old_output)

            # Extract global plot settings that should become per-tank
            plots = new_config['output'].get('plots', {})
            if 'show_reference_pressures' in plots:
                # This will be moved to per-tank plotting settings
                self.conversion_warnings.append(
                    "Global 'show_reference_pressures' setting moved to per-tank plotting configuration"
                )

    # Reverse conversion methods (new to old format)

    def _convert_analysis_metadata_reverse(self, new_config: Dict, old_config: Dict):
        """Convert analysis metadata back to old format."""
        analysis = new_config.get('analysis', {})

        old_config['analysis_name'] = analysis.get('name', 'Unnamed Analysis')
        old_config['description'] = analysis.get('description', 'No description')
        old_config['version'] = analysis.get('version', '1.0')

    def _convert_network_topology_reverse(self, new_config: Dict, old_config: Dict):
        """Convert network topology back to old flat format."""
        network = new_config.get('network', {})

        # Initialize old format sections
        old_config['network'] = {'tanks': []}
        old_config['geometry'] = {}
        old_config['tank_materials'] = {}
        old_config['coupling_rules'] = []

        # Convert nodes back to tanks
        nodes = network.get('nodes', [])
        for node in nodes:
            tank_id = node['node_id']

            # Network section
            tank_info = {
                'tank_id': tank_id,
                'tank_type': node.get('fluid', 'H2')
            }
            old_config['network']['tanks'].append(tank_info)

            # Geometry section
            geometry = {}
            if 'geometry' in node:
                geometry.update(node['geometry'])
            if 'initial_conditions' in node:
                init = node['initial_conditions']
                if 'pressure' in init:
                    geometry['initial_pressure'] = init['pressure']
                if 'temperature' in init:
                    geometry['initial_temperature'] = init['temperature']
                if 'density' in init:
                    geometry['initial_density'] = init['density']
            if 'operating_limits' in node:
                limits = node['operating_limits']
                if 'minimum_pressure' in limits:
                    geometry['minimum_pressure'] = limits['minimum_pressure']
                if 'venting_pressure' in limits:
                    geometry['venting_pressure'] = limits['venting_pressure']

            old_config['geometry'][tank_id] = geometry

            # Materials section
            if 'materials' in node:
                old_config['tank_materials'][tank_id] = copy.deepcopy(node['materials'])

        # Convert edges back to coupling rules
        edges = network.get('edges', [])
        for edge in edges:
            coupling = {
                'coupling_id': edge.get('edge_id', 'unnamed'),
                'coupling_type': edge.get('connection_type', 'simple_coupling'),
                'description': edge.get('description', ''),
                'participants': {
                    'source': edge.get('from_node'),
                    'target': edge.get('to_node')
                }
            }

            # Convert piping
            if 'piping' in edge:
                coupling['discharge_piping'] = copy.deepcopy(edge['piping'])

            # Convert flow physics
            if 'flow_physics' in edge:
                flow_physics = edge['flow_physics']
                coupling['flow_parameters'] = {}

                if 'orifice_flow' in flow_physics:
                    orifice = flow_physics['orifice_flow']
                    if 'orifice_diameter_m' in orifice:
                        coupling['flow_parameters']['orifice_diameter_m'] = orifice['orifice_diameter_m']

                if 'safety_limits' in flow_physics:
                    safety = flow_physics['safety_limits']
                    if 'max_flow_rate_kg_s' in safety:
                        coupling['flow_parameters']['max_flow_rate_kg_s'] = safety['max_flow_rate_kg_s']

            # Convert control parameters
            if 'control_parameters' in edge:
                coupling['control_parameters'] = copy.deepcopy(edge['control_parameters'])

            old_config['coupling_rules'].append(coupling)

        # Extract global stopping criteria (from first node that has it)
        for node in nodes:
            if 'stopping_criteria' in node:
                old_config['stopping_criteria'] = copy.deepcopy(node['stopping_criteria'])
                break

    def _convert_mission_config_reverse(self, new_config: Dict, old_config: Dict):
        """Convert mission config back to old format."""
        mission = new_config.get('mission', {})

        old_config['mission'] = {}
        if 'profile' in mission:
            old_config['mission']['profile'] = mission['profile']
        if 'ambient_temperature' in mission:
            old_config['mission']['ambient_temperature'] = mission['ambient_temperature']
        if 'assigned_to_node' in mission:
            old_config['mission']['assigned_to'] = mission['assigned_to_node']

        old_config['mission']['type'] = 'discharge'  # Default assumption

    def _convert_physics_config_reverse(self, new_config: Dict, old_config: Dict):
        """Convert physics config back to old format."""
        physics = new_config.get('physics', {})

        if physics:
            old_config['flow_physics'] = {}
            if 'fluid_properties' in physics:
                old_config['flow_physics']['fluid_properties'] = physics['fluid_properties']
            if 'numerical' in physics:
                old_config['flow_physics']['numerical'] = physics['numerical']

    def _convert_solver_config_reverse(self, new_config: Dict, old_config: Dict):
        """Convert solver config back to old format."""
        solver = new_config.get('solver', {})

        if solver:
            old_config['solver'] = copy.deepcopy(solver)

    def _convert_output_config_reverse(self, new_config: Dict, old_config: Dict):
        """Convert output config back to old format."""
        output = new_config.get('output', {})

        if output:
            old_config['output'] = copy.deepcopy(output)

    def get_conversion_warnings(self) -> List[str]:
        """Get any warnings generated during conversion."""
        return self.conversion_warnings

    def validate_conversion(self, original: Dict, converted: Dict, direction: str) -> bool:
        """
        Validate that conversion preserved essential information.

        Args:
            original: Original configuration
            converted: Converted configuration
            direction: 'old_to_new' or 'new_to_old'

        Returns:
            True if conversion appears successful

        Raises:
            ConfigurationError: If critical information was lost
        """
        try:
            if direction == 'old_to_new':
                # Check key information preservation
                self._validate_old_to_new_conversion(original, converted)
            else:
                # Check reverse conversion
                self._validate_new_to_old_conversion(original, converted)

            return True

        except Exception as e:
            raise ConfigurationError(f"Conversion validation failed: {e}")

    def _validate_old_to_new_conversion(self, old: Dict, new: Dict):
        """Validate old-to-new conversion preserved essential data."""
        # Check analysis name preserved
        old_name = old.get('analysis_name', '')
        new_name = new.get('analysis', {}).get('name', '')
        if old_name and old_name != new_name:
            raise ConfigurationError(f"Analysis name not preserved: {old_name} -> {new_name}")

        # Check tank count preserved
        old_geometry = old.get('geometry', {})
        new_nodes = new.get('network', {}).get('nodes', [])
        if len(old_geometry) != len(new_nodes):
            raise ConfigurationError(f"Tank count mismatch: {len(old_geometry)} -> {len(new_nodes)}")

        # Check coupling count preserved
        old_couplings = old.get('coupling_rules', [])
        new_edges = new.get('network', {}).get('edges', [])
        if len(old_couplings) != len(new_edges):
            raise ConfigurationError(f"Coupling count mismatch: {len(old_couplings)} -> {len(new_edges)}")

    def _validate_new_to_old_conversion(self, new: Dict, old: Dict):
        """Validate new-to-old conversion preserved essential data."""
        # Check analysis name preserved
        new_name = new.get('analysis', {}).get('name', '')
        old_name = old.get('analysis_name', '')
        if new_name and new_name != old_name:
            raise ConfigurationError(f"Analysis name not preserved: {new_name} -> {old_name}")

        # Check node/tank count preserved
        new_nodes = new.get('network', {}).get('nodes', [])
        old_geometry = old.get('geometry', {})
        if len(new_nodes) != len(old_geometry):
            raise ConfigurationError(f"Tank count mismatch: {len(new_nodes)} -> {len(old_geometry)}")


def load_config_with_adapter(config_path: Union[str, Path]) -> tuple[Dict[str, Any], str]:
    """
    Load configuration file with automatic format detection and conversion.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (config_dict, format) where format is 'old' or 'new'

    Raises:
        ConfigurationError: If loading or format detection fails
    """
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load YAML file {config_path}: {e}")

    adapter = ConfigurationAdapter()
    format_type = adapter.detect_format(raw_config)

    return raw_config, format_type


if __name__ == "__main__":
    # Example usage and testing
    import sys

    if len(sys.argv) != 3:
        print("Usage: python config_adapter.py <config_file.yaml> <output_format>")
        print("  output_format: 'new' or 'old'")
        sys.exit(1)

    config_path = sys.argv[1]
    target_format = sys.argv[2]

    try:
        # Load and detect format
        config, current_format = load_config_with_adapter(config_path)
        print(f"Detected format: {current_format}")

        # Convert if needed
        adapter = ConfigurationAdapter()

        if current_format == 'old' and target_format == 'new':
            converted = adapter.migrate_old_to_new(config)
            print("✅ Converted old format to new format")
        elif current_format == 'new' and target_format == 'old':
            converted = adapter.migrate_new_to_old(config)
            print("✅ Converted new format to old format")
        else:
            converted = config
            print(f"No conversion needed (already in {target_format} format)")

        # Show warnings
        warnings = adapter.get_conversion_warnings()
        if warnings:
            print("\nConversion warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")

        # Save converted config
        output_path = Path(config_path).with_suffix(f'.{target_format}_format.yaml')
        with open(output_path, 'w') as f:
            yaml.dump(converted, f, default_flow_style=False, indent=2)
        print(f"Saved converted config to: {output_path}")

    except ConfigurationError as e:
        print(f"❌ {e}")
        sys.exit(1)