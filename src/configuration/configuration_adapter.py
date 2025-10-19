"""
Configuration Format Adapter

This module provides conversion between old (flat) and new (network-based)
YAML configuration formats. It enables gradual migration without breaking
existing analyses.

Key Features:
- Automatic format detection
- Bidirectional conversion (old ↔ new)
- Preserves all configuration information
- Handles complex nested structures
- Maintains backward compatibility

Author: Configuration Migration Framework
Date: October 16, 2025
"""

from typing import Dict, Any, List, Optional
from copy import deepcopy


class ConfigurationAdapter:
    """
    Adapter that converts between old flat format and new network format.
    Allows gradual migration without breaking existing analyses.
    """

    def __init__(self):
        self.conversion_warnings = []

    def migrate_old_to_new(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert old flat format to new network format.

        Args:
            old_config: Configuration in old flat format

        Returns:
            Configuration in new network format
        """
        self.conversion_warnings = []
        new_config = {}

        # Convert analysis metadata
        new_config['analysis'] = self._convert_analysis_metadata(old_config)

        # Convert network topology
        new_config['network'] = self._convert_network_topology(old_config)

        # Convert mission configuration
        new_config['mission'] = self._convert_mission_config(old_config)

        # Convert physics configuration (if present)
        if 'physics' in old_config or 'flow_physics' in old_config:
            new_config['physics'] = self._convert_physics_config(old_config)

        # Convert solver configuration - ensure all parameters are present
        if 'solver' in old_config:
            new_config['solver'] = self._convert_solver_config(old_config['solver'])
        else:
            # Provide default solver configuration if missing
            new_config['solver'] = {
                'method': 'LSODA',
                'rtol': 1e-6,
                'atol': 1e-9,
                'time_step': 0.1,
                'max_step': 10.0
            }
            self.conversion_warnings.append("No solver configuration found, using defaults")

        # Convert output configuration - ensure all parameters are present
        if 'output' in old_config:
            new_config['output'] = self._convert_output_config(old_config['output'])
        else:
            # Provide default output configuration if missing
            new_config['output'] = {
                'save_plots': True,
                'save_data': True,
                'identifier': 'migrated_analysis'
            }
            self.conversion_warnings.append("No output configuration found, using defaults")

        return new_config

    def migrate_new_to_old(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert new network format to old flat format for legacy code.

        Args:
            new_config: Configuration in new network format

        Returns:
            Configuration in old flat format
        """
        self.conversion_warnings = []
        old_config = {}

        # Convert analysis metadata
        if 'analysis' in new_config:
            analysis = new_config['analysis']
            old_config['analysis_name'] = analysis.get('name', 'Unknown Analysis')
            old_config['description'] = analysis.get('description', '')
            old_config['version'] = analysis.get('version', '1.0')

        # Convert network topology
        if 'network' in new_config:
            self._convert_network_to_old_format(new_config['network'], old_config)

        # Convert mission configuration
        if 'mission' in new_config:
            old_config['mission'] = self._convert_mission_to_old_format(new_config['mission'])

        # Convert physics configuration
        if 'physics' in new_config:
            old_config['flow_physics'] = self._convert_physics_to_old_format(new_config['physics'])

        # Copy solver and output (should be compatible)
        for section in ['solver', 'output']:
            if section in new_config:
                old_config[section] = deepcopy(new_config[section])

        return old_config

    def _convert_analysis_metadata(self, old_config: Dict) -> Dict[str, Any]:
        """Convert analysis metadata from old to new format."""
        analysis = {}

        # Required fields with fallbacks for migration
        analysis['name'] = old_config.get('analysis_name', 'Migrated Analysis')
        analysis['description'] = old_config.get('description', 'Migrated from old format')
        analysis['version'] = old_config.get('version', '1.0')

        # Optional fields
        if 'author' in old_config:
            analysis['author'] = old_config['author']

        return analysis

    def _convert_network_topology(self, old_config: Dict) -> Dict[str, Any]:
        """Convert network topology from old to new format."""
        network = {'nodes': [], 'edges': []}

        # Get network info
        old_network = old_config.get('network', {})
        num_tanks = old_network.get('number_of_tanks', 1)

        # Convert tanks to nodes
        geometry = old_config.get('geometry', {})
        tank_materials = old_config.get('tank_materials', {})

        for tank_id in range(1, num_tanks + 1):
            node = self._convert_tank_to_node(tank_id, geometry, tank_materials, old_config)
            network['nodes'].append(node)

        # Convert coupling rules to edges
        coupling_rules = old_config.get('coupling_rules', [])
        for coupling_rule in coupling_rules:
            edge = self._convert_coupling_to_edge(coupling_rule, old_config)
            if edge:
                network['edges'].append(edge)

        return network

    def _convert_tank_to_node(self, tank_id: int, geometry: Dict, tank_materials: Dict,
                             old_config: Dict) -> Dict[str, Any]:
        """Convert a tank configuration to a network node."""
        node = {
            'node_id': tank_id,
            'type': 'tank',
            'description': f'Tank {tank_id}'
        }

        # Get tank geometry
        tank_geom = geometry.get(tank_id, {})

        # Determine fluid type from tank type or network config
        old_network = old_config.get('network', {})
        tank_info = None
        if 'tanks' in old_network:
            tank_list = old_network['tanks']
            tank_info = next((t for t in tank_list if t.get('tank_id') == tank_id), None)

        if tank_info and 'tank_type' in tank_info:
            node['fluid'] = tank_info['tank_type']
        else:
            # Try to infer from other information
            node['fluid'] = 'CH2'  # Default fallback
            self.conversion_warnings.append(f"Could not determine fluid type for tank {tank_id}, defaulting to CH2")

        # Convert geometry - extract all geometry parameters
        node['geometry'] = {}
        if 'phi' in tank_geom:
            value = tank_geom['phi']
            # Handle string expressions by evaluating them
            if isinstance(value, str):
                try:
                    value = eval(value)
                except:
                    self.conversion_warnings.append(f"Could not evaluate phi value '{value}' for tank {tank_id}")
            node['geometry']['phi'] = value
        if 'radius' in tank_geom:
            value = tank_geom['radius']
            # Handle string expressions by evaluating them
            if isinstance(value, str):
                try:
                    value = eval(value)
                except:
                    self.conversion_warnings.append(f"Could not evaluate radius value '{value}' for tank {tank_id}")
            node['geometry']['radius'] = value
        if tank_geom.get('mission_based_sizing'):
            node['geometry']['mission_based_sizing'] = True
        # Handle mission-based sizing if no explicit radius
        elif 'radius' not in tank_geom:
            node['geometry']['mission_based_sizing'] = True

        # Convert initial conditions - extract all initial state parameters
        node['initial_conditions'] = {}
        for param in ['initial_pressure', 'initial_temperature', 'initial_density']:
            if param in tank_geom:
                # Remove 'initial_' prefix for new format
                new_param = param.replace('initial_', '')
                value = tank_geom[param]
                # Handle string expressions (like '400e5') by evaluating them
                if isinstance(value, str):
                    try:
                        value = eval(value)
                    except:
                        # If evaluation fails, keep as string and warn
                        self.conversion_warnings.append(f"Could not evaluate {param} value '{value}' for tank {tank_id}")
                node['initial_conditions'][new_param] = value

        # Ensure temperature is present - if not provided, add a reasonable default
        if 'temperature' not in node['initial_conditions']:
            # For cryogenic fluids, use reasonable defaults based on fluid type
            fluid_type = node.get('fluid', 'CH2').upper()
            if 'CCH2' in fluid_type or 'CH2' in fluid_type:
                # Cryocompressed H2 typically around 40-50K
                node['initial_conditions']['temperature'] = 45.0
            elif 'LH2' in fluid_type:
                # Liquid H2 around 20-30K
                node['initial_conditions']['temperature'] = 25.0
            else:
                # Room temperature default for gaseous storage
                node['initial_conditions']['temperature'] = 288.15
            self.conversion_warnings.append(f"No initial temperature found for tank {tank_id}, using default {node['initial_conditions']['temperature']} K")

        # Convert operating limits - extract pressure limits
        node['operating_limits'] = {}
        for param in ['minimum_pressure', 'venting_pressure']:
            if param in tank_geom:
                value = tank_geom[param]
                # Handle string expressions (like '450e5') by evaluating them
                if isinstance(value, str):
                    try:
                        value = eval(value)
                    except:
                        # If evaluation fails, keep as string and warn
                        self.conversion_warnings.append(f"Could not evaluate {param} value '{value}' for tank {tank_id}")
                node['operating_limits'][param] = value

        # Convert materials with detailed specifications
        tank_mat = tank_materials.get(tank_id, {})
        if tank_mat:
            node['materials'] = self._serialize_materials(tank_mat)

            # Ensure liner has thickness if not already present
            if 'liner' in node['materials'] and 'thickness' not in node['materials']['liner']:
                liner_data = tank_mat.get('liner', {})
                if isinstance(liner_data, dict) and 'thickness' in liner_data:
                    node['materials']['liner']['thickness'] = liner_data['thickness']
                elif hasattr(liner_data, 'thickness'):
                    node['materials']['liner']['thickness'] = getattr(liner_data, 'thickness')
                else:
                    node['materials']['liner']['thickness'] = 0.001  # Default 1mm

            # Ensure insulation has required properties
            if 'insulation' in tank_mat:
                if 'insulation' not in node['materials']:
                    node['materials']['insulation'] = {}
                insulation_params = tank_mat['insulation']
                # Handle both dict and object types for insulation parameters
                if isinstance(insulation_params, dict):
                    for param in ['thickness', 'heat_transfer_coefficient']:
                        if param in insulation_params:
                            node['materials']['insulation'][param] = insulation_params[param]
                else:
                    # Handle object type (extract attributes)
                    for param in ['thickness', 'heat_transfer_coefficient']:
                        if hasattr(insulation_params, param):
                            node['materials']['insulation'][param] = getattr(insulation_params, param)

                # Ensure required insulation parameters are present
                if 'thickness' not in node['materials']['insulation']:
                    node['materials']['insulation']['thickness'] = 0.05  # Default 50mm
                    self.conversion_warnings.append(f"No insulation thickness found for tank {tank_id}, using default 0.05m")
                if 'heat_transfer_coefficient' not in node['materials']['insulation']:
                    node['materials']['insulation']['heat_transfer_coefficient'] = 0.025  # Default vacuum insulation
                    self.conversion_warnings.append(f"No insulation heat transfer coefficient found for tank {tank_id}, using default 0.025 W/m²K")

        # If no materials at all, provide minimal insulation spec
        if 'materials' not in node or not node['materials']:
            node['materials'] = {
                'insulation': {
                    'thickness': 0.05,
                    'heat_transfer_coefficient': 0.025
                }
            }
            self.conversion_warnings.append(f"No materials found for tank {tank_id}, providing default insulation")

        # Convert stopping criteria (if global, apply to all tanks)
        stopping_criteria = old_config.get('stopping_criteria', {})
        if stopping_criteria:
            node['stopping_criteria'] = deepcopy(stopping_criteria)

        # Convert plotting options (if any tank-specific settings exist)
        output = old_config.get('output', {})
        plots = output.get('plots', {})
        if plots:
            node['plotting'] = self._extract_tank_plotting_options(plots, tank_id)

        return node

    def _serialize_materials(self, materials: Dict) -> Dict:
        """
        Serialize materials to YAML-safe dictionary format.

        Converts NISTMaterial objects to dictionaries by extracting their
        key properties without Python object references.
        """
        serialized = {}

        for mat_type, material in materials.items():
            if hasattr(material, '__dict__'):
                # Convert material object to dictionary
                mat_dict = {}
                for attr, value in material.__dict__.items():
                    # Skip callable attributes and private attributes
                    if not callable(value) and not attr.startswith('_'):
                        # Handle specific function references
                        if hasattr(value, '__name__'):
                            # Store function name as string reference
                            mat_dict[attr] = f"function:{value.__name__}"
                        else:
                            mat_dict[attr] = value
                serialized[mat_type] = mat_dict
            else:
                # Already a dictionary, copy as-is
                serialized[mat_type] = deepcopy(material)

        return serialized

    def _convert_coupling_to_edge(self, coupling_rule: Dict, old_config: Dict) -> Optional[Dict[str, Any]]:
        """Convert a coupling rule to a network edge."""
        if not coupling_rule:
            return None

        edge = {}

        # Required edge information
        edge['edge_id'] = coupling_rule.get('coupling_id', 'unknown_coupling')
        edge['connection_type'] = coupling_rule.get('coupling_type', 'unknown')

        # Get participants
        participants = coupling_rule.get('participants', {})
        edge['from_node'] = participants.get('source')
        edge['to_node'] = participants.get('target')

        if not edge['from_node'] or not edge['to_node']:
            self.conversion_warnings.append(f"Coupling rule {edge['edge_id']} missing source or target")
            return None

        # Description
        if 'description' in coupling_rule:
            edge['description'] = coupling_rule['description']

        # Convert control parameters
        if 'control_parameters' in coupling_rule:
            edge['control_parameters'] = deepcopy(coupling_rule['control_parameters'])

        # Convert flow parameters to flow_physics
        if 'flow_parameters' in coupling_rule:
            edge['flow_physics'] = {}
            flow_params = coupling_rule['flow_parameters']

            # Convert orifice parameters
            if 'orifice_diameter_m' in flow_params or 'max_flow_rate_kg_s' in flow_params:
                edge['flow_physics']['orifice_flow'] = {}
                if 'orifice_diameter_m' in flow_params:
                    edge['flow_physics']['orifice_flow']['orifice_diameter_m'] = flow_params['orifice_diameter_m']

                # Convert flow limits to safety limits
                if 'max_flow_rate_kg_s' in flow_params:
                    if 'safety_limits' not in edge['flow_physics']:
                        edge['flow_physics']['safety_limits'] = {}
                    edge['flow_physics']['safety_limits']['max_flow_rate_kg_s'] = flow_params['max_flow_rate_kg_s']

        # Convert discharge piping
        if 'discharge_piping' in coupling_rule:
            edge['piping'] = deepcopy(coupling_rule['discharge_piping'])

        return edge

    def _convert_mission_config(self, old_config: Dict) -> Dict[str, Any]:
        """Convert mission configuration from old to new format."""
        old_mission = old_config.get('mission', {})
        if not old_mission:
            return {}

        new_mission = deepcopy(old_mission)

        # Convert assigned_to to assigned_to_node for new format
        if 'assigned_to' in new_mission:
            new_mission['assigned_to_node'] = new_mission.pop('assigned_to')

        # For single tank systems, assign mission to node 1 if not specified
        if 'assigned_to_node' not in new_mission and 'assigned_to' not in old_mission:
            # Check if it's a single tank system
            network = old_config.get('network', {})
            if network.get('number_of_tanks', 1) == 1:
                new_mission['assigned_to_node'] = 1

        return new_mission

    def _convert_physics_config(self, old_config: Dict) -> Dict[str, Any]:
        """Convert physics configuration from old to new format."""
        # Check for flow_physics section
        if 'flow_physics' in old_config:
            return deepcopy(old_config['flow_physics'])
        elif 'physics' in old_config:
            return deepcopy(old_config['physics'])
        else:
            return {}

    def _extract_tank_plotting_options(self, plots: Dict, tank_id: int) -> Dict[str, Any]:
        """Extract tank-specific plotting options."""
        # For now, just return empty dict as tank-specific plotting is not common in old format
        # Could be extended to handle tank-specific settings if they exist
        return {}

    def _convert_network_to_old_format(self, network: Dict, old_config: Dict) -> None:
        """Convert network nodes and edges back to old flat format."""
        nodes = network.get('nodes', [])
        edges = network.get('edges', [])

        # Create old format sections
        old_config['network'] = {
            'number_of_tanks': len([n for n in nodes if n.get('type') == 'tank']),
            'external_source': False,
            'external_sink': True,
            'tanks': []
        }

        old_config['geometry'] = {}
        old_config['tank_materials'] = {}

        # Convert nodes back to old format
        for node in nodes:
            if node.get('type') == 'tank':
                tank_id = node.get('node_id')

                # Add to tanks list
                old_config['network']['tanks'].append({
                    'tank_id': tank_id,
                    'tank_type': node.get('fluid', 'CH2')
                })

                # Convert geometry
                tank_geom = {}
                if 'geometry' in node:
                    tank_geom.update(node['geometry'])

                # Convert initial conditions
                if 'initial_conditions' in node:
                    for param, value in node['initial_conditions'].items():
                        tank_geom[f'initial_{param}'] = value

                # Convert operating limits
                if 'operating_limits' in node:
                    tank_geom.update(node['operating_limits'])

                old_config['geometry'][tank_id] = tank_geom

                # Convert materials
                if 'materials' in node:
                    old_config['tank_materials'][tank_id] = deepcopy(node['materials'])

        # Convert edges to coupling rules
        if edges:
            old_config['coupling_rules'] = []
            for edge in edges:
                coupling_rule = {
                    'coupling_id': edge.get('edge_id'),
                    'coupling_type': edge.get('connection_type'),
                    'participants': {
                        'source': edge.get('from_node'),
                        'target': edge.get('to_node')
                    }
                }

                if 'description' in edge:
                    coupling_rule['description'] = edge['description']

                if 'control_parameters' in edge:
                    coupling_rule['control_parameters'] = deepcopy(edge['control_parameters'])

                if 'activation_conditions' in edge:
                    # Convert activation_conditions to hysteresis format expected by SystemOrchestrator
                    activation = edge['activation_conditions']
                    coupling_rule['hysteresis'] = {}
                    if 'pressure_open_bar' in activation:
                        coupling_rule['hysteresis']['activation_threshold_bar'] = activation['pressure_open_bar']
                    if 'pressure_close_bar' in activation:
                        coupling_rule['hysteresis']['deactivation_threshold_bar'] = activation['pressure_close_bar']

                if 'piping' in edge:
                    coupling_rule['discharge_piping'] = deepcopy(edge['piping'])

                # Convert flow_physics back to flow_parameters
                if 'flow_physics' in edge:
                    flow_physics = edge['flow_physics']
                    coupling_rule['flow_parameters'] = {}

                    if 'orifice_flow' in flow_physics:
                        orifice = flow_physics['orifice_flow']
                        if 'orifice_diameter_m' in orifice:
                            coupling_rule['flow_parameters']['orifice_diameter_m'] = orifice['orifice_diameter_m']

                    if 'safety_limits' in flow_physics:
                        safety = flow_physics['safety_limits']
                        if 'max_flow_rate_kg_s' in safety:
                            coupling_rule['flow_parameters']['max_flow_rate_kg_s'] = safety['max_flow_rate_kg_s']

                old_config['coupling_rules'].append(coupling_rule)

    def _convert_mission_to_old_format(self, new_mission: Dict) -> Dict[str, Any]:
        """Convert mission configuration from new to old format."""
        old_mission = deepcopy(new_mission)

        # Convert assigned_to_node back to assigned_to
        if 'assigned_to_node' in old_mission:
            old_mission['assigned_to'] = old_mission.pop('assigned_to_node')

        return old_mission

    def _convert_physics_to_old_format(self, physics: Dict) -> Dict[str, Any]:
        """Convert physics configuration from new to old format."""
        return deepcopy(physics)

    def _convert_solver_config(self, solver_config: Dict) -> Dict[str, Any]:
        """Convert solver configuration, ensuring all required parameters are present."""
        converted = deepcopy(solver_config)

        # Ensure required solver parameters are present
        defaults = {
            'method': 'LSODA',
            'rtol': 1e-6,
            'atol': 1e-9
        }

        for key, default_value in defaults.items():
            if key not in converted:
                converted[key] = default_value
                self.conversion_warnings.append(f"Missing solver parameter '{key}', using default: {default_value}")

        return converted

    def _convert_output_config(self, output_config: Dict) -> Dict[str, Any]:
        """Convert output configuration, ensuring all required parameters are present."""
        converted = deepcopy(output_config)

        # Ensure required output parameters are present
        defaults = {
            'save_plots': True,
            'save_data': True
        }

        for key, default_value in defaults.items():
            if key not in converted:
                converted[key] = default_value
                self.conversion_warnings.append(f"Missing output parameter '{key}', using default: {default_value}")

        return converted

    def detect_format(self, config: Dict[str, Any]) -> str:
        """
        Detect configuration format.

        Args:
            config: Configuration dictionary

        Returns:
            "new" if network-based format, "old" if flat format
        """
        if (isinstance(config.get('network'), dict) and
            'nodes' in config['network']):
            return "new"
        else:
            return "old"

    def get_conversion_warnings(self) -> List[str]:
        """Get warnings generated during conversion."""
        return self.conversion_warnings.copy()

    def validate_conversion(self, original: Dict, converted: Dict) -> List[str]:
        """
        Validate that conversion preserved essential information.

        Args:
            original: Original configuration
            converted: Converted configuration

        Returns:
            List of validation issues found
        """
        issues = []

        # This is a placeholder for comprehensive conversion validation
        # Would check that tank counts match, key parameters preserved, etc.

        return issues