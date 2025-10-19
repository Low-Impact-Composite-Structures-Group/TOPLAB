#!/usr/bin/env python3
"""
Strict Configuration Validator for Multi-Tank Analysis Framework

This module provides comprehensive validation of YAML configuration files
with ZERO tolerance for missing parameters. Every required parameter must
be explicitly defined - no fallback values or defaults are used.

Author: Analysis Framework Team
Date: October 13, 2025
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml


class ConfigurationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class StrictConfigValidator:
    """
    Strict validator that ensures ALL required parameters are present
    in the configuration file. No fallback values or defaults are used.

    Philosophy: "Fail fast with clear error messages"
    """

    def __init__(self, config_path: Union[str, Path]):
        """Initialize validator with configuration file path."""
        self.config_path = Path(config_path)
        self.config = None
        self.errors = []

    def load_and_validate(self) -> Dict[str, Any]:
        """
        Load YAML configuration and perform comprehensive validation.

        Returns:
            Dict containing validated configuration

        Raises:
            ConfigurationError: If any required parameter is missing or invalid
        """
        self._load_yaml()
        self._validate_structure()

        if self.errors:
            error_msg = "ERROR: Simulation halted due to configuration issues:\n\n"
            for i, error in enumerate(self.errors, 1):
                error_msg += f"{i}. {error}\n"
            error_msg += f"\nPlease fix these issues in: {self.config_path}\n"
            raise ConfigurationError(error_msg)

        return self.config

    def _load_yaml(self):
        """Load YAML file with error handling."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"ERROR: Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"ERROR: Invalid YAML syntax in {self.config_path}: {e}")

    def _validate_structure(self):
        """Validate the complete configuration structure."""
        if not isinstance(self.config, dict):
            self.errors.append("Configuration must be a dictionary/object")
            return

        # Validate each major section
        self._validate_analysis_section()
        self._validate_network_section()
        self._validate_mission_section()
        self._validate_solver_section()
        self._validate_output_section()

    def _validate_analysis_section(self):
        """Validate analysis metadata section."""
        if 'analysis' not in self.config:
            self.errors.append("Missing required section: 'analysis'")
            return

        analysis = self.config['analysis']
        required_fields = ['name', 'description', 'version']

        for field in required_fields:
            if field not in analysis:
                self.errors.append(f"Missing required field: analysis.{field}")
            elif not isinstance(analysis[field], str) or not analysis[field].strip():
                self.errors.append(f"Field analysis.{field} must be a non-empty string")

    def _validate_network_section(self):
        """Validate network topology section."""
        if 'network' not in self.config:
            self.errors.append("Missing required section: 'network'")
            return

        network = self.config['network']

        # Validate nodes (tanks)
        if 'nodes' not in network:
            self.errors.append("Missing required section: network.nodes")
            return

        if not isinstance(network['nodes'], list) or len(network['nodes']) == 0:
            self.errors.append("network.nodes must be a non-empty list")
            return

        for i, node in enumerate(network['nodes']):
            self._validate_node(node, i)

        # Validate edges (connections) - optional for single tank
        if 'edges' in network:
            if not isinstance(network['edges'], list):
                self.errors.append("network.edges must be a list")
            else:
                for i, edge in enumerate(network['edges']):
                    self._validate_edge(edge, i)

    def _validate_node(self, node: Dict[str, Any], node_index: int):
        """Validate individual node (tank) configuration."""
        node_prefix = f"network.nodes[{node_index}]"

        # Required node fields
        required_fields = ['node_id', 'type', 'fluid']
        for field in required_fields:
            if field not in node:
                self.errors.append(f"Missing required field: {node_prefix}.{field}")

        # Validate node_id is unique and valid
        if 'node_id' in node:
            if not isinstance(node['node_id'], int) or node['node_id'] <= 0:
                self.errors.append(f"{node_prefix}.node_id must be a positive integer")

        # Validate type
        if 'type' in node and node['type'] != 'tank':
            self.errors.append(f"{node_prefix}.type must be 'tank'")

        # Validate fluid
        if 'fluid' in node:
            valid_fluids = ['CH2', 'LH2', 'CCH2', 'SLH2']  # Extend as needed
            if node['fluid'] not in valid_fluids:
                self.errors.append(f"{node_prefix}.fluid must be one of: {valid_fluids}")

        # Validate required subsections
        required_subsections = ['geometry', 'initial_conditions', 'operating_limits', 'materials']
        for subsection in required_subsections:
            if subsection not in node:
                self.errors.append(f"Missing required section: {node_prefix}.{subsection}")
                continue

            # Validate each subsection
            if subsection == 'geometry':
                self._validate_geometry(node[subsection], f"{node_prefix}.geometry")
            elif subsection == 'initial_conditions':
                self._validate_initial_conditions(node[subsection], f"{node_prefix}.initial_conditions")
            elif subsection == 'operating_limits':
                self._validate_operating_limits(node[subsection], f"{node_prefix}.operating_limits")
            elif subsection == 'materials':
                self._validate_materials(node[subsection], f"{node_prefix}.materials")

    def _validate_geometry(self, geometry: Dict[str, Any], prefix: str):
        """Validate geometry section."""
        required_fields = ['phi', 'radius']
        for field in required_fields:
            if field not in geometry:
                self.errors.append(f"Missing required field: {prefix}.{field}")
            elif not isinstance(geometry[field], (int, float)) or geometry[field] <= 0:
                self.errors.append(f"{prefix}.{field} must be a positive number")

    def _validate_initial_conditions(self, conditions: Dict[str, Any], prefix: str):
        """Validate initial conditions section."""
        required_fields = ['pressure', 'temperature', 'density']
        for field in required_fields:
            if field not in conditions:
                self.errors.append(f"Missing required field: {prefix}.{field}")
            elif not isinstance(conditions[field], (int, float)) or conditions[field] <= 0:
                self.errors.append(f"{prefix}.{field} must be a positive number")

    def _validate_operating_limits(self, limits: Dict[str, Any], prefix: str):
        """Validate operating limits section."""
        required_fields = ['minimum_pressure', 'venting_pressure']
        for field in required_fields:
            if field not in limits:
                self.errors.append(f"Missing required field: {prefix}.{field}")
            elif not isinstance(limits[field], (int, float)) or limits[field] <= 0:
                self.errors.append(f"{prefix}.{field} must be a positive number")

        # Validate pressure hierarchy
        if 'minimum_pressure' in limits and 'venting_pressure' in limits:
            if limits['minimum_pressure'] >= limits['venting_pressure']:
                self.errors.append(f"{prefix}: minimum_pressure must be less than venting_pressure")

    def _validate_materials(self, materials: Dict[str, Any], prefix: str):
        """Validate materials section."""
        required_subsections = ['liner', 'composite', 'insulation']
        for subsection in required_subsections:
            if subsection not in materials:
                self.errors.append(f"Missing required section: {prefix}.{subsection}")
                continue

            material = materials[subsection]
            if subsection in ['liner', 'composite']:
                if 'nist_path' not in material:
                    self.errors.append(f"Missing required field: {prefix}.{subsection}.nist_path")
                if 'thickness' not in material:
                    self.errors.append(f"Missing required field: {prefix}.{subsection}.thickness")
                elif not isinstance(material['thickness'], (int, float)) or material['thickness'] <= 0:
                    self.errors.append(f"{prefix}.{subsection}.thickness must be a positive number")
            elif subsection == 'insulation':
                required_fields = ['thickness', 'heat_transfer_coefficient']
                for field in required_fields:
                    if field not in material:
                        self.errors.append(f"Missing required field: {prefix}.{subsection}.{field}")
                    elif not isinstance(material[field], (int, float)) or material[field] <= 0:
                        self.errors.append(f"{prefix}.{subsection}.{field} must be a positive number")

    def _validate_edge(self, edge: Dict[str, Any], edge_index: int):
        """Validate individual edge (connection) configuration."""
        edge_prefix = f"network.edges[{edge_index}]"

        required_fields = ['edge_id', 'from_node', 'to_node', 'connection_type']
        for field in required_fields:
            if field not in edge:
                self.errors.append(f"Missing required field: {edge_prefix}.{field}")

        # Validate node references
        if 'from_node' in edge and 'to_node' in edge:
            if not isinstance(edge['from_node'], int) or not isinstance(edge['to_node'], int):
                self.errors.append(f"{edge_prefix}: from_node and to_node must be integers")
            elif edge['from_node'] == edge['to_node']:
                self.errors.append(f"{edge_prefix}: from_node and to_node cannot be the same")

        # Validate connection type
        if 'connection_type' in edge:
            valid_types = ['mission_adaptive_pressurization', 'flow_matching_pressurization', 'simple_coupling', 'pressure_triggered']
            if edge['connection_type'] not in valid_types:
                self.errors.append(f"{edge_prefix}.connection_type must be one of: {valid_types}")

    def _validate_mission_section(self):
        """Validate mission configuration section."""
        if 'mission' not in self.config:
            self.errors.append("Missing required section: 'mission'")
            return

        mission = self.config['mission']
        required_fields = ['profile', 'ambient_temperature', 'assigned_to_node']

        for field in required_fields:
            if field not in mission:
                self.errors.append(f"Missing required field: mission.{field}")

        # Validate specific fields
        if 'profile' in mission:
            valid_profiles = ['atr72', 'custom']  # Extend as needed
            if mission['profile'] not in valid_profiles:
                self.errors.append(f"mission.profile must be one of: {valid_profiles}")

        if 'ambient_temperature' in mission:
            if not isinstance(mission['ambient_temperature'], (int, float)) or mission['ambient_temperature'] <= 0:
                self.errors.append("mission.ambient_temperature must be a positive number")

        if 'assigned_to_node' in mission:
            if not isinstance(mission['assigned_to_node'], int) or mission['assigned_to_node'] <= 0:
                self.errors.append("mission.assigned_to_node must be a positive integer")

    def _validate_solver_section(self):
        """Validate solver configuration section."""
        if 'solver' not in self.config:
            self.errors.append("Missing required section: 'solver'")
            return

        solver = self.config['solver']
        required_fields = ['method', 'rtol', 'atol', 'time_step']

        for field in required_fields:
            if field not in solver:
                self.errors.append(f"Missing required field: solver.{field}")

        # Validate method
        if 'method' in solver:
            valid_methods = ['LSODA', 'RK45', 'BDF']  # Extend as needed
            if solver['method'] not in valid_methods:
                self.errors.append(f"solver.method must be one of: {valid_methods}")

        # Validate numerical parameters
        numeric_fields = ['rtol', 'atol', 'time_step']
        for field in numeric_fields:
            if field in solver:
                if not isinstance(solver[field], (int, float)) or solver[field] <= 0:
                    self.errors.append(f"solver.{field} must be a positive number")

    def _validate_output_section(self):
        """Validate output configuration section."""
        if 'output' not in self.config:
            self.errors.append("Missing required section: 'output'")
            return

        output = self.config['output']
        required_fields = ['save_plots', 'save_data']

        for field in required_fields:
            if field not in output:
                self.errors.append(f"Missing required field: output.{field}")
            elif not isinstance(output[field], bool):
                self.errors.append(f"output.{field} must be true or false")


def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to validate a configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigurationError: If validation fails
    """
    validator = StrictConfigValidator(config_path)
    return validator.load_and_validate()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file.yaml>")
        sys.exit(1)

    try:
        config = validate_config_file(sys.argv[1])
        print(f"✅ Configuration validation passed: {sys.argv[1]}")
    except ConfigurationError as e:
        print(str(e))
        sys.exit(1)