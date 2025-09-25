"""
Graph Factory for Multi-Tank System Configuration

This module provides the GraphFactory class that serves as the interface layer
between network definition and simulation implementation. It handles:
- Generic network configuration from dictionaries
- Network validation and error reporting
- Configuration saving/loading
- Visualization integration
- Bridge to physics simulation systems

The factory maintains separation of concerns between network topology and
physics simulation while ensuring reusability and maintainability.

Data Flow: GraphFactory → TankSystemGraph → MultiTankSystem → Physics Simulation

Authors: Dante Raso (2025)
"""

# Standard library imports
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
import pickle

# Third-party imports
import matplotlib.pyplot as plt

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
from tank_network_visualizer import (
    TankType, ConnectionType, FlowModel,
    TankConfiguration, FlowConnection, TankSystemGraph,
    TankNetworkVisualizer
)

# =================== GRAPH FACTORY CLASS ===================

class GraphFactory:
    """
    Factory class for creating and managing tank network configurations.

    Provides a clean interface between network definition and simulation
    implementation, supporting generic configuration from dictionaries,
    validation, persistence, and visualization.
    """

    def __init__(self):
        """Initialize the GraphFactory"""
        self.last_created_graph: Optional[TankSystemGraph] = None
        self.config_cache: Dict[str, Dict[str, Any]] = {}

    # =================== CORE FACTORY METHODS ===================

    def from_config(self, config: Dict[str, Any]) -> TankSystemGraph:
        """
        Create a TankSystemGraph from a configuration dictionary.

        Args:
            config: Configuration dictionary with structure:
                {
                    "system_name": str,
                    "tanks": [
                        {
                            "tank_id": str,
                            "tank_type": str,  # "CCH2", "CH2", or "SLH2"
                            "volume": float,
                            "initial_conditions": {"pressure": float, "temperature": float, ...},
                            "scenario_params": {...},
                            "thermal_params": {...},
                            "position": [float, float]  # Optional [x, y]
                        }, ...
                    ],
                    "connections": [
                        {
                            "connection_id": str,
                            "source": str,
                            "target": str,
                            "connection_type": str,  # "feed", "discharge", etc.
                            "flow_model": str,       # "constant", "orifice", etc.
                            "parameters": {...},
                            "bidirectional": bool    # Optional, default False
                        }, ...
                    ]
                }

        Returns:
            TankSystemGraph: Validated graph structure

        Raises:
            ValueError: If configuration is invalid
            KeyError: If required keys are missing
        """
        try:
            # Extract system information
            system_name = config.get("system_name", "Unnamed System")

            # Parse tanks
            tanks = []
            tank_configs = config.get("tanks", [])

            for tank_config in tank_configs:
                # Parse tank type
                tank_type_str = tank_config["tank_type"].upper()
                tank_type = TankType[tank_type_str] if hasattr(TankType, tank_type_str) else TankType.CCH2

                # Parse position (optional)
                position = tuple(tank_config.get("position", [0, 0]))

                tank = TankConfiguration(
                    tank_id=tank_config["tank_id"],
                    tank_type=tank_type,
                    volume=float(tank_config["volume"]),
                    initial_conditions=tank_config.get("initial_conditions", {}),
                    scenario_params=tank_config.get("scenario_params", {}),
                    thermal_params=tank_config.get("thermal_params", {}),
                    position=position
                )
                tanks.append(tank)

            # Parse connections
            connections = []
            connection_configs = config.get("connections", [])

            for conn_config in connection_configs:
                # Parse connection type
                conn_type_str = conn_config["connection_type"].upper()
                conn_type = ConnectionType[conn_type_str] if hasattr(ConnectionType, conn_type_str) else ConnectionType.TRANSFER

                # Parse flow model
                flow_model_str = conn_config["flow_model"].upper()
                flow_model = FlowModel[flow_model_str] if hasattr(FlowModel, flow_model_str) else FlowModel.CONSTANT

                connection = FlowConnection(
                    connection_id=conn_config["connection_id"],
                    source=conn_config["source"],
                    target=conn_config["target"],
                    connection_type=conn_type,
                    flow_model=flow_model,
                    parameters=conn_config.get("parameters", {}),
                    bidirectional=conn_config.get("bidirectional", False)
                )
                connections.append(connection)

            # Create and validate graph
            graph = TankSystemGraph(
                system_name=system_name,
                tanks=tanks,
                connections=connections
            )

            # Validate the graph
            is_valid, errors = graph.validate_graph()
            if not is_valid:
                raise ValueError(f"Invalid graph configuration: {'; '.join(errors)}")

            # Cache the graph
            self.last_created_graph = graph

            return graph

        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create graph from configuration: {e}")

    # =================== PREDEFINED CONFIGURATIONS ===================

    def get_cch2_prototype_config(self) -> Dict[str, Any]:
        """Get configuration for original CCH2 prototype (both tanks have all connections)"""
        return {
            "system_name": "CCH2 Prototype (Original)",
            "tanks": [
                {
                    "tank_id": "Tank_1",
                    "tank_type": "CCH2",
                    "volume": 0.5,
                    "initial_conditions": {"pressure": 400e5, "temperature": 53.25},
                    "scenario_params": {"stopping_density": 70.0, "scenario": "dormancy"},
                    "thermal_params": {"htc": 0.025},
                    "position": [-1.5, 0]
                },
                {
                    "tank_id": "Tank_2",
                    "tank_type": "CCH2",
                    "volume": 0.5,
                    "initial_conditions": {"pressure": 400e5, "temperature": 53.25},
                    "scenario_params": {"stopping_density": 5.8, "scenario": "discharge"},
                    "thermal_params": {"htc": 0.025},
                    "position": [1.5, 0]
                }
            ],
            "connections": [
                {
                    "connection_id": "T1_feed",
                    "source": "EXTERNAL_SOURCE",
                    "target": "Tank_1",
                    "connection_type": "feed",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                },
                {
                    "connection_id": "T1_discharge",
                    "source": "Tank_1",
                    "target": "EXTERNAL_SINK",
                    "connection_type": "discharge",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                },
                {
                    "connection_id": "T1_vent",
                    "source": "Tank_1",
                    "target": "ENVIRONMENT",
                    "connection_type": "vent",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                },
                {
                    "connection_id": "T2_feed",
                    "source": "EXTERNAL_SOURCE",
                    "target": "Tank_2",
                    "connection_type": "feed",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                },
                {
                    "connection_id": "T2_discharge",
                    "source": "Tank_2",
                    "target": "EXTERNAL_SINK",
                    "connection_type": "discharge",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.001}
                },
                {
                    "connection_id": "T2_vent",
                    "source": "Tank_2",
                    "target": "ENVIRONMENT",
                    "connection_type": "vent",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                }
            ]
        }

    def get_user_specified_prototype_config(self) -> Dict[str, Any]:
        """Get configuration for user specified prototype (Tank1=vent only, Tank2=discharge+vent)"""
        return {
            "system_name": "User Specified Prototype",
            "tanks": [
                {
                    "tank_id": "Tank_1",
                    "tank_type": "CCH2",
                    "volume": 0.5,
                    "initial_conditions": {"pressure": 400e5, "temperature": 53.25},
                    "scenario_params": {"stopping_density": 70.0, "scenario": "dormancy_only_vent"},
                    "thermal_params": {"htc": 0.025},
                    "position": [-1.5, 0]
                },
                {
                    "tank_id": "Tank_2",
                    "tank_type": "CCH2",
                    "volume": 0.5,
                    "initial_conditions": {"pressure": 400e5, "temperature": 53.25},
                    "scenario_params": {"stopping_density": 5.8, "scenario": "discharge_with_vent"},
                    "thermal_params": {"htc": 0.025},
                    "position": [1.5, 0]
                }
            ],
            "connections": [
                {
                    "connection_id": "T1_vent",
                    "source": "Tank_1",
                    "target": "ENVIRONMENT",
                    "connection_type": "vent",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                },
                {
                    "connection_id": "T2_discharge",
                    "source": "Tank_2",
                    "target": "EXTERNAL_SINK",
                    "connection_type": "discharge",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.001}
                },
                {
                    "connection_id": "T2_vent",
                    "source": "Tank_2",
                    "target": "ENVIRONMENT",
                    "connection_type": "vent",
                    "flow_model": "constant",
                    "parameters": {"rate": 0.0}
                }
            ]
        }

    # =================== CONVENIENCE METHODS ===================

    def create_cch2_prototype(self) -> TankSystemGraph:
        """Create original CCH2 prototype graph"""
        config = self.get_cch2_prototype_config()
        return self.from_config(config)

    def create_user_specified_prototype(self) -> TankSystemGraph:
        """Create user specified prototype graph"""
        config = self.get_user_specified_prototype_config()
        return self.from_config(config)

    # =================== PERSISTENCE METHODS ===================

    def save_config(self, config: Dict[str, Any], filepath: str) -> None:
        """
        Save configuration dictionary to file.

        Args:
            config: Configuration dictionary
            filepath: Path to save file (.json)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to: {filepath}")

    def load_config(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration dictionary from file.

        Args:
            filepath: Path to configuration file (.json)

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config = json.load(f)

        # Cache the loaded config
        self.config_cache[filepath.stem] = config

        print(f"Configuration loaded from: {filepath}")
        return config

    def save_graph(self, graph: TankSystemGraph, filepath: str) -> None:
        """
        Save TankSystemGraph to pickle file.

        Args:
            graph: TankSystemGraph to save
            filepath: Path to save file (.pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)

        print(f"Graph saved to: {filepath}")

    def load_graph(self, filepath: str) -> TankSystemGraph:
        """
        Load TankSystemGraph from pickle file.

        Args:
            filepath: Path to graph file (.pkl)

        Returns:
            TankSystemGraph: Loaded graph
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")

        with open(filepath, 'rb') as f:
            graph = pickle.load(f)

        self.last_created_graph = graph
        print(f"Graph loaded from: {filepath}")
        return graph

    # =================== VISUALIZATION METHODS ===================

    def visualize(self, graph: Optional[TankSystemGraph] = None,
                  figsize: Tuple[int, int] = (16, 10),
                  layout_method: str = 'spring',
                  show_labels: bool = True,
                  show_connection_details: bool = True,
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a tank network graph.

        Args:
            graph: Graph to visualize (uses last created if None)
            figsize: Figure size tuple
            layout_method: Layout algorithm ('spring', 'circular', 'hierarchical')
            show_labels: Whether to show node labels
            show_connection_details: Whether to show connection details
            save_path: Optional path to save visualization

        Returns:
            plt.Figure: The created figure

        Raises:
            ValueError: If no graph available to visualize
        """
        if graph is None:
            graph = self.last_created_graph

        if graph is None:
            raise ValueError("No graph available to visualize. Create or load a graph first.")

        # Use the existing visualizer
        visualizer = TankNetworkVisualizer(graph)
        fig = visualizer.visualize_network(
            figsize=figsize,
            layout_method=layout_method,
            show_labels=show_labels,
            show_connection_details=show_connection_details,
            save_path=save_path
        )

        return fig

    # =================== UTILITY METHODS ===================

    def get_graph_summary(self, graph: Optional[TankSystemGraph] = None) -> Dict[str, Any]:
        """
        Get summary information about a graph.

        Args:
            graph: Graph to summarize (uses last created if None)

        Returns:
            Dict with summary information
        """
        if graph is None:
            graph = self.last_created_graph

        if graph is None:
            raise ValueError("No graph available to summarize.")

        tank_types = [tank.tank_type.value for tank in graph.tanks]
        connection_types = [conn.connection_type.value for conn in graph.connections]

        return {
            "system_name": graph.system_name,
            "num_tanks": len(graph.tanks),
            "num_connections": len(graph.connections),
            "tank_types": list(set(tank_types)),
            "connection_types": list(set(connection_types)),
            "tank_ids": [tank.tank_id for tank in graph.tanks],
            "external_connections": sum(1 for conn in graph.connections
                                      if "EXTERNAL" in conn.source or "EXTERNAL" in conn.target
                                      or "ENVIRONMENT" in conn.source or "ENVIRONMENT" in conn.target)
        }

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary without creating the graph.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Check required top-level keys
            required_keys = ["tanks", "connections"]
            for key in required_keys:
                if key not in config:
                    errors.append(f"Missing required key: {key}")

            # Validate tanks
            if "tanks" in config:
                tank_ids = set()
                for i, tank in enumerate(config["tanks"]):
                    tank_required = ["tank_id", "tank_type", "volume"]
                    for key in tank_required:
                        if key not in tank:
                            errors.append(f"Tank {i}: Missing required key '{key}'")

                    if "tank_id" in tank:
                        if tank["tank_id"] in tank_ids:
                            errors.append(f"Duplicate tank_id: {tank['tank_id']}")
                        tank_ids.add(tank["tank_id"])

                    if "tank_type" in tank:
                        if tank["tank_type"].upper() not in [t.name for t in TankType]:
                            errors.append(f"Invalid tank_type: {tank['tank_type']}")

            # Validate connections
            if "connections" in config:
                connection_ids = set()
                for i, conn in enumerate(config["connections"]):
                    conn_required = ["connection_id", "source", "target", "connection_type", "flow_model"]
                    for key in conn_required:
                        if key not in conn:
                            errors.append(f"Connection {i}: Missing required key '{key}'")

                    if "connection_id" in conn:
                        if conn["connection_id"] in connection_ids:
                            errors.append(f"Duplicate connection_id: {conn['connection_id']}")
                        connection_ids.add(conn["connection_id"])

                    if "connection_type" in conn:
                        if conn["connection_type"].upper() not in [c.name for c in ConnectionType]:
                            errors.append(f"Invalid connection_type: {conn['connection_type']}")

                    if "flow_model" in conn:
                        if conn["flow_model"].upper() not in [f.name for f in FlowModel]:
                            errors.append(f"Invalid flow_model: {conn['flow_model']}")

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        return len(errors) == 0, errors

# =================== TESTING FUNCTION ===================

def main():
    """Test the GraphFactory functionality"""
    print("GRAPH FACTORY TESTING")
    print("="*50)

    # Initialize factory
    factory = GraphFactory()

    # Test 1: Create graphs from predefined configs
    print("\n1. Testing predefined configurations...")

    # Original prototype
    original_graph = factory.create_cch2_prototype()
    print(f"✅ Created original prototype: {factory.get_graph_summary()}")

    # User specified prototype
    user_graph = factory.create_user_specified_prototype()
    print(f"✅ Created user specified prototype: {factory.get_graph_summary()}")

    # Test 2: Save and load configurations
    print("\n2. Testing configuration persistence...")

    config = factory.get_user_specified_prototype_config()
    factory.save_config(config, "test_config.json")

    loaded_config = factory.load_config("test_config.json")
    loaded_graph = factory.from_config(loaded_config)
    print(f"✅ Loaded configuration: {factory.get_graph_summary()}")

    # Test 3: Visualization
    print("\n3. Testing visualization...")
    factory.visualize(original_graph, figsize=(16, 10))
    plt.show()

    # Test 4: Configuration validation
    print("\n4. Testing configuration validation...")
    is_valid, errors = factory.validate_config(config)
    print(f"Configuration valid: {'✅ YES' if is_valid else '❌ NO'}")
    if errors:
        for error in errors:
            print(f"   Error: {error}")

    print("\n" + "="*50)
    print("GRAPH FACTORY TESTING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main()