"""
Tank Network Visualization Tool

This script provides a visualization framework for multi-tank hydrogen storage systems
using NetworkX and matplotlib. It allows users to define tank networks with different
tank types and flow connections, then visualize the resulting graph topology.

This tool helps design and validate multi-tank system configurations before
implementing them in the full simulation framework.

Features:
- Support for different tank types (CCH2, CH2, SLH2)
- Various connection types (feed, discharge, transfer, vent)
- Interactive network visualization
- Configuration validation
- Export capabilities

Authors: Dante Raso (2025)
"""

# Standard library imports
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =================== TANK NETWORK DEFINITIONS ===================

class TankType(Enum):
    """Supported tank types for hydrogen storage"""
    CCH2 = "cryocompressed_h2"    # Cryocompressed H2 (like our current system)
    CH2 = "compressed_h2"         # Room temperature compressed H2
    SLH2 = "subcooled_liquid_h2"  # Subcooled liquid H2

class ConnectionType(Enum):
    """Types of connections between tanks and external systems"""
    FEED = "feed"           # External source → Tank
    DISCHARGE = "discharge" # Tank → External sink
    TRANSFER = "transfer"   # Tank → Tank
    VENT = "vent"          # Tank → Environment
    RETURN = "return"      # Tank → Tank (return line)

class FlowModel(Enum):
    """Flow models for connections"""
    CONSTANT = "constant"   # Constant mass flow rate
    ORIFICE = "orifice"    # Pressure-driven orifice flow
    PUMP = "pump"          # Active pumping
    VALVE = "valve"        # Controlled valve
    GRAVITY = "gravity"    # Gravity-driven flow

@dataclass
class TankConfiguration:
    """Configuration for individual tank"""
    tank_id: str
    tank_type: TankType
    volume: float                           # m³
    initial_conditions: Dict[str, float]    # {"pressure": Pa, "temperature": K, etc.}
    scenario_params: Dict[str, Any]         # Scenario-specific parameters
    thermal_params: Dict[str, float]        # Thermal model parameters
    position: Tuple[float, float] = (0, 0)  # For visualization (x, y)

@dataclass
class FlowConnection:
    """Flow connection between tanks or external systems"""
    connection_id: str
    source: str              # Tank ID or "EXTERNAL_SOURCE"
    target: str              # Tank ID or "EXTERNAL_SINK" or "ENVIRONMENT"
    connection_type: ConnectionType
    flow_model: FlowModel
    parameters: Dict[str, Any]  # Flow-specific parameters
    bidirectional: bool = False # Can flow both ways

@dataclass
class TankSystemGraph:
    """Complete tank system definition"""
    system_name: str
    tanks: List[TankConfiguration]
    connections: List[FlowConnection]

    def get_tank_by_id(self, tank_id: str) -> Optional[TankConfiguration]:
        """Get tank configuration by ID"""
        for tank in self.tanks:
            if tank.tank_id == tank_id:
                return tank
        return None

    def get_inflow_connections(self, tank_id: str) -> List[FlowConnection]:
        """Get all connections flowing into a tank"""
        return [conn for conn in self.connections if conn.target == tank_id]

    def get_outflow_connections(self, tank_id: str) -> List[FlowConnection]:
        """Get all connections flowing out of a tank"""
        return [conn for conn in self.connections if conn.source == tank_id]

    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Validate graph consistency and return errors"""
        errors = []

        # Check that all connection endpoints exist
        all_tank_ids = {tank.tank_id for tank in self.tanks}
        external_nodes = {"EXTERNAL_SOURCE", "EXTERNAL_SINK", "ENVIRONMENT"}
        valid_nodes = all_tank_ids | external_nodes

        for conn in self.connections:
            if conn.source not in valid_nodes:
                errors.append(f"Connection {conn.connection_id}: Unknown source '{conn.source}'")
            if conn.target not in valid_nodes:
                errors.append(f"Connection {conn.connection_id}: Unknown target '{conn.target}'")

        # Check for duplicate tank IDs
        tank_ids = [tank.tank_id for tank in self.tanks]
        if len(tank_ids) != len(set(tank_ids)):
            errors.append("Duplicate tank IDs found")

        # Check for duplicate connection IDs
        conn_ids = [conn.connection_id for conn in self.connections]
        if len(conn_ids) != len(set(conn_ids)):
            errors.append("Duplicate connection IDs found")

        return len(errors) == 0, errors

# =================== NETWORK VISUALIZATION ===================

class TankNetworkVisualizer:
    """Visualizer for tank network graphs"""

    # Color schemes for different elements
    TANK_COLORS = {
        TankType.CCH2: '#1f77b4',   # Blue
        TankType.CH2: '#ff7f0e',    # Orange
        TankType.SLH2: '#2ca02c'    # Green
    }

    CONNECTION_COLORS = {
        ConnectionType.FEED: '#d62728',      # Red
        ConnectionType.DISCHARGE: '#9467bd', # Purple
        ConnectionType.TRANSFER: '#8c564b',  # Brown
        ConnectionType.VENT: '#e377c2',      # Pink
        ConnectionType.RETURN: '#7f7f7f'     # Gray
    }

    EXTERNAL_COLOR = '#bcbd22'  # Olive for external nodes

    def __init__(self, system_graph: TankSystemGraph):
        """Initialize visualizer with system graph"""
        self.graph = system_graph
        self.nx_graph = None
        self.pos = None

    def create_networkx_graph(self) -> nx.DiGraph:
        """Convert tank system to NetworkX directed graph"""
        G = nx.DiGraph()

        # Add tank nodes
        for tank in self.graph.tanks:
            G.add_node(tank.tank_id,
                      node_type='tank',
                      tank_type=tank.tank_type,
                      volume=tank.volume,
                      **tank.initial_conditions,
                      **tank.scenario_params)

        # Add external nodes that are referenced
        external_nodes = set()
        for conn in self.graph.connections:
            if conn.source not in [tank.tank_id for tank in self.graph.tanks]:
                external_nodes.add(conn.source)
            if conn.target not in [tank.tank_id for tank in self.graph.tanks]:
                external_nodes.add(conn.target)

        for ext_node in external_nodes:
            G.add_node(ext_node, node_type='external')

        # Add connections as edges
        for conn in self.graph.connections:
            G.add_edge(conn.source, conn.target,
                      connection_id=conn.connection_id,
                      connection_type=conn.connection_type,
                      flow_model=conn.flow_model,
                      bidirectional=conn.bidirectional,
                      **conn.parameters)

            # Add reverse edge for bidirectional connections
            if conn.bidirectional:
                G.add_edge(conn.target, conn.source,
                          connection_id=conn.connection_id + "_reverse",
                          connection_type=conn.connection_type,
                          flow_model=conn.flow_model,
                          bidirectional=True,
                          **conn.parameters)

        self.nx_graph = G
        return G

    def calculate_layout(self, layout_method: str = 'spring') -> Dict[str, Tuple[float, float]]:
        """Calculate node positions for visualization"""
        if self.nx_graph is None:
            self.create_networkx_graph()

        # Use predefined positions if available
        predefined_pos = {}
        for tank in self.graph.tanks:
            if tank.position != (0, 0):
                predefined_pos[tank.tank_id] = tank.position

        if layout_method == 'spring':
            pos = nx.spring_layout(self.nx_graph, pos=predefined_pos,
                                 fixed=list(predefined_pos.keys()) if predefined_pos else None,
                                 k=4, iterations=100, scale=2.0)
        elif layout_method == 'circular':
            pos = nx.circular_layout(self.nx_graph, scale=2.0)
        elif layout_method == 'hierarchical':
            try:
                pos = nx.nx_agraph.graphviz_layout(self.nx_graph, prog='dot')
            except:
                # Fallback if graphviz not available
                pos = nx.spring_layout(self.nx_graph, k=4, iterations=100, scale=2.0)
        else:
            pos = nx.spring_layout(self.nx_graph, k=4, iterations=100, scale=2.0)

        self.pos = pos
        return pos

    def visualize_network(self, figsize: Tuple[int, int] = (16, 12),
                         layout_method: str = 'spring',
                         show_labels: bool = True,
                         show_connection_details: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive network visualization"""

        # Create NetworkX graph and calculate layout
        if self.nx_graph is None:
            self.create_networkx_graph()
        if self.pos is None:
            self.calculate_layout(layout_method)

        # Create figure with subplots to control layout better
        fig = plt.figure(figsize=figsize)

        # Main plot takes up most of the space, leave room for legend and summary
        ax = fig.add_subplot(111)

        # Calculate axis limits to ensure all nodes are visible with padding
        if self.pos:
            x_coords = [pos[0] for pos in self.pos.values()]
            y_coords = [pos[1] for pos in self.pos.values()]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding (30% extra space around nodes)
            x_range = x_max - x_min if x_max != x_min else 2.0
            y_range = y_max - y_min if y_max != y_min else 2.0
            padding_x = x_range * 0.3
            padding_y = y_range * 0.3

            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)

        # Draw tank nodes
        self._draw_tank_nodes(ax)

        # Draw external nodes
        self._draw_external_nodes(ax)

        # Draw connections
        self._draw_connections(ax, show_connection_details)

        # Add labels
        if show_labels:
            self._add_node_labels(ax)

        # Add legend (positioned outside plot area)
        self._add_legend(ax)

        # Set title and formatting
        ax.set_title(f"Tank Network: {self.graph.system_name}",
                    fontsize=18, fontweight='bold', pad=25)
        ax.set_aspect('equal')
        ax.axis('off')

        # Add system summary (positioned at bottom)
        self._add_system_summary(fig)

        # Adjust layout to prevent clipping
        plt.subplots_adjust(left=0.05, right=0.75, top=0.92, bottom=0.15)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to: {save_path}")

        return fig

    def _draw_tank_nodes(self, ax):
        """Draw tank nodes with tank-type specific styling"""
        for tank in self.graph.tanks:
            x, y = self.pos[tank.tank_id]
            color = self.TANK_COLORS[tank.tank_type]

            # Draw main tank circle (larger for better visibility)
            circle = plt.Circle((x, y), 0.25, color=color, alpha=0.8, ec='black', linewidth=3)
            ax.add_patch(circle)

            # Add tank type indicator
            tank_type_short = tank.tank_type.value.replace('_', '\n').upper()
            ax.text(x, y, tank_type_short, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

    def _draw_external_nodes(self, ax):
        """Draw external system nodes"""
        external_nodes = [node for node in self.nx_graph.nodes()
                         if self.nx_graph.nodes[node].get('node_type') == 'external']

        for node in external_nodes:
            x, y = self.pos[node]

            # Different shapes for different external types
            if 'SOURCE' in node:
                shape = 'square'
                symbol = '⊡'
            elif 'SINK' in node:
                shape = 'triangle'
                symbol = '△'
            elif 'ENVIRONMENT' in node:
                shape = 'diamond'
                symbol = '◇'
            else:
                shape = 'hexagon'
                symbol = '⬡'

            # Draw external node (larger for better visibility)
            if shape == 'square':
                rect = plt.Rectangle((x-0.15, y-0.15), 0.3, 0.3,
                                   color=self.EXTERNAL_COLOR, alpha=0.7, ec='black', linewidth=2)
                ax.add_patch(rect)
            elif shape == 'diamond':
                # Draw diamond shape for environment
                diamond_x = [x, x+0.15, x, x-0.15, x]
                diamond_y = [y+0.15, y, y-0.15, y, y+0.15]
                ax.plot(diamond_x, diamond_y, color='black', linewidth=2)
                ax.fill(diamond_x, diamond_y, color=self.EXTERNAL_COLOR, alpha=0.7)
            else:
                # Default circle for other types
                circle = plt.Circle((x, y), 0.15, color=self.EXTERNAL_COLOR, alpha=0.7, ec='black', linewidth=2)
                ax.add_patch(circle)

            ax.text(x, y, symbol, ha='center', va='center',
                   fontsize=18, fontweight='bold')

    def _draw_connections(self, ax, show_details: bool):
        """Draw flow connections with directional arrows"""
        for conn in self.graph.connections:
            # Get positions
            x1, y1 = self.pos[conn.source]
            x2, y2 = self.pos[conn.target]

            # Connection styling
            color = self.CONNECTION_COLORS[conn.connection_type]
            linestyle = '-' if not conn.bidirectional else '--'
            linewidth = 2

            # Draw arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->' if not conn.bidirectional else '<->',
                                     color=color, lw=linewidth, ls=linestyle))

            # Add connection label if requested
            if show_details:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

                # Offset label to avoid overlap
                offset_x = 0.05 * (y2 - y1) / np.sqrt((x2-x1)**2 + (y2-y1)**2) if (x2-x1)**2 + (y2-y1)**2 > 0 else 0
                offset_y = -0.05 * (x2 - x1) / np.sqrt((x2-x1)**2 + (y2-y1)**2) if (x2-x1)**2 + (y2-y1)**2 > 0 else 0

                label = f"{conn.connection_type.value}\n({conn.flow_model.value})"
                ax.text(mid_x + offset_x, mid_y + offset_y, label,
                       ha='center', va='center', fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def _add_node_labels(self, ax):
        """Add node ID labels"""
        for node in self.nx_graph.nodes():
            x, y = self.pos[node]

            # Position label below node (further down for larger nodes)
            ax.text(x, y - 0.4, node, ha='center', va='top',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))

    def _add_legend(self, ax):
        """Add legend for tank types and connection types"""
        # Tank type legend
        tank_patches = [mpatches.Patch(color=color, label=tank_type.value.replace('_', ' ').title())
                       for tank_type, color in self.TANK_COLORS.items()]

        # Connection type legend
        connection_patches = [mpatches.Patch(color=color, label=conn_type.value.title())
                            for conn_type, color in self.CONNECTION_COLORS.items()]

        # External node legend
        external_patch = mpatches.Patch(color=self.EXTERNAL_COLOR, label='External Systems')

        all_patches = tank_patches + [external_patch] + connection_patches

        # Position legend outside the plot area
        ax.legend(handles=all_patches, loc='center left', bbox_to_anchor=(1.05, 0.5),
                 title='Legend', title_fontsize=14, fontsize=11, frameon=True,
                 fancybox=True, shadow=True)

    def _add_system_summary(self, fig):
        """Add system summary information"""
        tank_types = set(tank.tank_type.value.replace('_', ' ').title() for tank in self.graph.tanks)
        summary_text = f"""System Summary:
Tanks: {len(self.graph.tanks)} • Connections: {len(self.graph.connections)}
Tank Types: {', '.join(tank_types)}"""

        fig.text(0.05, 0.05, summary_text, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.9, edgecolor='black'))

# =================== EXAMPLE CONFIGURATIONS ===================

def create_cch2_prototype_graph() -> TankSystemGraph:
    """Create the original 2-CCH2 prototype configuration"""

    tanks = [
        TankConfiguration(
            tank_id="Tank_1",
            tank_type=TankType.CCH2,
            volume=0.5,
            initial_conditions={"pressure": 400e5, "temperature": 53.25},
            scenario_params={"stopping_density": 70.0, "scenario": "dormancy"},
            thermal_params={"htc": 0.025},
            position=(-1, 0)
        ),
        TankConfiguration(
            tank_id="Tank_2",
            tank_type=TankType.CCH2,
            volume=0.5,
            initial_conditions={"pressure": 400e5, "temperature": 53.25},
            scenario_params={"stopping_density": 5.8, "scenario": "discharge"},
            thermal_params={"htc": 0.025},
            position=(1, 0)
        )
    ]

    connections = [
        # Tank 1 connections (dormancy - no active flows)
        FlowConnection(
            connection_id="T1_feed",
            source="EXTERNAL_SOURCE",
            target="Tank_1",
            connection_type=ConnectionType.FEED,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}
        ),
        FlowConnection(
            connection_id="T1_discharge",
            source="Tank_1",
            target="EXTERNAL_SINK",
            connection_type=ConnectionType.DISCHARGE,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}
        ),
        FlowConnection(
            connection_id="T1_vent",
            source="Tank_1",
            target="ENVIRONMENT",
            connection_type=ConnectionType.VENT,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}  # Configuration-dependent
        ),

        # Tank 2 connections (discharge)
        FlowConnection(
            connection_id="T2_feed",
            source="EXTERNAL_SOURCE",
            target="Tank_2",
            connection_type=ConnectionType.FEED,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}
        ),
        FlowConnection(
            connection_id="T2_discharge",
            source="Tank_2",
            target="EXTERNAL_SINK",
            connection_type=ConnectionType.DISCHARGE,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.001}
        ),
        FlowConnection(
            connection_id="T2_vent",
            source="Tank_2",
            target="ENVIRONMENT",
            connection_type=ConnectionType.VENT,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}  # Configuration-dependent
        )
    ]

    return TankSystemGraph(
        system_name="CCH2 Prototype (2 Independent Tanks)",
        tanks=tanks,
        connections=connections
    )

def create_user_specified_prototype() -> TankSystemGraph:
    """Create the user-specified 2-CCH2 tank configuration:
    - Tank 1: Dormancy mode, only vents to environment (no external feed/discharge)
    - Tank 2: Discharge mode, discharges to external sink + can vent to environment
    - No external sources (tanks assumed to start full)
    """

    tanks = [
        TankConfiguration(
            tank_id="Tank_1",
            tank_type=TankType.CCH2,
            volume=0.5,
            initial_conditions={"pressure": 400e5, "temperature": 53.25},
            scenario_params={"stopping_density": 70.0, "scenario": "dormancy_only_vent"},
            thermal_params={"htc": 0.025},
            position=(-1.5, 0)
        ),
        TankConfiguration(
            tank_id="Tank_2",
            tank_type=TankType.CCH2,
            volume=0.5,
            initial_conditions={"pressure": 400e5, "temperature": 53.25},
            scenario_params={"stopping_density": 5.8, "scenario": "discharge_with_vent"},
            thermal_params={"htc": 0.025},
            position=(1.5, 0)
        )
    ]

    connections = [
        # Tank 1 connections (dormancy - only venting)
        FlowConnection(
            connection_id="T1_vent",
            source="Tank_1",
            target="ENVIRONMENT",
            connection_type=ConnectionType.VENT,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}  # Configuration-dependent (A/B/C switching)
        ),

        # Tank 2 connections (discharge + vent capability)
        FlowConnection(
            connection_id="T2_discharge",
            source="Tank_2",
            target="EXTERNAL_SINK",
            connection_type=ConnectionType.DISCHARGE,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.001}  # 1 g/s constant discharge
        ),
        FlowConnection(
            connection_id="T2_vent",
            source="Tank_2",
            target="ENVIRONMENT",
            connection_type=ConnectionType.VENT,
            flow_model=FlowModel.CONSTANT,
            parameters={"rate": 0.0}  # Configuration-dependent (A/B/C switching)
        )
    ]

    return TankSystemGraph(
        system_name="User Specified 2-CCH2 Prototype",
        tanks=tanks,
        connections=connections
    )

def create_complex_example_graph() -> TankSystemGraph:
    """Create a more complex multi-tank system example"""

    tanks = [
        TankConfiguration(
            tank_id="Storage_1",
            tank_type=TankType.SLH2,
            volume=2.0,
            initial_conditions={"pressure": 5e5, "temperature": 20.0},
            scenario_params={"stopping_density": 50.0, "scenario": "storage"},
            thermal_params={"htc": 0.1},
            position=(-2, 1)
        ),
        TankConfiguration(
            tank_id="Storage_2",
            tank_type=TankType.SLH2,
            volume=2.0,
            initial_conditions={"pressure": 5e5, "temperature": 20.0},
            scenario_params={"stopping_density": 50.0, "scenario": "storage"},
            thermal_params={"htc": 0.1},
            position=(-2, -1)
        ),
        TankConfiguration(
            tank_id="Buffer",
            tank_type=TankType.CCH2,
            volume=0.5,
            initial_conditions={"pressure": 200e5, "temperature": 40.0},
            scenario_params={"stopping_density": 30.0, "scenario": "buffer"},
            thermal_params={"htc": 0.025},
            position=(0, 0)
        ),
        TankConfiguration(
            tank_id="Service",
            tank_type=TankType.CH2,
            volume=0.1,
            initial_conditions={"pressure": 350e5, "temperature": 298.15},
            scenario_params={"stopping_density": 20.0, "scenario": "service"},
            thermal_params={"htc": 0.01},
            position=(2, 0)
        )
    ]

    connections = [
        # External feed to storage tanks
        FlowConnection("feed_1", "EXTERNAL_SOURCE", "Storage_1",
                      ConnectionType.FEED, FlowModel.PUMP, {"capacity": 0.1}),
        FlowConnection("feed_2", "EXTERNAL_SOURCE", "Storage_2",
                      ConnectionType.FEED, FlowModel.PUMP, {"capacity": 0.1}),

        # Storage to buffer transfers
        FlowConnection("transfer_1", "Storage_1", "Buffer",
                      ConnectionType.TRANSFER, FlowModel.ORIFICE, {"area": 0.01}),
        FlowConnection("transfer_2", "Storage_2", "Buffer",
                      ConnectionType.TRANSFER, FlowModel.ORIFICE, {"area": 0.01}),

        # Buffer to service transfer
        FlowConnection("buffer_service", "Buffer", "Service",
                      ConnectionType.TRANSFER, FlowModel.VALVE, {"opening": 0.5}),

        # Service discharge
        FlowConnection("service_out", "Service", "EXTERNAL_SINK",
                      ConnectionType.DISCHARGE, FlowModel.CONSTANT, {"rate": 0.01}),

        # Venting connections
        FlowConnection("vent_buffer", "Buffer", "ENVIRONMENT",
                      ConnectionType.VENT, FlowModel.CONSTANT, {"rate": 0.0}),
        FlowConnection("vent_service", "Service", "ENVIRONMENT",
                      ConnectionType.VENT, FlowModel.CONSTANT, {"rate": 0.0}),

        # Return line (bidirectional)
        FlowConnection("return_line", "Service", "Buffer",
                      ConnectionType.RETURN, FlowModel.ORIFICE,
                      {"area": 0.005}, bidirectional=True)
    ]

    return TankSystemGraph(
        system_name="Complex Multi-Tank System",
        tanks=tanks,
        connections=connections
    )

# =================== INTERACTIVE CONFIGURATION BUILDER ===================

def build_custom_network() -> TankSystemGraph:
    """Interactive builder for custom tank networks"""
    print("\n🔧 CUSTOM NETWORK BUILDER")
    print("-" * 30)

    system_name = input("Enter system name: ").strip() or "Custom Tank System"

    tanks = []
    connections = []

    # Build tanks
    print(f"\n📦 Adding tanks to '{system_name}':")
    while True:
        tank_id = input(f"Tank ID (or 'done' to finish): ").strip()
        if tank_id.lower() == 'done':
            break

        print("Tank types: 1=CCH2, 2=CH2, 3=SLH2")
        tank_type_choice = input("Select tank type (1-3): ").strip()
        tank_type_map = {'1': TankType.CCH2, '2': TankType.CH2, '3': TankType.SLH2}
        tank_type = tank_type_map.get(tank_type_choice, TankType.CCH2)

        volume = float(input("Volume (m³): ") or "1.0")
        pressure = float(input("Initial pressure (bar): ") or "400") * 1e5  # Convert to Pa
        temperature = float(input("Initial temperature (K): ") or "53.25")

        # Optional positioning
        pos_x = float(input("X position (optional, press enter for auto): ") or "0")
        pos_y = float(input("Y position (optional, press enter for auto): ") or "0")

        tank = TankConfiguration(
            tank_id=tank_id,
            tank_type=tank_type,
            volume=volume,
            initial_conditions={"pressure": pressure, "temperature": temperature},
            scenario_params={"stopping_density": 20.0, "scenario": "custom"},
            thermal_params={"htc": 0.025},
            position=(pos_x, pos_y)
        )
        tanks.append(tank)
        print(f"✅ Added {tank_id} ({tank_type.value})")

    if not tanks:
        print("No tanks added. Using default configuration.")
        return create_cch2_prototype_graph()

    # Build connections
    print(f"\n🔗 Adding connections:")
    tank_ids = [t.tank_id for t in tanks]
    external_options = ["EXTERNAL_SOURCE", "EXTERNAL_SINK", "ENVIRONMENT"]
    all_nodes = tank_ids + external_options

    print(f"Available nodes: {', '.join(all_nodes)}")

    conn_id = 1
    while True:
        print(f"\nConnection {conn_id}:")
        source = input("Source node (or 'done' to finish): ").strip()
        if source.lower() == 'done':
            break

        target = input("Target node: ").strip()

        if source not in all_nodes or target not in all_nodes:
            print("Invalid node names. Try again.")
            continue

        print("Connection types: 1=Feed, 2=Discharge, 3=Transfer, 4=Vent, 5=Return")
        conn_type_choice = input("Select connection type (1-5): ").strip()
        conn_type_map = {
            '1': ConnectionType.FEED, '2': ConnectionType.DISCHARGE,
            '3': ConnectionType.TRANSFER, '4': ConnectionType.VENT, '5': ConnectionType.RETURN
        }
        conn_type = conn_type_map.get(conn_type_choice, ConnectionType.TRANSFER)

        print("Flow models: 1=Constant, 2=Orifice, 3=Pump, 4=Valve")
        flow_choice = input("Select flow model (1-4): ").strip()
        flow_map = {
            '1': FlowModel.CONSTANT, '2': FlowModel.ORIFICE,
            '3': FlowModel.PUMP, '4': FlowModel.VALVE
        }
        flow_model = flow_map.get(flow_choice, FlowModel.CONSTANT)

        bidirectional = input("Bidirectional? (y/N): ").strip().lower() == 'y'

        connection = FlowConnection(
            connection_id=f"conn_{conn_id}",
            source=source,
            target=target,
            connection_type=conn_type,
            flow_model=flow_model,
            parameters={"rate": 0.001},  # Default parameters
            bidirectional=bidirectional
        )
        connections.append(connection)
        print(f"✅ Added connection: {source} → {target}")
        conn_id += 1

    return TankSystemGraph(
        system_name=system_name,
        tanks=tanks,
        connections=connections
    )

# =================== MAIN TESTING FUNCTION ===================

def main():
    """Test the network visualization capabilities"""
    print("TANK NETWORK VISUALIZATION TOOL")
    print("="*50)

    # Ask user what they want to do
    print("\nSelect an option:")
    print("1. View Original CCH2 Prototype")
    print("2. View User Specified Prototype (Tank1=vent only, Tank2=discharge+vent)")
    print("3. Compare Both Prototypes")
    print("4. View Complex Example")
    print("5. Build Custom Network")
    print("6. View All Examples")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '5':
        # Custom network builder
        custom_graph = build_custom_network()

        is_valid, errors = custom_graph.validate_graph()
        print(f"\nGraph validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        if errors:
            for error in errors:
                print(f"Error: {error}")

        visualizer = TankNetworkVisualizer(custom_graph)
        fig = visualizer.visualize_network(layout_method='spring', figsize=(16, 12))
        plt.show()
        return

    elif choice == '1':
        graphs_to_show = [("Original CCH2 Prototype", create_cch2_prototype_graph())]
    elif choice == '2':
        graphs_to_show = [("User Specified Prototype", create_user_specified_prototype())]
    elif choice == '3':
        graphs_to_show = [
            ("Original CCH2 Prototype", create_cch2_prototype_graph()),
            ("User Specified Prototype", create_user_specified_prototype())
        ]
    elif choice == '4':
        graphs_to_show = [("Complex Example", create_complex_example_graph())]
    else:
        graphs_to_show = [
            ("Original CCH2 Prototype", create_cch2_prototype_graph()),
            ("User Specified Prototype", create_user_specified_prototype()),
            ("Complex Example", create_complex_example_graph())
        ]

    # Process selected graphs
    for i, (name, graph) in enumerate(graphs_to_show, 1):
        print(f"\n{i}. Testing {name} Configuration...")

        # Validate graph
        is_valid, errors = graph.validate_graph()
        print(f"   Graph validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        if errors:
            for error in errors:
                print(f"   Error: {error}")

        # Create visualizer and show network
        visualizer = TankNetworkVisualizer(graph)
        figsize = (16, 10) if len(graph.tanks) <= 2 else (18, 12)
        fig = visualizer.visualize_network(layout_method='spring', figsize=figsize)
        plt.show()

        # Print system summary
        print(f"\n{name} Summary:")
        print(f"   Tanks: {len(graph.tanks)}")
        print(f"   Connections: {len(graph.connections)}")

        # Show connection details for prototypes
        if "Prototype" in name:
            print(f"   Connection Details:")
            for conn in graph.connections:
                direction = "↔" if conn.bidirectional else "→"
                print(f"     • {conn.source} {direction} {conn.target} ({conn.connection_type.value})")

    # Special comparison for prototype differences
    if choice == '3' and len(graphs_to_show) == 2:
        print(f"\n🔍 KEY DIFFERENCES:")
        original = graphs_to_show[0][1]
        user_spec = graphs_to_show[1][1]

        print(f"   Original has {len(original.connections)} connections vs User Specified has {len(user_spec.connections)}")
        print(f"   🔹 Original: Both tanks have feed connections (dormant feeds)")
        print(f"   🔹 User Specified: No external feeds (tanks start full)")
        print(f"   🔹 Original: Tank 1 has discharge connection (rate=0)")
        print(f"   🔹 User Specified: Tank 1 only vents to environment")
        print(f"   🔹 Both: Tank 2 discharges to external sink")
        print(f"   🔹 Both: Both tanks can vent to environment")

    print("\n" + "="*50)
    print("NETWORK VISUALIZATION TESTING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main()