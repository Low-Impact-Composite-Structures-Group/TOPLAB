"""
Multi-Tank System Analysis Driver

A clean analysis script that uses the src/multi_tank framework for
coupled hydrogen tank system studies.

This driver demonstrates how to use the modular multi-tank framework
for specific analysis cases while keeping the reusable components
separate in src/multi_tank.
"""

# Standard library imports
import sys
import math
import time
from pathlib import Path

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Multi-tank framework imports
from src.multi_tank import (
    MultiTankSystem,
    MultiTankConfig,
    create_tank_from_fuel_mass,
    create_tank_from_mission
)

# Analysis framework imports
from src.mission.isochoric_missions import DischargeMission

# Graph-based network definition imports
from graph_factory import GraphFactory, TankSystemGraph
from tank_network_visualizer import TankNetworkVisualizer

# Plotting imports
from plotting.plot_style_sb import configure_plot_style


def create_dual_tank_network_config():
    """Create network configuration with CH2 and CCH2 tanks"""
    print("\n🔧 DUAL-TANK NETWORK CONFIGURATION")
    print("-" * 40)
    print("Tank 1: CH2 (700 bar, 330 K, 25 kg)")
    print("Tank 2: CCH2 (400 bar, 53.25 K, mission-based)")

    # ===== Tank 1: CH2 Tank (Compressed Hydrogen) =====
    print(f"\n🔧 Creating Tank 1 (CH2):")
    ch2_fuel_mass = 25.0  # kg (specified)
    ch2_initial_pressure = 700e5  # Pa (700 bar)
    ch2_initial_temperature = 330.0  # K
    ch2_operating_pressure = 800e5  # Pa (800 bar P_VENT)

    ch2_tank = create_tank_from_fuel_mass(
        fuel_mass=ch2_fuel_mass,
        initial_pressure=ch2_initial_pressure,
        initial_temperature=ch2_initial_temperature,
        operating_pressure=ch2_operating_pressure,
        safety_margin=1.1,
        liner_thickness=0.005,
        insulation_thickness=0.05,
    )

    # ===== Tank 2: CCH2 Tank (Cryocompressed Hydrogen) =====
    print(f"\n🔧 Creating Tank 2 (CCH2):")
    discharge_rate = 0.001  # kg/s
    mission_duration = 6 * 3600  # 6 hours in seconds

    discharge_mission = DischargeMission(
        discharge_rate=discharge_rate,
        duration=mission_duration
    )

    # Calculate tank geometry based on mission
    cch2_initial_pressure = 400e5  # Pa (400 bar)
    cch2_initial_temperature = 53.25  # K
    cch2_operating_pressure = 450e5  # Pa (450 bar P_VENT)

    print(f"CCH2 Mission Requirements:")
    print(f"  Discharge rate: {discharge_rate} kg/s")
    print(f"  Mission duration: {mission_duration / 3600:.1f} hours")

    cch2_tank, fuel_volume_required = create_tank_from_mission(
        mission=discharge_mission,
        initial_pressure=cch2_initial_pressure,
        initial_temperature=cch2_initial_temperature,
        operating_pressure=cch2_operating_pressure,
        safety_margin=1.2,
        liner_thickness=0.005,
        insulation_thickness=0.05,
    )

    print(f"\n✅ Dual-tank geometry calculated:")
    print(f"   Tank 1 (CH2): V={ch2_tank.volume:.4f} m³, R={ch2_tank.radius:.3f} m")
    print(f"   Tank 2 (CCH2): V={cch2_tank.volume:.4f} m³, R={cch2_tank.radius:.3f} m")

    # Create graph configuration using the existing user specified prototype
    factory = GraphFactory()
    graph = factory.create_user_specified_prototype()

    return graph, ch2_tank, cch2_tank


class GraphConfiguredMultiTankSystem:
    """
    A system that wraps MultiTankSystem with graph-based configuration.

    This class demonstrates how to configure the physics simulation
    using a graph-based tank network definition.
    """

    def __init__(self, graph: TankSystemGraph, mission_tank=None, ch2_tank=None):
        """Initialize with a tank system graph."""
        self.graph = graph
        self.mission_tank = mission_tank
        self.ch2_tank = ch2_tank

        # Create configuration that matches the graph
        self.config = self._create_config_from_graph()

        # Initialize the MultiTankSystem with graph-derived config
        tank_geometries = [ch2_tank, mission_tank] if ch2_tank and mission_tank else []
        self.physics_system = MultiTankSystem(self.config, tank_geometries)

    def _create_config_from_graph(self) -> MultiTankConfig:
        """Create MultiTankConfig from graph definition"""
        # Get tank parameters from graph
        tank1 = self.graph.tanks[0]  # Tank_1
        tank2 = self.graph.tanks[1]  # Tank_2

        # Create config with parameters extracted from graph
        config = MultiTankConfig()

        # Set stopping densities from graph
        config.HighPressureTank.STOPPING_DENSITY = tank1.scenario_params.get("stopping_density", 70.0)
        config.CryogenicTank.STOPPING_DENSITY = tank2.scenario_params.get("stopping_density", 5.8)

        # Configure scenarios based on graph connections
        tank1_discharge_rate = self._get_discharge_rate_from_graph("Tank_1")
        tank2_discharge_rate = self._get_discharge_rate_from_graph("Tank_2")

        print(f"Graph configuration extracted:")
        print(f"  Tank 1: stopping at {config.HighPressureTank.STOPPING_DENSITY} kg/m³")
        print(f"  Tank 2: stopping at {config.CryogenicTank.STOPPING_DENSITY} kg/m³")
        print(f"  Discharge rates: T1={tank1_discharge_rate:.4f} kg/s, T2={tank2_discharge_rate:.4f} kg/s")

        return config

    def _get_discharge_rate_from_graph(self, tank_id: str) -> float:
        """Get discharge rate for a tank from its graph connections"""
        discharge_rate = 0.0
        outflow_connections = self.graph.get_outflow_connections(tank_id)
        for conn in outflow_connections:
            if conn.connection_type.value == "discharge":
                discharge_rate += conn.parameters.get("rate", 0.0)
        return discharge_rate

    def run_simulation(self, solver_method: str = "LSODA"):
        """Run simulation using the graph-configured physics system."""
        print(f"\n🚀 Running graph-configured simulation with {solver_method} solver...")
        results = self.physics_system.run_analysis(solver_method)
        print(f"✅ Graph-configured simulation completed successfully!")
        return results

    def get_network_summary(self) -> dict:
        """Get summary of the network configuration"""
        return {
            "system_name": self.graph.system_name,
            "tank_count": len(self.graph.tanks),
            "connection_count": len(self.graph.connections),
            "tank_configurations": [
                {
                    "tank_id": tank.tank_id,
                    "volume": tank.volume,
                    "scenario": tank.scenario_params.get("scenario", "unknown"),
                    "stopping_density": tank.scenario_params.get("stopping_density", 0.0)
                }
                for tank in self.graph.tanks
            ],
            "connections": [
                {
                    "from": conn.source,
                    "to": conn.target,
                    "type": conn.connection_type.value,
                    "rate": conn.parameters.get("rate", 0.0)
                }
                for conn in self.graph.connections
            ]
        }


def main():
    """Main execution function"""
    print("="*80)
    print("MULTI-TANK COUPLED HYDROGEN SYSTEM ANALYSIS")
    print("Using Modular src/multi_tank Framework")
    print("="*80)

    # Create dual-tank configuration
    graph, ch2_tank, cch2_tank = create_dual_tank_network_config()

    # Display network summary
    print(f"\n📊 NETWORK SUMMARY:")
    print(f"   System: {graph.system_name}")
    print(f"   Tanks: {len(graph.tanks)}")
    print(f"   Connections: {len(graph.connections)}")

    # Create the graph-configured system
    system = GraphConfiguredMultiTankSystem(graph, cch2_tank, ch2_tank)

    # Display detailed network summary
    network_summary = system.get_network_summary()
    print(f"\n📋 Network Configuration Details:")
    for tank_config in network_summary["tank_configurations"]:
        print(f"   {tank_config['tank_id']}: {tank_config['scenario']} scenario, "
              f"V={tank_config['volume']:.4f} m³, "
              f"ρ_stop={tank_config['stopping_density']:.1f} kg/m³")

    for connection in network_summary["connections"]:
        print(f"   {connection['from']} → {connection['to']}: {connection['type']} "
              f"at {connection['rate']:.4f} kg/s")

    # Run the simulation
    try:
        print(f"\n🚀 Starting coupled multi-tank simulation...")
        start_time = time.time()

        results = system.run_simulation("RK45")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"   Total execution time: {total_time:.2f} seconds")
        print(f"   Final simulation time: {results.times[-1]/3600:.2f} hours")
        print(f"   Data points collected: {len(results.times)}")

        # Validate results
        validation = system.physics_system.validate_results()
        if validation['overall']:
            print(f"   ✅ Results validation: PASSED")
        else:
            print(f"   ⚠️ Results validation: Some checks failed")

        # Create plots
        try:
            import matplotlib.pyplot as plt

            # Plot 1: Tank states evolution
            tank_states_fig = system.physics_system.plot_results("ch2_cch2_tank_states.png")

            # Plot 2: Network topology
            visualizer = TankNetworkVisualizer(graph)
            network_fig = visualizer.visualize_network(
                figsize=(12, 8),
                layout_method='spring',
                show_labels=True,
                show_connection_details=True,
                save_path="ch2_cch2_network_topology.png"
            )

            # Show both plots using plt.show() - this will display all open figures
            plt.show()

            print(f"   📊 Tank states plot saved: ch2_cch2_tank_states.png")
            print(f"   🌐 Network topology plot saved: ch2_cch2_network_topology.png")

        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")
            # Print full traceback for debugging
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ SIMULATION FAILED: {e}")
        raise

    print(f"\n✅ Multi-tank analysis completed successfully!")
    return results


if __name__ == "__main__":
    # Run the analysis
    results = main()