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
import matplotlib.pyplot as plt



def create_dual_tank_network_config():
    """Create network configuration with CH2 and CCH2 tanks"""
    print("\n🔧 DUAL-TANK NETWORK CONFIGURATION")
    print("-" * 40)
    print("Tank 1: CH2 (700 bar, 330 K, 100 kg)")
    print("Tank 2: CCH2 (400 bar, 53.25 K, mission-based)")

    # ===== Tank 1: CH2 Tank (Compressed Hydrogen) =====
    print(f"\n🔧 Creating Tank 1 (CH2):")
    ch2_fuel_mass = 150.0  # kg (increased for better coupling capacity)
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

    # Use ATR72 mission profile instead of constant discharge
    cch2_initial_mass = 25.0  # kg initial estimate
    cch2_initial_temperature = 53.25  # K

    discharge_mission = DischargeMission.atr72_mission(
        initial_mass=cch2_initial_mass,
        initial_temperature=cch2_initial_temperature
    )

    # Calculate tank geometry based on ATR72 mission
    cch2_initial_pressure = 400e5  # Pa (400 bar)
    cch2_operating_pressure = 450e5  # Pa (450 bar P_VENT)

    # Calculate mission statistics
    total_duration = sum(section.duration for section in discharge_mission.sections)
    print(f"ATR72 Mission Requirements:")
    print(f"  Number of sections: {len(discharge_mission.sections)}")
    print(f"  Total mission duration: {total_duration / 3600:.2f} hours")
    print(f"  Average discharge rate: {discharge_mission.discharge_rate:.6f} kg/s")

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

    return graph, ch2_tank, cch2_tank, discharge_mission


class GraphConfiguredMultiTankSystem:
    """
    A system that wraps MultiTankSystem with graph-based configuration.

    This class demonstrates how to configure the physics simulation
    using a graph-based tank network definition.
    """

    def __init__(self, graph: TankSystemGraph, mission_tank=None, ch2_tank=None, atr72_mission=None):
        """Initialize with a tank system graph."""
        self.graph = graph
        self.mission_tank = mission_tank
        self.ch2_tank = ch2_tank
        self.atr72_mission = atr72_mission

        # Create configuration that matches the graph
        self.config = self._create_config_from_graph()

        # Initialize the MultiTankSystem with graph-derived config
        tank_geometries = [ch2_tank, mission_tank] if ch2_tank and mission_tank else []
        self.physics_system = MultiTankSystem(self.config, tank_geometries)

        # Override flow rate method if we have ATR72 mission
        if self.atr72_mission:
            self._setup_atr72_flow_rates()

        # Enable inter-tank coupling for pressure compensation
        self.physics_system.enable_tank_coupling(True)
        print("✅ Inter-tank coupling ENABLED for pressure compensation")

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

    def _setup_atr72_flow_rates(self):
        """Setup ATR72 mission flow rates for tank 2."""
        # Calculate total mission duration
        total_duration = sum(section.duration for section in self.atr72_mission.sections)

        # Calculate total fuel required for ATR72 mission
        total_fuel_consumed = 0.0
        for section in self.atr72_mission.sections:
            for flow in section.fuel_flows:
                if hasattr(flow, 'mass_flow'):
                    if isinstance(flow.mass_flow, list):
                        # Time-varying flow: trapezoidal integration
                        start_rate = abs(flow.mass_flow[0])
                        end_rate = abs(flow.mass_flow[-1])
                        avg_rate = (start_rate + end_rate) / 2.0
                        total_fuel_consumed += avg_rate * section.duration
                    else:
                        # Constant flow
                        total_fuel_consumed += abs(flow.mass_flow) * section.duration

        # Calculate precise initial mass to end exactly at minimum density
        tank2_volume = self.mission_tank.volume

        # Target final conditions - use the actual stopping density from configuration
        min_density = self.config.CryogenicTank.STOPPING_DENSITY  # kg/m³
        final_mass_target = min_density * tank2_volume

        # The initial mass should be: final target mass + fuel consumed
        # This accounts for the fact that we want to end with exactly the minimum mass
        total_fuel_required = final_mass_target + total_fuel_consumed

        # Add minimal operational margin (1%) to avoid numerical precision issues
        # but still end very close to the stopping density
        operational_margin = 1.01
        total_fuel_required *= operational_margin

        # Update config with ATR72 mission parameters and cryocompressed conditions
        self.config.CryogenicTank.MISSION_DURATION = total_duration
        self.config.CryogenicTank.INITIAL_MASS = total_fuel_required

        # No need to change initial conditions - previous conditions were already cryocompressed
        # 200+ bar and ~53 K is well in the cryocompressed region for hydrogen

        # Also update the global mission duration for both tanks
        self.config.HighPressureTank.MISSION_DURATION = total_duration

        print(f"   Precise initial mass calculation:")
        print(f"     ATR72 fuel consumed: {total_fuel_consumed:.2f} kg")
        print(f"     Target final mass (at min density): {final_mass_target:.2f} kg")
        print(f"     Operational margin: {operational_margin}x")
        print(f"     Total initial mass required: {total_fuel_required:.2f} kg")

        # Create time-varying flow rate function
        def atr72_flow_function(time: float) -> float:
            """Get ATR72 discharge rate at given time."""
            current_time = 0.0

            for section in self.atr72_mission.sections:
                section_end_time = current_time + section.duration

                if time <= section_end_time:
                    # We're in this section
                    section_time = time - current_time

                    # Get flow from this section
                    for flow in section.fuel_flows:
                        if hasattr(flow, 'mass_flow'):
                            if isinstance(flow.mass_flow, list):
                                # Time-varying flow: linear interpolation
                                start_rate = abs(flow.mass_flow[0])
                                end_rate = abs(flow.mass_flow[-1])
                                progress = section_time / section.duration if section.duration > 0 else 0
                                return start_rate + (end_rate - start_rate) * progress
                            else:
                                # Constant flow
                                return abs(flow.mass_flow)

                    return 0.0  # No flow found in this section

                current_time = section_end_time

            # Beyond mission duration
            return 0.0

        # Override the physics system's flow rate method for tank 2
        original_get_flow_rates = self.physics_system._get_flow_rates

        def atr72_aware_get_flow_rates(time: float, tank_index: int):
            if tank_index == 1:  # Tank 2 (CCH2 with ATR72 mission)
                inflow_rate = 0.0  # No inflow
                outflow_rate = atr72_flow_function(time)
                return inflow_rate, outflow_rate
            else:
                # Use original method for other tanks
                return original_get_flow_rates(time, tank_index)

        # Replace the method
        self.physics_system._get_flow_rates = atr72_aware_get_flow_rates

        print(f"✅ Tank 2 will use original cryocompressed conditions:")
        print(f"   Previous conditions (~200 bar, ~53 K) are already cryocompressed")
        print(f"   Expected mass: {total_fuel_required:.2f} kg (ATR72 requirement)")

        # Override initial state creation to correct Tank 2 mass and pressure for ATR72 mission
        original_create_initial_state = self.physics_system._create_initial_state

        def atr72_initial_state():
            """Create initial state with correct Tank 1 and Tank 2 pressures."""
            from CoolProp.CoolProp import PropsSI

            initial_state = original_create_initial_state()

            # ===== Fix Tank 1: Achieve target 700 bar =====
            tank1_state = initial_state.tank_states[0]
            tank1_original_mass = tank1_state.fuel_mass
            tank1_original_pressure = tank1_state.pressure
            tank1_target_pressure = 700e5  # 700 bar
            tank1_temperature = tank1_state.temperature  # Keep at 330 K

            try:
                # Calculate density needed for 700 bar at 330 K
                tank1_target_density = PropsSI('D', 'P', tank1_target_pressure, 'T', tank1_temperature, 'Hydrogen')
                tank1_required_mass = tank1_target_density * tank1_state.volume

                # Update Tank 1 state
                tank1_state.fuel_mass = tank1_required_mass
                tank1_state.pressure = tank1_target_pressure

                print(f"✅ Tank 1 corrected for 700 bar target:")
                print(f"   Original: P={tank1_original_pressure/1e5:.1f} bar, T={tank1_temperature:.1f} K, m={tank1_original_mass:.1f} kg")
                print(f"   Adjusted: P={tank1_target_pressure/1e5:.1f} bar, T={tank1_temperature:.1f} K, m={tank1_required_mass:.1f} kg")
                print(f"   Required density: {tank1_target_density:.1f} kg/m³")

            except Exception as e:
                print(f"⚠️ Could not adjust Tank 1 for 700 bar: {e}")

            # ===== Fix Tank 2: Achieve target 400 bar with ATR72 mass =====
            tank2_state = initial_state.tank_states[1]
            original_mass = tank2_state.fuel_mass
            original_pressure = tank2_state.pressure

            # Set the required mass
            tank2_state.fuel_mass = total_fuel_required

            # Calculate new density
            new_density = total_fuel_required / tank2_state.volume

            # Recalculate pressure at this density and temperature to maintain 400 bar
            # We want to start at 400 bar, so we need to adjust temperature to achieve this
            target_pressure = 400e5  # 400 bar
            current_temperature = tank2_state.temperature

            try:
                # Calculate what temperature gives us 400 bar at the required density
                adjusted_temperature = PropsSI('T', 'P', target_pressure, 'D', new_density, 'Hydrogen')

                # Update state with corrected values
                tank2_state.pressure = target_pressure
                tank2_state.temperature = adjusted_temperature
                tank2_state.solid_temperature = adjusted_temperature

                print(f"✅ Tank 2 corrected for ATR72 with 400 bar target:")
                print(f"   Original: P={original_pressure/1e5:.1f} bar, T={current_temperature:.1f} K, m={original_mass:.1f} kg")
                print(f"   Adjusted: P={target_pressure/1e5:.1f} bar, T={adjusted_temperature:.1f} K, m={total_fuel_required:.1f} kg")
                print(f"   Final density: {new_density:.1f} kg/m³")

            except Exception as e:
                print(f"⚠️ Could not adjust temperature for 400 bar target: {e}")
                print(f"   Using original temperature, pressure will be lower")
                # Recalculate pressure at original temperature
                try:
                    recalc_pressure = PropsSI('P', 'D', new_density, 'T', current_temperature, 'Hydrogen')
                    tank2_state.pressure = recalc_pressure
                    print(f"   Recalculated pressure: {recalc_pressure/1e5:.1f} bar")
                except:
                    print(f"   Could not recalculate pressure, keeping original")

            return initial_state

        self.physics_system._create_initial_state = atr72_initial_state

        print(f"✅ ATR72 flow profile configured for Tank 2")
        print(f"   Mission duration: {total_duration:.1f}s ({total_duration/3600:.2f}h)")
        print(f"   Total fuel required: {total_fuel_required:.2f} kg")

    def run_simulation(self, solver_method: str = "LSODA"):
        """Run simulation using the graph-configured physics system."""
        print(f"\n🚀 Running graph-configured simulation with {solver_method} solver...")
        results = self.physics_system.run_analysis(solver_method)
        print(f"✅ Graph-configured simulation completed successfully!")
        return results

    def plot_flow_rates(self, results, save_path: str = None) -> plt.Figure:
        """Plot simple two-tank flow rates: inflow (+), outflow (-), vent."""
        configure_plot_style()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        times_hours = [t / 3600 for t in results.times]

        # Extract flow data from stored tank states
        tank1_data = results._extract_tank_arrays(0)
        tank2_data = results._extract_tank_arrays(1)

        # Tank 1 Plot - Positive inflows, negative outflows
        inflow_total = tank1_data['inflow_rates'] + tank1_data['coupling_inflow_rates']
        outflow_total = -(tank1_data['outflow_rates'] + tank1_data['coupling_outflow_rates'])  # Make negative
        vent = -tank1_data['vent_rates']  # Make negative

        ax1.plot(times_hours, inflow_total, 'b-', label='Inflow', linewidth=2)
        ax1.plot(times_hours, outflow_total, 'r-', label='Outflow', linewidth=2)
        ax1.plot(times_hours, vent, 'g-', label='Vent', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_title('Tank 1 (CH2) Flow Rates')
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Flow Rate [g/s]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Tank 2 Plot - Positive inflows, negative outflows
        inflow_total = tank2_data['inflow_rates'] + tank2_data['coupling_inflow_rates']
        outflow_total = -(tank2_data['outflow_rates'] + tank2_data['coupling_outflow_rates'])  # Make negative
        vent = -tank2_data['vent_rates']  # Make negative

        ax2.plot(times_hours, inflow_total, 'b-', label='Inflow', linewidth=2)
        ax2.plot(times_hours, outflow_total, 'r-', label='Outflow (ATR72)', linewidth=2)
        ax2.plot(times_hours, vent, 'g-', label='Vent', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('Tank 2 (CCH2) Flow Rates')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Flow Rate [g/s]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Multi-Tank System Flow Analysis\nATR72 Mission Profile', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Flow analysis plot saved to: {save_path}")

        return fig

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
    graph, ch2_tank, cch2_tank, atr72_mission = create_dual_tank_network_config()

    # Display network summary
    print(f"\n📊 NETWORK SUMMARY:")
    print(f"   System: {graph.system_name}")
    print(f"   Tanks: {len(graph.tanks)}")
    print(f"   Connections: {len(graph.connections)}")

    # Create the graph-configured system with ATR72 mission
    system = GraphConfiguredMultiTankSystem(graph, cch2_tank, ch2_tank, atr72_mission)

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
            # Plot 1: Tank states evolution
            tank_states_fig = system.physics_system.plot_results("ch2_cch2_tank_states.png")

            # Plot 2: Flow rates analysis
            flow_fig = system.plot_flow_rates(results, "ch2_cch2_flow_analysis.png")

            # Plot 3: Network topology
            visualizer = TankNetworkVisualizer(graph)
            network_fig = visualizer.visualize_network(
                figsize=(12, 8),
                layout_method='spring',
                show_labels=True,
                show_connection_details=True,
                save_path="ch2_cch2_network_topology.png"
            )

            # Show all plots using plt.show() - this will display all open figures
            plt.show()

            print(f"   📊 Tank states plot saved: ch2_cch2_tank_states.png")
            print(f"   🔄 Flow analysis plot saved: ch2_cch2_flow_analysis.png")
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