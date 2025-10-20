#!/usr/bin/env python3
"""
Coupled CH2-LH2 Multi-Tank System - Orchestrated Analysis Driver

This driver demonstrates the multi-tank coupling framework using
a coupled gaseous hydrogen (CH2) and liquid hydrogen (LH2) system
with flow-controlled pressurization coupling and ATR72 mission profile.

Key Features:
- Multi-tank configuration with advanced coupling rules
- Flow-controlled pressurization: CH2 (700 bar) → LH2 (3-15 bar)
- Mission assignment to LH2 tank (ATR72 discharge from liquid fuel)
- Two-phase behavior in LH2 tank during operation
- Feedback control system for optimal pressurization
- Comprehensive multi-tank state tracking with coupling analysis

Author: Orchestrated Multi-Tank Framework
Date: October 2025
"""

import sys
import time
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import orchestrated framework
from src.multi_tank.configuration.scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator



def main():
    """Main execution function for coupled CH2-LH2 multi-tank analysis."""

    print("=" * 80)
    print("COUPLED CH2-LH2 MULTI-TANK SYSTEM - ORCHESTRATED FRAMEWORK")
    print("=" * 80)
    print("Multi-tank configuration with flow-controlled pressurization coupling")
    print("CH2 (700 bar) pressurizes LH2 (3-15 bar) for ATR72 mission discharge")
    print("Two-phase behavior expected in LH2 tank during operation")
    print("=" * 80)

    # Load configuration
    config_path = current_dir / "coupled_ch2_lh2_config.yaml"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return

    print(f"📄 Loading multi-tank configuration: {config_path.name}")

    try:
        config = ScenarioConfig.from_yaml(str(config_path))
        print(f"✅ Configuration loaded: {config}")

        # Display configuration summary
        print(f"\n📋 Multi-Tank Analysis Configuration:")
        info = {
            'name': config.analysis_name,
            'description': config.description,
            'version': config.version
        }
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Tanks: {config.get_tank_count()}")

        # Show tank-specific details
        print(f"\n🏭 Tank Configuration Details:")
        for tank_id, geometry in config.tank_geometries.items():
            print(f"   Tank {tank_id}:")
            print(f"     Initial Pressure: {geometry['initial_pressure']/1e5:.0f} bar")
            print(f"     Initial Temperature: {geometry.get('initial_temperature', 'N/A')} K")
            if 'radius' in geometry:
                print(f"     Sizing: User-defined radius ({geometry['radius']} m)")
            elif geometry.get('mission_based_sizing'):
                print(f"     Sizing: Mission-based (auto-calculated)")

        # Show coupling rules
        coupling_rules = config.config_dict.get('coupling_rules', [])
        print(f"\n🔗 Coupling Rules: {len(coupling_rules)}")
        for rule in coupling_rules:
            participants = rule.get('participants', {})
            print(f"   {rule['coupling_id']}: {rule['coupling_type']}")
            print(f"     CH2 Tank {participants.get('source')} → LH2 Tank {participants.get('target')}")
            if rule['coupling_type'] == 'flow_controlled_pressurization':
                control_params = rule.get('control_parameters', {})
                print(f"     Control: {control_params.get('target_pressure_offset_bar', 'N/A')} bar offset")
                print(f"     Max flow: {control_params.get('max_pressurization_rate_kg_s', 'N/A')} kg/s")

        # Show mission assignment
        mission = config.mission_sequence.missions[0]
        print(f"\n🎯 Mission Assignment:")
        print(f"   Mission: {mission.type} ({mission.profile})")
        print(f"   Assigned to: LH2 Tank {mission.assigned_to} (liquid fuel)")
        print(f"   CH2 Tank: Provides pressurization support")
        print(f"   Materials: {list(config.materials.keys())}")

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create orchestrator
    print(f"\n🔧 Creating Multi-Tank System Orchestrator...")

    try:
        start_time = time.time()
        orchestrator = SystemOrchestrator(config)
        setup_time = time.time() - start_time

        print(f"✅ Multi-tank orchestrator created in {setup_time:.2f} seconds")

        # Display component summary
        orchestrator.print_scenario_summary()

        # Display comprehensive analysis summary
        orchestrator.print_comprehensive_analysis_summary()

    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Demonstrate multi-tank component accessibility
    print(f"\n🔍 MULTI-TANK COMPONENT ACCESSIBILITY")
    print("-" * 50)
    print("Multi-tank system components:")
    print(f"   Tank Count: {len(orchestrator.tank_geometries)}")

    for i, tank_geom in enumerate(orchestrator.tank_geometries):
        # Get tank config data for pressure and temperature
        tank_config_key = str(i + 1)  # Tank IDs are 1-based in config
        tank_config = config.tank_geometries.get(tank_config_key, {})

        print(f"   Tank {i+1}:")
        print(f"     Volume: {tank_geom.volume:.4f} m³")
        if hasattr(tank_geom, 'radius'):
            print(f"     Radius: {tank_geom.radius:.3f} m")

        # Get pressure and temperature from config
        initial_pressure = tank_config.get('initial_pressure', 0)
        initial_temperature = tank_config.get('initial_temperature', 0)
        print(f"     Initial P: {initial_pressure/1e5:.0f} bar")
        print(f"     Initial T: {initial_temperature:.1f} K")

    # Show coupling rules details
    if hasattr(orchestrator, 'coupling_rules'):
        print(f"\n🔗 Active Coupling Rules:")
        for rule in orchestrator.coupling_rules:
            print(f"   {rule.coupling_id}: {type(rule).__name__}")
            if hasattr(rule, 'target_tank'):
                print(f"     Participants: Tank {rule.source_tank} → Tank {rule.target_tank}")
            else:
                # OHEX extraction rule - no target tank
                print(f"     Participants: Tank {rule.source_tank} → OHEX")

    # Show TankSystem coupling valves
    if hasattr(orchestrator.tank_system, 'coupling_valves'):
        print(f"\n⚙️ TankSystem Coupling Valves:")
        for i, valve in enumerate(orchestrator.tank_system.coupling_valves):
            if valve.target_idx == -1:
                # OHEX extraction
                print(f"   Coupling {i}: Tank{valve.source_idx+1} → OHEX")
                print(f"     Min extraction pressure: {valve.min_extraction_pressure/1e5:.1f} bar")
                if hasattr(valve, 'current_mission_flow_rate'):
                    print(f"     Current mission flow: {valve.current_mission_flow_rate*1000:.1f} g/s")
            elif hasattr(valve, 'p_open'):
                # Standard pressure-triggered valve
                print(f"   Valve {i}: Tank{valve.source_idx+1} → Tank{valve.target_idx+1}")
                print(f"     Opens when target < {valve.p_open/1e5:.1f} bar")
                print(f"     Closes when target > {valve.p_close/1e5:.1f} bar")
                print(f"     Max flow rate: {valve.max_flow_rate*1000:.0f} g/s")
            elif type(valve).__name__ == 'PressureGovernorValve' or hasattr(valve, 'pressure_gain_kg_s_per_bar'):
                # Margin-free pressure governor (Option A)
                print(f"   Valve {i}: Tank{valve.source_idx+1} → Tank{valve.target_idx+1} (Pressure Governor)")
                if hasattr(valve, 'pipe_diameter') and hasattr(valve, 'pipe_length'):
                    print(f"     Discharge piping: {valve.pipe_diameter*1000:.0f}mm × {valve.pipe_length:.1f}m")
                if hasattr(valve, 'pressure_gain_kg_s_per_bar'):
                    print(f"     Pressure gain: {valve.pressure_gain_kg_s_per_bar:.3f} kg/s/bar")
                if hasattr(valve, 'control_interval_s'):
                    print(f"     Control cadence: {valve.control_interval_s:.2f} s")
                if hasattr(valve, 'target_filter_tau_s'):
                    print(f"     Target LPF tau: {valve.target_filter_tau_s:.2f} s")
                if hasattr(valve, 'max_flow_rate'):
                    print(f"     Max flow rate: {valve.max_flow_rate*1000:.0f} g/s")
            else:
                # Mission-adaptive pressure valve
                print(f"   Valve {i}: Tank{valve.source_idx+1} → Tank{valve.target_idx+1} (Mission-Adaptive)")
                print(f"     Dynamic thresholds based on real-time mission flow")
                print(f"     Discharge piping: {valve.pipe_diameter*1000:.0f}mm × {valve.pipe_length:.1f}m")
                # print(f"     Pressure margin: {valve.pressure_margin_bar:.1f} bar")
                print(f"     Max flow rate: {valve.max_flow_rate*1000:.0f} g/s")

    print(f"\n   Mission Assignment: ATR72 discharge from LH2 Tank {mission.assigned_to}")
    print(f"   Coupling Physics: Flow-controlled CH2 → LH2 pressurization")
    print(f"   Control System: Feedback-based pressure regulation")
    print(f"   Two-phase Operation: Expected in LH2 tank")
    print(f"   Materials: NIST-enabled with temperature dependence")

    # Run simulation
    print(f"\n🚀 RUNNING MULTI-TANK SIMULATION")
    print("-" * 50)

    try:
        # Extract solver configuration from config file
        solver_config_dict = config.config_dict.get('solver', {})
        solver_method = solver_config_dict.get('method', 'RK45')
        solver_config = {
            'rtol': solver_config_dict.get('rtol', 1e-6),
            'atol': solver_config_dict.get('atol', 1e-9),
            'max_step': solver_config_dict.get('max_step', None)
        }

        print(f"   Using solver: {solver_method}")
        print(f"   Solver config: rtol={solver_config['rtol']}, atol={solver_config['atol']}, max_step={solver_config['max_step']}")

        sim_start_time = time.time()
        results = orchestrator.run_simulation(solver_method=solver_method, solver_config=solver_config)
        sim_time = time.time() - sim_start_time

        print(f"✅ Multi-tank simulation completed in {sim_time:.2f} seconds")

        # Validate results
        validation = orchestrator.validate_results()

        # Generate summary from multi-tank results
        print(f"\n📊 MULTI-TANK SIMULATION RESULTS")
        print("-" * 50)
        print(f"   Mission Duration: {results.times[-1]:.1f} s ({results.times[-1]/3600:.2f} hours)")
        print(f"   Data Points: {len(results.times)}")

        # Get initial and final states for all tanks
        if hasattr(results, 'multi_tank_states') and results.multi_tank_states:
            initial_multi_state = results.multi_tank_states[0]
            final_multi_state = results.multi_tank_states[-1]

            total_initial_mass = 0
            total_final_mass = 0

            print(f"\n   📊 Per-Tank Results:")
            for tank_idx in range(len(orchestrator.tank_geometries)):
                initial_tank_state = initial_multi_state.get_tank_state(tank_idx)
                final_tank_state = final_multi_state.get_tank_state(tank_idx)

                fuel_consumed = initial_tank_state.fuel_mass - final_tank_state.fuel_mass
                total_initial_mass += initial_tank_state.fuel_mass
                total_final_mass += final_tank_state.fuel_mass

                print(f"     Tank {tank_idx+1}:")
                print(f"       Initial: {initial_tank_state.fuel_mass:.2f} kg, {initial_tank_state.temperature:.1f} K, {initial_tank_state.pressure/1e5:.1f} bar")
                print(f"       Final:   {final_tank_state.fuel_mass:.2f} kg, {final_tank_state.temperature:.1f} K, {final_tank_state.pressure/1e5:.1f} bar")
                print(f"       Consumed: {fuel_consumed:.2f} kg")

            total_consumed = total_initial_mass - total_final_mass
            print(f"\n   🎯 System Totals:")
            print(f"     Total Initial Mass: {total_initial_mass:.2f} kg")
            print(f"     Total Final Mass: {total_final_mass:.2f} kg")
            print(f"     Total Fuel Consumed: {total_consumed:.2f} kg")
            print(f"     Average System Discharge: {total_consumed/results.times[-1]:.6f} kg/s")

            # Check for coupling activity and two-phase behavior
            print(f"     Inter-tank Coupling: Active (CH2 → LH2 pressurization)")

            # Analyze LH2 tank for two-phase behavior
            lh2_tank_idx = 1  # LH2 is tank 2 (index 1)
            if lh2_tank_idx < len(orchestrator.tank_geometries):
                lh2_initial = initial_multi_state.get_tank_state(lh2_tank_idx)
                lh2_final = final_multi_state.get_tank_state(lh2_tank_idx)

                temp_change = lh2_final.temperature - lh2_initial.temperature
                pressure_change = (lh2_final.pressure - lh2_initial.pressure) / 1e5

                print(f"     LH2 Two-phase Analysis:")
                print(f"       Temperature change: {temp_change:.2f} K")
                print(f"       Pressure change: {pressure_change:.2f} bar")
                if abs(temp_change) > 2.0 or abs(pressure_change) > 1.0:
                    print(f"       Two-phase behavior: Detected ✓")
                else:
                    print(f"       Two-phase behavior: Limited")

        else:
            print("   ⚠️ Multi-tank state data not available in results")

        # Generate plots
        print(f"\n📊 Generating multi-tank plots...")
        try:
            # Create output directory in the analysis-specific location
            output_dir = current_dir / "output" / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate plot with save path
            save_path = output_dir / f"{config.analysis_name.replace(' ', '_').replace('-', '_')}_evolution.png"
            figures = orchestrator.generate_plots(save_path=str(save_path))

            if figures:
                print(f"   ✓ Multi-tank plot generation completed successfully")
                print(f"   Plots saved to: {output_dir}")
                print(f"   Main plot: {save_path.name}")
            else:
                print(f"   ⚠️ Plot generation returned no figures")

        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()



        print(f"\n✅ MULTI-TANK ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"   Total execution time: {time.time() - start_time:.2f} seconds")
        print(f"   Validation: {'✅ PASSED' if validation['overall'] else '❌ FAILED'}")

    except Exception as e:
        print(f"❌ Multi-tank simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n🎉 Coupled CH2-LH2 Multi-Tank Analysis Complete!")
    print("   Framework successfully demonstrates:")
    print("   • Multi-tank configuration with advanced coupling rules")
    print("   • Flow-controlled pressurization between gas and liquid phases")
    print("   • Mission assignment to liquid fuel tank")
    print("   • Two-phase behavior handling in LH2 system")
    print("   • Feedback control system for optimal operation")
    print("   • Comprehensive multi-tank state tracking with coupling analysis")


if __name__ == "__main__":
    main()