#!/usr/bin/env python3
"""
Single Tank LH2 Configuration - Orchestrated Analysis Driver

This driver demonstrates the single-tank framework with low-pressure
liquid hydrogen (LH2) configuration for two-phase operation.

Key Features:
- Low pressure operation (1-15 bar)
- Two-phase behavior expected
- Cryogenic temperatures (18-25K)
- Lower pressure than sLH2 (subcooled LH2)

Author: Orchestrated Framework
Date: October 2025
"""

import sys
import time
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import orchestrated framework
from src.configuration.scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator


def main():
    """Main execution function for single tank LH2 analysis."""

    print("=" * 80)
    print("SINGLE TANK LH2 CONFIGURATION - ORCHESTRATED FRAMEWORK")
    print("=" * 80)
    print("Low-pressure liquid hydrogen with two-phase behavior")
    print("Pressure range: 1-15 bar, Temperature range: 18-25K")
    print("=" * 80)

    # Load configuration
    config_path = current_dir / "single_tank_lh2_config.yaml"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return

    print(f"📄 Loading LH2 configuration: {config_path.name}")

    try:
        config = ScenarioConfig.from_yaml(str(config_path))
        print(f"✅ Configuration loaded: {config}")

        # Display configuration summary
        print(f"\n📋 LH2 Analysis Configuration:")
        info = {
            'name': config.analysis_name,
            'description': config.description,
            'version': config.version
        }
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Tanks: {config.get_tank_count()}")

        # Show tank-specific details for LH2
        tank_geometry = config.tank_geometries[1]  # Key is converted to int in parsing
        print(f"\n🏭 LH2 Tank Configuration:")
        print(f"   Initial Pressure: {float(tank_geometry['initial_pressure'])/1e5:.1f} bar")
        print(f"   Initial Temperature: {float(tank_geometry['initial_temperature']):.1f} K")
        print(f"   Initial Density: {float(tank_geometry['initial_density']):.1f} kg/m³")
        print(f"   Venting Pressure: {float(tank_geometry['venting_pressure'])/1e5:.1f} bar")
        print(f"   Minimum Pressure: {float(tank_geometry['minimum_pressure'])/1e5:.1f} bar")

        # Show mission details
        mission = config.mission_sequence.missions[0]
        print(f"\n🎯 Mission Configuration:")
        print(f"   Mission: {mission.type} ({mission.profile})")
        print(f"   Ambient Temperature: {mission.ambient_temperature} K")
        print(f"   Materials: {list(config.materials.keys())}")

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create orchestrator
    print(f"\n🔧 Creating LH2 System Orchestrator...")

    try:
        start_time = time.time()
        orchestrator = SystemOrchestrator(config)
        setup_time = time.time() - start_time

        print(f"✅ LH2 orchestrator created in {setup_time:.2f} seconds")

        # Display component summary
        orchestrator.print_scenario_summary()

        # Display comprehensive analysis summary
        orchestrator.print_comprehensive_analysis_summary()

    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Show LH2-specific configuration details
    print(f"\n🔍 LH2 SYSTEM ANALYSIS")
    print("-" * 50)
    print("LH2 system characteristics:")

    tank_geom = orchestrator.tank_geometries[0]
    tank_config = config.tank_geometries[1]  # Key is converted to int in parsing

    print(f"   Tank Volume: {tank_geom.volume:.4f} m³")
    print(f"   Operating Pressure Range: {float(tank_config['minimum_pressure'])/1e5:.1f} - {float(tank_config['venting_pressure'])/1e5:.1f} bar")
    print(f"   Operating Temperature: ~{float(tank_config['initial_temperature']):.1f} K")
    print(f"   Two-phase behavior: Expected during operation")
    print(f"   Storage type: Low-pressure liquid hydrogen")

    # Run simulation
    print(f"\n🚀 RUNNING LH2 SIMULATION")
    print("-" * 50)

    try:
        # Extract solver configuration
        solver_config_dict = config.config_dict.get('solver', {})
        solver_method = solver_config_dict.get('method', 'RK45')
        solver_config = {
            'rtol': solver_config_dict.get('rtol', 1e-6),
            'atol': solver_config_dict.get('atol', 1e-9),
            'max_step': solver_config_dict.get('max_step', None)
        }

        print(f"   Using solver: {solver_method}")
        print(f"   Solver config: rtol={solver_config['rtol']}, atol={solver_config['atol']}")

        sim_start_time = time.time()
        results = orchestrator.run_simulation(solver_method=solver_method, solver_config=solver_config)
        sim_time = time.time() - sim_start_time

        print(f"✅ LH2 simulation completed in {sim_time:.2f} seconds")

        # Validate results
        validation = orchestrator.validate_results()

        # Generate summary
        print(f"\n📊 LH2 SIMULATION RESULTS")
        print("-" * 50)
        print(f"   Mission Duration: {results.times[-1]:.1f} s ({results.times[-1]/3600:.2f} hours)")
        print(f"   Data Points: {len(results.times)}")

        # Tank state summary
        if hasattr(results, 'multi_tank_states') and results.multi_tank_states:
            initial_state = results.multi_tank_states[0].get_tank_state(0)
            final_state = results.multi_tank_states[-1].get_tank_state(0)

            fuel_consumed = initial_state.fuel_mass - final_state.fuel_mass

            print(f"\n   🏭 LH2 Tank Results:")
            print(f"     Initial: {initial_state.fuel_mass:.2f} kg, {initial_state.temperature:.1f} K, {initial_state.pressure/1e5:.1f} bar")
            print(f"     Final:   {final_state.fuel_mass:.2f} kg, {final_state.temperature:.1f} K, {final_state.pressure/1e5:.1f} bar")
            print(f"     Consumed: {fuel_consumed:.2f} kg")
            print(f"     Average Flow: {fuel_consumed/results.times[-1]:.6f} kg/s")

            # Check for two-phase behavior indicators
            if hasattr(results, 'temperatures') and hasattr(results, 'pressures'):
                temp_range = max(results.temperatures) - min(results.temperatures)
                pressure_range = (max(results.pressures) - min(results.pressures)) / 1e5

                print(f"\n   📈 Two-phase Analysis:")
                print(f"     Temperature range: {temp_range:.2f} K")
                print(f"     Pressure range: {pressure_range:.2f} bar")
                if temp_range > 5.0 or pressure_range > 2.0:
                    print(f"     Two-phase behavior: Detected ✓")
                else:
                    print(f"     Two-phase behavior: Limited")
            else:
                print(f"\n   📈 Two-phase Analysis:")
                print(f"     Temperature and pressure data not available in results")
                print(f"     Two-phase behavior: Analysis not possible")

        # Generate plots
        print(f"\n📊 Generating LH2 plots...")
        try:
            # Create output directory in the analysis-specific location
            output_dir = current_dir / "output" / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            save_path = output_dir / f"{config.analysis_name.replace(' ', '_')}_evolution.png"
            figures = orchestrator.generate_plots(save_path=str(save_path))

            if figures:
                print(f"   ✓ LH2 plot generation completed successfully")
                print(f"   Plots saved to: {output_dir}")
                print(f"   Main plot: {save_path.name}")
            else:
                print(f"   ⚠️ Plot generation returned no figures")

        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n✅ LH2 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"   Total execution time: {time.time() - start_time:.2f} seconds")
        print(f"   Validation: {'✅ PASSED' if validation['overall'] else '❌ FAILED'}")

    except Exception as e:
        print(f"❌ LH2 simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n🎉 Single Tank LH2 Analysis Complete!")
    print("   Framework successfully demonstrates:")
    print("   • Low-pressure liquid hydrogen operation")
    print("   • Two-phase behavior handling")
    print("   • Cryogenic temperature management")
    print("   • Comprehensive state tracking")


if __name__ == "__main__":
    main()