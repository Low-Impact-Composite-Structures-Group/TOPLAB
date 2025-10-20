#!/usr/bin/env python3
"""
Single Tank SLH2 Benchmark - Orchestrated Analysis Driver

This driver demonstrates the new orchestrated multi-tank framework
using a single subcooled liquid hydrogen tank with ATR72 mission profile.

Key Features:
- Configuration-driven parameter specification
- Semi-exposed physics components for transparency
- NIST materials with temperature-dependent properties
- Direct access to all src/ components
- Comprehensive results validation

Author: Orchestrated Multi-Tank Framework
Date: September 2025
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
    """Main execution function for single tank SLH2 analysis."""

    print("=" * 80)
    print("SINGLE TANK SLH2 BENCHMARK - ORCHESTRATED FRAMEWORK")
    print("=" * 80)
    print("Configuration-driven subcooled liquid hydrogen analysis")
    print("Using semi-exposed src/ components with NIST materials")
    print("=" * 80)

    # Load configuration
    config_path = current_dir / "single_tank_slh2_config.yaml"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return

    print(f"📄 Loading configuration: {config_path.name}")

    try:
        config = ScenarioConfig.from_yaml(str(config_path))
        print(f"✅ Configuration loaded: {config}")

        # Display configuration summary
        print(f"\n📋 Analysis Configuration:")
        info = {
            'name': config.analysis_name,
            'description': config.description,
            'version': config.version
        }
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Tanks: {config.get_tank_count()}")
        print(f"   Mission: {config.mission_sequence.missions[0].profile}")
        print(f"   Materials: {list(config.materials.keys())}")

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return

    # Create orchestrator
    print(f"\n🔧 Creating System Orchestrator...")

    try:
        start_time = time.time()
        orchestrator = SystemOrchestrator(config)
        setup_time = time.time() - start_time

        print(f"✅ Orchestrator created in {setup_time:.2f} seconds")

        # Display component summary
        orchestrator.print_scenario_summary()

    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Demonstrate component accessibility
    print(f"\n🔍 COMPONENT ACCESSIBILITY DEMONSTRATION")
    print("-" * 50)
    print("All src/ components are directly accessible:")
    print(f"   Tank Count: {len(orchestrator.tank_geometries)}")
    if orchestrator.tank_geometries:
        print(f"   Tank Volume: {orchestrator.tank_geometries[0].volume:.4f} m³")
        if hasattr(orchestrator.tank_geometries[0], 'radius'):
            print(f"   Tank Radius: {orchestrator.tank_geometries[0].radius:.3f} m")
    print(f"   Mission Type: {config.mission_sequence.missions[0].type}")
    print(f"   Mission Profile: {config.mission_sequence.missions[0].profile}")
    print(f"   Materials: NIST-enabled with temperature dependence")

    # Show material temperature dependence
    print(f"\n🌡️ NIST Material Temperature Dependence:")
    test_temps = [50, 100, 200, 300]  # K
    print(f"   Temperature [K]: {' '.join([f'{T:6.0f}' for T in test_temps])}")

    try:
        if 'liner' in config.materials:
            liner_material = config.materials['liner']
            liner_cp = [liner_material.get_specific_heat(T) for T in test_temps]
            print(f"   Liner Cp [J/kg·K]: {' '.join([f'{cp:6.0f}' for cp in liner_cp])}")

        if 'composite' in config.materials:
            composite_material = config.materials['composite']
            composite_cp = [composite_material.get_specific_heat(T) for T in test_temps]
            print(f"   Composite Cp [J/kg·K]: {' '.join([f'{cp:6.0f}' for cp in composite_cp])}")

    except Exception as e:
        print(f"   ⚠️ Material property calculation failed: {e}")

    # Run simulation
    print(f"\n🚀 RUNNING SIMULATION")
    print("-" * 50)

    try:
        sim_start_time = time.time()
        results = orchestrator.run_simulation()
        sim_time = time.time() - sim_start_time

        print(f"✅ Simulation completed in {sim_time:.2f} seconds")

        # Validate results
        validation = orchestrator.validate_results()

        # Generate summary from TankSystem results
        print(f"\n📊 SIMULATION RESULTS SUMMARY")
        print("-" * 50)
        print(f"   Mission Duration: {results.times[-1]:.1f} s ({results.times[-1]/3600:.2f} hours)")
        print(f"   Data Points: {len(results.times)}")

        # Get initial and final states from the first tank
        if hasattr(results, 'multi_tank_states') and results.multi_tank_states:
            initial_multi_state = results.multi_tank_states[0]
            final_multi_state = results.multi_tank_states[-1]

            initial_tank_state = initial_multi_state.get_tank_state(0)
            final_tank_state = final_multi_state.get_tank_state(0)

            fuel_consumed = initial_tank_state.fuel_mass - final_tank_state.fuel_mass

            print(f"   Initial Mass: {initial_tank_state.fuel_mass:.2f} kg")
            print(f"   Final Mass: {final_tank_state.fuel_mass:.2f} kg")
            print(f"   Fuel Consumed: {fuel_consumed:.2f} kg")
            print(f"   Initial Temperature: {initial_tank_state.temperature:.1f} K")
            print(f"   Final Temperature: {final_tank_state.temperature:.1f} K")
            print(f"   Initial Pressure: {initial_tank_state.pressure/1e5:.1f} bar")
            print(f"   Final Pressure: {final_tank_state.pressure/1e5:.1f} bar")
            print(f"   Average Discharge: {fuel_consumed/results.times[-1]:.6f} kg/s")
        else:
            print("   ⚠️ Multi-tank state data not available in results")

        # Generate plots
        print(f"\n📊 Generating plots...")
        try:
            # Create output directory
            output_dir = Path("output/plots")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate plot with save path
            save_path = output_dir / f"{config.analysis_name.replace(' ', '_')}_evolution.png"
            figures = orchestrator.generate_plots(save_path=str(save_path))

            if figures:
                print(f"   ✓ Plot generation completed successfully")
                print(f"   Plots saved to: {output_dir}")
            else:
                print(f"   ⚠️ Plot generation returned no figures")

        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")

        # Generate comprehensive results report
        print(f"\n📋 Generating comprehensive results report...")
        try:
            report_file = orchestrator.save_comprehensive_results()
            print(f"   ✓ Comprehensive report generated successfully")
            print(f"   Report saved to: {report_file}")
        except Exception as e:
            print(f"   ⚠️ Comprehensive report generation failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"   Total execution time: {time.time() - start_time:.2f} seconds")
        print(f"   Validation: {'✅ PASSED' if validation['overall'] else '❌ FAILED'}")

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n🎉 Single Tank SLH2 Analysis Complete!")
    print("   Framework successfully demonstrates:")
    print("   • Configuration-driven parameter specification")
    print("   • Semi-exposed src/ component access")
    print("   • NIST materials with temperature dependence")
    print("   • Comprehensive validation and results")


if __name__ == "__main__":
    main()