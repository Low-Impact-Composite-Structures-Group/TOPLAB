"""
Common driver function for multi-tank hydrogen storage analyses.

This module provides a unified execution function for all multi-tank analyses,
eliminating code duplication across driver scripts and ensuring consistent
output formatting.

Author: Dante Raso
"""

import sys
import time
from pathlib import Path
from typing import Optional

from src.configuration.scenario_configuration import ScenarioConfig
from src.orchestration.system_orchestrator import SystemOrchestrator


def run_analysis(
    config_path: Path,
    analysis_name: Optional[str] = None,
    show_material_props: bool = False,
    verbose: bool = False
) -> dict:
    """
    Execute a multi-tank hydrogen storage analysis.

    This function provides the complete analysis workflow:
    1. Load YAML configuration
    2. Create system orchestrator
    3. Run simulation
    4. Validate results
    5. Generate plots
    6. Save comprehensive results report

    Args:
        config_path: Path to YAML configuration file
        analysis_name: Optional override for analysis display name
        show_material_props: If True, display material property temperature dependence
        verbose: If True, show additional diagnostic information

    Returns:
        dict: Analysis results containing:
            - 'success': bool
            - 'results': simulation results object (if successful)
            - 'validation': validation results dict
            - 'execution_time': total execution time in seconds
            - 'error': error message (if failed)
    """
    start_time = time.time()

    print("=" * 80)
    if analysis_name:
        print(f"{analysis_name.upper()}")
    print("=" * 80)

    # Validate configuration file exists
    if not config_path.exists():
        error_msg = f"ERROR: Configuration file not found: {config_path}"
        print(error_msg)
        return {'success': False, 'error': error_msg}

    print(f"Loading configuration: {config_path.name}")

    # Load configuration
    try:
        config = ScenarioConfig.from_yaml(str(config_path))
        print(f"SUCCESS: Configuration loaded")

        # Display configuration summary
        print(f"\nAnalysis Configuration:")
        print(f"  Name: {config.analysis_name}")
        print(f"  Description: {config.description}")
        print(f"  Tanks: {config.get_tank_count()}")
        print(f"  Mission: {config.mission_sequence.missions[0].profile}")
        print(f"  Materials: {', '.join(config.materials.keys())}")

    except Exception as e:
        error_msg = f"ERROR: Configuration loading failed: {e}"
        print(error_msg)
        return {'success': False, 'error': str(e)}

    # Create orchestrator
    print(f"\nCreating System Orchestrator...")

    try:
        setup_start = time.time()
        orchestrator = SystemOrchestrator(config)
        setup_time = time.time() - setup_start

        print(f"SUCCESS: Orchestrator created in {setup_time:.2f} seconds")

        if verbose:
            orchestrator.print_scenario_summary()

    except Exception as e:
        error_msg = f"ERROR: Orchestrator creation failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

    # Optional: Display material temperature dependence
    if show_material_props:
        print(f"\nNIST Material Temperature Dependence:")
        test_temps = [50, 100, 200, 300]  # K
        print(f"  Temperature [K]: {' '.join([f'{T:6.0f}' for T in test_temps])}")

        try:
            if 'liner' in config.materials:
                liner = config.materials['liner']
                liner_cp = [liner.get_specific_heat(T) for T in test_temps]
                print(f"  Liner Cp [J/kg·K]: {' '.join([f'{cp:6.0f}' for cp in liner_cp])}")

            if 'composite' in config.materials:
                composite = config.materials['composite']
                comp_cp = [composite.get_specific_heat(T) for T in test_temps]
                print(f"  Composite Cp [J/kg·K]: {' '.join([f'{cp:6.0f}' for cp in comp_cp])}")

        except Exception as e:
            print(f"  WARNING: Material property calculation failed: {e}")

    # Run simulation
    print(f"\nRUNNING SIMULATION")
    print("-" * 60)

    try:
        sim_start = time.time()
        results = orchestrator.run_simulation()
        sim_time = time.time() - sim_start

        print(f"SUCCESS: Simulation completed in {sim_time:.2f} seconds")

        # Validate results
        validation = orchestrator.validate_results()

        # Display results summary
        print(f"\nSIMULATION RESULTS")
        print("-" * 60)
        print(f"  Mission Duration: {results.times[-1]:.1f} s ({results.times[-1]/3600:.2f} hours)")
        print(f"  Data Points: {len(results.times)}")

        # Extract tank state information
        if hasattr(results, 'multi_tank_states') and results.multi_tank_states:
            initial_state = results.multi_tank_states[0].get_tank_state(0)
            final_state = results.multi_tank_states[-1].get_tank_state(0)
            fuel_consumed = initial_state.fuel_mass - final_state.fuel_mass

            print(f"  Initial Mass: {initial_state.fuel_mass:.2f} kg")
            print(f"  Final Mass: {final_state.fuel_mass:.2f} kg")
            print(f"  Fuel Consumed: {fuel_consumed:.2f} kg")
            print(f"  Initial Temperature: {initial_state.temperature:.1f} K")
            print(f"  Final Temperature: {final_state.temperature:.1f} K")
            print(f"  Initial Pressure: {initial_state.pressure/1e5:.1f} bar")
            print(f"  Final Pressure: {final_state.pressure/1e5:.1f} bar")
            print(f"  Average Discharge: {fuel_consumed/results.times[-1]:.6f} kg/s")

    except Exception as e:
        error_msg = f"ERROR: Simulation failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

    # Generate plots
    print(f"\nGenerating plots...")
    try:
        # Get output directory from config or use default
        config_dict = config.config_dict
        output_path = config_dict.get('output', {}).get('plots', {}).get('save_path', 'output/plots')
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"{config.analysis_name.replace(' ', '_')}_evolution.png"
        figures = orchestrator.generate_plots(save_path=str(save_path))

        if figures:
            print(f"  SUCCESS: Generated {len(figures)} plots")
            print(f"  Plots saved to: {output_dir}")
        else:
            print(f"  WARNING: Plot generation returned no figures")

    except Exception as e:
        print(f"  WARNING: Plot generation failed: {e}")

    # Save comprehensive results report
    print(f"\nGenerating comprehensive results report...")
    try:
        report_file = orchestrator.save_comprehensive_results()
        print(f"  SUCCESS: Report saved to {report_file}")
    except Exception as e:
        print(f"  WARNING: Report generation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    # Final summary
    execution_time = time.time() - start_time
    validation_passed = validation.get('overall', False)

    print(f"\nANALYSIS COMPLETE")
    print(f"  Total execution time: {execution_time:.2f} seconds")
    print(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")

    return {
        'success': True,
        'results': results,
        'validation': validation,
        'execution_time': execution_time
    }
