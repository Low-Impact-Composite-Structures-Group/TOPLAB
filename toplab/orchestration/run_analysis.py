"""
Common driver function for multi-tank hydrogen storage analyses.

This module provides a unified execution function for all multi-tank analyses,
eliminating code duplication across driver scripts and ensuring consistent
output formatting.

Author: Dante Raso
"""

import sys
import time
import os
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional

from toplab.configuration.scenario_configuration import ScenarioConfig
from toplab.orchestration.system_orchestrator import SystemOrchestrator


def _is_output_silent(config: ScenarioConfig) -> bool:
    return bool(config.config_dict.get('output', {}).get('silent', False))


def _stdout_context(silent: bool):
    if not silent:
        return nullcontext()
    return open(os.devnull, 'w')


def run_analysis(
    config_path: Path,
    analysis_name: Optional[str] = None,
    show_material_props: bool = False,
    verbose: bool = False,
    verbosity: str = "summary"
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
    effective_verbosity = "debug" if verbose else (verbosity or "summary")

    # Validate configuration file exists
    if not config_path.exists():
        error_msg = f"ERROR: Configuration file not found: {config_path}"
        print(error_msg)
        return {'success': False, 'error': error_msg}

    # Load configuration
    try:
        config = ScenarioConfig.from_yaml(str(config_path))

    except Exception as e:
        error_msg = f"ERROR: Configuration loading failed: {e}"
        print(error_msg)
        return {'success': False, 'error': str(e)}

    silent_output = _is_output_silent(config)
    if silent_output:
        effective_verbosity = "quiet"

    if not silent_output:
        print("=" * 80)
        if analysis_name:
            print(f"{analysis_name.upper()}")
        print("=" * 80)
        print(f"Loading configuration: {config_path.name}")
        print(f"SUCCESS: Configuration loaded")
        print(f"\nAnalysis Configuration:")
        print(f"  Name: {config.analysis_name}")
        print(f"  Description: {config.description}")
        print(f"  Tanks: {config.get_tank_count()}")
        print(f"  Mission: {config.mission_sequence.missions[0].profile}")
        print(f"  Materials: {', '.join(config.materials.keys())}")

    try:
        with _stdout_context(silent_output) as stdout_target:
            stderr_context = redirect_stderr(stdout_target) if silent_output else nullcontext()
            stdout_redirect = redirect_stdout(stdout_target) if silent_output else nullcontext()
            with stdout_redirect, stderr_context:
                if not silent_output:
                    print(f"\nCreating System Orchestrator...")

                setup_start = time.time()
                orchestrator = SystemOrchestrator(config, verbosity=effective_verbosity)
                setup_time = time.time() - setup_start

                if not silent_output:
                    print(f"SUCCESS: Orchestrator created in {setup_time:.2f} seconds")

                if verbose and not silent_output:
                    orchestrator.print_scenario_summary()

                if show_material_props and not silent_output:
                    print(f"\nNIST Material Temperature Dependence:")
                    test_temps = [50, 100, 200, 300]
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

                if not silent_output:
                    print(f"\nRUNNING SIMULATION")
                    print("-" * 60)

                sim_start = time.time()
                results = orchestrator.run_simulation()
                sim_time = time.time() - sim_start

                if not silent_output:
                    print(f"SUCCESS: Simulation completed in {sim_time:.2f} seconds")

                validation = orchestrator.validate_results()

                if not silent_output:
                    print(f"\nSIMULATION RESULTS")
                    print("-" * 60)
                    print(f"  Mission Duration: {results.times[-1]:.1f} s ({results.times[-1]/3600:.2f} hours)")
                    print(f"  Data Points: {len(results.times)}")

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

                if not silent_output:
                    print(f"\nGenerating plots...")

                config_dict = config.config_dict
                output_path = config_dict.get('output', {}).get('plots', {}).get('save_path', 'output/plots')
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                save_path = output_dir / f"{config.analysis_name.replace(' ', '_')}_evolution.png"
                figures = orchestrator.generate_plots(save_path=str(save_path))

                if not silent_output:
                    if figures:
                        print(f"  SUCCESS: Generated {len(figures)} plots")
                        print(f"  Plots saved to: {output_dir}")
                    else:
                        print(f"  WARNING: Plot generation returned no figures")

                if not silent_output:
                    print(f"\nGenerating comprehensive results report...")

                report_file = orchestrator.save_comprehensive_results()
                if not silent_output:
                    print(f"  SUCCESS: Report saved to {report_file}")

    except Exception as e:
        error_msg = f"ERROR: Analysis execution failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

    execution_time = time.time() - start_time
    validation_passed = validation.get('overall', False)

    if not silent_output:
        print(f"\nANALYSIS COMPLETE")
        print(f"  Total execution time: {execution_time:.2f} seconds")
        print(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")

    return {
        'success': True,
        'results': results,
        'validation': validation,
        'execution_time': execution_time
    }
