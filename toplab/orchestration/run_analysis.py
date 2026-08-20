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

    packaging_result = None  # populated below if 'packaging:' block is present

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

                # --- Optional: aft fuselage packaging ---
                packaging_cfg = config.config_dict.get('packaging')
                packaging_result = None
                if packaging_cfg:
                    if not silent_output:
                        print(f"\nAFT FUSELAGE PACKAGING")
                        print("-" * 60)
                    packaging_result = _run_aft_packaging(
                        orchestrator=orchestrator,
                        packaging_cfg=packaging_cfg,
                        report_file=report_file,
                        config_path=config_path,
                        silent=silent_output,
                    )

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
        'execution_time': execution_time,
        'packaging': packaging_result,
    }


# ---------------------------------------------------------------------------
# Aft fuselage packaging helper
# ---------------------------------------------------------------------------

def _run_aft_packaging(
    orchestrator: SystemOrchestrator,
    packaging_cfg: dict,
    report_file: str | None,
    config_path: Path,
    *,
    silent: bool = False,
) -> dict:
    """Run the aft fuselage packaging step and append results to the report.

    Returns a dict with keys ``feasible``, ``placements``, and ``result``.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for file output
    import matplotlib.pyplot as plt

    from toplab.packaging.aft_placement import (
        AftFuselageDimensions,
        place_tanks_in_aft,
        plot_aft_placement,
    )

    # --- Parse aft fuselage dimensions ---
    dims_cfg = packaging_cfg.get("aft_fuselage_dimensions", {})
    try:
        dims = AftFuselageDimensions(
            d1=float(dims_cfg["d1"]),
            d2=float(dims_cfg["d2"]),
            d3=float(dims_cfg["d3"]),
            l1=float(dims_cfg["l1"]),
            l2=float(dims_cfg["l2"]),
            l3=float(dims_cfg["l3"]),
            epsilon=float(dims_cfg.get("epsilon", 0.05)),
        )
    except KeyError as exc:
        print(f"  ERROR: Missing required aft_fuselage_dimensions key: {exc}")
        return {"feasible": False, "placements": [], "result": None}

    generate_3d_plot = bool(packaging_cfg.get("generate_3d_plot", False))
    save_pickle = bool(packaging_cfg.get("save_pickle", False))

    # --- Gather outer tank dimensions ---
    outer_radii: list[float] = []
    half_cyl_lengths: list[float] = []

    for i, tank_geom in enumerate(orchestrator.tank_geometries):
        try:
            tank_props = orchestrator.tank_system._get_tank_properties(
                tank_geom, f"Tank{i + 1}", i
            )
            outer_radius = float(tank_props["outer_diameter"]) / 2.0
        except Exception:
            outer_radius = float(tank_geom.radius)
            if not silent:
                print(
                    f"  WARNING: Could not compute outer radius for tank {i + 1}; "
                    "using inner radius as fallback."
                )
        outer_radii.append(outer_radius)
        half_cyl_lengths.append(tank_geom.cylindrical_section_length / 2.0)

    n_tanks = len(outer_radii)

    if not silent:
        for i in range(n_tanks):
            total_outer_length = 2.0 * (outer_radii[i] + half_cyl_lengths[i])
            print(
                f"  Tank {i + 1}: outer radius = {outer_radii[i]:.3f} m, "
                f"total outer length = {total_outer_length:.3f} m"
            )
        print(
            f"  Aft dimensions: d1={dims.d1} m, d2={dims.d2} m, d3={dims.d3} m, "
            f"l1={dims.l1} m, l2={dims.l2} m, l3={dims.l3} m, ε={dims.epsilon} m"
        )

    # --- Run placement ---
    result = place_tanks_in_aft(
        outer_radii=outer_radii,
        half_cyl_lengths=half_cyl_lengths,
        dims=dims,
    )

    if not silent:
        status = "FEASIBLE" if result.feasible else "INFEASIBLE"
        print(f"\n  Placement result: {status}")
        print(f"  {result.message}")
        for p in result.placements:
            half_total = result.half_outer_lengths[p.tank_index]
            x_start = p.x_center - half_total
            x_end = p.x_center + half_total
            viol_str = (
                f"violation = {p.max_violation:.4f} m" if not p.feasible else "OK"
            )
            print(
                f"    Tank {p.tank_index + 1}: x_centre = {p.x_center:.3f} m  "
                f"[{x_start:.3f}, {x_end:.3f}]  {viol_str}"
            )

    # --- Append packaging section to the results report ---
    if report_file:
        try:
            _append_packaging_report(result, report_file)
            if not silent:
                print(f"\n  Packaging summary appended to: {report_file}")
        except Exception as exc:
            if not silent:
                print(f"  WARNING: Could not append packaging report: {exc}")

    # --- Optional 3-D plot ---
    if generate_3d_plot:
        try:
            fig, _ = plot_aft_placement(result)
            plot_dir = Path(config_path).parent / "output" / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "aft_placement_3d.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")

            if save_pickle:
                import pickle
                pkl_path = plot_path.with_suffix(".pkl")
                try:
                    with open(pkl_path, "wb") as fh:
                        pickle.dump(fig, fh)
                except Exception as pkl_exc:
                    if not silent:
                        print(f"  WARNING: Pickle save failed: {pkl_exc}")
            plt.close(fig)
        except Exception as exc:
            if not silent:
                print(f"  WARNING: 3-D plot generation failed: {exc}")

    return {
        "feasible": result.feasible,
        "placements": [
            {"tank_index": p.tank_index, "x_center": p.x_center, "feasible": p.feasible}
            for p in result.placements
        ],
        "result": result,
    }


def _append_packaging_report(result, report_file: str) -> None:
    """Append a packaging summary block to an existing results text report."""
    from toplab.packaging.aft_placement import AftFuselageDimensions

    dims = result.dims
    sep = "=" * 80

    lines = [
        "",
        sep,
        "AFT FUSELAGE PACKAGING",
        sep,
        "",
        "Aft Fuselage Dimensions",
        "-" * 40,
        f"  d1 (aft bulkhead diameter)  : {dims.d1:.4f} m",
        f"  d2 (intermediate diameter)  : {dims.d2:.4f} m",
        f"  d3 (forward tip diameter)   : {dims.d3:.4f} m",
        f"  l1 (cylinder length)        : {dims.l1:.4f} m",
        f"  l2 (first cone length)      : {dims.l2:.4f} m",
        f"  l3 (second cone length)     : {dims.l3:.4f} m",
        f"  epsilon (clearance margin)  : {dims.epsilon:.4f} m",
        f"  Total aft length            : {dims.total_length:.4f} m",
        "",
        "Tank Placement",
        "-" * 40,
    ]

    for p in result.placements:
        half_total = result.half_outer_lengths[p.tank_index]
        x_start = p.x_center - half_total
        x_end = p.x_center + half_total
        R_out = result.outer_radii[p.tank_index]
        status = "FEASIBLE" if p.feasible else f"INFEASIBLE (violation = {p.max_violation:.4f} m)"
        lines += [
            f"  Tank {p.tank_index + 1}:",
            f"    Outer radius              : {R_out:.4f} m",
            f"    Total outer length        : {2.0 * half_total:.4f} m",
            f"    x_centre (from aft datum) : {p.x_center:.4f} m",
            f"    x extent                  : [{x_start:.4f}, {x_end:.4f}] m",
            f"    Status                    : {status}",
        ]

    lines += [
        "",
        f"Overall result: {'FEASIBLE' if result.feasible else 'INFEASIBLE'}",
        f"Message: {result.message}",
        "",
    ]

    with open(report_file, "a") as fh:
        fh.write("\n".join(lines) + "\n")
