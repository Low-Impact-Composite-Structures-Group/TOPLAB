"""Figure generation for a finished run."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from CoolProp.CoolProp import PropsSI

from toplab.utilities.output_control import silent_when_quiet
from toplab.post.hex import calculate_ohex_requirements, calculate_ihex_requirements, compute_per_stream_outflows
from toplab.post.streams import coupling_stream_history
from toplab.orchestration.run_result import RunResult


@silent_when_quiet
def generate_plots(run: RunResult, save_path: str = None) -> Any:
    """
    Generate plots from simulation results.

    Args:
        save_path: Optional path to save plot file

    Returns:
        Figure object from plotting
    """
    if not run.results:
        raise ValueError("No results available. Run simulation first.")

    print("Generating multi-tank analysis plots...")

    try:
        # Import the new plotting module
        from toplab.plotting.multi_tank_plotting import DelftColourPlotter

        # Get plotting configuration from config
        plot_config = run.config.output_plots
        style_config = plot_config.get('style', {})

        # Get output formats from config (defaults to ['png'] if not specified)
        output_config = run.config.output
        output_formats = output_config.get('plot_formats', ['png'])
        enable_pgf = 'pgf' in output_formats

        if enable_pgf:
            print(f"   PGF output enabled for LaTeX compatibility")

        # Load font configuration and update global font settings
        font_config = style_config.get('font', {})
        if font_config:
            from toplab.plotting.plot_style_sb import update_font_settings

            master_size = font_config.get('master_size')
            legend_size = font_config.get('legend_size')
            font_name = font_config.get('font_name')

            # Update global font settings if specified in config
            update_font_settings(
                master_size=master_size,
                legend_size=legend_size,
                font_name=font_name
            )

            print(f"   Font configuration loaded: master={master_size}, legend={legend_size}, font='{font_name}'")

        # Create plotter with analysis name and styling options from config
        plotter = DelftColourPlotter(
            analysis_name=run.config.name,
            use_greyscale=style_config.get('use_greyscale', False),
            enable_multi_tank_overlay=style_config.get('enable_multi_tank_overlay', False),
            output_formats=output_formats,
            enable_pgf=enable_pgf
        )

        # Get figure_prefix from config (if specified)
        figure_prefix = plot_config.get('figure_prefix', None)
        if figure_prefix:
            print(f"   Using figure prefix from config: '{figure_prefix}'")

        return generate_single_mission_plots(run, plotter, save_path, figure_prefix)

    except Exception as e:
        print(f"   WARNING: Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_single_mission_plots(run: RunResult, plotter, save_path: str = None, figure_prefix: str = None):
    """Generate plots for single mission analysis."""
    figures = []

    # Check if multi-tank overlay is enabled and we have multiple tanks
    num_tanks = len(run.tank_geometries)
    use_overlay = plotter.enable_multi_tank_overlay and num_tanks > 1

    if use_overlay:
        print(f"   Plotting multi-tank evolution (overlay mode, {num_tanks} tanks)...")

        # Create reference lines from first tank configuration (for overlay mode)
        tank_config_data = run.config.tanks[0].geometry_data()
        ref_line_config = run.config.output_plots.get('reference_lines', {})
        reference_lines = plotter.create_reference_lines_from_config(tank_config_data, ref_line_config)

        # Add mission ambient temperature if available
        mission = run.config.missions[0]
        reference_lines['T_ambient'] = mission.ambient_temperature

        # Generate single overlay plot for all tanks
        overlay_save_path = None
        if save_path:
            from pathlib import Path
            save_dir = Path(save_path).parent
            save_ext = Path(save_path).suffix or '.png'
            base_name = figure_prefix if figure_prefix else Path(save_path).stem
            overlay_save_path = save_dir / f"{base_name}_evolution_all_tanks{save_ext}"

        # Get event lines configuration from tank_states config
        tank_states_config = run.config.output_plots.get('tank_states', {})
        event_lines = tank_states_config.get('event_lines', [])

        # Get axis limits and legend locations from config
        xlim = tank_states_config.get('xlim', None)
        ylim_pressure = tank_states_config.get('ylim_pressure', None)
        ylim_temperature = tank_states_config.get('ylim_temperature', None)
        ylim_density = tank_states_config.get('ylim_density', None)
        legend_locations = tank_states_config.get('legend_locations', None)

        fig = plotter.plot_tank_evolution(
            results=run.results,
            tank_index=0,  # Not used in overlay mode
            reference_lines=reference_lines,
            reference_lines_config=ref_line_config,
            event_lines=event_lines,
            save_path=str(overlay_save_path) if overlay_save_path else None,
            overlay_all_tanks=True,
            xlim=xlim,
            ylim_pressure=ylim_pressure,
            ylim_temperature=ylim_temperature,
            ylim_density=ylim_density,
            legend_locations=legend_locations
        )
        figures.append(fig)

    # Generate per-tank plots (individual evolution plots if not overlaying, plus other plot types)
    for tank_idx in range(num_tanks):
        # Create reference lines from tank configuration
        tank_config_data = run.config.tanks[tank_idx].geometry_data()
        ref_line_config = run.config.output_plots.get('reference_lines', {})
        reference_lines = plotter.create_reference_lines_from_config(tank_config_data, ref_line_config)

        # Add mission ambient temperature if available
        mission = run.config.missions[0]
        reference_lines['T_ambient'] = mission.ambient_temperature

        # Generate individual tank evolution plot (only if not using overlay mode)
        if not use_overlay:
            print(f"   Plotting Tank {tank_idx + 1} evolution...")

            # Read tank state plotting configuration
            tank_states_config = run.config.output_plots.get('tank_states', {})
            event_lines = tank_states_config.get('event_lines', [])

            # Axis limits and legend locations
            xlim = tank_states_config.get('xlim', None)
            ylim_pressure = tank_states_config.get('ylim_pressure', None)
            ylim_temperature = tank_states_config.get('ylim_temperature', None)
            ylim_density = tank_states_config.get('ylim_density', None)
            legend_locations = tank_states_config.get('legend_locations', None)

            # Separate figures config and dpi
            separate_figures = tank_states_config.get('separate_figures', False)
            dpi = tank_states_config.get('dpi', 900)

            # File/dir naming
            tank_save_path = None
            save_dir = None
            filename_prefix = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem
                ts_filename = tank_states_config.get('filename', 'evolution')

                if separate_figures:
                    # Provide a prefix; function will append _pressure/_temperature/_density and keep extension from save_path
                    filename_prefix = f"{base_name}_{ts_filename}_tank{tank_idx + 1}"
                else:
                    tank_save_path = save_dir / f"{base_name}_{ts_filename}_tank{tank_idx + 1}{save_ext}"
            result_figs = plotter.plot_tank_evolution(
                results=run.results,
                tank_index=tank_idx,
                reference_lines=reference_lines,
                reference_lines_config=ref_line_config,
                event_lines=event_lines,
                save_path=str(tank_save_path) if (tank_save_path and not separate_figures) else None,
                xlim=xlim,
                ylim_pressure=ylim_pressure,
                ylim_temperature=ylim_temperature,
                ylim_density=ylim_density,
                legend_locations=legend_locations,
                separate_figures=separate_figures,
                save_dir=str(save_dir) if (save_dir and separate_figures) else None,
                filename_prefix=filename_prefix,
                dpi=dpi
            )
            # Handle return type (dict of figures when separate_figures=True)
            if isinstance(result_figs, dict):
                figures.extend(result_figs.values())
            else:
                figures.append(result_figs)

        # Generate density-temperature plot for this tank
        dt_save_path = None
        if save_path:
            from pathlib import Path
            save_dir = Path(save_path).parent
            save_ext = Path(save_path).suffix or '.png'
            base_name = figure_prefix if figure_prefix else Path(save_path).stem
            dt_save_path = save_dir / f"{base_name}_density_temperature_tank{tank_idx + 1}{save_ext}"

        # Get density-temperature plot configuration
        dt_config = run.config.output_plots.get('density_temperature', {})

        # Extract plot parameters from config with defaults
        isobar_pressures = dt_config.get('isobar_pressures', [450, 400, 100, 15, 5])
        show_reference_pressures = dt_config.get('show_reference_pressures', True)
        include_saturation_line = dt_config.get('include_saturation_line', True)
        include_isobars = dt_config.get('include_isobars', True)

        # Only pass reference pressures if config allows it
        ref_pressures = reference_lines if show_reference_pressures else None

        # Respect the configuration settings for saturation line and isobars
        # (The old logic was unnecessarily disabling these for multi-tank systems)
        show_saturation = include_saturation_line
        show_isobars = include_isobars

        # Extract valve events for multi-tank systems
        valve_events = None
        if len(run.tank_geometries) > 1:
            try:
                valve_events = plotter._extract_valve_events(run.results)
            except Exception as e:
                print(f"   WARNING: Valve event extraction failed for tank {tank_idx}: {e}")

        # Get arrow configuration from density_temperature config
        dt_config = run.config.output_plots.get('density_temperature', {})
        arrow_position = dt_config.get('arrow_position', 0.25)
        arrow_size = dt_config.get('arrow_size', 1.0)
        valve_marker_size = dt_config.get('valve_marker_size', 8.0)
        legend_location = dt_config.get('legend_location', 'upper right')
        xlim = dt_config.get('xlim', None)
        ylim = dt_config.get('ylim', None)

        dt_fig = plotter.plot_density_temperature(
            results=run.results,
            tank_index=tank_idx,
            include_saturation_line=show_saturation,
            include_isobars=show_isobars,
            isobar_pressures=isobar_pressures,
            reference_pressures=ref_pressures,
            valve_events=valve_events,
            arrow_position=arrow_position,
            arrow_size=arrow_size,
            valve_marker_size=valve_marker_size,
            xlim=xlim,
            ylim=ylim,
            save_path=str(dt_save_path) if dt_save_path else None,
            legend_location=legend_location
        )
        figures.append(dt_fig)

        # Get mass flow plot configuration
        mf_config = run.config.output_plots.get('mass_flows', {})

        # Generate mass flow plot for this tank
        mf_save_path = None
        if save_path:
            from pathlib import Path
            save_dir = Path(save_path).parent
            save_ext = Path(save_path).suffix or '.png'
            base_name = figure_prefix if figure_prefix else Path(save_path).stem

            # Use filename from config if available
            mf_filename = mf_config.get('filename', 'mass_flows')
            mf_save_path = save_dir / f"{base_name}_{mf_filename}_tank{tank_idx + 1}{save_ext}"

        # Check if mass flow plots are enabled
        if mf_config.get('enabled', True):
            # Extract plot parameters from config with defaults
            include_venting_flow = mf_config.get('include_venting_flow', True)
            include_coupling_flows = mf_config.get('include_coupling_flows', True)
            xlim = mf_config.get('xlim', None)
            ylim = mf_config.get('ylim', None)
            legend_location = mf_config.get('legend_location', 'best')

            per_stream = compute_per_stream_outflows(run, tank_idx)

            mf_fig = plotter.plot_mass_flows(
                results=run.results,
                tank_index=tank_idx,
                include_venting_flow=include_venting_flow,
                include_coupling_flows=include_coupling_flows,
                save_path=str(mf_save_path) if mf_save_path else None,
                xlim=xlim,
                ylim=ylim,
                legend_location=legend_location,
                per_stream_outflows=per_stream,
            )
            figures.append(mf_fig)

        # Generate thermal-network heat-flow plot for this tank
        thermal_flow_config = run.config.output_plots.get('heat_flows', {})
        if thermal_flow_config.get('enabled', True):
            thermal_flow_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem
                thermal_flow_filename = thermal_flow_config.get('filename', 'heat_flows')
                thermal_flow_save_path = save_dir / f"{base_name}_{thermal_flow_filename}_tank{tank_idx + 1}{save_ext}"

            #store the include dictionary for the heat flow plot
            include_dict = {
                'Ambient to shell': thermal_flow_config.get('include', {}).get('amb_to_shell', True),
                'Shell to insulation': thermal_flow_config.get('include', {}).get('shell_to_insulation', True),
                'Insulation to structure': thermal_flow_config.get('include', {}).get('insulation_to_structure', True),
                'Structure to H2': thermal_flow_config.get('include', {}).get('structure_to_h2', True),
            }

            thermal_flow_fig = plotter.plot_heat_flows(
                results=run.results,
                thermal_model=run.tank_system.thermal_models[tank_idx],
                tank_index=tank_idx,
                save_path=str(thermal_flow_save_path) if thermal_flow_save_path else None,
                xlim=thermal_flow_config.get('xlim'),
                ylim=thermal_flow_config.get('ylim'),
                legend_location=thermal_flow_config.get('legend_location', 'best'),
                include=include_dict,
            )
            figures.append(thermal_flow_fig)

        # Get heat exchanger plot configuration
        hex_config = run.config.output_plots.get('heat_exchanger_requirements', {})

        # Generate heat exchanger plot for this tank (if enabled)
        if hex_config.get('enabled', False):
            hex_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_ext = Path(save_path).suffix or '.png'
                base_name = figure_prefix if figure_prefix else Path(save_path).stem

                # Use filename from config if available
                hex_filename = hex_config.get('filename', 'hex')
                hex_save_path = save_dir / f"{base_name}_{hex_filename}_tank{tank_idx + 1}{save_ext}"

            # Calculate OHEX data if needed
            qdot_ohex = calculate_ohex_requirements(run, tank_idx) if hex_config.get('include_ohex', True) else []

            # Calculate iHEX data if needed
            qdot_ihex = calculate_ihex_requirements(run, tank_idx) if hex_config.get('include_ihex', True) else [0.0] * len(run.results.times)

            # Prepare heat exchanger data
            heat_exchanger_data = {
                'times': run.results.times / 3600.0,  # Convert to hours
                'ihex_requirements': qdot_ihex,
                'ohex_requirements': qdot_ohex
            }

            xlim = hex_config.get('xlim', None)
            ylim = hex_config.get('ylim', None)
            legend_location = hex_config.get('legend_location', 'best')

            hex_fig = plotter.plot_heat_exchanger_requirements(
                heat_exchanger_data=heat_exchanger_data,
                tank_index=tank_idx,
                include_ohex=hex_config.get('include_ohex', True),
                include_total=hex_config.get('include_total', True),
                save_path=str(hex_save_path) if hex_save_path else None,
                xlim=xlim,
                ylim=ylim,
                legend_location=legend_location
            )
            figures.append(hex_fig)

    # Generate pressure requirements plot for mission-adaptive systems (not per-tank)
    pressure_req_config = run.config.output_plots.get('pressure_requirements', {})
    pressure_req_enabled = pressure_req_config.get('enabled', False)

    if pressure_req_enabled:
        pressure_req_save_path = None
        if save_path:
            from pathlib import Path
            save_dir = Path(save_path).parent
            save_ext = Path(save_path).suffix or '.png'
            base_name = figure_prefix if figure_prefix else Path(save_path).stem

            # Use filename from config if available
            pressure_req_filename = pressure_req_config.get('filename', 'pressure_requirements')
            pressure_req_save_path = save_dir / f"{base_name}_{pressure_req_filename}{save_ext}"

        # Get tank index from config (default to 1 for LH2 tank)
        tank_index = pressure_req_config.get('tank_index', 1)

        try:
            pressure_req_fig = plotter.plot_pressure_requirements(
                coupling_edges=run.tank_system.coupling_valves,
                results=run.results,
                minimum_pressure_pa=run.tank_system.config.tanks[tank_index].P_MIN,
                tank_index=tank_index,
                save_path=str(pressure_req_save_path) if pressure_req_save_path else None
            )
            figures.append(pressure_req_fig)
        except Exception as e:
            print(f"   WARNING: Pressure requirements plot failed: {e}")

    # Count plots generated
    mass_flows_enabled = run.config.output_plots.get('mass_flows', {}).get('enabled', True)
    heat_flows_enabled = run.config.output_plots.get('heat_flows', {}).get('enabled', True)
    heat_exchanger_enabled = run.config.output_plots.get('heat_exchanger_requirements', {}).get('enabled', False)

    # -------------------------------------------------------------------
    # Peripheral component chain plots (HEX, compressor, pump…)
    # -------------------------------------------------------------------
    from toplab.coupling import ProportionalSplitCoupling, PressureTriggeredDischarge

    # Only attempt if the tank_system has coupling valves
    if hasattr(run, 'tank_system') and run.tank_system is not None:
        # --- Component chain plots (one per coupling with components) ---
        try:
            stream_histories = coupling_stream_history(run.tank_system, run.results)
            for sh in stream_histories:
                # Only plot peripheral-component charts for the 5% split chain
                if sh.get('stream_type', 'split_chain') != 'split_chain':
                    continue
                if not sh['components']:
                    continue
                pc_save_path = None
                if save_path:
                    from pathlib import Path
                    save_dir = Path(save_path).parent
                    save_ext = Path(save_path).suffix or '.png'
                    base_name = figure_prefix if figure_prefix else Path(save_path).stem
                    safe_id = sh['coupling_id'].replace(' ', '_').replace('/', '_').replace('→', '_to_')
                    pc_save_path = save_dir / f"{base_name}_components_{safe_id}{save_ext}"

                pc_fig = plotter.plot_peripheral_components(
                    stream_history=sh,
                    save_path=str(pc_save_path) if pc_save_path else None,
                )
                figures.append(pc_fig)

                # --- HEX / compressor power & energy plot ---
                try:
                    _pwr_cfg = run.config.output_plots.get('hex_compressor_power', {})
                    if _pwr_cfg.get('enabled', True):
                        pwr_save_path = None
                        if save_path:
                            from pathlib import Path as _Path
                            _save_dir = _Path(save_path).parent
                            _save_ext = _Path(save_path).suffix or '.png'
                            _base_name = figure_prefix if figure_prefix else _Path(save_path).stem
                            _safe_id = sh['coupling_id'].replace(' ', '_').replace('/', '_').replace('→', '_to_')
                            _pwr_fname = _pwr_cfg.get('filename', 'power')
                            pwr_save_path = _save_dir / f"{_base_name}_{_pwr_fname}_{_safe_id}{_save_ext}"

                        pwr_fig = plotter.plot_hex_compressor_power(
                            stream_history=sh,
                            save_path=str(pwr_save_path) if pwr_save_path else None,
                            xlim=_pwr_cfg.get('xlim', None),
                            ylim_power=_pwr_cfg.get('ylim_power', None),
                            ylim_energy=_pwr_cfg.get('ylim_energy', None),
                            legend_location=_pwr_cfg.get('legend_location_power', 'best'),
                            fluid=run.config.tanks[sh['source_tank']].coolprop,
                        )
                        figures.append(pwr_fig)
                except Exception as _pwr_e:
                    print(f"   WARNING: HEX/compressor power plot failed: {_pwr_e}")

        except Exception as e:
            print(f"   WARNING: Peripheral component plots failed: {e}")
            import traceback
            traceback.print_exc()

        # --- Fuel cell inlet plot: find the PressureTriggeredDischarge sink ---
        try:
            discharge_tank_idx = None
            for valve in run.tank_system.coupling_valves:
                if isinstance(valve, PressureTriggeredDischarge):
                    discharge_tank_idx = valve.source_tank
                    break

            if discharge_tank_idx is not None:
                _fc_cfg = run.config.output_plots.get('fuel_cell_inlet', {})
                if _fc_cfg.get('enabled', True):
                    fc_save_path = None
                    if save_path:
                        from pathlib import Path
                        save_dir = Path(save_path).parent
                        save_ext = Path(save_path).suffix or '.png'
                        base_name = figure_prefix if figure_prefix else Path(save_path).stem
                        _fc_fname = _fc_cfg.get('filename', 'fuel_cell_inlet')
                        fc_save_path = save_dir / f"{base_name}_{_fc_fname}{save_ext}"

                    fc_fig = plotter.plot_fuel_cell_inlet(
                        results=run.results,
                        source_tank_idx=discharge_tank_idx,
                        stream_histories=stream_histories,
                        save_path=str(fc_save_path) if fc_save_path else None,
                        xlim=_fc_cfg.get('xlim', None),
                        ylim_mdot=_fc_cfg.get('ylim_mdot', None),
                        ylim_temp=_fc_cfg.get('ylim_temp', None),
                        ylim_pres=_fc_cfg.get('ylim_pres', None),
                        legend_location=_fc_cfg.get('legend_location', 'best'),
                    )
                    figures.append(fc_fig)
        except Exception as e:
            print(f"   WARNING: Fuel cell inlet plot failed: {e}")
            import traceback
            traceback.print_exc()

        # --- Flow-distribution mixer plots: one per mixer junction ---
        _mx_cfg = run.config.output_plots.get('mixer_flows', {})
        if run.network.mixers and _mx_cfg.get('enabled', True):
            try:
                from pathlib import Path
                from toplab.post.mixer import mixer_stream_history
                for mixer_history in mixer_stream_history(run.network, run.tank_system, run.results):
                    mx_save_path = None
                    if save_path:
                        base_name = figure_prefix if figure_prefix else Path(save_path).stem
                        mx_fname = _mx_cfg.get('filename', 'mixer_flows')
                        mx_save_path = (Path(save_path).parent /
                                        f"{base_name}_{mx_fname}_{mixer_history.mixer_id}"
                                        f"{Path(save_path).suffix or '.png'}")
                    figures.append(plotter.plot_mixer_flows(
                        mixer_history,
                        save_path=str(mx_save_path) if mx_save_path else None,
                        xlim=_mx_cfg.get('xlim', None),
                        legend_location=_mx_cfg.get('legend_location', 'best'),
                    ))
            except Exception as e:
                print(f"   WARNING: Mixer flow plot failed: {e}")
                import traceback
                traceback.print_exc()
    # -------------------------------------------------------------------

    plots_per_tank = 2  # Always: evolution + density-temperature
    if mass_flows_enabled:
        plots_per_tank += 1
    if heat_flows_enabled:
        plots_per_tank += 1
    if heat_exchanger_enabled:
        plots_per_tank += 1

    total_plots = len(run.results.tank_metadata) * plots_per_tank
    if pressure_req_enabled:
        total_plots += 1  # Add pressure requirements plot

    plot_types = ['evolution', 'density-temperature']
    if mass_flows_enabled:
        plot_types.append('mass flows')
    if heat_flows_enabled:
        plot_types.append('thermal heat flows')
    if heat_exchanger_enabled:
        plot_types.append('heat exchanger requirements')
    if pressure_req_enabled:
        plot_types.append('pressure requirements')

    print(f"   Generated {len(figures)} tank plots ({plots_per_tank} plots per tank: {' + '.join(plot_types)})")

    # Return the first figure (or all figures if multiple tanks)
    return figures[0] if len(figures) == 1 else figures
