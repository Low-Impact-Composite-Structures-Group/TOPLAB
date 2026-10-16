"""Text summaries and the saved results report for a finished run."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from CoolProp.CoolProp import PropsSI

from toplab.utilities.output_control import silent_when_quiet
from toplab.post.hex import calculate_ohex_requirements, calculate_ihex_requirements
from toplab.network import SINK
from toplab.orchestration.run_result import RunResult


def get_scenario_summary(run: RunResult) -> Dict[str, Any]:
    """Get comprehensive summary of the scenario configuration and results."""
    summary = {
        'scenario': {
            'name': run.config.name,
            'description': run.config.description,
            'version': run.config.version
        },
        'tanks': {
            'count': len(run.tank_geometries),
            'geometries': [
                {
                    'volume': tank.volume,
                    'radius': getattr(tank, 'radius', 'N/A')
                }
                for tank in run.tank_geometries
            ]
        },
        'missions': {
            'count': len(run.config.missions),
            'sequences': [
                {
                    'type': mission.type,
                    'profile': mission.profile,
                    'ambient_temperature': mission.ambient_temperature
                }
                for mission in run.config.missions
            ]
        },
        'tank_materials': {
            str(tank.node_id): sorted(tank.materials) for tank in run.config.tanks
        }
    }

    # Add results summary if available
    if run.results:
        final_multi_state = run.results.multi_tank_states[-1]
        summary['results'] = {
            'final_time_hours': run.results.times[-1] / 3600,
            'data_points': len(run.results.times),
            'final_tank_states': [
                {
                    'fuel_mass': final_multi_state.get_tank_state(i).fuel_mass,
                    'h2_temperature': final_multi_state.get_tank_state(i).h2_temperature,
                    'density': final_multi_state.get_tank_state(i).density
                }
                for i in range(len(run.tank_geometries))
            ]
        }

    if run.validation_results:
        summary['validation'] = run.validation_results

    return summary


def get_comprehensive_analysis_summary(run: RunResult) -> Dict[str, Any]:
    """Get comprehensive analysis summary with all input/output parameters for benchmarking."""
    import numpy as np

    if not run.results or not run.results.multi_tank_states:
        raise RuntimeError("No simulation results available. Run simulation first.")

    summary = {
        'analysis_name': run.config.name,
        'tank_count': len(run.tank_geometries),
        'tanks': {}
    }

    for tank_idx in range(len(run.tank_geometries)):
        tank_name = f"Tank_{tank_idx + 1}"
        # Geometry and mass properties already computed during TankSystem initialisation
        tank_props = run.tank_system.tank_properties(tank_idx)

        # TankSystemConfig holds the values actually used in the simulation
        tank_sys_cfg = run.tank_system.config.tanks[tank_idx]

        # Per-tank material thicknesses via the same path used by _get_tank_properties
        mat_cfg = run.config.tanks[tank_idx].materials
        liner_thickness_m     = float(mat_cfg['liner']['thickness'])
        insulation_thickness_m = float(mat_cfg['insulation']['thickness'])

        initial_state = run.results.multi_tank_states[0].get_tank_state(tank_idx)

        # === DIRECT INPUTS ===
        initial_pressure_bar  = float(tank_sys_cfg.P_INIT) / 1e5
        initial_temperature_K = float(tank_sys_cfg.T_INIT)
        tank_volume_m3        = float(tank_props['volume'])
        initial_density_kg_m3 = float(initial_state.fuel_mass) / tank_volume_m3
        vent_pressure_bar     = float(tank_sys_cfg.P_VENT) / 1e5
        min_pressure_bar      = float(tank_sys_cfg.P_MIN) / 1e5

        if run.tank_system.fluid_kind(tank_idx) == 'CH2':
            min_density_kg_m3 = 0.0  # CH2 uses pressure-based stopping
        else:
            min_density_kg_m3 = run.tank_system.min_density(tank_idx)

        direct_inputs = {
            'initial_pressure_bar':   initial_pressure_bar,
            'initial_temperature_K':  initial_temperature_K,
            'initial_density_kg_m3':  initial_density_kg_m3,
            'vent_pressure_bar':      vent_pressure_bar,
            'min_pressure_bar':       min_pressure_bar,
            'min_density_kg_m3':      min_density_kg_m3,
            'liner_thickness_m':      liner_thickness_m,
            'insulation_thickness_m': insulation_thickness_m,
        }

        # === PREPROCESSED INPUTS ===
        fluid_name = run.config.tanks[tank_idx].coolprop
        P_crit_bar = PropsSI("Pcrit", fluid_name) / 1e5
        if initial_pressure_bar <= P_crit_bar:
            T_sat = float(PropsSI("T", "P", initial_pressure_bar * 1e5, "Q", 0, fluid_name))
        else:
            T_sat = float(PropsSI("Tcrit", fluid_name))
        delta_T = initial_temperature_K - T_sat

        preprocessed_inputs = {
            'T_sat_K':   T_sat,
            'delta_T_K': delta_T,
        }

        # === DIRECT OUTPUTS ===
        tank_radius_m    = float(tank_props['radius'])
        tank_diameter_m  = float(tank_props['inner_diameter'])
        surface_area_m2  = float(tank_props['inner_surface_area'])
        wall_thickness_m = float(tank_props['thickness_wall'])
        liner_mass_kg    = float(tank_props['liner_mass'])
        wall_mass_kg     = float(tank_props['wall_mass'])
        total_tank_mass_kg = liner_mass_kg + wall_mass_kg
        fuel_mass_kg     = float(initial_state.fuel_mass)

        direct_outputs = {
            'tank_volume_m3':      tank_volume_m3,
            'tank_radius_m':       tank_radius_m,
            'tank_diameter_m':     tank_diameter_m,
            'surface_area_m2':     surface_area_m2,
            'wall_thickness_m':    wall_thickness_m,
            'fuel_mass_kg':        fuel_mass_kg,
            'liner_mass_kg':       liner_mass_kg,
            'wall_mass_kg':        wall_mass_kg,
            'total_tank_mass_kg':  total_tank_mass_kg,
        }

        # === POSTPROCESSED OUTPUTS ===
        times_array = np.asarray(run.results.times, dtype=float)
        mission_duration_hours = float(times_array[-1]) / 3600.0

        vent_pressure_pa = vent_pressure_bar * 1e5
        tank_series = run.results.get_tank_series(tank_idx)
        time_to_vent_hours = mission_duration_hours  # default: never vented
        for i, state in enumerate(tank_series.states):
            if float(state.pressure) >= vent_pressure_pa:
                time_to_vent_hours = float(times_array[i]) / 3600.0
                break

        # Gravimetric efficiency = fuel / (fuel + structure)
        structure_mass = liner_mass_kg + wall_mass_kg
        mass_efficiency = fuel_mass_kg / (fuel_mass_kg + structure_mass)

        # Volumetric efficiency = inner volume / outer volume (capsule-correct, from tank_props)
        outer_volume_m3 = float(tank_props['outer_volume'])
        volumetric_efficiency = tank_volume_m3 / outer_volume_m3 if outer_volume_m3 > 0.0 else 0.0

        fuel_energy_mj = fuel_mass_kg * 120.0  # H2 LHV ~120 MJ/kg

        qdot_ohex = calculate_ohex_requirements(run, tank_idx)
        if len(qdot_ohex) == len(times_array) and len(times_array) > 1:
            dt = np.diff(times_array)
            ohex_energy_mj = float(np.sum(np.asarray(qdot_ohex[:-1]) * dt)) / 1e6
        else:
            ohex_energy_mj = 0.0

        qdot_ihex = calculate_ihex_requirements(run, tank_idx)
        if len(qdot_ihex) == len(times_array) and len(times_array) > 1:
            dt = np.diff(times_array)
            ihex_energy_mj = float(np.sum(np.asarray(qdot_ihex[:-1]) * dt)) / 1e6
        else:
            ihex_energy_mj = 0.0

        total_energy_mj = ohex_energy_mj + ihex_energy_mj

        postprocessed_outputs = {
            'mission_duration_hours': mission_duration_hours,
            'time_to_vent_hours':     time_to_vent_hours,
            'mass_efficiency':        mass_efficiency,
            'volumetric_efficiency':  volumetric_efficiency,
            'fuel_energy_MJ':         fuel_energy_mj,
            'ohex_energy_MJ':         ohex_energy_mj,
            'ihex_energy_MJ':         ihex_energy_mj,
            'total_energy_MJ':        total_energy_mj,
        }

        summary['tanks'][tank_name] = {
            'direct_inputs':         direct_inputs,
            'preprocessed_inputs':   preprocessed_inputs,
            'direct_outputs':        direct_outputs,
            'postprocessed_outputs': postprocessed_outputs,
        }

    return summary


def print_scenario_summary(run):
    """Print comprehensive scenario summary."""
    summary = get_scenario_summary(run)

    print("\n🔍 SCENARIO CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Analysis: {summary['scenario']['name']}")
    print(f"Description: {summary['scenario']['description']}")
    print(f"Version: {summary['scenario']['version']}")
    print(f"Tanks: {summary['tanks']['count']}")
    print(f"Missions: {summary['missions']['count']}")
    print(f"Materials: {summary['materials']['count']} ({', '.join(summary['materials']['types'])})")

    if 'results' in summary:
        print(f"\nResults:")
        print(f"  Final time: {summary['results']['final_time_hours']:.2f} hours")
        print(f"  Data points: {summary['results']['data_points']}")
        for i, tank_state in enumerate(summary['results']['final_tank_states']):
            print(f"  Tank {i+1}: m={tank_state['fuel_mass']:.2f}kg, "
                  f"T={tank_state['temperature']:.1f}K, "
                  f"ρ={tank_state['density']:.1f}kg/m³")

    if 'validation' in summary:
        overall = summary['validation']['overall']
        print(f"  Validation: {'PASSED' if overall else 'FAILED'}")

    print("=" * 60)


def print_comprehensive_analysis_summary(run):
    """Print comprehensive analysis parameter summary table."""
    summary = get_comprehensive_analysis_summary(run)

    print(f"\nCOMPREHENSIVE ANALYSIS SUMMARY: {summary['analysis_name']}")
    print("=" * 100)

    # Header
    header = f"{'Parameter':<35} {'Unit':<12}"
    for tank_name in summary['tanks'].keys():
        header += f"{tank_name:<15}"
    print(header)
    print("-" * 100)

    # Define parameter groups and their display information
    param_groups = [
        ("DIRECT INPUTS", [
            ('initial_pressure_bar', 'bar', 'Initial Pressure'),
            ('initial_temperature_K', 'K', 'Initial Temperature'),
            ('initial_density_kg_m3', 'kg/m³', 'Initial Density'),
            ('vent_pressure_bar', 'bar', 'Vent Pressure'),
            ('min_pressure_bar', 'bar', 'Minimum Pressure'),
            ('min_density_kg_m3', 'kg/m³', 'Minimum Density'),
            ('liner_thickness_m', 'm', 'Liner Thickness'),
            ('insulation_thickness_m', 'm', 'Insulation Thickness')
        ]),
        ("PREPROCESSED INPUTS", [
            ('T_sat_K', 'K', 'Saturation Temperature'),
            ('delta_T_K', 'K', 'ΔT from Saturation')
        ]),
        ("DIRECT OUTPUTS", [
            ('tank_volume_m3', 'm³', 'Tank Volume'),
            ('tank_radius_m', 'm', 'Tank Radius'),
            ('tank_diameter_m', 'm', 'Tank Diameter'),
            ('surface_area_m2', 'm²', 'Surface Area'),
            ('wall_thickness_m', 'm', 'Wall Thickness'),
            ('fuel_mass_kg', 'kg', 'Fuel Mass'),
            ('liner_mass_kg', 'kg', 'Liner Mass'),
            ('wall_mass_kg', 'kg', 'Wall Mass'),
            ('total_tank_mass_kg', 'kg', 'Total Tank Mass')
        ]),
        ("POSTPROCESSED OUTPUTS", [
            ('mission_duration_hours', 'hours', 'Mission Duration'),
            ('time_to_vent_hours', 'hours', 'Time to Vent'),
            ('mass_efficiency', '-', 'Mass Efficiency'),
            ('volumetric_efficiency', '-', 'Volumetric Efficiency'),
            ('fuel_energy_MJ', 'MJ', 'Fuel Energy'),
            ('ohex_energy_MJ', 'MJ', 'OHEX Energy'),
            ('ihex_energy_MJ', 'MJ', 'iHEX Energy'),
            ('total_energy_MJ', 'MJ', 'Total Energy')
        ])
    ]

    # Print each parameter group
    for group_name, params in param_groups:
        print(f"\n{group_name}")
        print("-" * 50)

        for param_key, unit, display_name in params:
            row = f"{display_name:<35} {unit:<12}"

            for tank_name in summary['tanks'].keys():
                tank_data = summary['tanks'][tank_name]

                # Find the parameter in the appropriate group
                value = None
                for group_key in ['direct_inputs', 'preprocessed_inputs', 'direct_outputs', 'postprocessed_outputs']:
                    if param_key in tank_data[group_key]:
                        value = tank_data[group_key][param_key]
                        break

                if value is not None:
                    # Format based on parameter type
                    if 'efficiency' in param_key:
                        formatted_value = f"{value:.3f}"
                    elif 'energy' in param_key or 'MJ' in unit:
                        formatted_value = f"{value:.1f}"
                    elif 'hours' in unit:
                        formatted_value = f"{value:.2f}"
                    elif 'bar' in unit:
                        formatted_value = f"{value:.0f}"
                    elif param_key in ['tank_volume_m3', 'surface_area_m2']:
                        formatted_value = f"{value:.3f}"
                    elif param_key in ['tank_radius_m', 'tank_diameter_m']:
                        formatted_value = f"{value:.2f}"
                    elif 'thickness' in param_key:
                        formatted_value = f"{value:.3f}"
                    elif 'mass' in param_key:
                        formatted_value = f"{value:.1f}"
                    elif 'density' in param_key:
                        formatted_value = f"{value:.1f}"
                    elif 'temperature' in param_key.lower() or unit == 'K':
                        formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = "N/A"

                row += f"{formatted_value:<15}"

            print(row)

    print("\n" + "=" * 100)
    print("Comprehensive analysis summary complete")
    print("   📝 All input parameters, tank geometry, and performance metrics displayed")


@silent_when_quiet
def save_comprehensive_results(run) -> str:
    """
    Generate and save comprehensive results report to text file.

    This method runs after simulation completion and generates a complete
    analysis report using the exact data sources specified in the requirements table.

    Returns:
        str: Path to the saved results file
    """
    from datetime import datetime
    from pathlib import Path

    if not run.results:
        raise RuntimeError("No simulation results available. Run simulation first.")

    output_config = run.config.output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mission_sections = []
    if getattr(run, 'mission_profile', None) and getattr(run.mission_profile, 'sections', None):
        mission_sections = [section.duration for section in run.mission_profile.sections]

    times_array = np.asarray(run.results.times, dtype=float)
    total_duration_s = float(times_array[-1]) if times_array.size > 0 else 0.0
    data_points = int(times_array.size)

    def _fmt(value: Any, decimals: int = 2, default: str = "N/A") -> str:
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.flat[0]
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            value = value[0]
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isnan(value):
            return default
        return f"{value:.{decimals}f}"

    def _line(label: str, value: Any) -> str:
        return f"{label:<32} {value}\n"

    def _get_tank_props(tank_geometry: Any, tank_index: int) -> Dict[str, Any]:
        if hasattr(run, 'tank_system') and run.tank_system:
            try:
                return run.tank_system.tank_properties(tank_index)
            except Exception:
                return {}
        return {}

    total_initial_mass = 0.0
    total_final_mass = 0.0
    total_fuel_consumed = 0.0
    total_dry_mass = 0.0
    total_inner_volume = 0.0
    total_outer_volume = 0.0
    total_ihex_kwh = 0.0
    total_ohex_kwh = 0.0
    total_hex_kwh = 0.0

    per_tank_blocks = []
    for tank_index, tank_geometry in enumerate(run.tank_geometries):
        tank_name = f"Tank_{tank_index + 1}"
        tank_config = run.config.tanks[tank_index].geometry_data()
        design = run.designs[tank_index]
        tank_material_config = run.config.tanks[tank_index].materials
        insulation_config = tank_material_config.get('insulation', {})
        initial_state = run.results.multi_tank_states[0].get_tank_state(tank_index)
        final_state = run.results.multi_tank_states[-1].get_tank_state(tank_index)
        tank_props = _get_tank_props(tank_geometry, tank_index)

        initial_mass = float(initial_state.fuel_mass)
        final_mass = float(final_state.fuel_mass)
        fuel_consumed = initial_mass - final_mass

        try:
            qdot_ihex = calculate_ihex_requirements(run, tank_index)
            ihex_kwh = np.trapz(np.abs(np.asarray(qdot_ihex, dtype=float)), times_array) / 3.6e6 if len(qdot_ihex) == data_points else 0.0
        except Exception:
            ihex_kwh = 0.0

        try:
            qdot_ohex = calculate_ohex_requirements(run, tank_index)
            ohex_kwh = np.trapz(np.asarray(qdot_ohex, dtype=float), times_array) / 3.6e6 if len(qdot_ohex) == data_points else 0.0
        except Exception:
            ohex_kwh = 0.0

        hex_kwh = ihex_kwh + ohex_kwh
        liner_mass = float(tank_props.get('liner_mass', 0.0))
        composite_wall_mass = float(tank_props.get('wall_mass', 0.0))
        insulation_mass = float(tank_props.get('foam_mass', 0.0))
        outer_shell_mass = float(tank_props.get('shell_mass', 0.0))
        dry_mass = liner_mass + composite_wall_mass + insulation_mass + outer_shell_mass

        inner_volume = float(getattr(tank_geometry, 'volume', tank_props.get('volume', 0.0)))
        outer_volume = float(tank_props.get('outer_volume', 0.0))

        total_initial_mass += initial_mass
        total_final_mass += final_mass
        total_fuel_consumed += fuel_consumed
        total_dry_mass += dry_mass
        total_inner_volume += inner_volume
        total_outer_volume += outer_volume
        total_ihex_kwh += ihex_kwh
        total_ohex_kwh += ohex_kwh
        total_hex_kwh += hex_kwh

        block = []
        block.append(f"{tank_name.upper()}\n")
        block.append("-" * 80 + "\n")
        block.append(_line("Configured pressure [bar]:", _fmt(tank_config.get('initial_pressure', 0.0) / 1e5, 1)))
        block.append(_line("Configured temperature [K]:", _fmt(design.initial_temperature, 2)))
        block.append(_line("Configured density [kg/m^3]:", _fmt(design.initial_density, 2)))
        block.append(_line("Minimum pressure [bar]:", _fmt(design.minimum_pressure / 1e5, 1)))
        block.append(_line("Venting pressure [bar]:", _fmt(design.venting_pressure / 1e5, 1)))
        block.append(_line("Inner volume [m^3]:", _fmt(inner_volume, 4)))
        block.append(_line("Outer volume [m^3]:", _fmt(outer_volume, 4)))
        block.append(_line("Tank radius [m]:", _fmt(getattr(tank_geometry, 'radius', 0.0), 3)))
        block.append(_line("Tank phi (L/R) [-]:", _fmt(getattr(tank_geometry, 'phi', 0.0), 3)))
        block.append(_line("Cylindrical length [m]:", _fmt(getattr(tank_geometry, 'cylindrical_section_length', 0.0), 3)))
        block.append(_line("Total length [m]:", _fmt(getattr(tank_geometry, 'total_height', 0.0), 3)))
        block.append(_line("Inner area [m^2]:", _fmt(tank_props.get('inner_surface_area', 0.0), 3)))
        block.append(_line("Outer area [m^2]:", _fmt(tank_props.get('outer_surface_area', 0.0), 3)))
        block.append(_line("Liner thickness [m]:", _fmt(tank_material_config.get('liner', {}).get('thickness'), 4)))
        block.append(_line("Composite wall thickness [m]:", _fmt(tank_props.get('wall_thickness', 0.0), 4)))
        block.append(_line("Insulation thickness [m]:", _fmt(insulation_config.get('thickness'), 4)))
        block.append(_line("Outer shell thickness [m]:", _fmt(insulation_config.get('shell_thickness'), 4)))
        block.append(_line("Liner mass [kg]:", _fmt(liner_mass, 2)))
        block.append(_line("Composite wall mass [kg]:", _fmt(composite_wall_mass, 2)))
        block.append(_line("Insulation mass [kg]:", _fmt(insulation_mass, 2)))
        block.append(_line("Outer shell mass [kg]:", _fmt(outer_shell_mass, 2)))
        block.append(_line("Total dry mass [kg]:", _fmt(dry_mass, 2)))
        block.append(_line("Initial fuel mass [kg]:", _fmt(initial_mass, 2)))
        block.append(_line("Final fuel mass [kg]:", _fmt(final_mass, 2)))
        block.append(_line("Fuel consumed [kg]:", _fmt(fuel_consumed, 2)))
        block.append(_line("Initial pressure actual [bar]:", _fmt(initial_state.pressure / 1e5, 2)))
        block.append(_line("Initial H2 temperature actual [K]:", _fmt(initial_state.h2_temperature, 2)))
        block.append(_line("Initial density actual [kg/m^3]:", _fmt(initial_state.density, 2)))
        block.append(_line("Final H2 temperature [K]:", _fmt(final_state.h2_temperature, 2)))
        block.append(_line("Final pressure [bar]:", _fmt(final_state.pressure / 1e5, 2)))
        block.append(_line("Final density [kg/m^3]:", _fmt(final_state.density, 2)))
        block.append(_line("Mission duration [h]:", _fmt(total_duration_s / 3600.0, 3)))
        block.append(_line("iHEX energy [kWh]:", _fmt(ihex_kwh, 3)))
        block.append(_line("OHEX energy [kWh]:", _fmt(ohex_kwh, 3)))
        block.append(_line("Total HEX energy [kWh]:", _fmt(hex_kwh, 3)))
        per_tank_blocks.append("".join(block))

    report_lines = []
    report_lines.append(f"{run.config.name} - Results Report\n")
    report_lines.append(f"Generated: {timestamp}\n")
    report_lines.append(f"Description: {run.config.description}\n\n")

    report_lines.append("SYSTEM OVERVIEW\n")
    report_lines.append("=" * 80 + "\n")
    report_lines.append(_line("Tank count:", len(run.tank_geometries)))
    report_lines.append(_line("Coupling rule count:", len(run.couplings)))
    report_lines.append(_line("Mission count:", len(run.config.missions)))
    report_lines.append(_line("Mission sections:", len(mission_sections)))
    report_lines.append(_line("Mission duration [h]:", _fmt(total_duration_s / 3600.0, 3)))
    report_lines.append(_line("Data points:", data_points))
    if mission_sections:
        mission_duration_str = ", ".join(_fmt(duration, 1) for duration in mission_sections)
        report_lines.append(_line("Mission section durations [s]:", mission_duration_str))
    report_lines.append("\n")

    report_lines.append("SYSTEM TOTALS\n")
    report_lines.append("=" * 80 + "\n")
    report_lines.append(_line("Initial fuel mass [kg]:", _fmt(total_initial_mass, 2)))
    report_lines.append(_line("Final fuel mass [kg]:", _fmt(total_final_mass, 2)))
    report_lines.append(_line("Fuel consumed [kg]:", _fmt(total_fuel_consumed, 2)))
    report_lines.append(_line("Total dry mass [kg]:", _fmt(total_dry_mass, 2)))
    report_lines.append(_line("Total inner volume [m^3]:", _fmt(total_inner_volume, 4)))
    report_lines.append(_line("Total outer volume [m^3]:", _fmt(total_outer_volume, 4)))
    total_system_mass = total_initial_mass + total_dry_mass
    gravimetric_efficiency = total_initial_mass / total_system_mass if total_system_mass > 0 else 0.0
    volumetric_efficiency = total_inner_volume / total_outer_volume if total_outer_volume > 0 else 0.0
    report_lines.append(_line("Gravimetric efficiency [-]:", _fmt(gravimetric_efficiency, 4)))
    report_lines.append(_line("Volumetric efficiency [-]:", _fmt(volumetric_efficiency, 4)))
    report_lines.append(_line("Total iHEX energy [kWh]:", _fmt(total_ihex_kwh, 3)))
    report_lines.append(_line("Total OHEX energy [kWh]:", _fmt(total_ohex_kwh, 3)))
    report_lines.append(_line("Total HEX energy [kWh]:", _fmt(total_hex_kwh, 3)))
    if total_duration_s > 0:
        report_lines.append(_line("Average net fuel use [kg/s]:", _fmt(total_fuel_consumed / total_duration_s, 6)))
    report_lines.append("\n")

    report_lines.append("COUPLING SUMMARY\n")
    report_lines.append("=" * 80 + "\n")
    if run.couplings:
        for idx, coupling in enumerate(run.couplings, start=1):
            source = run.network.tank_ids[coupling.source]
            target = 'sink' if coupling.target == SINK else run.network.tank_ids[coupling.target]
            report_lines.append(
                f"{idx}. {coupling.edge_id}: {coupling.coupling_type} (source={source}, target={target})\n"
            )
    else:
        report_lines.append("No coupling rules configured.\n")
    report_lines.append("\n")

    report_lines.append("PER-TANK DETAILS\n")
    report_lines.append("=" * 80 + "\n\n")
    report_lines.append("\n".join(per_tank_blocks))
    report_lines.append("\n")
    report_lines.append("=" * 80 + "\n")
    report_lines.append("Report generated from SystemOrchestrator summary data and simulation results.\n")

    report_text = "".join(report_lines)

    # Create output directory and save file
    output_dir = Path(output_config.get('results_directory', 'output/results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"comprehensive_report.txt"

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"COMPREHENSIVE RESULTS SUMMARY:")
    print(f"   Tanks: {len(run.tank_geometries)}")
    print(f"   Fuel consumed: {total_fuel_consumed:.2f} kg")
    print(f"   Total iHEX Energy: {total_ihex_kwh:.3f} kWh")
    print(f"   Total OHEX Energy: {total_ohex_kwh:.3f} kWh")
    print(f"   Total HEX Energy: {total_hex_kwh:.3f} kWh")

    return str(output_file)
