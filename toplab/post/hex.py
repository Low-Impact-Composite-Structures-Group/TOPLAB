"""Heat-exchanger loads and per-stream flows derived from simulation results.

Each function takes the finished run (the orchestrator) and reads its results; nothing is mutated.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from CoolProp.CoolProp import PropsSI

from toplab.utilities.output_control import silent_when_quiet
from toplab.orchestration.run_result import RunResult


def calculate_ohex_requirements(run: RunResult, tank_index: int = 0) -> List[float]:
    """
    Calculate OHEX (Outboard Heat Exchanger) heat requirements.

    Uses enthalpy difference method: Q_oHEX = mdot * (h_target - h_current)
    where h_target is at standard fuel cell inlet conditions.

    Args:
        tank_index: Index of tank to calculate for

    Returns:
        List of OHEX heat requirements [W] for each time point
    """
    if not run.results:
        return []

    # Get OHEX target conditions from config
    hex_config = run.config.output_plots.get('heat_exchanger_requirements', {})
    target_temp = float(hex_config.get('ohex_target_temperature', 200.0))  # K
    target_press = float(hex_config.get('ohex_target_pressure', 20e5))    # Pa

    # Log fallback usage for heat exchanger parameters
    if 'ohex_target_temperature' not in hex_config:
        print(f"WARNING: Using fallback: ohex_target_temperature = {target_temp} K (now available in YAML config)")
    if 'ohex_target_pressure' not in hex_config:
        print(f"WARNING: Using fallback: ohex_target_pressure = {target_press/1e5:.0f} bar (now available in YAML config)")

    fluid = run.config.tanks[tank_index].coolprop
    try:
        # Calculate target enthalpy (constant for all time points)
        h_target = PropsSI("Hmass", "T", target_temp, "P", target_press, fluid)
    except Exception as e:
        print(f"   WARNING: Could not calculate OHEX target enthalpy: {e}")
        return [0.0] * len(run.results.times)

    qdot_ohex = []
    tank_series = run.results.get_tank_series(tank_index)

    # Add statistics tracking
    max_q_ohex = 0.0
    valid_points = 0
    print(f"   Calculating OHEX for {len(tank_series.states)} time points...")

    for i, state in enumerate(tank_series.states):
        try:
            # Get current state conditions
            T_current = state.h2_temperature
            m_current = state.fuel_mass

            # Get tank volume for density calculation
            tank_volume = run.tank_geometries[tank_index].volume
            rho_current = m_current / tank_volume  # kg/m³

            # Get actual mass flow rate used during simulation (stored in tank state)
            mass_rate = abs(state.outflow_rate)

            if T_current > 0 and rho_current > 0 and mass_rate > 0:
                # Calculate current pressure and enthalpy with saturation-aware helpers
                try:
                    from toplab.fluids.coolprop_safe import safe_pressure_from_T_rho, safe_enthalpy
                    p_current = safe_pressure_from_T_rho(T_current, rho_current, fluid)
                    # Assume gas enthalpy for outlet stream when in two-phase
                    h_current = safe_enthalpy(T_current, p_current, assume_gas_when_twophase=True, fluid=fluid)
                except Exception:
                    # Fallback to direct CoolProp calls
                    p_current = PropsSI("P", "T", T_current, "Dmass", rho_current, fluid)
                    h_current = PropsSI("Hmass", "T", T_current, "P", p_current, fluid)

                # Calculate OHEX heat requirement
                q_ohex = mass_rate * (h_target - h_current)  # [W]
                q_ohex_final = max(0.0, q_ohex)  # Ensure non-negative
                qdot_ohex.append(q_ohex_final)

                # Track statistics
                if q_ohex_final > max_q_ohex:
                    max_q_ohex = q_ohex_final
                if q_ohex_final > 0:
                    valid_points += 1
            else:
                qdot_ohex.append(0.0)

        except Exception:
            # Silently handle CoolProp errors (common at extreme conditions)
            qdot_ohex.append(0.0)

    print(f"   OHEX calculated: {valid_points}/{len(qdot_ohex)} points, max = {max_q_ohex:.0f}W")
    return qdot_ohex


def calculate_ihex_requirements(run: RunResult, tank_index: int = 0) -> List[float]:
    """
    iHEX heat requirement [W] of one tank at every result time.

    This is the Configuration B discharge heat computed in the isochoric energy balance
    (the heat needed to hold the pressure constraint). It is recorded per tank at each
    accepted integration point, so it lines up one-to-one with ``run.results.times``.
    """
    if not run.results:
        return []
    history = run.tank_system.discharge_heat_history
    if history.shape != (len(run.tank_geometries), len(run.results.times)):
        return [0.0] * len(run.results.times)
    qdot_ihex = history[tank_index].tolist()
    peak = max((abs(q) for q in qdot_ihex), default=0.0)
    print(f"   iHEX: {sum(abs(q) > 1e-3 for q in qdot_ihex)}/{len(qdot_ihex)} active points, max = {peak:.0f}W")
    return qdot_ihex


def compute_per_stream_outflows(run: RunResult, tank_index: int) -> Optional[Dict[str, np.ndarray]]:
    """
    For tanks with multiple discharge streams (e.g. FC + GT dual-flow), return a dict
    mapping each stream label → outflow array [g/s, negative convention] at simulation times.
    Returns None when only a single stream is present.
    """
    if not run.results:
        return None
    streams = run.streams.get(tank_index, [])
    if len(streams) < 2:
        return None

    from toplab.demand import outflow_demand
    times = run.results.times
    result = {}
    for stream in streams:
        demand = outflow_demand(stream.mission).scaled(stream.fraction)
        flows_gs = np.array([-demand.mass_flow(t) * 1000.0 for t in times])
        result[stream.label] = flows_gs
    return result
