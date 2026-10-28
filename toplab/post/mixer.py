"""Mixed-stream state at flow-distribution mixer junctions, derived from simulation results.

Each tank feeding a mixer discharges at its own (T, P, h). The mixer throttles every inlet
isenthalpically to the outlet pressure and merges them adiabatically, so

    m_out = sum(m_i)          h_out = sum(m_i * h_i) / m_out          T_out = T(P_out, h_out)

``P_out`` is the mixer's configured ``outlet_pressure_bar`` or, when unset, the lowest inlet
pressure at that instant (a regulator cannot raise pressure). The mixer itself adds no heat, so
the enthalpy flux in equals the enthalpy flux out; the residuals returned here verify that
against the demand actually requested of the tanks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from CoolProp.CoolProp import PropsSI

from toplab.network.network import Mixer, Network

_MIN_FLOW_GS = 1e-9  # below this a stream is treated as absent


@dataclass
class MixerInlet:
    tank_id: int
    mdot_gs: np.ndarray
    temperature_K: np.ndarray   # tank gas temperature (stream leaves at tank state)
    pressure_bar: np.ndarray    # tank pressure
    enthalpy_J_kg: np.ndarray


@dataclass
class MixerStreamHistory:
    mixer_id: int
    times_h: np.ndarray
    inlets: List[MixerInlet]
    mdot_gs: np.ndarray                 # merged stream
    temperature_K: np.ndarray
    pressure_bar: np.ndarray
    enthalpy_J_kg: np.ndarray
    demand_gs: np.ndarray               # mission demand requested of the mixer's tanks
    mass_residual_gs: np.ndarray        # sum of inlet flows - demand
    enthalpy_residual_W: np.ndarray     # outlet enthalpy flux h(P_out, T_out)*m - sum of inlet enthalpy fluxes

    def cumulative_mass_kg(self) -> dict:
        """Mass passed by each inlet, the outlet and the demand, from t=0 [kg]."""
        t = self.times_h * 3600.0
        def integral(q):
            q = np.nan_to_num(q)
            return np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(t))]) / 1000.0
        out = {f"tank_{i.tank_id}": integral(i.mdot_gs) for i in self.inlets}
        out["outlet"] = integral(self.mdot_gs)
        out["demand"] = integral(self.demand_gs)
        return out


def mix_streams(mdots_kg_s: Sequence[float], enthalpies: Sequence[float],
                pressures_pa: Sequence[float], fluid: str, outlet_pressure: Optional[float] = None):
    """Merge streams; returns ``(mdot, h, P_out, T_out)`` or ``None`` when nothing flows.

    Streams below ``_MIN_FLOW_GS`` are ignored. ``outlet_pressure`` is capped at the lowest
    inlet pressure of the streams that are actually flowing.
    """
    flowing = [k for k, m in enumerate(mdots_kg_s) if np.isfinite(m) and m * 1000.0 > _MIN_FLOW_GS]
    if not flowing:
        return None
    m = sum(mdots_kg_s[k] for k in flowing)
    h = sum(mdots_kg_s[k] * enthalpies[k] for k in flowing) / m
    p_limit = min(pressures_pa[k] for k in flowing)
    p_out = p_limit if outlet_pressure is None else min(outlet_pressure, p_limit)
    return m, h, p_out, PropsSI("T", "P", p_out, "Hmass", h, fluid)


def mixer_stream_history(network: Network, system, results) -> List[MixerStreamHistory]:
    """One :class:`MixerStreamHistory` per mixer junction in ``network``."""
    return [_mixer_history(mixer, network, system, results) for mixer in network.mixers.values()]


def _mixer_history(mixer: Mixer, network: Network, system, results) -> MixerStreamHistory:
    indices = [network.index_of(tank_id) for tank_id in mixer.fractions]
    fluids = {system._coolprop_fluids[i] for i in indices}
    if len(fluids) != 1:
        raise ValueError(f"Mixer {mixer.node_id}: tanks use different fluids {sorted(fluids)}")
    fluid = fluids.pop()

    times = np.asarray(results.times, dtype=float)
    n = len(times)
    raw = [results._extract_tank_arrays(i) for i in indices]

    inlets = []
    for tank_id, i, tank in zip(mixer.fractions, indices, raw):
        h = np.array([PropsSI("Hmass", "T", T, "Dmass", rho, fluid)
                      for T, rho in zip(tank["h2_temperatures"], tank["densities"])])
        inlets.append(MixerInlet(tank_id, np.asarray(tank["outflow_rates"], dtype=float),
                                 np.asarray(tank["h2_temperatures"], dtype=float),
                                 np.asarray(tank["pressures"], dtype=float), h))

    demand = np.array([sum(system._get_outflow_rate(t, i) for i in indices) for t in times]) * 1000.0

    out_m, out_T, out_P, out_h, out_h_state = (np.full(n, np.nan) for _ in range(5))
    for k in range(n):
        merged = mix_streams([s.mdot_gs[k] / 1000.0 for s in inlets],
                             [s.enthalpy_J_kg[k] for s in inlets],
                             [s.pressure_bar[k] * 1e5 for s in inlets], fluid, mixer.outlet_pressure)
        if merged is None:
            out_m[k] = 0.0
            continue
        m, h, p, T = merged
        out_m[k], out_h[k], out_P[k], out_T[k] = m * 1000.0, h, p / 1e5, T
        try:
            out_h_state[k] = PropsSI("Hmass", "P", p, "T", T, fluid)
        except ValueError:  # (P, T) exactly on the saturation curve is ambiguous
            out_h_state[k] = h

    flux_in = sum(np.nan_to_num(s.mdot_gs) / 1000.0 * s.enthalpy_J_kg for s in inlets)
    # Enthalpy re-evaluated from the mixed (P, T) state, so the residual tests the mixing result
    # rather than restating the weighted average that produced it.
    flux_out = np.nan_to_num(out_m) / 1000.0 * np.nan_to_num(out_h_state)
    return MixerStreamHistory(
        mixer_id=mixer.node_id, times_h=times / 3600.0, inlets=inlets, mdot_gs=out_m,
        temperature_K=out_T, pressure_bar=out_P, enthalpy_J_kg=out_h, demand_gs=demand,
        mass_residual_gs=sum(s.mdot_gs for s in inlets) - demand,
        enthalpy_residual_W=flux_out - flux_in)
