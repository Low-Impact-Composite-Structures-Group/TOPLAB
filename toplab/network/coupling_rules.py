"""
Translate :class:`~toplab.network.network.CouplingEdge` objects into the rule dicts consumed by
``TankSystem._setup_coupling_rules``.

Transitional: once each edge type registers its own builder (refactor Phase 6), this module goes away.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from toplab.fluids.valve_flow import ValvePath, path_from_options
from toplab.network.network import CouplingEdge

BAR = 1e5


def to_tank_system_rules(couplings: List[CouplingEdge]) -> List[Dict[str, Any]]:
    rules = []
    for coupling in couplings:
        builder = _BUILDERS.get(coupling.coupling_type)
        if builder is None:
            print(f"WARNING: Unsupported coupling type '{coupling.coupling_type}' - skipping")
            continue
        rule = builder(coupling)
        for key in ("peripheral_components", "main_conditioning_components", "discharge_conditioning"):
            if coupling.options.get(key):
                rule[key] = coupling.options[key]
        rules.append(rule)
    return rules


def _valve_params(c: CouplingEdge, default_flow: float, default_orifice: float) -> Dict[str, Any]:
    flow = c.options.get("flow_parameters", {})
    return {
        "max_flow_rate": flow.get("max_flow_rate_kg_s", default_flow),
        "orifice_diameter": flow.get("orifice_diameter_m", default_orifice),
    }


def _valve_path(c: CouplingEdge, default_orifice: float) -> ValvePath:
    """Pipe/orifice description of a pressure-triggered valve edge."""
    if c.options.get("valve"):
        return path_from_options(c.options)
    flow = c.options.get("flow_parameters", {})
    return ValvePath(orifice_diameter_m=float(flow.get("orifice_diameter_m", default_orifice)))


def _controlled_valve(rule_type: str, default_id: str) -> Callable[[CouplingEdge], Dict[str, Any]]:
    def build(c: CouplingEdge) -> Dict[str, Any]:
        return {
            "type": rule_type,
            "source_tank": c.source,
            "target_tank": c.target,
            "mission_profile": {},  # injected by TankSystem from the demands
            "discharge_piping": c.options.get("discharge_piping", {}),
            "control_params": c.options.get("control_parameters", {}),
            **_valve_params(c, 0.005, 0.001),
            "coupling_id": c.edge_id or default_id,
        }
    return build


def _flow_controlled_pressurization(c: CouplingEdge) -> Dict[str, Any]:
    hysteresis = c.options.get("hysteresis", {})
    for key in ("activation_threshold_bar", "deactivation_threshold_bar"):
        if key not in hysteresis:
            raise RuntimeError(
                f"Missing required hysteresis field '{key}' for flow_controlled_pressurization "
                f"coupling '{c.edge_id}'."
            )
    return {
        "type": "pressure_triggered_valve",
        "source_tank": c.source,
        "target_tank": c.target,
        "opening_pressure": hysteresis["activation_threshold_bar"] * BAR,
        "closing_pressure": hysteresis["deactivation_threshold_bar"] * BAR,
        "valve_path": _valve_path(c, 0.01),
    }


def _pressure_compensation(c: CouplingEdge) -> Dict[str, Any]:
    thresholds = c.options.get("activation_conditions", {})
    if thresholds:
        open_bar = float(thresholds.get("pressure_open_bar", 20.0))
        close_bar = float(thresholds.get("pressure_close_bar", 21.0))
    else:
        hysteresis = c.options.get("hysteresis", {})
        open_bar = float(hysteresis.get("activation_threshold_bar", 20.0))
        close_bar = float(hysteresis.get("deactivation_threshold_bar", 21.0))

    if close_bar <= open_bar:
        print(f"WARNING: Adjusting close threshold of '{c.edge_id}': {close_bar} <= {open_bar} bar -> +1 bar gap")
        close_bar = open_bar + 1.0

    return {
        "type": "pressure_triggered_valve",
        "source_tank": c.source,
        "target_tank": c.target,
        "opening_pressure": open_bar * BAR,
        "closing_pressure": close_bar * BAR,
        "valve_path": _valve_path(c, 0.002),
        "valve_time_constant_s": float(thresholds.get("valve_time_constant_s", 0.5)),
        "coupling_id": c.edge_id,
    }


def _flow_matching_pressurization(c: CouplingEdge) -> Dict[str, Any]:
    rule = _controlled_valve("mass_flow_pid_valve", "flow_matching_pressurization")(c)
    rule["mission_profile"] = c.options.get("mission_parameters", {}).get("mission_profile", {})
    return rule


def _ohex_extraction(c: CouplingEdge) -> Dict[str, Any]:
    mission = c.options.get("mission_parameters", {})
    extraction = c.options.get("extraction_parameters", {})
    return {
        "type": "ohex_extraction",
        "source_tank": c.source,
        "mission_profile": mission.get("mission_profile", {}),
        "min_extraction_pressure": extraction.get("min_extraction_pressure_bar", 3.0) * BAR,
        "coupling_id": c.edge_id,
    }


def _proportional_split(c: CouplingEdge) -> Dict[str, Any]:
    fraction = c.options.get("split_fraction")
    return {
        "type": "proportional_split",
        "source_tank": c.source,
        "target_tank": c.target,
        "split_fraction": float(fraction if fraction is not None else 0.5),
        "coupling_id": c.edge_id,
    }


def _pressure_triggered_discharge(c: CouplingEdge) -> Dict[str, Any]:
    activation = c.options.get("activation_conditions", {})
    flow = c.options.get("flow_parameters", {})
    return {
        "type": "pressure_triggered_discharge",
        "source_tank": c.source,
        "open_pressure": float(activation.get("pressure_open_bar", 11.0)) * BAR,
        "close_pressure": float(activation.get("pressure_close_bar", 10.0)) * BAR,
        "max_flow_rate": float(flow.get("max_flow_rate_kg_s", 0.05)),
        "coupling_id": c.edge_id,
    }


_BUILDERS: Dict[str, Callable[[CouplingEdge], Dict[str, Any]]] = {
    "flow_controlled_pressurization": _flow_controlled_pressurization,
    "mission_adaptive_pressurization": _controlled_valve(
        "mission_adaptive_pressure_valve", "mission_adaptive_pressurization"),
    "pressure_governor": _controlled_valve("pressure_governor", "pressure_governor"),
    "feedforward_pressure_enforcer": _controlled_valve(
        "feedforward_pressure_enforcer", "feedforward_pressure_enforcer"),
    "flow_matching_pressurization": _flow_matching_pressurization,
    "pressure_compensation": _pressure_compensation,
    "ohex_extraction": _ohex_extraction,
    "proportional_split": _proportional_split,
    "pressure_triggered_discharge": _pressure_triggered_discharge,
}
