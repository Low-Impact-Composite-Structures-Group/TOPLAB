"""Parse and validate an analysis YAML file into an :class:`AnalysisConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from toplab.config.schema import (
    AnalysisConfig,
    EdgeSpec,
    JunctionSpec,
    MissionSpec,
    SinkSpec,
    TankSpec,
)

_EDGE_STRUCTURAL_KEYS = {
    "edge_id", "connection_type", "participants", "from_node", "to_node",
    "flow_fraction", "split_fraction",
}


class ConfigError(ValueError):
    """Raised for malformed analysis configuration."""


def load_config(path: Union[str, Path]) -> AnalysisConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return parse_config(raw, source_path=path)


def parse_config(raw: Dict[str, Any], source_path: Optional[Path] = None) -> AnalysisConfig:
    if not isinstance(raw, dict) or "network" not in raw:
        raise ConfigError(f"{source_path or 'config'}: expected a mapping with a 'network' section")

    analysis = raw.get("analysis", {}) or {}
    tanks, junctions, sinks = _parse_nodes((raw["network"] or {}).get("nodes", []) or [])
    edges = [_parse_edge(e) for e in (raw["network"] or {}).get("edges", []) or []]
    missions = _parse_missions(raw, source_path)

    if not tanks:
        raise ConfigError("At least one tank node is required")

    return AnalysisConfig(
        name=analysis.get("name", "Unnamed Analysis"),
        description=analysis.get("description", ""),
        version=str(analysis.get("version", "1.0")),
        tanks=tanks,
        junctions=junctions,
        sinks=sinks,
        edges=edges,
        missions=missions,
        physics=raw.get("physics") or raw.get("flow_physics") or {},
        solver=raw.get("solver") or {},
        output=raw.get("output") or {},
        packaging=raw.get("packaging"),
        dormancy_check=raw.get("dormancy_check"),
        stopping_criteria=raw.get("stopping_criteria") or {},
        source_path=source_path,
    )


def _parse_nodes(nodes: List[Dict[str, Any]]):
    tanks: List[TankSpec] = []
    junctions: List[JunctionSpec] = []
    sinks: List[SinkSpec] = []
    seen = set()
    for node in nodes:
        node_id = node.get("node_id")
        if node_id is None:
            raise ConfigError(f"Network node without node_id: {node}")
        if node_id in seen:
            raise ConfigError(f"Duplicate node_id {node_id!r}")
        seen.add(node_id)

        kind = node.get("type")
        if kind == "tank":
            tanks.append(TankSpec(
                node_id=node_id,
                fluid=str(node.get("fluid", "")).upper(),
                geometry=dict(node.get("geometry") or {}),
                initial_conditions=dict(node.get("initial_conditions") or {}),
                operating_limits=dict(node.get("operating_limits") or {}),
                materials=dict(node.get("materials") or {}),
                stopping_criteria=dict(node.get("stopping_criteria") or {}),
                plotting=dict(node.get("plotting") or {}),
                description=node.get("description", ""),
                coolprop_fluid=str(node.get("coolprop_fluid", "")),
            ))
            try:
                tanks[-1].coolprop
            except ValueError as exc:
                raise ConfigError(f"Tank {node_id!r}: {exc}") from exc
        elif kind == "junction":
            role = str(node.get("junction_role") or node.get("role") or "").lower()
            junctions.append(JunctionSpec(node_id, role, {
                k: v for k, v in node.items() if k not in {"node_id", "type", "junction_role", "role"}
            }))
        elif kind == "sink":
            sinks.append(SinkSpec(node_id, {k: v for k, v in node.items() if k not in {"node_id", "type"}}))
        else:
            raise ConfigError(f"Node {node_id!r}: unknown type {kind!r} (expected tank, junction or sink)")
    return tanks, junctions, sinks


def _parse_edge(edge: Dict[str, Any]) -> EdgeSpec:
    participants = edge.get("participants") or {}
    source = participants.get("source", edge.get("from_node"))
    target = participants.get("target", edge.get("to_node"))
    if source is None or target is None:
        raise ConfigError(f"Edge {edge.get('edge_id')!r} needs a source and a target")

    def _opt_float(key: str) -> Optional[float]:
        value = edge.get(key)
        return None if value is None else float(value)

    return EdgeSpec(
        edge_id=edge.get("edge_id"),
        connection_type=edge.get("connection_type"),
        source=source,
        target=target,
        flow_fraction=_opt_float("flow_fraction"),
        split_fraction=_opt_float("split_fraction"),
        options={k: v for k, v in edge.items() if k not in _EDGE_STRUCTURAL_KEYS},
    )


def _parse_missions(raw: Dict[str, Any], source_path: Optional[Path]) -> List[MissionSpec]:
    if "mission_sequence" in raw:
        raise ConfigError("'mission_sequence' is no longer supported; use 'mission' or a 'missions' list")

    entries = raw.get("missions")
    if entries is None:
        single = raw.get("mission")
        if not single:
            raise ConfigError("A 'mission' section or a 'missions' list is required")
        entries = [single]
    if not isinstance(entries, list) or not entries:
        raise ConfigError("'missions' must be a non-empty list")

    return [_parse_mission(m, source_path) for m in entries]


_MISSION_KNOWN_KEYS = {"type", "profile", "ambient_temperature", "assigned_to_node", "parameters"}


def _parse_mission(entry: Dict[str, Any], source_path: Optional[Path]) -> MissionSpec:
    if not entry.get("type") or not entry.get("profile"):
        raise ConfigError(f"Mission needs both 'type' and 'profile': {entry}")

    parameters = dict(entry.get("parameters") or {})
    if parameters.get("csv_file"):
        parameters["csv_file"] = str(_resolve_csv(parameters["csv_file"], source_path))

    return MissionSpec(
        type=str(entry["type"]),
        profile=str(entry["profile"]),
        ambient_temperature=float(entry.get("ambient_temperature", 288.15)),
        assigned_to_node=entry.get("assigned_to_node"),
        parameters=parameters,
        options={k: v for k, v in entry.items() if k not in _MISSION_KNOWN_KEYS},
    )


def _resolve_csv(csv_file: str, source_path: Optional[Path]) -> Path:
    csv_path = Path(csv_file)
    if csv_path.is_absolute() or source_path is None:
        return csv_path
    config_dir = source_path.parent
    resolved = config_dir / csv_path
    if resolved.exists():
        return resolved
    # Staged jobs copy inputs next to the configuration file.
    staged = config_dir / csv_path.name
    return staged if staged.exists() else resolved
