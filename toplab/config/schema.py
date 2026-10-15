"""
Typed analysis configuration.

A YAML file is parsed exactly once (see :func:`toplab.config.loader.load_config`) into an
:class:`AnalysisConfig`. Everything downstream consumes these objects; nothing re-reads the file.

Sections whose structure is owned by a later refactor phase (``physics``, ``solver``,
``output``, ``packaging``, ``dormancy_check``) are kept as plain dicts for now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

NodeId = Any  # YAML node ids are ints today; strings are allowed too.


@dataclass(frozen=True)
class TankSpec:
    node_id: NodeId
    fluid: str
    geometry: Dict[str, Any] = field(default_factory=dict)
    initial_conditions: Dict[str, Any] = field(default_factory=dict)
    operating_limits: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    stopping_criteria: Dict[str, Any] = field(default_factory=dict)
    plotting: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    coolprop_fluid: str = ""

    @property
    def coolprop(self) -> str:
        """CoolProp fluid name used for every property call on this tank."""
        from toplab.fluids.fluid import resolve_coolprop_fluid

        return resolve_coolprop_fluid(self.fluid, self.coolprop_fluid or None)

    def geometry_data(self) -> Dict[str, Any]:
        """Flat geometry/initial-condition/limit dict (transitional; replaced by pure sizing)."""
        data: Dict[str, Any] = dict(self.geometry)
        for key, value in self.initial_conditions.items():
            data[f"initial_{key}"] = value
        data.update(self.operating_limits)
        if "minimum_density" in self.stopping_criteria:
            data["minimum_density"] = float(self.stopping_criteria["minimum_density"])
        data["fluid"] = self.fluid
        return data

    def nist_materials(self) -> Dict[str, Any]:
        """Instantiate the NIST material objects declared in this tank's ``materials`` block."""
        from toplab.materials.nist_materials import get_material_by_nist_path

        loaded: Dict[str, Any] = {}
        for name, data in self.materials.items():
            if name == "safety_margin" or not isinstance(data, dict):
                continue
            nist_path = data.get("nist_path")
            if not nist_path:
                continue
            angle_deg = data.get("winding_angle")
            angle_rad = math.radians(angle_deg) if angle_deg is not None else None
            loaded[name] = get_material_by_nist_path(nist_path, winding_angle=angle_rad)
        return loaded


@dataclass(frozen=True)
class JunctionSpec:
    node_id: NodeId
    role: str = ""
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SinkSpec:
    node_id: NodeId
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: Optional[str]
    connection_type: Optional[str]
    source: NodeId
    target: NodeId
    flow_fraction: Optional[float] = None
    split_fraction: Optional[float] = None
    # Everything else on the edge (activation_conditions, flow_parameters, components, ...).
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MissionSpec:
    type: str
    profile: str
    ambient_temperature: float = 288.15
    assigned_to_node: Optional[NodeId] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    # Profile-specific extras such as flow_rate, duration, key for ``constant_flow``.
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    name: str
    description: str
    version: str
    tanks: List[TankSpec]
    junctions: List[JunctionSpec]
    sinks: List[SinkSpec]
    edges: List[EdgeSpec]
    missions: List[MissionSpec]
    physics: Dict[str, Any] = field(default_factory=dict)
    solver: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    packaging: Optional[Dict[str, Any]] = None
    dormancy_check: Optional[Dict[str, Any]] = None
    stopping_criteria: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def tank(self, node_id: NodeId) -> TankSpec:
        for tank in self.tanks:
            if tank.node_id == node_id:
                return tank
        raise KeyError(f"No tank with node_id {node_id!r}")

    @property
    def tank_ids(self) -> List[NodeId]:
        return [t.node_id for t in self.tanks]

    @property
    def output_plots(self) -> Dict[str, Any]:
        return (self.output or {}).get("plots", {}) or {}
