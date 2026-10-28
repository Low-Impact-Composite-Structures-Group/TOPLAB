"""
Typed network of tanks, junctions, sinks and edges.

The Network owns node-id → tank-index resolution (no ``id - 1`` arithmetic elsewhere), the
flow-distribution mixers, and the translation of the declarative edge list into coupling edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from toplab.config.loader import ConfigError
from toplab.config.schema import AnalysisConfig, EdgeSpec, NodeId

SINK = -1  # target index of a coupling edge that discharges out of the system


@dataclass(frozen=True)
class Mixer:
    """Junction that merges tank outflows; ``fractions`` maps tank node id -> share (sums to 1).

    ``outlet_pressure`` [Pa] is the pressure the merged stream is throttled to (junction option
    ``outlet_pressure_bar``); ``None`` means the lowest inlet pressure at each instant.
    """
    node_id: NodeId
    fractions: Dict[NodeId, float]
    outlet_pressure: Optional[float] = None


@dataclass(frozen=True)
class CouplingEdge:
    """An inter-tank (or tank-to-sink) coupling, expressed in tank indices."""
    edge_id: str
    coupling_type: str
    source: int
    target: int  # tank index, or SINK
    options: Dict[str, Any] = field(default_factory=dict)


_COMPONENT_KEYS = ("peripheral_components", "main_conditioning_components", "discharge_conditioning")


class Network:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._tank_index: Dict[NodeId, int] = {t.node_id: i for i, t in enumerate(config.tanks)}
        self._junctions = {j.node_id: j for j in config.junctions}
        self._sink_ids = {s.node_id for s in config.sinks}
        self._edges = list(config.edges)

        known = set(self._tank_index) | set(self._junctions) | self._sink_ids
        for edge in self._edges:
            for end in (edge.source, edge.target):
                if end not in known:
                    raise ConfigError(f"Edge {edge.edge_id!r} references unknown node {end!r}")

        self.mixers: Dict[NodeId, Mixer] = self._find_mixers()

    # -- tank indexing -------------------------------------------------------------------------
    @property
    def tank_ids(self) -> List[NodeId]:
        return list(self._tank_index)

    @property
    def tank_count(self) -> int:
        return len(self._tank_index)

    def index_of(self, node_id: NodeId) -> int:
        try:
            return self._tank_index[node_id]
        except KeyError:
            raise ConfigError(f"Node {node_id!r} is not a tank") from None

    def is_tank(self, node_id: NodeId) -> bool:
        return node_id in self._tank_index

    # -- demand routing ------------------------------------------------------------------------
    def demand_targets(self, node_id: Optional[NodeId]) -> Dict[int, float]:
        """Tank index -> share of the demand assigned to ``node_id`` (default: the first tank)."""
        if node_id is None:
            return {0: 1.0}
        if node_id in self._tank_index:
            return {self._tank_index[node_id]: 1.0}
        if node_id in self.mixers:
            return {self._tank_index[t]: f for t, f in self.mixers[node_id].fractions.items()}
        raise ConfigError(
            f"Mission assigned to node {node_id!r}, which is neither a tank nor a flow mixer"
        )

    # -- mixers --------------------------------------------------------------------------------
    def _incoming(self, node_id: NodeId) -> List[EdgeSpec]:
        return [e for e in self._edges if e.target == node_id]

    def _outgoing(self, node_id: NodeId) -> List[EdgeSpec]:
        return [e for e in self._edges if e.source == node_id]

    def _find_mixers(self) -> Dict[NodeId, Mixer]:
        """Mixer junctions fed only by plain ``discharge`` edges from tanks."""
        mixers: Dict[NodeId, Mixer] = {}
        for node_id, junction in self._junctions.items():
            if junction.role != "mixer":
                continue
            incoming = self._incoming(node_id)
            if not incoming:
                continue
            if not all(self.is_tank(e.source) and (e.connection_type or "discharge") == "discharge"
                       for e in incoming):
                continue

            sources = [e.source for e in incoming]
            if len(set(sources)) != len(sources):
                raise ConfigError(f"Mixer {node_id!r} has more than one edge from the same tank")

            given = [e.flow_fraction for e in incoming]
            if all(f is None for f in given):
                fractions = [1.0 / len(incoming)] * len(incoming)
            elif any(f is None for f in given):
                raise ConfigError(f"Mixer {node_id!r}: set 'flow_fraction' on all incoming edges or on none")
            else:
                fractions = [float(f) for f in given]
                if any(f < 0.0 for f in fractions):
                    raise ConfigError(f"Mixer {node_id!r}: flow_fraction values must be >= 0")
                if abs(sum(fractions) - 1.0) > 1e-6:
                    raise ConfigError(
                        f"Mixer {node_id!r}: flow_fraction values must sum to 1 (got {sum(fractions):.6g})"
                    )
            outlet_bar = junction.options.get("outlet_pressure_bar")
            if outlet_bar is not None and float(outlet_bar) <= 0.0:
                raise ConfigError(f"Mixer {node_id!r}: outlet_pressure_bar must be > 0")
            mixers[node_id] = Mixer(node_id, dict(zip(sources, fractions)),
                                    None if outlet_bar is None else float(outlet_bar) * 1e5)
        return mixers

    # -- coupling edges ------------------------------------------------------------------------
    def coupling_edges(self) -> List[CouplingEdge]:
        """Translate the edge list into tank-to-tank / tank-to-sink coupling edges."""
        couplings: List[CouplingEdge] = []
        handled = set()  # ids of consumed EdgeSpecs

        def mark(edges: List[EdgeSpec]) -> None:
            handled.update(id(e) for e in edges)

        for junction in self.config.junctions:
            node_id = junction.node_id
            incoming, outgoing = self._incoming(node_id), self._outgoing(node_id)

            if node_id in self.mixers:
                mark(incoming + outgoing)
                continue
            if not incoming or not outgoing:
                continue

            source_edge = next((e for e in incoming if self.is_tank(e.source)), None)
            if source_edge is None:
                continue
            source = self.index_of(source_edge.source)

            if junction.role == "mixer" or any(e.connection_type == "pressure_triggered_discharge" for e in incoming):
                discharge_edge = next(
                    (e for e in incoming if e.connection_type == "pressure_triggered_discharge"), source_edge)
                outlet_edge = next((e for e in outgoing if self.is_tank(e.target)), outgoing[0])
                coupling = self._coupling(discharge_edge, "pressure_triggered_discharge", source, SINK)
                conditioning = outlet_edge.options.get("discharge_conditioning",
                                                       outlet_edge.options.get("components", []))
                if conditioning:
                    coupling.options["discharge_conditioning"] = conditioning
                couplings.append(coupling)
                mark(incoming + outgoing)
                continue

            split_edges = [e for e in outgoing if e.split_fraction is not None]
            if not split_edges:
                continue
            peripheral = min(split_edges, key=lambda e: e.split_fraction)
            main = next((e for e in outgoing if e is not peripheral), peripheral)
            if not self.is_tank(peripheral.target):
                continue

            coupling = self._coupling(peripheral, "proportional_split", source, self.index_of(peripheral.target))
            main_components = main.options.get("main_conditioning_components", main.options.get("components", []))
            peripheral_components = peripheral.options.get(
                "peripheral_components", peripheral.options.get("components", []))
            if main_components:
                coupling.options["main_conditioning_components"] = main_components
            if peripheral_components:
                coupling.options["peripheral_components"] = peripheral_components
            couplings.append(coupling)
            mark(incoming + outgoing)

        for edge in self._edges:
            if id(edge) in handled:
                continue
            ctype = edge.connection_type or "pressure_compensation"
            if ctype == "pressure_triggered_discharge" and self.is_tank(edge.source):
                couplings.append(self._coupling(edge, ctype, self.index_of(edge.source), SINK))
            elif self.is_tank(edge.source) and self.is_tank(edge.target):
                couplings.append(
                    self._coupling(edge, ctype, self.index_of(edge.source), self.index_of(edge.target)))
        return couplings

    def _coupling(self, edge: EdgeSpec, coupling_type: str, source: int, target: int) -> CouplingEdge:
        options = dict(edge.options)
        for key in ("activation_conditions", "control_parameters", "flow_parameters",
                    "flow_physics", "discharge_piping"):
            options.setdefault(key, {})
        options["peripheral_components"] = edge.options.get("peripheral_components",
                                                            edge.options.get("components", []))
        options["main_conditioning_components"] = edge.options.get("main_conditioning_components", [])
        options["discharge_conditioning"] = edge.options.get("discharge_conditioning", [])
        options["split_fraction"] = edge.split_fraction
        edge_id = edge.edge_id or f"coupling_{edge.source}_{edge.target}"
        return CouplingEdge(edge_id, coupling_type, source, target, options)
