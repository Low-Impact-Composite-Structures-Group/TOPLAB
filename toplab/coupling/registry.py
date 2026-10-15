"""Edge-type registry: ``connection_type`` name -> coupling class.

A new edge type is one file with a class decorated by ``@register_edge("name")`` that
implements ``from_edge``; nothing else in the package needs editing.
"""

from typing import Callable, Dict, Type

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling
from toplab.network.network import CouplingEdge
from toplab.peripheral_components.factory import build_peripheral_component_chain

EDGE_TYPES: Dict[str, Type[InterTankCoupling]] = {}


def register_edge(*names: str) -> Callable[[Type[InterTankCoupling]], Type[InterTankCoupling]]:
    def decorate(cls: Type[InterTankCoupling]) -> Type[InterTankCoupling]:
        for name in names:
            if name in EDGE_TYPES and EDGE_TYPES[name] is not cls:
                raise ValueError(f"Edge type {name!r} is already registered to {EDGE_TYPES[name].__name__}")
            EDGE_TYPES[name] = cls
        return cls
    return decorate


def build_coupling(edge: CouplingEdge, ctx: EdgeContext) -> InterTankCoupling:
    """Build the coupling for ``edge`` and attach its conditioning component chains."""
    cls = EDGE_TYPES.get(edge.coupling_type)
    if cls is None:
        raise ConfigError(
            f"Edge {edge.edge_id!r}: unknown connection_type {edge.coupling_type!r}. "
            f"Available: {', '.join(sorted(EDGE_TYPES))}")
    coupling = cls.from_edge(edge, ctx)
    _attach_conditioning(coupling, edge.options)
    return coupling


def _attach_conditioning(coupling: InterTankCoupling, options: dict) -> None:
    main = options.get("main_conditioning_components") or []
    split = options.get("peripheral_components") or []
    discharge = options.get("discharge_conditioning") or []

    # ODE chain: main conditioning (whole outflow) followed by edge-specific components.
    if main or split:
        coupling.set_component_chain(build_peripheral_component_chain(main + split))
    # Chains kept separately for post-processing (the stream that bypasses the edge components,
    # and the conditioning applied after a sink discharge).
    if main and hasattr(coupling, "main_conditioning_chain"):
        coupling.main_conditioning_chain = build_peripheral_component_chain(main)
    if discharge and hasattr(coupling, "discharge_conditioning_chain"):
        coupling.discharge_conditioning_chain = build_peripheral_component_chain(discharge)
