"""Inter-tank coupling edges: one module per edge type, selected through the edge registry."""

from toplab.coupling.base import EdgeContext, InterTankCoupling
from toplab.coupling.registry import EDGE_TYPES, build_coupling, register_edge

# Importing the modules registers their edge types.
from toplab.legacy.feedforward_enforcer import FeedforwardPressureEnforcer
from toplab.coupling.ohex_extraction import OHEXExtractionCoupling
from toplab.coupling.pressure_triggered_discharge import PressureTriggeredDischarge
from toplab.coupling.pressure_triggered_valve import PressureTriggeredValve
from toplab.coupling.proportional_split import ProportionalSplitCoupling

__all__ = [
    "EDGE_TYPES", "EdgeContext", "InterTankCoupling", "build_coupling", "register_edge",
    "FeedforwardPressureEnforcer", "OHEXExtractionCoupling", "PressureTriggeredDischarge",
    "PressureTriggeredValve", "ProportionalSplitCoupling",
]
