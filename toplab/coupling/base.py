"""Base class and build context for inter-tank coupling edges."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from toplab.legacy.flow_physics import FlowPhysics
from toplab.network.network import CouplingEdge
from toplab.peripheral_components.base import PeripheralFlowState


@dataclass
class EdgeContext:
    """System facts an edge may need when it is built from its config."""
    minimum_pressures: Dict[int, float]              # sized minimum pressure [Pa] per tank index
    fluids: Dict[int, str]                           # CoolProp fluid per tank index
    flow_physics: Optional[FlowPhysics] = None       # only for the controlled valves (Phase 7)
    mission_data: Dict[str, Any] = field(default_factory=dict)  # time_s / flow_rate_kg_s of the primary mission


@dataclass
class ModeSwitch:
    """A discrete mode change of an edge, triggered when a tank pressure crosses a threshold.

    ``direction`` is +1 for a rising crossing and -1 for a falling one. The solver turns each switch
    into a terminal event and calls ``apply`` when it fires, then restarts the integration.
    """
    tank_idx: int
    pressure_pa: float
    direction: int
    apply: Callable[[], None]
    label: str = ""


class InterTankCoupling:
    """Base class for inter-tank mass transfer mechanisms.

    Subclasses register under one or more ``connection_type`` names with ``register_edge`` and
    implement ``from_edge``, which parses and validates their own options.
    """

    @classmethod
    def from_edge(cls, edge: CouplingEdge, ctx: EdgeContext) -> "InterTankCoupling":
        raise NotImplementedError(f"{cls.__name__} must implement from_edge()")

    def __init__(self, source_idx: int, target_idx: int, coupling_id: str = None):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.coupling_id = coupling_id or f"Coupling_{source_idx}→{target_idx}"
        self.is_active = False
        self.component_chain = []
        # Names the system uses when routing flows between tanks (target -1 = sink)
        self.source_tank = source_idx
        self.target_tank = target_idx

    # --- Solver interface -------------------------------------------------------------------
    # Edge-owned continuous states (e.g. valve opening) are appended to the system state vector.
    # ``flow`` must be a pure function of (t, tank_states, x); discrete mode changes happen only
    # through ``mode_switches`` events, never inside the right-hand side.
    n_states = 0

    def initialize(self, tank_states: List) -> List[float]:
        """Set the initial mode from the t0 tank states and return the initial edge states."""
        return []

    def state_derivatives(self, t: float, tank_states: List, x) -> List[float]:
        return []

    def flow(self, t: float, tank_states: List, x) -> float:
        """Mass flow [kg/s] from source to target (target -1 = sink)."""
        target = tank_states[self.target_idx] if self.target_idx >= 0 else None
        return self.calculate_flow(tank_states[self.source_idx], target, t)

    def mode_switches(self) -> List[ModeSwitch]:
        """Switches that can fire from the current mode."""
        return []

    def evaluate(self, time_s, source_tank, dest_tank):
        """Evaluate the coupling (base implementation - should be overridden)."""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate mass flow rate [kg/s] when coupling is active."""
        raise NotImplementedError("Subclasses must implement calculate_flow_rate()")

    def set_component_chain(self, component_chain: List) -> None:
        self.component_chain = list(component_chain or [])

    def get_delivered_enthalpy(self, source_state, target_state, mass_flow_rate: float) -> float:
        if mass_flow_rate <= 0.0:
            return 0.0

        stream = PeripheralFlowState.from_tank_state(source_state, mass_flow_rate)
        target_pressure = None
        if target_state is not None:
            target_pressure = getattr(target_state, 'pressure', None)
            if target_pressure is None and hasattr(target_state, 'compute_pressure'):
                target_pressure = target_state.compute_pressure()

        for component in self.component_chain:
            stream = component.process_stream(stream, target_pressure=target_pressure)

        return stream.resolved().enthalpy


def as_float(x: Any) -> float:
    """Coerce a scalar, array or sequence to a Python float."""
    try:
        return float(x)
    except Exception:
        try:
            return float(x[0])
        except Exception:
            import numpy as np
            arr = np.asarray(x)
            return float(arr.item() if arr.shape == () else arr.flatten()[0])
