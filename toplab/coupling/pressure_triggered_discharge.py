"""Hysteresis discharge valve from a tank to a sink."""

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling
from toplab.coupling.registry import register_edge
from toplab.network.network import CouplingEdge


@register_edge("pressure_triggered_discharge")
class PressureTriggeredDischarge(InterTankCoupling):
    """
    Pressure-triggered discharge valve: removes mass from the source tank and
    routes it to a downstream sink (target_tank = -1) when source pressure
    exceeds *open_pressure*.  A hysteresis band closes the valve again when
    pressure falls below *close_pressure*.

    Topology::

        [source] --valve(P_src > P_open)--> [sink / fuel cell]

    This is the "return path" for a GH2 buffer: excess pressure bleeds through
    the valve toward the merger junction and ultimately the fuel cell.

    Parameters
    ----------
    source_idx : int    (0-based index of the source tank)
    open_pressure : float   [Pa]  Valve opens when source P ≥ this value.
    close_pressure : float  [Pa]  Valve closes when source P ≤ this value.
    max_flow_rate : float   [kg/s] Maximum discharge rate when fully open.
    coupling_id : str, optional
    """

    def __init__(self, source_idx: int,
                 open_pressure: float,
                 close_pressure: float,
                 max_flow_rate: float = 0.05,
                 coupling_id: str = None):
        # target_tank = -1 signals "discharge to sink" in _calculate_coupling_flows
        super().__init__(source_idx, -1, coupling_id or f"P_discharge_{source_idx}→sink")
        if close_pressure >= open_pressure:
            raise ValueError(
                f"close_pressure ({close_pressure/1e5:.1f} bar) must be < "
                f"open_pressure ({open_pressure/1e5:.1f} bar)"
            )
        self.open_pressure = open_pressure
        self.close_pressure = close_pressure
        self.max_flow_rate = max_flow_rate
        self._valve_open = False
        self.discharge_conditioning_chain = []  # components applied after discharge (before fuel cell)

    def _update_valve_state(self, source_pressure: float) -> None:
        if self._valve_open and source_pressure <= self.close_pressure:
            self._valve_open = False
        elif not self._valve_open and source_pressure >= self.open_pressure:
            self._valve_open = True

    def _compute_flow_rate(self, source_pressure: float) -> float:
        """Linear ramp from 0 at close_pressure to max_flow_rate at open_pressure."""
        if not self._valve_open:
            return 0.0
        band = max(self.open_pressure - self.close_pressure, 1.0)
        fraction = min(1.0, max(0.0, (source_pressure - self.close_pressure) / band))
        return self.max_flow_rate * fraction

    def calculate_flow(self, source_state, target_state, t: float) -> float:
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        p_src = getattr(source_state, 'pressure', None) or 0.0
        self._update_valve_state(p_src)
        return self._compute_flow_rate(p_src)

    def calculate_flow_rate(self, t: float, tank_states: list) -> float:
        src = tank_states[self.source_idx]
        return self.calculate_flow(src, None, t)

    def evaluate(self, time_s, source_tank, dest_tank):
        return self.calculate_flow(source_tank, dest_tank, time_s)

    @classmethod
    def from_edge(cls, edge: CouplingEdge, ctx: EdgeContext) -> "PressureTriggeredDischarge":
        activation = edge.options.get("activation_conditions", {})
        flow = edge.options.get("flow_parameters", {})
        try:
            return cls(edge.source,
                       open_pressure=float(activation.get("pressure_open_bar", 11.0)) * 1e5,
                       close_pressure=float(activation.get("pressure_close_bar", 10.0)) * 1e5,
                       max_flow_rate=float(flow.get("max_flow_rate_kg_s", 0.05)),
                       coupling_id=edge.edge_id)
        except ValueError as err:
            raise ConfigError(f"Edge {edge.edge_id!r}: {err}") from err
