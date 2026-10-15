"""Routes a fixed fraction of a tank's mission discharge to another tank."""

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling
from toplab.coupling.registry import register_edge
from toplab.network.network import CouplingEdge


@register_edge("proportional_split")
class ProportionalSplitCoupling(InterTankCoupling):
    """
    Proportional-split coupling: routes a fixed fraction of the source tank's
    mission-driven discharge to the target tank via an optional component chain.

    Topology::

        [source] --split_fraction * mdot_mission(t)--> [component chain] --> [target]

    Typical use: divert 5 % of an LH2 discharge through a HEX + compressor
    into a GH2 buffer tank.

    Parameters
    ----------
    source_idx : int   (0-based)
    target_idx : int   (0-based)
    split_fraction : float   Fraction of source mission discharge to route to target (0–1).
    coupling_id : str, optional
    """

    def __init__(self, source_idx: int, target_idx: int,
                 split_fraction: float,
                 coupling_id: str = None):
        super().__init__(source_idx, target_idx, coupling_id or f"split_{source_idx}→{target_idx}")
        if not (0.0 < split_fraction <= 1.0):
            raise ValueError(f"split_fraction must be in (0, 1]; got {split_fraction}")
        self.split_fraction = split_fraction
        self.mission_times = None
        self.mission_flow_rates = None
        self.main_conditioning_chain = []  # components applied to full Tank 1 outflow (95% path)

    # ------------------------------------------------------------------
    # Mission profile injection
    # ------------------------------------------------------------------

    def set_mission_profile(self, mission_data: dict) -> None:
        """Store mission time/flow arrays for discharge-rate lookup."""
        times = mission_data.get('time_s', [])
        flows = mission_data.get('flow_rate_kg_s', mission_data.get('flow_kg_s', []))
        if times and flows:
            self.mission_times = list(times)
            self.mission_flow_rates = list(flows)
            print(f"   ProportionalSplitCoupling '{self.coupling_id}': "
                  f"mission loaded ({len(self.mission_times)} pts, "
                  f"max={max(self.mission_flow_rates):.4f} kg/s)")

    def _get_source_discharge_rate(self, t: float) -> float:
        """Return the mission discharge rate at time *t* [s].

        The ``(time_s, flow_rate_kg_s)`` arrays built by
        ``_extract_mission_profile_data`` represent a **step function**:
        ``flow_rates[i+1]`` is the constant rate during the interval
        ``[times[i], times[i+1])``.  Returning the interval value (rather
        than linearly interpolating) avoids an artificial ramp from 0 at t=0.
        """
        if not self.mission_times or not self.mission_flow_rates:
            return 0.0
        if t >= self.mission_times[-1]:
            return self.mission_flow_rates[-1]
        for i in range(len(self.mission_times) - 1):
            if self.mission_times[i] <= t < self.mission_times[i + 1]:
                # Step function: rate is the value at the END of the interval
                return self.mission_flow_rates[i + 1]
        # Before first time point — use first non-zero rate
        return self.mission_flow_rates[1] if len(self.mission_flow_rates) > 1 else 0.0

    # ------------------------------------------------------------------
    # InterTankCoupling interface
    # ------------------------------------------------------------------

    def calculate_flow_rate(self, t: float, tank_states: list) -> float:
        return self.split_fraction * self._get_source_discharge_rate(t)

    def calculate_flow(self, source_state, target_state, t: float) -> float:
        return self.split_fraction * self._get_source_discharge_rate(t)

    def evaluate(self, time_s, source_tank, dest_tank):
        return self.calculate_flow(source_tank, dest_tank, time_s)

    @classmethod
    def from_edge(cls, edge: CouplingEdge, ctx: EdgeContext) -> "ProportionalSplitCoupling":
        fraction = edge.options.get("split_fraction")
        try:
            coupling = cls(edge.source, edge.target,
                           split_fraction=float(0.5 if fraction is None else fraction),
                           coupling_id=edge.edge_id)
        except ValueError as err:
            raise ConfigError(f"Edge {edge.edge_id!r}: {err}") from err
        if ctx.mission_data:
            coupling.set_mission_profile(ctx.mission_data)
        return coupling
