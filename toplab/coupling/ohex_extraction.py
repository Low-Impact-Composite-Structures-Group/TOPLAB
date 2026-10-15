"""OHEX extraction: draws the mission flow out of a tank into the outboard heat exchanger."""

from typing import List

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling
from toplab.coupling.registry import register_edge
from toplab.network.network import CouplingEdge


@register_edge("ohex_extraction")
class OHEXExtractionCoupling(InterTankCoupling):
    """OHEX (Outboard Heat Exchanger) extraction coupling for mission fuel demand."""

    def __init__(self, source_idx: int, mission_profile: dict,
                 min_extraction_pressure: float = 3.0e5,
                 coupling_id: str = None):
        # OHEX extraction is unidirectional (source → OHEX), so target_idx = -1 (no target tank)
        super().__init__(source_idx, -1, coupling_id)
        self.mission_profile = mission_profile
        self.min_extraction_pressure = min_extraction_pressure
        self.current_mission_flow_rate = 0.0

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Check if OHEX extraction should be active."""
        # Get mission flow rate requirement
        mission_flow_rate = self._get_mission_flow_rate(t)

        # Only activate if mission requires fuel
        if mission_flow_rate <= 0:
            return False

        # Check if source tank has sufficient pressure for extraction
        source_state = tank_states[self.source_idx]
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()

        source_pressure = source_state.pressure
        return source_pressure >= self.min_extraction_pressure

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate OHEX extraction flow rate based on mission demand."""
        # Get mission flow rate requirement
        mission_flow_rate = self._get_mission_flow_rate(t)
        self.current_mission_flow_rate = mission_flow_rate

        # Check activation conditions
        if not self.evaluate(t, tank_states):
            return 0.0

        # Return mission-required flow rate (positive = extraction from source)
        return mission_flow_rate

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Update source state pressure
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()

        # Create mock tank_states list for compatibility
        tank_states = [None] * (self.source_idx + 1)
        tank_states[self.source_idx] = source_state

        # Call existing calculate_flow_rate method
        return self.calculate_flow_rate(t, tank_states)

    def _get_mission_flow_rate(self, time: float) -> float:
        """
        Get mission flow rate requirement at given time.

        Args:
            time: Current simulation time [s]

        Returns:
            Required mass flow rate [kg/s]
        """
        if not self.mission_profile:
            return 0.0

        # Extract time points and flow rates from mission profile
        time_points = self.mission_profile.get('time_s', [])
        flow_rates = self.mission_profile.get('flow_rate_kg_s', [])

        if not time_points or not flow_rates:
            return 0.0

        # Simple linear interpolation
        if time <= time_points[0]:
            return flow_rates[0]
        elif time >= time_points[-1]:
            return flow_rates[-1]
        else:
            # Find surrounding points and interpolate
            for i in range(len(time_points) - 1):
                if time_points[i] <= time <= time_points[i + 1]:
                    # Linear interpolation
                    t1, t2 = time_points[i], time_points[i + 1]
                    f1, f2 = flow_rates[i], flow_rates[i + 1]
                    return f1 + (f2 - f1) * (time - t1) / (t2 - t1)

        return 0.0

    @classmethod
    def from_edge(cls, edge: CouplingEdge, ctx: EdgeContext) -> "OHEXExtractionCoupling":
        mission = edge.options.get("mission_parameters", {})
        extraction = edge.options.get("extraction_parameters", {})
        return cls(edge.source,
                   mission_profile=mission.get("mission_profile", {}),
                   min_extraction_pressure=float(extraction.get("min_extraction_pressure_bar", 3.0)) * 1e5,
                   coupling_id=edge.edge_id)
