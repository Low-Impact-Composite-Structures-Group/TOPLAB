from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple, Union
import bisect

class InteractionRule(ABC):
    """Base class for all tank interaction rules.

    Controls when and how much flow occurs between tanks.
    """
    def __init__(self):
        self._transfer_active = False
        self._max_flow_rate = None

    @abstractmethod
    def evaluate(self,
                tank_states: list,
                current_time: float,
                mission_data=None) -> float:
        """Evaluate the rule and return the desired flow rate.

        Args:
            tank_states: List of tank states (typically source is 0, target is 1)
            current_time: Current simulation time in seconds
            mission_data: Optional mission section data

        Returns:
            Flow rate in kg/s (positive means flow from source to target)
        """
        pass

    def start_transfer(self):
        """Start fuel transfer between tanks."""
        self._transfer_active = True

    def stop_transfer(self):
        """Stop fuel transfer between tanks."""
        self._transfer_active = False

    def is_transfer_active(self) -> bool:
        """Check if transfer is currently active."""
        return self._transfer_active

    def set_max_flow_rate(self, rate: float):
        """Set maximum allowed flow rate."""
        self._max_flow_rate = rate

    def apply_limits(self, flow_rate: float) -> float:
        """Apply flow rate limits (if set) and check if transfer is active."""
        if not self._transfer_active:
            return 0.0

        if self._max_flow_rate is not None:
            return min(flow_rate, self._max_flow_rate)

        return flow_rate


class PrescribedFlow(InteractionRule):
    """Rule with predefined flow behavior, like time-based flow profiles.

    This handles cases where the flow pattern is known in advance.
    """
    def __init__(self, max_flow_rate: Optional[float] = None, active: bool = True):
        """
        Args:
            max_flow_rate: Optional cap on maximum flow rate
            active: Whether transfer starts active
        """
        super().__init__()
        self._max_flow_rate = max_flow_rate
        self._transfer_active = active


class TimeBasedFlow(PrescribedFlow):
    """Flow that changes at specific time points.

    Example:
        0-100s: 0.05 kg/s
        100-200s: 0.1 kg/s
        200+: 0 kg/s (no flow)
    """
    def __init__(self,
                time_flow_pairs: List[Tuple[float, float]],
                max_flow_rate: Optional[float] = None,
                active: bool = True):
        """
        Args:
            time_flow_pairs: List of (time, flow_rate) pairs, sorted by time
                Times are in seconds, flow rates in kg/s
            max_flow_rate: Optional cap on maximum flow rate
            active: Whether transfer starts active
        """
        super().__init__(max_flow_rate, active)
        # Ensure time points are sorted
        self.time_points = [t for t, _ in sorted(time_flow_pairs)]
        self.flow_rates = [f for _, f in sorted(time_flow_pairs)]

    def evaluate(self, tank_states: list, current_time: float, mission_data=None) -> float:
        """Return flow rate based on current time"""
        if not self._transfer_active:
            return 0.0

        # Handle case where time is before first time point
        if current_time <= self.time_points[0]:
            return self.apply_limits(self.flow_rates[0])

        # Handle case where time is after last time point
        if current_time >= self.time_points[-1]:
            return self.apply_limits(self.flow_rates[-1])

        # Find the time points that bracket the current time
        idx = bisect.bisect_right(self.time_points, current_time) - 1
        t1, t2 = self.time_points[idx], self.time_points[idx + 1]
        f1, f2 = self.flow_rates[idx], self.flow_rates[idx + 1]

        # Linear interpolation between time points
        flow = f1 + (f2 - f1) * (current_time - t1) / (t2 - t1)
        return self.apply_limits(flow)


class MissionBasedFlow(PrescribedFlow):
    """Flow based on mission requirements (current implementation).

    Calculates flow from Tank 1 to Tank 2 based on Tank 2's mission outflow.
    """
    def __init__(self,
                safety_factor: float = 0.8,
                max_flow_rate: Optional[float] = None,
                active: bool = True):
        """
        Args:
            safety_factor: Factor to apply to mission outflow (0.8 means transfer is 80% of outflow)
            max_flow_rate: Optional maximum flow rate in kg/s
            active: Whether transfer starts active
        """
        super().__init__(max_flow_rate, active)
        self.safety_factor = safety_factor

    def evaluate(self, tank_states: list, current_time: float, mission_data=None) -> float:
        """Return flow based on mission outflow requirements"""
        if not self._transfer_active or mission_data is None:
            return 0.0

        # Extract mission outflow (similar to existing implementation)
        mission_outflow = 0.0
        from src.mission.mission_sections import OutFlow

        for flow in mission_data.fuel_flows:
            if isinstance(flow, OutFlow):
                if isinstance(flow.mass_flow, list):
                    from src.dynamics.dynamic_analysis import MissionSectionAnalysis
                    # We need section_iter and steps from somewhere
                    section_iter = getattr(mission_data, 'current_step', 0)
                    steps = getattr(mission_data, 'total_steps', 1)
                    mission_outflow += MissionSectionAnalysis.interpolate_mass_flows(
                        flow.mass_flow, section_iter, steps
                    )
                else:
                    mission_outflow += flow.mass_flow

        # Calculate transfer flow (existing logic)
        transfer_flow = abs(mission_outflow) * self.safety_factor
        return self.apply_limits(transfer_flow)


class ConditionalFlow(InteractionRule):
    """Flow based on run-time conditions of the tanks.

    This handles cases where flow control depends on the evolving state.
    """
    def __init__(self, max_flow_rate: Optional[float] = None, active: bool = True):
        """
        Args:
            max_flow_rate: Optional cap on maximum flow rate
            active: Whether transfer starts active
        """
        super().__init__()
        self._max_flow_rate = max_flow_rate
        self._transfer_active = active
        self._conditions = []
        self._default_flow = 0.0

    def add_condition(self,
                     condition_func: Callable[[list, float, Optional[object]], bool],
                     flow_rate: Union[float, Callable[[list, float, Optional[object]], float]]):
        """Add a condition and its corresponding flow rate.

        Args:
            condition_func: Function that returns True if condition is met
            flow_rate: Either a fixed flow rate or a function that calculates the flow
        """
        self._conditions.append((condition_func, flow_rate))
        return self  # Allow chaining

    def set_default_flow(self, flow_rate: float):
        """Set the default flow rate when no conditions are met."""
        self._default_flow = flow_rate
        return self  # Allow chaining

    def evaluate(self, tank_states: list, current_time: float, mission_data=None) -> float:
        """Evaluate all conditions and return the flow rate of the first matching condition."""
        if not self._transfer_active:
            return 0.0

        for condition_func, flow_rate in self._conditions:
            if condition_func(tank_states, current_time, mission_data):
                # Flow rate can be either a value or a function
                if callable(flow_rate):
                    return self.apply_limits(flow_rate(tank_states, current_time, mission_data))
                else:
                    return self.apply_limits(flow_rate)

        # No conditions matched, use default
        return self.apply_limits(self._default_flow)


# Common condition functions for convenience
def pressure_above(tank_idx: int, threshold: float):
    """Create a condition that checks if tank pressure is above threshold."""
    return lambda states, *args: states[tank_idx].pressure > threshold

def pressure_below(tank_idx: int, threshold: float):
    """Create a condition that checks if tank pressure is below threshold."""
    return lambda states, *args: states[tank_idx].pressure < threshold

def fill_above(tank_idx: int, threshold: float):
    """Create a condition that checks if tank fill is above threshold."""
    return lambda states, *args: states[tank_idx].fill > threshold

def fill_below(tank_idx: int, threshold: float):
    """Create a condition that checks if tank fill is below threshold."""
    return lambda states, *args: states[tank_idx].fill < threshold

def time_after(time_threshold: float):
    """Create a condition that checks if current time is after threshold."""
    return lambda _, time, *args: time > time_threshold

def time_before(time_threshold: float):
    """Create a condition that checks if current time is before threshold."""
    return lambda _, time, *args: time < time_threshold


# Flow calculator functions for conditional rules
def mission_based_flow(safety_factor: float = 0.8):
    """Calculate flow based on mission requirements."""
    def calculator(states, time, mission_data):
        if mission_data is None:
            return 0.0

        # Extract mission outflow
        mission_outflow = 0.0
        from src.mission.mission_sections import OutFlow

        for flow in mission_data.fuel_flows:
            if isinstance(flow, OutFlow):
                if isinstance(flow.mass_flow, list):
                    from src.dynamics.dynamic_analysis import MissionSectionAnalysis
                    section_iter = getattr(mission_data, 'current_step', 0)
                    steps = getattr(mission_data, 'total_steps', 1)
                    mission_outflow += MissionSectionAnalysis.interpolate_mass_flows(
                        flow.mass_flow, section_iter, steps
                    )
                else:
                    mission_outflow += flow.mass_flow

        return abs(mission_outflow) * safety_factor

    return calculator