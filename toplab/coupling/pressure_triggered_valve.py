"""Hysteresis-controlled valve between two tanks (pipe / orifice / pipe flow model)."""

from typing import List

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling, ModeSwitch
from toplab.coupling.registry import register_edge
from toplab.network.network import CouplingEdge
from toplab.fluids.valve_flow import path_from_options
from toplab.fluids.valve_flow import ValvePath, valve_mass_flow


@register_edge("pressure_compensation", "flow_controlled_pressurization", "pressure_triggered_valve")
class PressureTriggeredValve(InterTankCoupling):
    """Hysteresis valve between two tanks; flow from the adiabatic pipe/orifice/pipe model."""

    def __init__(self, source_idx: int, target_idx: int,
                 p_open: float, p_close: float,
                 path: ValvePath,
                 coupling_id: str = None,
                 valve_time_constant_s: float = 0.5):
        super().__init__(source_idx, target_idx, coupling_id)
        if p_close <= p_open:
            raise ValueError(f"deactivation_threshold ({p_close/1e5:.1f} bar) must be > activation_threshold ({p_open/1e5:.1f} bar)")
        # Opens when the target pressure falls to p_open, closes when it rises to p_close.
        self.p_open = p_open
        self.p_close = p_close
        self.path = path
        self.valve_time_constant_s = valve_time_constant_s
        # With a lag the opening is an ODE state: tau d(alpha)/dt = alpha_cmd - alpha.
        self.n_states = 1 if valve_time_constant_s > 1e-9 else 0

    @property
    def commanded_opening(self) -> float:
        return 1.0 if self.is_active else 0.0

    def initialize(self, tank_states: List) -> List[float]:
        p_target = tank_states[self.target_idx].compute_pressure()
        self.is_active = p_target <= self.p_open
        return [self.commanded_opening] if self.n_states else []

    def state_derivatives(self, t: float, tank_states: List, x) -> List[float]:
        return [(self.commanded_opening - x[0]) / self.valve_time_constant_s] if self.n_states else []

    def mode_switches(self) -> List[ModeSwitch]:
        if self.is_active:
            return [ModeSwitch(self.target_idx, self.p_close, +1, lambda: setattr(self, "is_active", False),
                               f"{self.coupling_id} closes")]
        return [ModeSwitch(self.target_idx, self.p_open, -1, lambda: setattr(self, "is_active", True),
                           f"{self.coupling_id} opens")]

    def flow(self, t: float, tank_states: List, x) -> float:
        """Mass flow [kg/s] through the pipe/orifice/pipe path at the current valve opening."""
        opening = min(1.0, max(0.0, float(x[0]))) if self.n_states else self.commanded_opening
        if opening <= 1e-6:
            return 0.0

        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]
        if source_state.fuel_mass <= 0.0:
            return 0.0

        P1, P2 = source_state.compute_pressure(), target_state.compute_pressure()
        if P1 <= P2:
            return 0.0

        return valve_mass_flow(
            self.path,
            fluid=source_state.fluid,
            p_source=P1,
            temperature=source_state.h2_temperature,
            density=source_state.fuel_mass / source_state.tank.volume,
            p_target=P2,
            opening=opening,
        )

    def calculate_flow(self, source_state, target_state, t):
        """Fully-open flow for post-processing callers that have no edge state."""
        states = [None] * (max(self.source_idx, self.target_idx) + 1)
        states[self.source_idx], states[self.target_idx] = source_state, target_state
        return self.flow(t, states, [1.0] if self.n_states else [])

    @classmethod
    def from_edge(cls, edge: CouplingEdge, ctx: EdgeContext) -> "PressureTriggeredValve":
        """Options: ``activation_conditions`` {pressure_open_bar, pressure_close_bar,
        valve_time_constant_s} (or ``hysteresis`` {activation_threshold_bar, deactivation_threshold_bar}),
        plus ``valve`` {orifice_diameter_m | orifice_area_m2, discharge_coefficient} with optional
        ``pipe_upstream`` / ``pipe_downstream`` {diameter_m, length_m, friction_factor | roughness_m, loss_coefficient}."""
        o = edge.options
        if o.get("activation_conditions"):
            thresholds, open_key, close_key = o["activation_conditions"], "pressure_open_bar", "pressure_close_bar"
        elif o.get("hysteresis"):
            thresholds, open_key, close_key = o["hysteresis"], "activation_threshold_bar", "deactivation_threshold_bar"
        else:
            raise ConfigError(f"Edge {edge.edge_id!r}: needs 'activation_conditions' with open/close pressures")
        for key in (open_key, close_key):
            if key not in thresholds:
                raise ConfigError(f"Edge {edge.edge_id!r}: missing '{key}'")

        if o.get("valve"):
            try:
                path = path_from_options(o)
            except ValueError as err:
                raise ConfigError(f"Edge {edge.edge_id!r}: {err}") from err
        elif "orifice_diameter_m" in o.get("flow_parameters", {}):
            path = ValvePath(orifice_diameter_m=float(o["flow_parameters"]["orifice_diameter_m"]))
        else:
            raise ConfigError(f"Edge {edge.edge_id!r}: needs a 'valve' block with an orifice size")

        try:
            return cls(edge.source, edge.target,
                       p_open=float(thresholds[open_key]) * 1e5,
                       p_close=float(thresholds[close_key]) * 1e5,
                       path=path,
                       coupling_id=edge.edge_id,
                       valve_time_constant_s=float(thresholds.get("valve_time_constant_s", 0.5)))
        except ValueError as err:
            raise ConfigError(f"Edge {edge.edge_id!r}: {err}") from err
