"""Hysteresis-controlled valve between two tanks (pipe / orifice / pipe flow model)."""

import math
from typing import List

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling
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
        # Opens when the target pressure falls to p_open, closes when it rises to p_close.
        self.p_open = p_open
        self.p_close = p_close
        self.path = path

        self.activation_threshold = p_open
        self.deactivation_threshold = p_close

        # First-order valve dynamics; the opening scales the orifice area.
        self.valve_time_constant_s = valve_time_constant_s
        self._valve_coefficient = 0.0
        self._last_coeff_time = None

        if p_close <= p_open:
            raise ValueError(f"deactivation_threshold ({p_close/1e5:.1f} bar) must be > activation_threshold ({p_open/1e5:.1f} bar)")

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Evaluate valve state with hysteresis logic and update opening coefficient.

        Valve opens when target pressure ≤ activation_threshold
        Valve closes when target pressure ≥ deactivation_threshold
        Valve opening follows first-order dynamics: τ dα/dt = α_target - α
        """
        target_state = tank_states[self.target_idx]

        if target_state.pressure is None:
            target_state.compute_pressure()

        target_pressure = target_state.pressure

        # Determine target valve state (fully open or fully closed)
        if not self.is_active and target_pressure <= self.activation_threshold:
            self.is_active = True
            print(f"t={t/3600:.2f}h: Valve {self.source_idx}→{self.target_idx} OPENING (P={target_pressure/1e5:.1f} bar ≤ {self.activation_threshold/1e5:.1f} bar)")

        elif self.is_active and target_pressure >= self.deactivation_threshold:
            self.is_active = False
            print(f"t={t/3600:.2f}h: Valve {self.source_idx}→{self.target_idx} CLOSING (P={target_pressure/1e5:.1f} bar ≥ {self.deactivation_threshold/1e5:.1f} bar)")

        # Update valve coefficient with first-order lag
        target_coeff = 1.0 if self.is_active else 0.0

        if self._last_coeff_time is None:
            # First call - initialize
            self._valve_coefficient = target_coeff
            self._last_coeff_time = t
        else:
            # Apply first-order dynamics: α(t) = α(t-Δt) + (Δt/τ)(α_target - α(t-Δt))
            dt = max(0.0, t - self._last_coeff_time)
            if self.valve_time_constant_s > 1e-9 and dt > 0.0:
                alpha = min(1.0, dt / self.valve_time_constant_s)
                self._valve_coefficient += alpha * (target_coeff - self._valve_coefficient)
            else:
                # Instantaneous response if tau = 0
                self._valve_coefficient = target_coeff
            self._last_coeff_time = t

        # Clamp coefficient to valid range
        self._valve_coefficient = max(0.0, min(1.0, self._valve_coefficient))

        # Debug: Show valve state periodically
        # if abs(t % 60) < 0.1:  # Every 60 seconds
        #     print(f"  Valve state at t={t/3600:.3f}h: coeff={self._valve_coefficient:.3f}, P_target={target_pressure/1e5:.1f}bar")

        return self._valve_coefficient > 1e-6  # Active if any opening

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Mass flow [kg/s] through the pipe/orifice/pipe path at the current valve opening."""
        if self._valve_coefficient <= 1e-6:
            return 0.0

        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        if source_state.fuel_mass <= 0.0:
            return 0.0

        if source_state.pressure is None:
            source_state.compute_pressure()
        if target_state.pressure is None:
            target_state.compute_pressure()

        P1, P2 = source_state.pressure, target_state.pressure

        if P1 <= P2:
            return 0.0

        return valve_mass_flow(
            self.path,
            fluid=source_state.fluid,
            p_source=P1,
            temperature=source_state.h2_temperature,
            density=source_state.fuel_mass / source_state.tank.volume,
            p_target=P2,
            opening=self._valve_coefficient,
        )

    def update_valve_state(self, target_pressure, t):
        """Update valve open/close state based on hysteresis logic

        Opens when target_pressure ≤ activation_threshold
        Closes when target_pressure ≥ deactivation_threshold
        """
        if not self.is_active:
            # Valve opens when target pressure drops to or below activation threshold
            if target_pressure <= self.activation_threshold:
                self.is_active = True
                print(f"  Valve {self.source_idx}→{self.target_idx} OPENED: P={target_pressure/1e5:.1f} ≤ {self.activation_threshold/1e5:.1f} bar")
        else:
            # Valve closes when target pressure rises to or above deactivation threshold
            if target_pressure >= self.deactivation_threshold:
                self.is_active = False
                print(f"  Valve {self.source_idx}→{self.target_idx} CLOSED: P={target_pressure/1e5:.1f} ≥ {self.deactivation_threshold/1e5:.1f} bar")

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Ensure pressures are up to date before evaluating
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Build a temporary tank_states list so we can reuse evaluate()/calculate_flow_rate
        tank_states = [None] * max(self.source_idx + 1, self.target_idx + 1)
        tank_states[self.source_idx] = source_state
        tank_states[self.target_idx] = target_state

        # Always drive the hysteresis state machine and update _valve_coefficient
        self.evaluate(t, tank_states)

        # Gate on coefficient (not is_active) so the closing ramp is also applied
        if self._valve_coefficient <= 1e-6:
            return 0.0

        # Otherwise, compute flow with the selected physics and capacity limits
        return self.calculate_flow_rate(t, tank_states)

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
