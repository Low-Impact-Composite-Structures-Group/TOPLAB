"""
Inter-tank coupling mechanisms for multi-tank systems.

This module provides base classes and specific implementations for mass transfer
between tanks in a multi-tank hydrogen storage system.
"""

import math
from typing import List


class InterTankCoupling:
    """Base class for inter-tank mass transfer mechanisms."""

    def __init__(self, source_idx: int, target_idx: int, coupling_id: str = None):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.coupling_id = coupling_id or f"Coupling_{source_idx}→{target_idx}"
        self.is_active = False

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Determine if coupling should be active at current conditions."""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate mass flow rate [kg/s] when coupling is active."""
        raise NotImplementedError("Subclasses must implement calculate_flow_rate()")


class PressureTriggeredValve(InterTankCoupling):
    """Pressure-triggered valve with choked flow physics and hysteresis control."""

    def __init__(self, source_idx: int, target_idx: int,
                 p_open: float, p_close: float,
                 max_flow_rate: float = 0.005,
                 orifice_diameter: float = 0.002,
                 coupling_id: str = None):
        super().__init__(source_idx, target_idx, coupling_id)
        self.p_open = p_open
        self.p_close = p_close
        self.max_flow_rate = max_flow_rate
        self.effective_area = math.pi * (orifice_diameter / 2)**2

        if p_close <= p_open:
            raise ValueError(f"p_close ({p_close/1e5:.1f} bar) must be > p_open ({p_open/1e5:.1f} bar)")

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Evaluate valve state with hysteresis logic."""
        target_state = tank_states[self.target_idx]

        if target_state.pressure is None:
            target_state.compute_pressure()

        target_pressure = target_state.pressure

        if not self.is_active and target_pressure < self.p_open:
            self.is_active = True
            print(f"t={t/3600:.2f}h: Valve {self.source_idx}→{self.target_idx} OPENED (P={target_pressure/1e5:.1f} bar < {self.p_open/1e5:.1f} bar)")

        elif self.is_active and target_pressure > self.p_close:
            self.is_active = False
            print(f"t={t/3600:.2f}h: Valve {self.source_idx}→{self.target_idx} CLOSED (P={target_pressure/1e5:.1f} bar > {self.p_close/1e5:.1f} bar)")

        return self.is_active

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate choked flow rate using compressible gas physics."""
        if not self.is_active:
            return 0.0

        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        if source_state.fuel_mass < 1.0:
            return 0.0

        if source_state.pressure is None:
            source_state.compute_pressure()
        if target_state.pressure is None:
            target_state.compute_pressure()

        P1, P2 = source_state.pressure, target_state.pressure

        if P1 <= P2:
            return 0.0

        T1 = source_state.temperature
        rho1 = source_state.fuel_mass / source_state.tank.volume

        # Gas properties
        gamma = 1.4  # Heat capacity ratio for hydrogen
        R_specific = 4124  # J/(kg⋅K) specific gas constant for hydrogen
        P_crit_ratio = (2/(gamma+1))**(gamma/(gamma-1))  # Critical pressure ratio ≈ 0.528
        discharge_coeff = 0.6  # Discharge coefficient for sharp-edged orifice

        if P2/P1 < P_crit_ratio:
            # Choked flow - sonic velocity condition
            sonic_velocity = math.sqrt(gamma * R_specific * T1)
            flow_rate = discharge_coeff * self.effective_area * rho1 * sonic_velocity
        else:
            # Subsonic flow
            velocity = math.sqrt(2 * (P1 - P2) / rho1)
            flow_rate = discharge_coeff * self.effective_area * rho1 * velocity

        # Apply valve capacity limit
        flow_rate = min(flow_rate, self.max_flow_rate)

        # Safety limit: prevent excessive mass transfer rate
        max_safe_flow = 0.1 * source_state.fuel_mass
        flow_rate = min(flow_rate, max_safe_flow)

        return flow_rate