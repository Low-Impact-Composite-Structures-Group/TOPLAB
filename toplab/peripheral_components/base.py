"""Peripheral component abstractions for multi-tank flow conditioning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from CoolProp.CoolProp import PropsSI


@dataclass(frozen=True)
class PeripheralFlowState:
    """Thermodynamic state of a coupling stream between tanks/components."""

    pressure: float
    temperature: float
    mass_flow_rate: float
    fluid: str = "PARAHYD"
    enthalpy: Optional[float] = None
    entropy: Optional[float] = None
    density: Optional[float] = None

    def with_updates(self, **kwargs) -> "PeripheralFlowState":
        return replace(self, **kwargs)

    def resolved(self) -> "PeripheralFlowState":
        enthalpy = self.enthalpy
        entropy = self.entropy
        density = self.density

        if enthalpy is None:
            try:
                enthalpy = PropsSI("Hmass", "P", self.pressure, "T", self.temperature, self.fluid)
            except ValueError:
                # (P, T) is on the saturation curve — phase is ambiguous.
                # Fall back to (Dmass, P) which uniquely resolves the state.
                if self.density is not None:
                    enthalpy = PropsSI("Hmass", "Dmass", self.density, "P", self.pressure, self.fluid)
                else:
                    raise
        if entropy is None:
            try:
                entropy = PropsSI("Smass", "P", self.pressure, "T", self.temperature, self.fluid)
            except ValueError:
                if self.density is not None:
                    entropy = PropsSI("Smass", "Dmass", self.density, "P", self.pressure, self.fluid)
                else:
                    raise
        if density is None:
            density = PropsSI("Dmass", "P", self.pressure, "T", self.temperature, self.fluid)

        return replace(self, enthalpy=enthalpy, entropy=entropy, density=density)

    @classmethod
    def from_tank_state(
        cls,
        tank_state,
        mass_flow_rate: float,
        fluid: str = "PARAHYD",
    ) -> "PeripheralFlowState":
        pressure = tank_state.pressure
        if pressure is None and hasattr(tank_state, "compute_pressure"):
            pressure = tank_state.compute_pressure()
        density = getattr(tank_state, "density", None)
        if density is None:
            density = tank_state.fuel_mass / tank_state.tank.volume
        return cls(
            pressure=pressure,
            temperature=tank_state.h2_temperature,
            mass_flow_rate=mass_flow_rate,
            fluid=fluid,
            density=density,
        ).resolved()


class PeripheralComponent:
    """Base class for idealized flow-conditioning components."""

    component_type = "peripheral_component"

    def process_stream(
        self,
        stream: PeripheralFlowState,
        *,
        target_pressure: Optional[float] = None,
        target_temperature: Optional[float] = None,
    ) -> PeripheralFlowState:
        raise NotImplementedError("Peripheral components must implement process_stream().")