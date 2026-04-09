"""Idealized compressor component for multi-tank coupling streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from CoolProp.CoolProp import PropsSI

from .base import PeripheralComponent, PeripheralFlowState


@dataclass(frozen=True)
class CompressorParameters:
    efficiency: float = 1.0
    outlet_pressure: Optional[float] = None
    pressure_ratio: Optional[float] = None


class Compressor(PeripheralComponent):
    component_type = "compressor"

    def __init__(self, parameters: CompressorParameters):
        if parameters.efficiency <= 0.0:
            raise ValueError("Compressor efficiency must be positive.")
        self.parameters = parameters

    def process_stream(
        self,
        stream: PeripheralFlowState,
        *,
        target_pressure: Optional[float] = None,
        target_temperature: Optional[float] = None,
    ) -> PeripheralFlowState:
        resolved = stream.resolved()
        outlet_pressure = self.parameters.outlet_pressure
        if outlet_pressure is None and self.parameters.pressure_ratio is not None:
            outlet_pressure = resolved.pressure * self.parameters.pressure_ratio
        if outlet_pressure is None:
            outlet_pressure = target_pressure
        if outlet_pressure is None:
            raise ValueError("Compressor requires outlet_pressure, pressure_ratio, or target_pressure.")

        outlet_pressure = max(outlet_pressure, resolved.pressure)
        s_in = resolved.entropy if resolved.entropy is not None else PropsSI(
            "Smass", "P", resolved.pressure, "T", resolved.temperature, resolved.fluid
        )
        h_in = resolved.enthalpy if resolved.enthalpy is not None else PropsSI(
            "Hmass", "P", resolved.pressure, "T", resolved.temperature, resolved.fluid
        )
        h_out_isentropic = PropsSI("Hmass", "P", outlet_pressure, "Smass", s_in, resolved.fluid)
        h_out = h_in + (h_out_isentropic - h_in) / self.parameters.efficiency
        outlet_temperature = PropsSI("T", "P", outlet_pressure, "Hmass", h_out, resolved.fluid)
        outlet_entropy = PropsSI("Smass", "P", outlet_pressure, "Hmass", h_out, resolved.fluid)
        outlet_density = PropsSI("Dmass", "P", outlet_pressure, "Hmass", h_out, resolved.fluid)

        if target_temperature is not None:
            outlet_temperature = target_temperature
            h_out = PropsSI("Hmass", "P", outlet_pressure, "T", outlet_temperature, resolved.fluid)
            outlet_entropy = PropsSI("Smass", "P", outlet_pressure, "T", outlet_temperature, resolved.fluid)
            outlet_density = PropsSI("Dmass", "P", outlet_pressure, "T", outlet_temperature, resolved.fluid)

        return resolved.with_updates(
            pressure=outlet_pressure,
            temperature=outlet_temperature,
            enthalpy=h_out,
            entropy=outlet_entropy,
            density=outlet_density,
        )