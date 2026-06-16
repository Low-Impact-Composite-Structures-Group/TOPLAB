"""Ideal heat exchanger component for multi-tank coupling streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from CoolProp.CoolProp import PropsSI

from .base import PeripheralComponent, PeripheralFlowState


@dataclass(frozen=True)
class IdealHeatExchangerParameters:
    target_temperature: Optional[float] = None
    pressure_drop: float = 0.0


class IdealHeatExchanger(PeripheralComponent):
    component_type = "ideal_heat_exchanger"

    def __init__(self, parameters: IdealHeatExchangerParameters):
        self.parameters = parameters

    def process_stream(
        self,
        stream: PeripheralFlowState,
        *,
        target_pressure: Optional[float] = None,
        target_temperature: Optional[float] = None,
    ) -> PeripheralFlowState:
        resolved = stream.resolved()
        outlet_pressure = max(
            resolved.pressure - self.parameters.pressure_drop,
            1.0,
        )
        outlet_temperature = (
            target_temperature
            if target_temperature is not None
            else self.parameters.target_temperature
        )
        if outlet_temperature is None:
            outlet_temperature = resolved.temperature

        outlet_enthalpy = PropsSI(
            "Hmass", "P", outlet_pressure, "T", outlet_temperature, resolved.fluid
        )
        outlet_entropy = PropsSI(
            "Smass", "P", outlet_pressure, "T", outlet_temperature, resolved.fluid
        )
        outlet_density = PropsSI(
            "Dmass", "P", outlet_pressure, "T", outlet_temperature, resolved.fluid
        )
        return resolved.with_updates(
            pressure=outlet_pressure,
            temperature=outlet_temperature,
            enthalpy=outlet_enthalpy,
            entropy=outlet_entropy,
            density=outlet_density,
        )