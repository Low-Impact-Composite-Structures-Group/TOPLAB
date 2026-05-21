"""Isenthalpic pressure regulator (throttle valve) for multi-tank coupling streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from CoolProp.CoolProp import PropsSI

from .base import PeripheralComponent, PeripheralFlowState


@dataclass(frozen=True)
class PressureRegulatorParameters:
    outlet_pressure: float  # [Pa]


class PressureRegulator(PeripheralComponent):
    """Isenthalpic (throttle) expansion to a fixed outlet pressure.

    The enthalpy is conserved: h_out = h_in.  Temperature, entropy, and
    density are re-derived from (P_out, h_out) via CoolProp.

    If outlet_pressure >= inlet pressure the stream is returned unchanged
    (a regulator cannot raise pressure).
    """

    component_type = "pressure_regulator"

    def __init__(self, parameters: PressureRegulatorParameters):
        if parameters.outlet_pressure <= 0:
            raise ValueError("PressureRegulator outlet_pressure must be positive.")
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

        # No expansion needed if outlet >= inlet
        if outlet_pressure >= resolved.pressure:
            return resolved

        h_in = resolved.enthalpy  # already resolved by .resolved()

        # Isenthalpic process: h_out = h_in
        outlet_temperature = PropsSI("T",      "P", outlet_pressure, "Hmass", h_in, resolved.fluid)
        outlet_entropy      = PropsSI("Smass", "P", outlet_pressure, "Hmass", h_in, resolved.fluid)
        outlet_density      = PropsSI("Dmass", "P", outlet_pressure, "Hmass", h_in, resolved.fluid)

        return resolved.with_updates(
            pressure=outlet_pressure,
            temperature=outlet_temperature,
            enthalpy=h_in,          # conserved
            entropy=outlet_entropy,
            density=outlet_density,
        )
