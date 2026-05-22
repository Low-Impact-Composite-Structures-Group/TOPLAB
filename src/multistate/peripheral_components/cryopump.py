"""Cryopump peripheral component for multi-tank and legacy refuel workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from CoolProp.CoolProp import PropsSI

from src.fluids.hydrogen_retrievers import HydrogenRetriever

from .base import PeripheralComponent, PeripheralFlowState


@dataclass(frozen=True)
class CryopumpParameters:
    """Parameters for cryopump outlet calculations."""

    reservoir_pressure: float = 3.0e5
    efficiency: float = 0.78


class CryoPumpModel(PeripheralComponent):
    """Cryogenic liquid-hydrogen pump model."""

    component_type = "cryopump"

    def __init__(
        self,
        parameters: CryopumpParameters = CryopumpParameters(),
        enable_cache: bool = True,
    ):
        self.parameters = parameters
        self.enable_cache = enable_cache
        self._cache: Dict[float, Any] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_info(self) -> dict:
        return {
            "enabled": bool(self.enable_cache),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "size": int(len(self._cache)),
        }

    def compute_pump_outlet_hydrogen(self, tank_pressure: float):
        if self.enable_cache and tank_pressure in self._cache:
            self._cache_hits += 1
            return self._cache[tank_pressure]
        if self.enable_cache:
            self._cache_misses += 1

        fluid = "Hydrogen"
        P1 = self.parameters.reservoir_pressure
        P2 = tank_pressure
        eta_p = self.parameters.efficiency

        h1 = PropsSI("H", "P", P1, "Q", 0, fluid)
        s1 = PropsSI("S", "P", P1, "Q", 0, fluid)
        h2s = PropsSI("H", "P", P2, "S", s1, fluid)
        h2 = h1 + (h2s - h1) / eta_p
        T2 = PropsSI("T", "P", P2, "H", h2, fluid)

        hydrogen = HydrogenRetriever().get_hydrogen_properties(pressure=P2, temperature=T2)

        if self.enable_cache:
            self._cache[tank_pressure] = hydrogen

        return hydrogen

    def process_stream(
        self,
        stream: PeripheralFlowState,
        *,
        target_pressure: Optional[float] = None,
        target_temperature: Optional[float] = None,
    ) -> PeripheralFlowState:
        outlet_pressure = target_pressure or stream.pressure
        outlet_hydrogen = self.compute_pump_outlet_hydrogen(outlet_pressure)
        return stream.with_updates(
            pressure=outlet_pressure,
            temperature=outlet_hydrogen.temperature,
            enthalpy=outlet_hydrogen.enthalpy,
            density=outlet_hydrogen.density,
            entropy=getattr(outlet_hydrogen, "entropy", None),
        )


default_cryopump = CryoPumpModel()


def compute_pump_outlet_hydrogen(tank_pressure: float):
    return default_cryopump.compute_pump_outlet_hydrogen(tank_pressure)