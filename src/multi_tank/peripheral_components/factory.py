"""Factory helpers for multi-tank peripheral component chains."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import PeripheralComponent
from .compressor import Compressor, CompressorParameters
from .cryopump import CryoPumpModel, CryopumpParameters
from .ideal_heat_exchanger import IdealHeatExchanger, IdealHeatExchangerParameters


def build_peripheral_component(component_config: Dict[str, Any]) -> PeripheralComponent:
    component_type = component_config.get("type")
    params = component_config.get("parameters", {})

    if component_type == "ideal_heat_exchanger":
        return IdealHeatExchanger(IdealHeatExchangerParameters(**params))
    if component_type == "compressor":
        return Compressor(CompressorParameters(**params))
    if component_type == "cryopump":
        return CryoPumpModel(CryopumpParameters(**params))

    raise ValueError(f"Unsupported peripheral component type: {component_type}")


def build_peripheral_component_chain(component_configs: List[Dict[str, Any]]) -> List[PeripheralComponent]:
    return [build_peripheral_component(component_config) for component_config in component_configs]