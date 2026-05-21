"""Factory helpers for multi-tank peripheral component chains."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import PeripheralComponent
from .compressor import Compressor, CompressorParameters
from .cryopump import CryoPumpModel, CryopumpParameters
from .ideal_heat_exchanger import IdealHeatExchanger, IdealHeatExchangerParameters
from .pressure_regulator import PressureRegulator, PressureRegulatorParameters


def build_peripheral_component(component_config: Dict[str, Any]) -> PeripheralComponent:
    component_type = component_config.get("type")
    # Support both nested {type: ..., parameters: {k: v}} and flat {type: ..., k: v}
    if "parameters" in component_config:
        params = component_config["parameters"]
    else:
        params = {k: v for k, v in component_config.items() if k != "type"}

    if component_type == "ideal_heat_exchanger":
        return IdealHeatExchanger(IdealHeatExchangerParameters(**params))
    if component_type == "compressor":
        return Compressor(CompressorParameters(**params))
    if component_type == "cryopump":
        return CryoPumpModel(CryopumpParameters(**params))
    if component_type == "pressure_regulator":
        return PressureRegulator(PressureRegulatorParameters(**params))

    raise ValueError(f"Unsupported peripheral component type: {component_type}")


def build_peripheral_component_chain(component_configs: List[Dict[str, Any]]) -> List[PeripheralComponent]:
    return [build_peripheral_component(component_config) for component_config in component_configs]