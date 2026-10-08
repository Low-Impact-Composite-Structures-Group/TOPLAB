"""Insulation model selection: bundles conductivity integral, density and specific heat."""
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from toplab.materials import mli_properties, rohacell_properties


@dataclass(frozen=True)
class InsulationModel:
    name: str
    density: float  # [kg/m³]; equivalent (lumped) density for MLI
    integrated_conductivity: Callable[[float, float], float]  # (T_low, T_high) -> [W/m]
    specific_heat: Callable[[float], float]  # T [K] -> [J/kg/K]


ROHACELL = InsulationModel(
    name="Rohacell 51A",
    density=rohacell_properties.DENSITY,
    integrated_conductivity=rohacell_properties.integrated_thermal_conductivity,
    specific_heat=rohacell_properties.specific_heat,
)


def make_mli_model(
    A: float = mli_properties.DEFAULT_A,
    B: float = mli_properties.DEFAULT_B,
    density: float = mli_properties.DEFAULT_EFFECTIVE_DENSITY,
    specific_heat: float = mli_properties.DEFAULT_SPECIFIC_HEAT,
) -> InsulationModel:
    return InsulationModel(
        name="MLI (equivalent)",
        density=float(density),
        integrated_conductivity=lambda lo, hi: mli_properties.integrated_thermal_conductivity(lo, hi, A, B),
        specific_heat=lambda T: float(specific_heat),
    )


def insulation_model_from_config(insulation_config: Optional[Mapping]) -> InsulationModel:
    """Build the model from the YAML ``insulation`` block (``model``: rohacell | mli)."""
    cfg = insulation_config or {}
    model = str(cfg.get("model", "rohacell")).lower()
    if model == "rohacell":
        return ROHACELL
    if model == "mli":
        kwargs = {k: float(cfg[k]) for k in ("A", "B", "density", "specific_heat") if k in cfg}
        return make_mli_model(**kwargs)
    raise ValueError(f"Unknown insulation model '{model}' (expected 'rohacell' or 'mli')")
