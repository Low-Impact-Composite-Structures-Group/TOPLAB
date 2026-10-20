"""Supported insulation model selection and shared thermal properties."""
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from toplab.materials import modified_lockheed, rohacell_properties


@dataclass(frozen=True)
class InsulationModel:
    name: str
    density: float  # [kg/m³]; effective lumped density for MLI
    integrated_conductivity: Optional[Callable[[float, float], float]]  # (T_low, T_high) -> [W/m]
    specific_heat: Callable[[float], float]  # T [K] -> [J/kg/K]
    heat_flux: Optional[Callable[[float, float, float], float]] = None  # (T_hot, T_cold, thickness) -> [W/m²]


ROHACELL = InsulationModel(
    name="Rohacell 51A",
    density=rohacell_properties.DENSITY,
    integrated_conductivity=rohacell_properties.integrated_thermal_conductivity,
    specific_heat=rohacell_properties.specific_heat,
)


def make_modified_lockheed_model(
    layer_density: float = modified_lockheed.DEFAULT_LAYER_DENSITY,
    pressure_torr: float = modified_lockheed.DEFAULT_PRESSURE_TORR,
    emissivity: float = modified_lockheed.DEFAULT_EMISSIVITY,
    Cs: float = modified_lockheed.DEFAULT_CS,
    Cr: float = modified_lockheed.DEFAULT_CR,
    Cg: float = modified_lockheed.DEFAULT_CG,
    degradation_factor: float = modified_lockheed.DEFAULT_DEGRADATION_FACTOR,
    density: float = modified_lockheed.DEFAULT_EFFECTIVE_DENSITY,
    specific_heat: float = modified_lockheed.DEFAULT_SPECIFIC_HEAT,
) -> InsulationModel:
    return InsulationModel(
        name="MLI (Modified Lockheed)",
        density=float(density),
        integrated_conductivity=None,
        specific_heat=lambda T: float(specific_heat),
        heat_flux=lambda T_hot, T_cold, thickness: modified_lockheed.modified_lockheed_heat_flux(
            T_hot=T_hot,
            T_cold=T_cold,
            thickness=thickness,
            layer_density=layer_density,
            pressure_torr=pressure_torr,
            emissivity=emissivity,
            Cs=Cs,
            Cr=Cr,
            Cg=Cg,
            degradation_factor=degradation_factor,
        ),
    )


def insulation_model_from_config(insulation_config: Optional[Mapping]) -> InsulationModel:
    """Build the insulation model from the YAML ``insulation`` block.

    ``model`` is the preferred selector; ``type`` is accepted for existing
    configurations. Supported models are ``foam``/``rohacell`` and
    ``modified_lockheed``. The equivalent ``mli`` formulation is unsupported.
    """
    cfg = insulation_config or {}
    model = str(cfg.get("model", cfg.get("type", "foam"))).strip().lower()
    if model in {"foam", "rohacell"}:
        return ROHACELL
    if model == "mli":
        raise ValueError(
            "The equivalent MLI insulation model is not working and is unsupported; "
            "use 'modified_lockheed' instead"
        )
    if model == "modified_lockheed":
        parameter_names = (
            "layer_density",
            "pressure_torr",
            "Cs",
            "Cr",
            "Cg",
            "degradation_factor",
            "density",
            "specific_heat",
        )
        kwargs = {key: float(cfg[key]) for key in parameter_names if key in cfg}
        # `emissivity` is the shell emissivity in legacy configs, so it only
        # means the MLI emissivity when `emissivity_shell` is given as well.
        if "emissivity_mli" in cfg:
            kwargs["emissivity"] = float(cfg["emissivity_mli"])
        elif "emissivity" in cfg and "emissivity_shell" in cfg:
            kwargs["emissivity"] = float(cfg["emissivity"])
        return make_modified_lockheed_model(**kwargs)
    raise ValueError(
        f"Unknown insulation model '{model}' "
        "(expected 'foam', 'rohacell' or 'modified_lockheed')"
    )
