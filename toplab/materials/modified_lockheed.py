"""NASA Modified Lockheed heat-flux correlation for multilayer insulation.

Source:
Hastings, L. J., Hedayat, A., and Brown, T. M.,
"Analytical Modeling and Test Correlation of Variable Density
Multilayer Insulation for Cryogenic Storage,"
NASA/TM-2004-213175

"""

import math
from typing import NamedTuple


DEFAULT_CS = 2.4e-4
DEFAULT_CR = 4.944e-10
DEFAULT_CG = 1.46e4
DEFAULT_EMISSIVITY = 0.031
DEFAULT_LAYER_DENSITY = 16.0  # [layers/cm]
DEFAULT_PRESSURE_TORR = 1.0e-5  # [torr]
DEFAULT_DEGRADATION_FACTOR = 1.0

DEFAULT_EFFECTIVE_DENSITY = 300.0   # [kg/m^3], fitted equivalent density
DEFAULT_SPECIFIC_HEAT = 1000.0  # [J/kg/K], provisional constant (order of Mylar/Dacron)


class ModifiedLockheedComponents(NamedTuple):
    q_solid: float
    q_radiation: float
    q_gas: float
    q_total: float
    number_of_layers: float
    average_temperature: float


def modified_lockheed_heat_flux_components(
    T_hot: float,
    T_cold: float,
    thickness: float,
    layer_density: float = DEFAULT_LAYER_DENSITY,
    pressure_torr: float = DEFAULT_PRESSURE_TORR,
    emissivity: float = DEFAULT_EMISSIVITY,
    Cs: float = DEFAULT_CS,
    Cr: float = DEFAULT_CR,
    Cg: float = DEFAULT_CG,
    degradation_factor: float = DEFAULT_DEGRADATION_FACTOR,
) -> ModifiedLockheedComponents:
    """Return Modified Lockheed component heat fluxes and diagnostic values.

    Temperatures are in K, ``thickness`` in m, ``layer_density`` in layers/cm,
    and ``pressure_torr`` in torr. Heat fluxes are returned in W/m².
    """
    named_values = {
        "T_hot": T_hot,
        "T_cold": T_cold,
        "thickness": thickness,
        "layer_density": layer_density,
        "pressure_torr": pressure_torr,
        "emissivity": emissivity,
        "Cs": Cs,
        "Cr": Cr,
        "Cg": Cg,
        "degradation_factor": degradation_factor,
    }
    for name, value in named_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if T_hot <= 0.0 or T_cold <= 0.0:
        raise ValueError("T_hot and T_cold must be greater than 0 K")
    if T_hot < T_cold:
        raise ValueError("T_hot must be greater than or equal to T_cold")
    if thickness <= 0.0:
        raise ValueError("thickness must be greater than 0 m")
    if layer_density <= 0.0:
        raise ValueError("layer_density must be greater than 0 layers/cm")
    if pressure_torr < 0.0:
        raise ValueError("pressure_torr must be greater than or equal to 0 torr")
    if degradation_factor <= 0.0:
        raise ValueError("degradation_factor must be greater than 0")

    thickness_cm = 100.0 * thickness
    number_of_layers = layer_density * thickness_cm
    if number_of_layers <= 0.0:
        raise ValueError("The effective number of MLI layers must be greater than 0")

    average_temperature = 0.5 * (T_hot + T_cold)
    temperature_factor = (
        0.017
        + 7.0e-6 * (800.0 - average_temperature)
        + 2.28e-2 * math.log(average_temperature)
    )

    q_solid = (
        Cs
        * temperature_factor
        * layer_density**2.63
        * (T_hot - T_cold)
        / number_of_layers
    )
    q_radiation = (
        Cr
        * emissivity
        * (T_hot**4.67 - T_cold**4.67)
        / number_of_layers
    )
    q_gas = Cg * pressure_torr * (T_hot**0.52 - T_cold**0.52) / number_of_layers

    q_solid *= degradation_factor
    q_radiation *= degradation_factor
    q_gas *= degradation_factor
    return ModifiedLockheedComponents(
        q_solid=q_solid,
        q_radiation=q_radiation,
        q_gas=q_gas,
        q_total=q_solid + q_radiation + q_gas,
        number_of_layers=number_of_layers,
        average_temperature=average_temperature,
    )


def modified_lockheed_heat_flux(
    T_hot: float,
    T_cold: float,
    thickness: float,
    layer_density: float = DEFAULT_LAYER_DENSITY,
    pressure_torr: float = DEFAULT_PRESSURE_TORR,
    emissivity: float = DEFAULT_EMISSIVITY,
    Cs: float = DEFAULT_CS,
    Cr: float = DEFAULT_CR,
    Cg: float = DEFAULT_CG,
    degradation_factor: float = DEFAULT_DEGRADATION_FACTOR,
) -> float:
    """Return Modified Lockheed heat flux in W/m²."""
    return modified_lockheed_heat_flux_components(
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
    ).q_total
