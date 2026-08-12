from __future__ import annotations

import math

from toplab.fluids.convective_mediums import ConvectiveMedium


GRAVIMETRIC_ACCELERATION = 9.80665
UNIVERSAL_GAS_CONSTANT = 8.314
MOLAR_MASS_AIR = 0.02897
STANDARD_SEA_LEVEL_TEMPERATURE = 288.15
TROPOPAUSE_TEMPERATURE = 217
UNIVERSAL_AIR_CONSTANT = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_AIR


class ISA:
    standard_sea_level_temperature = STANDARD_SEA_LEVEL_TEMPERATURE
    lapse_rate = 0.0065

    def __init__(self, altitude: float, temperature: float = None) -> None:
        self.altitude = altitude
        if temperature:
            self.standard_sea_level_temperature = temperature

    def __eq__(self, other: "ISA") -> bool:
        return (
            self.temperature == other.temperature
            and self.standard_sea_level_temperature == other.standard_sea_level_temperature
        )

    @property
    def specific_heat_constant_pressure(self) -> float:
        return 1005

    @property
    def thermal_conductivity(self) -> float:
        return 0.025

    @property
    def temperature(self) -> float:
        if self.altitude <= 11e3:
            return self.standard_sea_level_temperature - self.altitude * self.lapse_rate
        if self.altitude > 20e3:
            raise ValueError("Altitude out of bound...")
        return TROPOPAUSE_TEMPERATURE

    @property
    def pressure(self) -> float:
        if self.altitude <= 11e3:
            exponent = GRAVIMETRIC_ACCELERATION / (self.lapse_rate * UNIVERSAL_AIR_CONSTANT)
            temperature_ratio = self.temperature / self.standard_sea_level_temperature
            sea_level_pressure = 101.3e3
            return sea_level_pressure * temperature_ratio ** exponent
        tropopause = ISA.tropopause()
        return tropopause.pressure * math.e ** (
            (GRAVIMETRIC_ACCELERATION * (tropopause.altitude - self.altitude))
            / (UNIVERSAL_AIR_CONSTANT * tropopause.temperature)
        )

    @property
    def density(self) -> float:
        return self.pressure / (UNIVERSAL_AIR_CONSTANT * self.temperature)

    @property
    def dynamic_viscosity(self) -> float:
        kinematic_viscosity = (
            -2.079 * 10 ** -6
            + 2.777 * 10 ** -8 * self.temperature
            + 1.077 * 10 ** -10 * self.temperature ** 2
        )
        return kinematic_viscosity * self.density

    @property
    def thermal_expansion_coefficient(self):
        return 1 / self.temperature

    @classmethod
    def tropopause(cls) -> "ISA":
        return cls(11e3)


def get_ISA_air_properties(altitude: float, temperature: float = None) -> ConvectiveMedium:
    isa = ISA(altitude) if temperature is None else ISA(altitude, temperature=temperature)
    return ConvectiveMedium(
        isa.temperature,
        isa.pressure,
        isa.density,
        isa.dynamic_viscosity,
        isa.specific_heat_constant_pressure,
        isa.thermal_conductivity,
    )