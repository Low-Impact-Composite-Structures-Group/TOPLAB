"""International Standard Atmosphere is used to compute the properties 
of air, for the provided altitude. The properties of air are use in the 
heat transfer modes on the external of the tank.

Fuel Tank - International Standard Atmosphere
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


import math

from src.fluids.convective_mediums import ConvectiveMedium

GRAVIMETRIC_ACCELERATION = 9.80665      # [m/s^2]
UNIVERSAL_GAS_CONSTANT = 8.314          # [J/mol K]
MOLAR_MASS_AIR = 0.02897                # [kg/mol]
STANDARD_SEA_LEVEL_TEMPERATURE = 288.15 # [K]
TROPOPAUSE_TEMPERATURE = 217            # [K]
UNIVERSAL_AIR_CONSTANT = UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_AIR


class ISA(object):

    standard_sea_level_temperature = STANDARD_SEA_LEVEL_TEMPERATURE
    lapse_rate = 0.0065

    def __init__(
        self, altitude: float, temperature: float=None
    ) -> None:
        """ISA is used to determine the properties of air, required
        as a convective medium at the outside of the storage vessel.

        Args:
            altitude (float): Altitude at which the properties are to be
            determined.
            temperature (float, optional): Alternative temperature for
            the given altitude. This is generally done at sea level.
            Defaults to None.
        """
        self.altitude = altitude

        # Change the sea level temperature when one is provided, this
        # is done in some papers to create a more extreme initial
        # pressure build-up
        if temperature:
            self.standard_sea_level_temperature = temperature

    def __eq__(self, __o: "ISA") -> bool:
        return (
            self.temperature == __o.temperature
            and (
                self.standard_sea_level_temperature
                == __o.standard_sea_level_temperature
            )
        )

    @property
    def specific_heat_constant_pressure(self) -> float:
        """This value is assumed at sea level for simplicity.

        Returns:
            float: Specific heat constant pressure
        """
        return 1005

    @property
    def thermal_conductivity(self) -> float:
        """This value is assumed at sea level for simplicity.

        Returns:
            float: Thermal conductivity
        """
        return 0.025
    
    @property
    def temperature(self) -> float:
        """Compute the temperature at altitude"""
        if self.altitude <= 11E3:
            temperature = (
                self.standard_sea_level_temperature
                - self.altitude * self.lapse_rate
            )
            return temperature
        if self.altitude > 20e3:
            raise ValueError("Altitude out of bound...")
        return TROPOPAUSE_TEMPERATURE
    
    @property
    def pressure(self) -> float:
        """Compute the pressure at altitude"""
        if self.altitude <= 11E3:
            exponent = (
                GRAVIMETRIC_ACCELERATION
                / (self.lapse_rate * UNIVERSAL_AIR_CONSTANT)
            )
            temperature_ratio = (
                self.temperature / self.standard_sea_level_temperature
            )
            sea_level_pressure = 101.3E3
            return sea_level_pressure * temperature_ratio ** exponent
        tropopause = ISA.tropopause()
        return tropopause.pressure * math.e ** (
            (GRAVIMETRIC_ACCELERATION * (
                tropopause.altitude - self.altitude
            ))
            / (UNIVERSAL_AIR_CONSTANT * tropopause.temperature)
        )

    @property
    def density(self) -> float:
        """Method to compute the density of air"""
        return self.pressure / (
            UNIVERSAL_AIR_CONSTANT * self.temperature
        )

    @property
    def dynamic_viscosity(self) -> float:
        """Method to compute the viscosity of air"""
        # The method is based on the work of Colozza 2002
        kinematic_viscosity = (
        -2.079 * 10 ** -6
        + 2.777 * 10 ** -8 * self.temperature
        + 1.077 * 10 ** -10 * self.temperature ** 2
        )
        return kinematic_viscosity * self.density
    
    @property
    def thermal_expansion_coefficient(self):
        """Method to compute the thermal expansion coefficient"""
        # The method is applicable for ideal gasses Colozza 2002
        return 1 / self.temperature

    @classmethod
    def tropopause(cls) -> "ISA":
        """Method to define the properties are the tropopause.
        """
        return cls(11E3)


def get_ISA_air_properties(
    altitude: float, temperature:float=None
) -> ConvectiveMedium:
    """Function to get the properties of air at the provided altitude 
    and eventually for the corrected sea level temperature.

    Args:
        altitude (float): Altitude at which the properties are to be
        determined.
        temperature (float, optional): Alternative sea level temperature.
        Defaults to None.

    Returns:
        ConvectiveMedium: Instance of the Convective Medium dataclass,
        containing all the attributes required to perform convective
        computations with air.
    """
    if temperature is None:
        isa = ISA(altitude)
    else:
        isa = ISA(altitude, temperature=temperature)
    return ConvectiveMedium(
        isa.temperature,
        isa.pressure,
        isa.density,
        isa.dynamic_viscosity,
        isa.specific_heat_constant_pressure,
        isa.thermal_conductivity
    )


def main():
    
    pass


if __name__ == "__main__":
    main()


# End