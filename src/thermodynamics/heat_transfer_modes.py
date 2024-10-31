"""This module is used to define the heat transfer modes of that can
occur in the fuel tank. This entails convective motions and radiation.
Different types of convective motions are defined, with the main split
being passive or forced. The heat transfer modes differ in the
computation of the Nussult number, which in turn is required to compute
the heat transfer coefficient, needed to obtain the thermal resistance.

Fuel Tank - Heat Transfer Modes
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


GRAVITATIONAL_ACCELERATION = 9.81
STEPHAN_BOLTZMANN_CONSTANT = 5.67e-8
RADIATION_EMITTANCE = 0.95


from abc import abstractmethod
from dataclasses import dataclass
import math
from typing import Protocol


class ConvectiveMedium(Protocol):
    dynamic_viscosity: float
    specific_heat_constant_pressure: float
    thermal_conductivity: float
    temperature: float

    @property
    @abstractmethod
    def prantl_number(self):
        ...


class NaturalConvectiveMedium(ConvectiveMedium):
    kinematic_viscosity: float
    temperature: float
    thermal_expansion_coefficient: float


class ForcedConvectiveMedium(ConvectiveMedium):
    density: float


class HeatTransferMode(Protocol):

    @property
    @abstractmethod
    def heat_transfer_coefficient(self) -> float:
        """Property to compute the heat transfer coefficient of the heat
        transfer mode.

        Returns:
            float: Heat transfer coefficient of the heat transfer mode.
        """
        ...


@dataclass
class Radiation(HeatTransferMode):
    """Class to define radiation.

    Args:
        skin_temp (float): Skin temperature in Kelvin
        ambient_tamp (float): Ambient temperature in Kelvin
        emittance: (float): Radiative emittance, depends on the type of
        paint used. 0.95 for white 0.09 for unpainted. Defaults to 0.95.
    """
    skin_temp: float
    ambient_temp: float
    emittance: float = 0.95

    @property
    def heat_transfer_coefficient(self) -> float:
        fac1 = STEPHAN_BOLTZMANN_CONSTANT * self.emittance
        fac2 = self.skin_temp ** 2 + self.ambient_temp ** 2
        fac3 = self.skin_temp + self.ambient_temp
        return fac1 * fac2 * fac3


@dataclass
class NaturalConvection(HeatTransferMode):
    """Natural convection, which in later stages is to be used to define
    different types of natural convection.

    Args:
        medium: (ConvectiveMedium): Convective medium for the passive
        convection.
        characteristic_dimension: (float): Characteristic dimension for
        the convection.
        surface_temperature: (float): Surface temperature with which the
        convection interacts.
    """
    medium: NaturalConvectiveMedium
    characteristic_dimension: float
    surface_temperature: float

    @property
    def temperature_delta(self) -> float:
        """Property to compute the temperature difference between the
        convective medium and the surface temperature. Note that the
        difference is given as an absolute value.

        Returns:
            float: Temperature difference
        """
        return abs(self.medium.temperature - self.surface_temperature)


    @property
    @abstractmethod
    def nussult_number(self) -> float:
        """Property to compute the nussult number of the convective
        motion. The definition depends on the type of convection.

        Returns:
            float: Nussult number of the convection.
        """
        ...

    @property
    def heat_transfer_coefficient(self) -> float:
        if self.characteristic_dimension is None:
            raise ValueError("characteristic_dimension is not set")
        if self.characteristic_dimension <= 0:
            return float("inf")
        return (
            self.nussult_number * self.medium.thermal_conductivity
            / self.characteristic_dimension
        )

    @property
    def rayleigh_number(self) -> float:
        # A small temp difference is forced to avoid zero division, this
        # may be required to initialise the temperature convergence
        # computation
        if self.temperature_delta == 0:
            return 1e-13
        return (
            GRAVITATIONAL_ACCELERATION
            * self.medium.thermal_expansion_coefficient
            * self.temperature_delta
            * self.characteristic_dimension ** 3
            * self.medium.prantl_number
            / self.medium.kinematic_viscosity
        )


@dataclass
class ForcedConvection(NaturalConvection):
    velocity: float
    medium: ForcedConvectiveMedium

    @property
    def reynolds_number(self) -> float:
        """Property to define the Reynolds number of the forced
        convection.

        Returns:
            float: Reynolds number.
        """
        return (
            self.medium.density
            * self.velocity
            * self.characteristic_dimension
            / self.medium.dynamic_viscosity
        )

    @property
    def nussult_number(self) -> float:
        return (
            0.03625
            * self.medium.prantl_number ** (0.43)
            * self.reynolds_number ** (0.8)
        )


class LiquidPhaseConvection(NaturalConvection):
    """Class to define natural liquid convection. The current
    implementation is to be used for hydrogen convection and is based on
    the work Hochstein et al. 1986, and is the method as implemented by
    Verstraete 2009.
    """

    @property
    def nussult_number(self) -> float:
        return 0.0605 * self.rayleigh_number ** (1 / 3)


class GasPhaseConvection(NaturalConvection):
    """Class to define natural gas convection. The current
    implementation is to be used for gas hydrogen convection and is
    based on the work of Brewer 1991, and is the method as implemented
    by Verstraete 2009, and all the other works on hydrogen fuel tanks.
    """

    @property
    def nussult_number(self) -> float:
        return 17


class NaturalCylinderConvection(NaturalConvection):

    @property
    def nussult_number(self) -> float:
        return (
            0.6
            + (
                0.387 * self.rayleigh_number ** (1 / 6)
            ) / (
                1 + (0.559 / self.medium.prantl_number) ** (9 / 16)
            ) ** (8 / 27)
        ) ** 2


class NaturalSphereConvection(NaturalConvection):

    @property
    def nussult_number(self) -> float:
        return (
            2 + (
                0.589 * self.rayleigh_number ** (1 / 4)
            ) / (
                1 + (0.469 / self.medium.prantl_number) ** (9 / 16)
            ) ** (4 / 9)
        )


class RohsenowNaturalConvection(NaturalConvection):

    @property
    def nussult_number(self) -> float:
        m = 10
        return (self.nussult_l ** m + self.nussult_t ** m) ** (1 / m)

    @property
    def nussult_l(self):
        return 2 * self.f / math.log(1 + 2 * self.f / self.nussult_T)

    @property
    def nussult_t(self):
        return self.c_barred_l * self.rayleigh_number ** (1 / 3)

    @property
    def f(self):
        return 1 - 0.13 / (self.nussult_T ** (0.16))

    @property
    def nussult_T(self):
        return 0.772 * self.c_barred_l * self.rayleigh_number ** (1 / 4)

    @property
    def c_barred_l(self):
        return (
            0.671 / (
                1 + (0.492 / self.medium.prantl_number) ** (9 / 16)
            ) ** (4 / 9)
        )


class ChurchillNaturalConvection(NaturalConvection):

    @property
    def nussult_number(self) -> float:
        return math.sqrt(
            0.825
            + (
                0.387
                * self.rayleigh_number ** (1 / 6)
            ) / (
                1 + (
                    0.437 / self.medium.prantl_number
                ) ** (9 / 16)
            ) ** (8 / 27)
        )


class FujiiNaturalConvection(NaturalConvection):

    @property
    def nussult_number(self) -> float:
        return 0.56 * self.rayleigh_number ** (1 / 4)


def main():
    pass


if __name__ == "__main__":
    main()


# End
