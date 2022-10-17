"""This module enables the definition of thermal resistances and defines
classes which enable the coupling between the different thermal
resistances.

Fuel Tank - Thermal Resistances
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ThermalResistance:
    """Thermal resistance is used to compute the thermal resistance of
    a heat transfer mode.

    Args:
        heat_transfer_coefficient (float): Heat transfer coefficient of
        the heat transfer mode.
        surface_area (float): Surface are along which the heat transfer
        mode acts.
    """
    heat_transfer_coefficient: float
    surface_area: float

    @property
    def value(self) -> float:
        """Compute the value of the thermal resistance.

        This also accounts for null values in the surface area or the 
        heat transfer coefficient, which would lead to infinite 
        resistance.

        Returns:
            float: Value of the thermal resistance
        """
        if (
            self.heat_transfer_coefficient == 0
            or self.surface_area == 0
        ):
            return float("inf")
        return 1 / (self.heat_transfer_coefficient * self.surface_area)


@dataclass
class ResistanceCoupling(Protocol):
    """Define coupled (Thermal) resistances. Can be used to compute the
    equivalent resistance.
    """

    @abstractmethod
    def compute_equivalent_resistance(
        self, resistances: list[float]
    ) -> float:
        """Compute the equivalent resistance of the resistances.

        Args:
            resistances (List[float]): List with values of the single 
            resistances.
        """
        pass


class ParallelResistances(ResistanceCoupling):
    """Define (Thermal) resistances coupled in parallel. Can be used to
    compute the equivalent resistance.
    """

    def compute_equivalent_resistance(
        self, resistances: list[float]
    ) -> float:
        den = 0
        for resistance in resistances:
            if resistance == 0:
                return 0
            den += 1 / resistance
        return 1 / den


class SeriesResistances(ResistanceCoupling):
    """Define (Thermal) resistances coupled in series. Can be used to
    compute the equivalent resistance.
    """

    def compute_equivalent_resistance(
        self, resistances: list[float]
    ) -> float:
        return sum(resistances)


def main():
    pass


if __name__ == "__main__":
    main()


# End
