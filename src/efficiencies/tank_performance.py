

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from src.efficiencies.efficiency_computers import (GravimetricEfficiency,
                                                   VolumetricEfficiency)


class FuelTank(Protocol):
    volume: float


class Insulation(Protocol):
    ...


class TankStates(Protocol):
    
    @property
    @abstractmethod
    def first_state(self):
        ...


@dataclass
class TankPerformance:
    tank: FuelTank
    insulation: Insulation
    tank_states: TankStates

    @property
    def volumetric_efficiency(self):
        return (
            VolumetricEfficiency(self.tank, self.insulation).efficiency
        )

    @property
    def gravimetric_efficiency(self):
        return (
            GravimetricEfficiency(
                self.tank, self.insulation, self.tank_states.first_state
            ).efficiency
        )

    @property
    def volume(self):
        return self.tank.volume


def main():
    pass


if __name__ == "__main__":
    main()


# End
