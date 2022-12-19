

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from src.efficiencies.efficiency_computers import (GravimetricEfficiency, GravimetricEfficiencyComputer,
                                                   VolumetricEfficiency, VolumetricEfficiencyComputer)


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

    volumetric_computer: VolumetricEfficiencyComputer = VolumetricEfficiency
    gravimetric_computer: GravimetricEfficiencyComputer = GravimetricEfficiency

    @property
    def volumetric_efficiency(self):
        return self.volumetric_computer.compute_efficiency(
            self.tank, self.insulation
        )

    @property
    def gravimetric_efficiency(self):
        return self.gravimetric_computer.compute_efficiency(
            self.tank, self.insulation, self.tank_states.first_state
        )

    @property
    def volume(self):
        return self.tank.volume


def main():
    pass


if __name__ == "__main__":
    main()


# End
