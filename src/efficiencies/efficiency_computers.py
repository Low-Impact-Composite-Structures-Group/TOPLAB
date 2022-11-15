


from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol


class FuelTank(Protocol):
    volume: float
    surface_area: float
    structural_mass: float
    structural_volume: float


class Insulation(Protocol):
    density: float
    thickness: float


class TankState(Protocol):
    fuel_mass: float


@dataclass
class Efficiency(Protocol):
    tank: FuelTank
    insulation: Insulation

    @property
    @abstractmethod
    def efficiency(self) -> float:
        ...
    
    @property
    def insulation_volume(self):
        return self.insulation.thickness * self.tank.surface_area


@dataclass
class GravimetricEfficiency(Efficiency):
    initial_state: TankState

    @property
    def efficiency(self) -> float:
        return self.fuel_mass / (self.fuel_mass + self.system_mass)

    @property
    def fuel_mass(self) -> float:
        return self.initial_state.fuel_mass

    @property
    def system_mass(self) -> float:
        return self.tank.structural_mass + self.insulation_mass

    @property
    def insulation_mass(self):
        return (
            self.insulation_volume
            * self.insulation.density
        )

    
@dataclass
class VolumetricEfficiency(Efficiency):

    @property
    def efficiency(self) -> float:
        return self.tank.volume / self.system_volume
    
    @property
    def system_volume(self):
        return (
            self.tank.volume
            + self.tank.structural_volume
            + self.insulation_volume
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
