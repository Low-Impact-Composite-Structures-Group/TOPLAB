


from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol


class FuelTank(Protocol):
    volume: float
    radius: float
    total_length: float
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
        return (
            self.tank.structural_mass
            + self.insulation_mass
        )

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
        return self.fuel_volume / self.system_volume

    @property
    def fuel_volume(self):
        return self.tank.volume
    
    @property
    def system_volume(self):
        return (
            self.tank.volume
            + self.tank.structural_volume
            + self.insulation_volume
        )


@dataclass
class SquareVolumetricEfficiency(VolumetricEfficiency):

    @property
    def system_volume(self):
        return self.effective_area * self.tank.total_length

    @property
    def effective_area(self):
        return self.effective_diameter ** 2

    @property
    def effective_diameter(self):
        return self.effective_radius * 2

    @property
    def effective_radius(self):
        return self.tank.radius + self.insulation.thickness


class HexagonVolumetricEfficiency(SquareVolumetricEfficiency):

    @property
    def effective_area(self):
        return 2 * (3) ** (1 / 3) * self.effective_radius ** 2


def main():
    pass


if __name__ == "__main__":
    main()


# End
