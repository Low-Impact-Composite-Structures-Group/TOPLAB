


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

    number_of_tanks: int = 1

    @property
    def efficiency(self) -> float:
        return self.fuel_mass / (self.fuel_mass + self.system_mass)

    @property
    def fuel_mass(self) -> float:
        return self.initial_state.fuel_mass * self.number_of_tanks

    @property
    def system_mass(self) -> float:
        return (
            self.tank.structural_mass * self.number_of_tanks
            + self.insulation_mass
        )

    @property
    def insulation_mass(self):
        return (
            self.insulation_volume
            * self.insulation.density
            * self.number_of_tanks
        )

    
@dataclass
class VolumetricEfficiency(Efficiency):

    number_of_tanks: int = 1

    @property
    def efficiency(self) -> float:
        return (
            self.tank.volume * self.number_of_tanks / self.system_volume
        )
    
    @property
    def system_volume(self):
        return (
            (
                self.tank.volume
                + self.tank.structural_volume
                + self.insulation_volume
            ) * self.number_of_tanks
        )


@dataclass
class SquareVolumetricEfficiency(Efficiency):

    number_of_tanks: int = 1

    @property
    def efficiency(self) -> float:
        return (
            self.tank.volume * self.number_of_tanks / self.system_volume
        )
    
    @property
    def system_volume(self):
        return self.effective_body_volume * self.number_of_tanks

    @property
    def effective_body_volume(self):
        return (
            ((self.tank.radius + self.insulation.thickness) * 2) ** 2
            * self.tank.total_length
        )



def main():
    pass


if __name__ == "__main__":
    main()


# End
