


import math
from abc import abstractmethod
from typing import Protocol


class FuelTank(Protocol):
    volume: float
    radius: float
    total_length: float
    surface_area: float
    structural_mass: float
    structural_volume: float
    thickness: float


class Insulation(Protocol):
    density: float
    thickness: float

    @abstractmethod
    def compute_mass(self, surface_area: float) -> float:
        ...

    @abstractmethod
    def compute_volume(self, surface_area: float) -> float:
        ...


class TankState(Protocol):
    fuel_mass: float


class GravimetricEfficiencyComputer(Protocol):

    @abstractmethod
    def compute_efficiency(
        self,
        tank: FuelTank,
        insulation: Insulation,
        initial_state: TankState
    ) -> float:
        ...


class GravimetricEfficiency(GravimetricEfficiencyComputer):

    def compute_efficiency(
        self,
        tank: FuelTank,
        insulation: Insulation,
        initial_state: TankState
    ) -> float:
        fuel_mass = initial_state.fuel_mass
        system_mass = self.compute_system_mass(tank, insulation)
        
        return fuel_mass / (fuel_mass + system_mass)

    def compute_system_mass(
        self, tank: FuelTank, insulation: Insulation
    ) -> float:
        
        return (
            tank.structural_mass
            + insulation.compute_mass(tank.surface_area)
        )


class VolumetricEfficiencyComputer(Protocol):

    @abstractmethod
    def compute_efficiency(
        self, tank: FuelTank, insulation: Insulation
    ) -> float:
        ...

    
class VolumetricEfficiency(VolumetricEfficiencyComputer):

    def compute_efficiency(
        self,
        tank: FuelTank,
        insulation: Insulation
    ) -> float:
        fuel_volume = tank.volume
        system_volume = self.compute_system_volume(tank, insulation)
        
        return fuel_volume / system_volume
    
    def compute_system_volume(
        self,
        tank: FuelTank,
        insulation: Insulation
    ):
        return (
            tank.volume
            + tank.structural_volume
            + insulation.compute_volume(tank.surface_area)
        )


class SquareVolumetricEfficiency(VolumetricEfficiency):

    def compute_system_volume(
        self,
        tank: FuelTank,
        insulation: Insulation
    ):

        return (
            self.compute_effective_area(tank, insulation)
            * tank.total_length
        )

    def compute_effective_area(
        self,
        tank: FuelTank,
        insulation: Insulation
    ):

        return self.compute_effective_diameter(tank, insulation) ** 2

    def compute_effective_diameter(
        self,
        tank: FuelTank,
        insulation: Insulation
    ):
        return self.compute_effective_radius(tank, insulation) * 2

    def compute_effective_radius(
        self,
        tank: FuelTank,
        insulation: Insulation
    ):
        return tank.radius + insulation.thickness + tank.thickness


class HexagonVolumetricEfficiency(SquareVolumetricEfficiency):

    def compute_effective_area(
        self,
        tank: FuelTank,
        insulation: Insulation
    ):
        radius = self.compute_effective_radius(tank, insulation)
        a = 2 * radius * math.tan(math.pi / 12)
        return 3 * math.sqrt(3) * a ** 2 / 2


def main():
    pass


if __name__ == "__main__":
    main()


# End
