


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

        mass = tank.structural_mass

        if insulation is not None:
            mass += insulation.compute_mass(tank.surface_area)

        return mass


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

        volume = tank.volume + tank.structural_volume
        if insulation is not None:
            volume += insulation.compute_volume(tank.surface_area)

        return volume


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
        return 2 * math.sqrt(3) * radius ** 2


class GravimetricEfficiencyComputerFactory():

    _computers = {
        "basic": GravimetricEfficiency(),
    }

    @property
    def _available(self):
        return ", ".join(self._computers.keys())

    def create_efficiency_computer(self, computer: str) -> GravimetricEfficiencyComputer:
        comp = self._computers.get(computer)

        if comp is None:
            raise ValueError(
                f"{computer} is an invalid gravimetric efficiency computer.\n"
                f"Available computers are: {self._available}."
            )

        return comp


class VolumetricEfficiencyComputerFactory():

    _computers = {
        "basic": VolumetricEfficiency(),
        "square": SquareVolumetricEfficiency(),
        "hexagon": HexagonVolumetricEfficiency()
    }

    @property
    def _available(self):
        return ", ".join(self._computers.keys())

    def create_efficiency_computer(self, computer: str) -> VolumetricEfficiencyComputer:
        comp = self._computers.get(computer)

        if comp is None:
            raise ValueError(
                f"{computer} is an invalid volumetric efficiency computer.\n"
                f"Available computers are: {self._available}."
            )

        return comp


def main():
    pass


if __name__ == "__main__":
    main()


# End
