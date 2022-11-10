
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from src.fluids.hydrogen_retrievers import HydrogenRetriever


class Hydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen
    density: float


class StateDerivatives(Protocol):
    ...


@dataclass
class InitialState:
    pressure: float
    temperature: float
    fill: float


@dataclass
class TargetState:
    pressure: float
    temperature: float
    fill: float
    mass: float


@dataclass
class TankState:
    temperature: float
    pressure: float
    fill: float
    fuel_height: float
    volume: float
        
    @property
    def liquid_mass(self) -> float:
        if self.fill == 0:
            return 0
        return self.volume * self.fill * self.hydrogen.liquid.density

    @property
    def gas_mass(self) -> float:
        if self.fill == 1:
            return 0
        return (
            self.volume
            * (1 - self.fill)
            * self.hydrogen.gas.density
        )
        
    @property
    def fuel_mass(self) -> float:
        return self.gas_mass + self.liquid_mass
    
    @property
    def is_full(self):
        return self.fill >= 1

    @property
    def is_empty(self):
        return self.fill == 0 or self.fuel_height == 0

    def __post_init__(self) -> None:
        self.get_hydrogen_properties()

        if self.pressure is None:
            self.pressure = self.hydrogen.pressure
        if self.temperature is None:
            self.temperature = self.hydrogen.temperature

    def get_hydrogen_properties(self) -> Hydrogen:
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        return self.hydrogen

    def set_state_derivatives(
        self,
        state_derivatives: StateDerivatives
    ) -> StateDerivatives:
        self.derivatives = state_derivatives
        return self.derivatives

    def set_thermal_capacity(
        self,
        tank_thermal_capacity: float
    ) -> float:
        self.tank_thermal_capacity = tank_thermal_capacity
        return self.tank_thermal_capacity

    def set_heat_flux(
        self,
        heat_flux: float
    ) -> float:
        self.heat_flux = heat_flux
        return self.heat_flux


def main():
    pass


if __name__ == "__main__":
    main()

# End
