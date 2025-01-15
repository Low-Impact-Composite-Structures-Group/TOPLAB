from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from src.fluids.hydrogen_retrievers import HydrogenRetriever

SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5 


class Hydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen
    density: float


class StateDerivatives(Protocol):
    pressure: float
    temperature: float
    gas_mass: float
    liquid_mass: float
    venting_mass: float
    heat_flux: float


class FuelFlow(Protocol):
    ...


class Tank(Protocol):
    volume: float

    @abstractmethod
    def compute_fuel_height(self, fuel_volume: float):
        ...


class DynamicModel(Protocol):

    @abstractmethod
    def compute_state_derivatives(
        self, tank_sate: TankState, fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        ...


@dataclass
class InitialState:
    pressure: float
    temperature: float
    fill: float

    def __post_init__(self):
        self.hydrogen = self.get_hydrogen_properties()

    def get_hydrogen_properties(self) -> Hydrogen:
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        return self.hydrogen

    def compute_fuel_mass(self, tank_volume: float) -> float:
        if self.fill == 0.0 or self.fill == 1.0:
            return self.hydrogen.density * tank_volume
        return tank_volume * (
            self.fill * self.hydrogen.liquid.density
            + (1 - self.fill) * self.hydrogen.gas.density
        )


@dataclass
class TargetState:
    max_pressure: float
    min_pressure: float
    min_temperature: float
    fill: float
    mass: float


@dataclass
class TankState:
    tank: Tank
    temperature: float
    pressure: float
    fuel_mass: float

    @property
    def volume(self):
        return self.tank.volume
        
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
    def fill(self):
        if self.phase == "gas":
            return 0.0
        if self.phase == "liquid":
            return 1.0
        
        # Ensure that divide by zero is not possible
        if self.volume == 0:
            raise ValueError("Volume cannot be zero")
        
        # Ensure densities are valid
        if self.hydrogen.liquid.density <= self.hydrogen.gas.density:
            raise ValueError("Liquid density must be greater than gas density")
        
        fill_value = (
            (self.fuel_mass / self.volume - self.hydrogen.gas.density)
            / (self.hydrogen.liquid.density - self.hydrogen.gas.density)
        )
         # Ensure fill value is not negative
        if fill_value < 0:
            fill_value = 0
        
        return fill_value
    
    @property
    def fuel_volume(self):
        return self.fill * self.volume

    @property
    def fuel_height(self):
        if self.fuel_volume <= 0:
            return 0
        return self.tank.compute_fuel_height(self.fuel_volume)
    
    @property
    def is_full(self):
        return self.fill >= 1

    @property
    def is_empty(self):
        return self.fill == 0 or self.fuel_height == 0

    @property
    def phase(self) -> str:
        hydrogen_state = self.hydrogen.phase
        if "liquid" in hydrogen_state:
            return "liquid"
        if hydrogen_state == "twophase":
            return "twophase"
        return "gas"

    def __post_init__(self) -> None:
        self.get_hydrogen_properties()
        self.complete_state_properties()

    def complete_state_properties(self):
        if self.pressure is None:
            self.pressure = self.hydrogen.pressure
        if self.temperature is None:
            self.temperature = self.hydrogen.temperature

    def get_hydrogen_properties(self) -> Hydrogen:
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        return self.hydrogen

    def compute_state_derivatives(
        self,
        dynamic_model: DynamicModel,
        fuel_flows: list[FuelFlow],
        heat_flux: float,
        tank_thermal_capacity: float
    ) -> StateDerivatives:
        self.heat_flux = heat_flux
        self.tank_thermal_capacity = tank_thermal_capacity
        self.derivatives = dynamic_model.compute_state_derivatives(
            self, fuel_flows
        )
        return self.derivatives


@dataclass
class TankStates:
    states: list[TankState]
    timestep: float

    def __add__(self, other: TankStates) -> TankStates:
        if len(self.states) == 0:
            self.states = other.states
            return self
        if self.states[-1] == other.states[0]:
            self.states += other.states[1:]
            return self
        self.states += other.states
        return self

    def add_tank_state(self, tank_state: TankState) -> list[TankState]:
        self.states.append(tank_state)
        return self.states

    @property
    def timesteps_in_hours(self):
        return [
            i * self.timestep * SECONDS_TO_HOURS
            for i, _ in enumerate(self.pressures)
        ]

    @property
    def last_state(self):
        return self.states[-1]

    @property
    def first_state(self):
        return self.states[0]

    @property
    def pressures_in_bar(self):
        return [
            pressure * PASCAL_TO_BAR for pressure in self.pressures
        ]

    @property
    def pressures(self):
        return [state.pressure for state in self.states]

    @property
    def temperatures(self):
        return [state.temperature for state in self.states]

    @property
    def pressure_derivatives(self):
        return [state.derivatives.pressure for state in self.states]

    @property
    def temperature_derivatives(self):
        return [state.derivatives.temperature for state in self.states]

    @property
    def initial_temperature(self) -> float:
        return self.states[0].temperature

    @property
    def last_pressure(self):
        return self.last_state.pressure

    @property
    def last_temperature(self):
        return self.last_state.temperature

    @property
    def last_fill(self):
        return self.last_state.fill

    @property
    def max_pressure(self):
        return max(self.pressures)

    @property
    def average_temperature(self):
        return mean(self.temperatures)

    @property
    def min_temperature(self):
        return min(self.temperatures)

    @property
    def hydrogens(self) -> list[Hydrogen]:
        return [state.hydrogen for state in self.states]

    @property
    def fills(self) -> list[float]:
        return [state.fill for state in self.states]

    @property
    def volumes(self) -> list[float]:
        return [state.volume for state in self.states]

    @property
    def liquid_masses(self) -> list[float]:
        masses = [
            fill * volume * hydrogen.liquid.density
            if fill != 0 else 0
            for fill, volume, hydrogen in zip(
                self.fills, self.volumes, self.hydrogens
            )
        ]
        for mass, fill, volume, hydrogen in zip(masses, self.fills, self.volumes, self.hydrogens):
            if mass < 0:
                raise ValueError(f"Negative liquid mass detected: mass={mass}, volume={volume}, fill={fill}, density={hydrogen.liquid.density}")
        return masses

    @property
    def gas_masses(self) -> list[float]:
        masses = [
            (1 - fill) * volume * hydrogen.gas.density
            if fill < 1 else 0
            for fill, volume, hydrogen in zip(
                self.fills, self.volumes, self.hydrogens
            )
        ]
        for mass, fill, volume, hydrogen in zip(masses, self.fills, self.volumes, self.hydrogens):
            if mass < 0:
                raise ValueError(f"Negative gas mass detected: mass={mass}, volume={volume}, fill={fill}, density={hydrogen.gas.density}")
        return masses

    @property
    def total_masses(self) -> list[float]:
        masses = [
            liquid_mass + gas_mass
            for liquid_mass, gas_mass in zip(self.liquid_masses, self.gas_masses)
        ]
        for mass, liquid_mass, gas_mass, fill, volume, hydrogen in zip(masses, self.liquid_masses, self.gas_masses, self.fills, self.volumes, self.hydrogens):
            if mass < 0:
                raise ValueError(f"Negative total mass detected: mass={mass}, liquid_mass={liquid_mass}, gas_mass={gas_mass}, volume={volume}, fill={fill}, liquid_density={hydrogen.liquid.density}, gas_density={hydrogen.gas.density}")
        return masses

    @property
    def state_derivatives(self):
        return [
            state.derivatives
            if hasattr(state, "derivatives")
            else self.states[i-1].derivatives
            for i, state in enumerate(self.states[:-1])
        ]

    @property
    def required_fluxes(self):
        return [
            derivative.heat_flux
            for derivative in self.state_derivatives
        ]


def main():
    pass


if __name__ == "__main__":
    main()

# End
