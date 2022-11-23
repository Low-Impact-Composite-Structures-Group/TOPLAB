
from __future__ import annotations

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


class DynamicModel(Protocol):

    def compute_state_derivatives(
        self, tank_sate: TankState, fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        ...

@dataclass
class InitialState:
    pressure: float
    temperature: float
    fill: float

    def get_hydrogen_properties(self) -> Hydrogen:
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        return self.hydrogen


@dataclass
class TargetState:
    max_pressure: float
    min_pressure: float
    min_temperature: float
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
        return [
            fill * volume * hydrogen.liquid.density
            if fill != 0 else 0
            for fill, volume, hydrogen in zip(
                self.fills, self.volumes, self.hydrogens
            )
        ]

    @property
    def gas_masses(self) -> list[float]:
        return [
            (1 - fill) * volume * hydrogen.gas.density
            for fill, volume, hydrogen in zip(
                self.fills, self.volumes, self.hydrogens
            )
        ]

    @property
    def total_masses(self):
        return [
            liquid_mass + gas_mass
            for liquid_mass, gas_mass in zip(
                self.liquid_masses, self.gas_masses
            )
        ]

    @property
    def state_derivatives(self):
        return [state.derivatives for state in self.states[:-1]]

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
