from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from src.multistate.fluids.hydrogen_retrievers import IsochoricHydrogenRequester


class Tank(Protocol):
    volume: float

    @abstractmethod
    def compute_fuel_height(self, fuel_volume: float):
        ...


@dataclass
class IsochoricTankState:
    """
    Isochoric tank state for the multistate solver.

    This keeps the multistate state vector local to src/multistate while
    temporarily reusing shared fluid/property helpers from the compatibility
    layer in src/fluids.
    """

    tank: Tank
    fuel_mass: float
    temperature: float
    solid_temperature: float
    pressure: float = None
    configuration: str = "A"
    scenario: str = "DISCHARGE"
    hydrogen: object = None
    derivatives: IsochoricStateDerivatives = None
    inflow_rate: float = 0.0
    outflow_rate: float = 0.0
    vent_rate: float = 0.0
    coupling_inflow_rate: float = 0.0
    coupling_outflow_rate: float = 0.0

    @property
    def volume(self):
        return self.tank.volume

    @property
    def density(self):
        return self.fuel_mass / self.volume

    @property
    def state_vector(self):
        return [self.fuel_mass, self.temperature, self.solid_temperature]

    @classmethod
    def from_state_vector(cls, tank: Tank, state_vector: list, **kwargs):
        m, temperature, solid_temperature = state_vector
        return cls(
            tank=tank,
            fuel_mass=m,
            temperature=temperature,
            solid_temperature=solid_temperature,
            **kwargs,
        )

    def __post_init__(self):
        self.get_hydrogen_properties()
        self.compute_pressure()

    def get_hydrogen_properties(self):
        if self.hydrogen is None or self._needs_hydrogen_update():
            requester = IsochoricHydrogenRequester()
            if self.pressure is None:
                self.compute_pressure()
            self.hydrogen = requester.get_hydrogen_properties(
                self.pressure,
                self.temperature,
                self.density,
            )

    def compute_pressure(self):
        if self.pressure is None:
            try:
                from src.multistate.fluids.coolprop_safe import safe_pressure_from_T_rho

                if self.temperature <= 0:
                    self.pressure = 1e5
                    return
                self.pressure = safe_pressure_from_T_rho(
                    self.temperature,
                    self.density,
                    "hydrogen",
                )
            except Exception:
                self.pressure = 1e5

    def _needs_hydrogen_update(self) -> bool:
        if self.hydrogen is None:
            return True

        temp_change = abs(self.temperature - self.hydrogen.temperature) / self.hydrogen.temperature
        density_change = abs(self.density - self.hydrogen.density) / self.hydrogen.density
        return temp_change > 0.01 or density_change > 0.01

    def update_from_state_vector(self, state_vector: list):
        self.fuel_mass, self.temperature, self.solid_temperature = state_vector
        self.compute_pressure()
        self.get_hydrogen_properties()

    def is_configuration_B(self, p_min: float) -> bool:
        return self.pressure <= p_min

    def is_configuration_C(self, p_vent: float) -> bool:
        return self.pressure >= p_vent

    def determine_configuration(self, p_min: float, p_vent: float) -> str:
        if self.is_configuration_C(p_vent):
            return "C"
        if self.is_configuration_B(p_min):
            return "B"
        return "A"

    def get_effective_cv(self) -> float:
        if self.hydrogen is not None:
            return self.hydrogen.get_effective_cv()

        from CoolProp.CoolProp import PropsSI

        return PropsSI("Cvmass", "T", self.temperature, "Dmass", self.density, "hydrogen")


@dataclass
class IsochoricStateDerivatives:
    fuel_mass_derivative: float
    temperature_derivative: float
    solid_temperature_derivative: float
    heat_flux: float = 0.0
    discharge_heat_flux: float = 0.0
    alpha_s: float = 0.0

    @property
    def state_derivative_vector(self):
        return [
            self.fuel_mass_derivative,
            self.temperature_derivative,
            self.solid_temperature_derivative,
        ]


@dataclass
class IsochoricInitialState:
    fuel_mass: float
    temperature: float
    solid_temperature: float
    pressure: float = None
    scenario: str = "DISCHARGE"

    def to_isochoric_tank_state(self, tank: Tank) -> IsochoricTankState:
        return IsochoricTankState(
            tank=tank,
            fuel_mass=self.fuel_mass,
            temperature=self.temperature,
            solid_temperature=self.solid_temperature,
            pressure=self.pressure,
            scenario=self.scenario,
        )

    def get_state_vector(self):
        return [self.fuel_mass, self.temperature, self.solid_temperature]


@dataclass
class IsochoricTankStates:
    states: list[IsochoricTankState]
    timestep: float

    def __add__(self, other: "IsochoricTankStates") -> "IsochoricTankStates":
        if len(self.states) == 0:
            self.states = other.states
            return self
        if len(other.states) > 0:
            self.states += other.states
        return self

    def add_state(self, state: IsochoricTankState):
        self.states.append(state)

    @property
    def last_state(self) -> IsochoricTankState:
        return self.states[-1]

    @property
    def first_state(self) -> IsochoricTankState:
        return self.states[0]

    @property
    def times(self) -> list[float]:
        return [i * self.timestep for i in range(len(self.states))]

    @property
    def fuel_masses(self) -> list[float]:
        return [state.fuel_mass for state in self.states]

    @property
    def temperatures(self) -> list[float]:
        return [state.temperature for state in self.states]

    @property
    def solid_temperatures(self) -> list[float]:
        return [state.solid_temperature for state in self.states]

    @property
    def pressures(self) -> list[float]:
        return [state.pressure for state in self.states]

    @property
    def densities(self) -> list[float]:
        return [state.density for state in self.states]

    @property
    def configurations(self) -> list[str]:
        return [state.configuration for state in self.states]

    @property
    def max_pressure(self) -> float:
        return max(self.pressures)

    @property
    def min_temperature(self) -> float:
        return min(self.temperatures)

    @property
    def state_derivatives(self) -> list[IsochoricStateDerivatives]:
        return [state.derivatives for state in self.states if state.derivatives is not None]