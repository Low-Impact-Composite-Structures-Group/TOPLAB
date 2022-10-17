

from dataclasses import dataclass
from typing import Protocol

from src.thermodynamics.tank_states import InitialState, TankState, TargetState


class FuelTank(Protocol):
    volume: float

    def compute_fuel_height(self, fuel_volume: float) -> float:
        ...

    def compute_thermal_capacity(self, temperature: float) -> float:
        ...


class FuelFlow(Protocol):
    ...


class StateDerivatives(Protocol):
    pressure: float
    temperature: float
    gas_mass: float
    liquid_mass: float
    venting_mass: float
    heat_flux: float


class MultistepMethod(Protocol):
    timestep: float

    def compute_new_value(
        self,
        derivatives: list[float],
        current_value: float
    ) -> float:
        ...


class DynamicModel(Protocol):

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        ...


class DynamicModelFactory(Protocol):

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: TargetState
    ) -> DynamicModel:
        ...


class ConvectiveMedium:
    ...


class MissionSection(Protocol):
    duration: float
    fuel_flows: list[FuelFlow]
    ambient: ConvectiveMedium
    flight_speed: float

    def number_of_timesteps(self, timestep: float) -> float:
        ...

    def final_timestep(self, timestep: float) -> float:
        ...


class StoppingCriterion(Protocol):

    def is_met(
        self,
        tank_state: TankState,
        target_state: TargetState
    ) -> bool:
        ...


class ThermodynamicModel(Protocol):

    def compute_heat_flux(
        self,
        tank: FuelTank,
        state: TankState,
        mission_section: MissionSection
    ) -> tuple[float, list]:
        ...


@dataclass
class AnalyseMissionSection:
    tank: FuelTank
    initial: InitialState
    mission_section: MissionSection
    stopping_criteria: list[StoppingCriterion]
    target_conditions: TargetState
    multistep_method: MultistepMethod
    dynamic_model_factory: DynamicModelFactory
    thermal_model: ThermodynamicModel

    heat_flux_factor: float = 1.0

    def __post_init__(self) -> None:
        self.tank_states: list[TankState] = [
            self.define_tank_state(
                self.initial.pressure,
                self.initial.temperature,
                self.initial.fill
            )
        ]
    
    @property
    def timesteps(self) -> int:
        return self.mission_section.number_of_timesteps(
            self.multistep_method.timestep
        )

    @property
    def last_tank_state(self) -> TankState:
        return self.tank_states[-1]

    def define_tank_state(
        self, pressure: float, temperature: float, fill: float
    ) -> TankState:
        state = TankState(
            temperature,
            pressure,
            fill,
            self.tank.compute_fuel_height(self.tank.volume * fill),
            self.tank.volume
        )
        heat_flux, temperatures = self.thermal_model.compute_heat_flux(
            self.tank, state, self.mission_section
        )
        state.set_thermal_capacity(
            self.tank.compute_thermal_capacity(temperatures[0])
        )
        state.set_heat_flux(heat_flux)
        dynamic_model = self.dynamic_model_factory.get_dynamic_model(
            state, self.target_conditions
        )
        state.set_state_derivatives(
            dynamic_model.compute_state_derivatives(
                state,
                self.mission_section.fuel_flows
            )
        )
        return state

    def compute_new_pressure(self) -> float:
        pressure_derivatives = [
            state.derivatives.pressure
            for state in self.tank_states
        ]
        new_pressure = self.multistep_method.compute_new_value(
            pressure_derivatives,
            self.tank_states[-1].pressure
        )
        return new_pressure

    def compute_new_temperature(self) -> float:
        temperature_derivatives = [
            state.derivatives.temperature
            for state in self.tank_states
        ]
        return self.multistep_method.compute_new_value(
            temperature_derivatives,
            self.tank_states[-1].temperature
        )

    def compute_new_fill(self) -> float:
        if self.last_tank_state.derivatives.liquid_mass == 0:
            return self.last_tank_state.fill
        return (
            self.last_tank_state.fill
            + self.last_tank_state.derivatives.liquid_mass
            / self.last_tank_state.hydrogen.liquid.density
            / self.tank.volume
            * self.multistep_method.timestep
        )

    def stopping_criterion_is_met(self) -> bool:
        for criterion in self.stopping_criteria:
            if criterion.is_met(
                self.tank_states[-1], self.target_conditions
            ):
                return True
        return False
        
    def analyse_mission_section(self) -> list[TankState]:

        for _ in range(self.timesteps):

            # Define the new tank state
            self.tank_states.append(
                self.define_tank_state(
                    self.compute_new_pressure(),
                    self.compute_new_temperature(),
                    self.compute_new_fill()
                )
            )
            
            # Check if one of the stopping criteria has been met, if so,
            # exit the iteration and return the tank states
            if self.stopping_criterion_is_met():
                return self.tank_states
        return self.tank_states


def main():
    pass


if __name__ == "__main__":
    main()


# End
