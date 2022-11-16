

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.thermodynamics.tank_states import (InitialState, TankState,
                                            TankStates, TargetState)

# The thermal capacity of the tank depends on the mass of the tank, as
# such this needs to be iterated as the operating pressure of the 
# tank is refined. Here the maximum amount of iterations are defined
# and the percentage change in capacity of the tank
MAX_THERMAL_CAPACITY_ITERATIONS = 5
THERMAL_CAPACITY_THRESHOLD = 1              # This is as a percentage


class FuelTank(Protocol):
    volume: float

    def compute_fuel_height(self, fuel_volume: float) -> float:
        ...

    def compute_thermal_capacity(self, temperature: float) -> float:
        ...

    def set_operating_pressure(self, pressure: float) -> float:
        ...


class Hydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen


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
    fuel_flows: list[FuelFlow | OutFlow]
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


class OutFlow(Protocol):
    mass_flow: float
    phase: str

@dataclass
class FuelFlow:
    mass_flow: float
    hydrogen: Hydrogen


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

    def define_tank_states(self):
        self.tank_states = TankStates(
            list(), self.multistep_method.timestep
        )
        self.set_up_new_tank_state(
            self.initial.pressure,
            self.initial.temperature,
            self.initial.fill
        )
        self.compute_state_derivatives()
    
    @property
    def timesteps(self) -> int:
        return self.mission_section.number_of_timesteps(
            self.multistep_method.timestep
        )

    def define_fuel_flows(self) -> list[FuelFlow]:
        fuel_flows = [
            fuel_flow
            if fuel_flow.mass_flow > 0 else
            FuelFlow(
                fuel_flow.mass_flow,
                self.tank_states.last_state.hydrogen.get_phase(
                    fuel_flow.phase
                )
            )
            for fuel_flow in self.mission_section.fuel_flows
        ]
        return fuel_flows

    def set_up_new_tank_state(
        self, pressure: float, temperature: float, fill: float
    ) -> TankState:
        self.tank_states.add_tank_state(
            TankState(
                temperature,
                pressure,
                fill,
                self.tank.compute_fuel_height(self.tank.volume * fill),
                self.tank.volume
            )
        )
        return self.tank_states.last_state
    
    def compute_state_derivatives(self):
        heat_flux, temperatures = self.thermal_model.compute_heat_flux(
            self.tank, self.tank_states.last_state, self.mission_section
        )
        dynamic_model = self.dynamic_model_factory.get_dynamic_model(
            self.tank_states.last_state, self.target_conditions
        )
        self.tank_states.last_state.compute_state_derivatives(
            dynamic_model,
            self.define_fuel_flows(),
            heat_flux * self.heat_flux_factor,
            self.tank.compute_thermal_capacity(temperatures[0])
        )
        return self.tank_states.last_state
 
    def compute_new_pressure(self) -> float:
        new_pressure = self.multistep_method.compute_new_value(
            self.tank_states.pressure_derivatives,
            self.tank_states.last_pressure
        )
        return new_pressure

    def compute_new_temperature(self) -> float:
        return self.multistep_method.compute_new_value(
            self.tank_states.temperature_derivatives,
            self.tank_states.last_temperature
        )

    def compute_new_fill(self) -> float:
        if self.tank_states.last_state.derivatives.liquid_mass == 0:
            return self.tank_states.last_state.fill
        new_fill = (
            self.tank_states.last_state.fill
            + self.tank_states.last_state.derivatives.liquid_mass
            / self.tank_states.last_state.hydrogen.liquid.density
            / self.tank.volume
            * self.multistep_method.timestep
        )
        # This line is added, as the step may be such that a negative 
        # fill is obtained. To avoid this the fill is simply set to 0
        if new_fill < 0:
            print(f"Negative fill value: {new_fill}. Fill forced to 0")
            return 0
        return new_fill
        
    def analyse_mission_section(self) -> TankStates:

        for _ in range(self.timesteps):

            self.set_up_new_tank_state(
                self.compute_new_pressure(),
                self.compute_new_temperature(),
                self.compute_new_fill()
            )

            if self.stopping_criterion_is_met():
                return self.tank_states
            self.compute_state_derivatives()

        return self.tank_states

    def stopping_criterion_is_met(self) -> bool:
        for criterion in self.stopping_criteria:
            if criterion.is_met(
                self.tank_states.last_state, self.target_conditions
            ):
                return True
        return False

    def thermal_capacity_has_converged(self) -> bool:
        old_thermal_capacity = self.tank.compute_thermal_capacity(
            self.tank_states.average_temperature
        )
        self.tank.set_operating_pressure(self.tank_states.max_pressure)
        new_thermal_capacity = self.tank.compute_thermal_capacity(
            self.tank_states.average_temperature
        )
        percentage_change = (
            (old_thermal_capacity - new_thermal_capacity)
            / old_thermal_capacity * 100
        )
        if abs(percentage_change) <= THERMAL_CAPACITY_THRESHOLD:
            return True
        return False

    def perform_analysis(self):

        for i in range(MAX_THERMAL_CAPACITY_ITERATIONS):
            self.define_tank_states()
            self.analyse_mission_section()
            if self.thermal_capacity_has_converged():
                print(
                    f"Thermal capacity has converged in {i} iterations"
                )
                return self.tank_states
        
        raise ValueError("Thermal capacity has failed to converge...")

    
def main():
    pass


if __name__ == "__main__":
    main()


# End
