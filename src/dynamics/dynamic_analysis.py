

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from src.dynamics.stopping_criteria import TankIsEmpty
from src.mission.mission import Mission, MissionSection

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


class MissionSectionAnalysis:

    @classmethod
    def initialise_tank_states(
        cls,
        initial_state: InitialState,
        tank: FuelTank,
        timestep: float
    ) -> TankStates:
        initial_state = TankState(
            initial_state.temperature,
            initial_state.pressure,
            initial_state.fill,
            tank.compute_fuel_height(
            tank.volume * initial_state.fill
            ),
            tank.volume
        )
        return TankStates([initial_state], timestep)

    @staticmethod
    def define_fuel_flows(
        fuel_flows: list[FuelFlow | OutFlow], tank_state: TankState
    ) -> list[FuelFlow]:
        return [
            fuel_flow
            if fuel_flow.mass_flow > 0
            else FuelFlow(
                fuel_flow.mass_flow,
                tank_state.hydrogen.get_phase(fuel_flow.phase)
            )
            for fuel_flow in fuel_flows
        ]

    @classmethod
    def compute_state_derivatives(
        cls,
        tank: FuelTank,
        thermal_model: ThermodynamicModel,
        tank_state: TankState,
        mission_section: MissionSection,
        dynamic_model_factory: DynamicModelFactory,
        target_conditions: TargetState,
        heat_flux_factor: float
    ) -> TankState:
        heat_flux, temperatures = thermal_model.compute_heat_flux(
            tank, tank_state, mission_section
        )
        dynamic_model = dynamic_model_factory.get_dynamic_model(
            tank_state, target_conditions
        )
        tank_state.compute_state_derivatives(
            dynamic_model,
            cls.define_fuel_flows(
                mission_section.fuel_flows,
                tank_state
            ),
            heat_flux * heat_flux_factor,
            tank.compute_thermal_capacity(temperatures[0])
        )
        return tank_state
   
    @staticmethod
    def stopping_criterion_is_met(
        stopping_criteria: list[StoppingCriterion],
        current_state: TankState,
        target_state: TargetState
    ) -> bool:
        for criterion in stopping_criteria:
            if criterion.is_met(
                current_state, target_state
            ):
                return True
        return False

    @classmethod
    def compute_new_temperature(
        cls,
        multistep_method: MultistepMethod,
        tank_states: TankStates
    ) -> float:
        new_temperature = multistep_method.compute_new_value(
                tank_states.temperature_derivatives,
                tank_states.last_temperature
            )
        
        return new_temperature

    @classmethod
    def compute_new_pressure(
        cls,
        multistep_method: MultistepMethod,
        tank_states: TankStates
    ) -> float:
        new_pressure = multistep_method.compute_new_value(
                tank_states.pressure_derivatives,
                tank_states.last_pressure
            )
        
        return new_pressure

    @staticmethod
    def compute_new_fill(
        last_state: TankState,
        timestep: float
    ) -> float:
        if last_state.derivatives.liquid_mass == 0:
            return last_state.fill
        new_fill = (
            last_state.fill
            + last_state.derivatives.liquid_mass
            / last_state.hydrogen.liquid.density
            / last_state.volume
            * timestep
        )
        # This line is added, as the step may be such that a negative 
        # fill is obtained. To avoid this the fill is simply set to 0
        if new_fill < 0:
            print(f"Negative fill value: {new_fill}. Fill forced to 0")
            return 0
        return new_fill
   
    @classmethod
    def analyse_section(
        cls,
        tank: FuelTank,
        initial: InitialState,
        mission_section: MissionSection,
        stopping_criteria: list[StoppingCriterion],
        target_conditions: TargetState,
        multistep_method: MultistepMethod,
        dynamic_model_factory: DynamicModelFactory,
        thermal_model: ThermodynamicModel,
        heat_flux_factor: float
    ) -> TankStates:

        tank_states = cls.initialise_tank_states(
            initial,
            tank,
            multistep_method.timestep
        )

        steps = mission_section.number_of_timesteps(
            multistep_method.timestep
        )
        for _ in range(steps):

            cls.compute_state_derivatives(
                tank,
                thermal_model,
                tank_states.last_state,
                mission_section,
                dynamic_model_factory,
                target_conditions,
                heat_flux_factor
            )

            new_fill = cls.compute_new_fill(
                tank_states.last_state,
                multistep_method.timestep
            )
            tank_states.add_tank_state(
                TankState(
                    cls.compute_new_temperature(
                        multistep_method, tank_states
                    ),
                    cls.compute_new_pressure(
                        multistep_method, tank_states
                    ),
                    new_fill,
                    tank.compute_fuel_height(
                        tank.volume * new_fill
                    ),
                    tank.volume
                )
            )

            if cls.stopping_criterion_is_met(
                stopping_criteria,
                tank_states.last_state,
                target_conditions
            ):
                return tank_states

        return tank_states


class MissionAnalysis:

    @classmethod
    def perform_analysis(
        cls,
        tank: FuelTank,
        initial_state: InitialState,
        mission: Mission,
        stopping_criteria: list[StoppingCriterion],
        target_conditions: TargetState,
        multistep_method: MultistepMethod,
        dynamic_model_factory: DynamicModelFactory,
        thermal_model: ThermodynamicModel,
        heat_flux_factor: float
    ) -> TankStates:

        # Iterate till the thermal capacity has converge
        for i in range(MAX_THERMAL_CAPACITY_ITERATIONS):

            # Define initial state of the tank
            initial = initial_state
            tank_states = TankStates(list(), multistep_method.timestep)

            for mission_section in mission.sections:

                tank_states += MissionSectionAnalysis().analyse_section(
                    tank,
                    initial,
                    mission_section,
                    stopping_criteria,
                    target_conditions,
                    multistep_method,
                    dynamic_model_factory,
                    thermal_model,
                    heat_flux_factor
                )

                initial = InitialState(
                    tank_states.last_pressure,
                    tank_states.last_temperature,
                    tank_states.last_fill
                )

            # Check for convergence in the thermal capacity of the tank
            if cls.thermal_capacity_has_converged(tank, tank_states):
                print(f"Thermal capacity converged in {i} iterations")
                return tank_states
        raise ValueError("Thermal capacity has failed to converge")

    @classmethod
    def thermal_capacity_has_converged(
        cls, tank: FuelTank, tank_states: TankStates
    ) -> bool:

        # Compute old thermal capacity
        old = tank.compute_thermal_capacity(
            tank_states.average_temperature
        )

        # Update thermal capacity 
        tank.set_operating_pressure(tank_states.max_pressure)
        new = tank.compute_thermal_capacity(
            tank_states.average_temperature
        )

        # Compute percentage change and verify convergence
        if abs((old - new) / old) * 100 <= THERMAL_CAPACITY_THRESHOLD:
            return True
        return False


class DrainingAnalysis:

    @classmethod
    def perform_analysis(
        cls,
        tank: FuelTank,
        fuel_mass_flow: float,
        fuel_flow_state: float,
        initial_state: InitialState,
        multistep_method: MultistepMethod,
        dynamic_model_factory: DynamicModelFactory,
        thermal_model: ThermodynamicModel,
        heat_flux_factor: float
    ) -> TankStates:

        # Definition of the mission
        mission_section = MissionSection.draining(
            fuel_mass_flow, fuel_flow_state
        )
        mission = Mission([mission_section])

        # Definition of stopping criteria
        stopping_criteria = [TankIsEmpty()]

        # Define the target sate
        pressure = None
        temperature = None
        fill = 0.0
        mass = None
        target_conditions = TargetState(
            pressure,
            temperature,
            fill,
            mass
        )

        return MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            stopping_criteria,
            target_conditions,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor
        )

    
def main():
    pass


if __name__ == "__main__":
    main()


# End
