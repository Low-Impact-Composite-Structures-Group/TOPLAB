

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from src.dynamics.stopping_criteria import LowerPressureReached, MaxPressureReached, NoFuelMass, TankIsEmpty
from src.mission.mission import Mission, MissionSection
from src.mission.mission_sections import FuelFlow, OutFlow
from src.thermodynamics.tank_states import (InitialConditions, TankState,
                                            TankStates, OperationalEnvelope)

# The thermal capacity of the tank depends on the mass of the tank, as
# such this needs to be iterated as the operating pressure of the 
# tank is refined. Here the maximum amount of iterations are defined
# and the percentage change in capacity of the tank
THERMAL_CAPACITY_THRESHOLD = 1              # This is as a percentage
# LOWER_MASS_LIMIT = 500   

class FuelTank(Protocol):
    volume: float

    @abstractmethod
    def compute_fuel_height(self, fuel_volume: float) -> float:
        ...
    
    @abstractmethod
    def compute_thermal_capacity(self, temperature: float) -> float:
        ...
    
    @abstractmethod
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

    @abstractmethod
    def compute_new_value(
        self,
        derivatives: list[float],
        current_value: float
    ) -> float:
        ...


class DynamicModel(Protocol):

    @classmethod
    @abstractmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        ...


class DynamicModelFactory(Protocol):

    @abstractmethod
    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperationalEnvelope
    ) -> DynamicModel:
        ...


# class ConvectiveMedium:
#     ...


class StoppingCriterion(Protocol):

    @abstractmethod
    def is_met(
        self,
        tank_state: TankState,
        target_state: OperationalEnvelope
    ) -> bool:
        ...


class ThermodynamicModel(Protocol):

    @abstractmethod
    def compute_heat_flux(
        self,
        tank: FuelTank,
        state: TankState,
        mission_section: MissionSection
    ) -> tuple[float, list]:
        ...


# class OutFlow(Protocol):
#     mass_flow: float
#     phase: str

@dataclass
class FuelFlow:
    mass_flow: float
    hydrogen: Hydrogen


class MissionSectionAnalysis:

    @classmethod
    def initialise_tank_states(
        cls,
        initial_state: InitialConditions,
        tank: FuelTank,
        timestep: float
    ) -> TankStates:
        initial_state = TankState(
            tank,
            initial_state.temperature,
            initial_state.pressure,
            initial_state.compute_fuel_mass(tank.volume)
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
        target_conditions: OperationalEnvelope,
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
        target_state: OperationalEnvelope
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

    @classmethod
    def compute_new_mass(
        cls,
        timestep: float,
        tank_states: TankStates
    ) -> float:
        new_mass = (
                tank_states.last_state.fuel_mass
                + (
                    tank_states.last_state.derivatives.liquid_mass
                    + tank_states.last_state.derivatives.gas_mass
                ) * timestep
            )
        return new_mass
   
    @classmethod
    def analyse_section(
        cls,
        tank: FuelTank,
        initial: InitialConditions,
        mission_section: MissionSection,
        stopping_criteria: list[StoppingCriterion],
        target_conditions: OperationalEnvelope,
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
            new_values = (
                cls.compute_new_temperature(
                    multistep_method, tank_states
                ),
                cls.compute_new_pressure(
                    multistep_method, tank_states
                ),
                cls.compute_new_mass(
                    multistep_method.timestep, tank_states
                )
            )
            if any(value < 0 for value in new_values):
                return tank_states
            tank_states.add_tank_state(
                TankState(tank, *new_values)
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
        initial_state: InitialConditions,
        mission: Mission,
        stopping_criteria: list[StoppingCriterion],
        operational_envelope: OperationalEnvelope,
        multistep_method: MultistepMethod,
        dynamic_model_factory: DynamicModelFactory,
        thermal_model: ThermodynamicModel,
        heat_flux_factor: float,
        thermal_capacity_convergence: bool = False,
        max_thermal_capacity_iterations: int = 5,
        user_update: bool = False,
    ) -> TankStates:

        # Iterate till the thermal capacity has converge
        for i in range(max_thermal_capacity_iterations):

            # Define initial state of the tank
            initial = InitialConditions(initial_state.pressure, initial_state.temperature, initial_state.fill)
            tank_states = TankStates(list(), multistep_method.timestep)

            for mission_section in mission.sections:

                tank_states += MissionSectionAnalysis().analyse_section(
                    tank,
                    initial,
                    mission_section,
                    stopping_criteria,
                    operational_envelope,
                    multistep_method,
                    dynamic_model_factory,
                    thermal_model,
                    heat_flux_factor
                )

                initial = InitialConditions(
                    tank_states.last_pressure,
                    tank_states.last_temperature,
                    tank_states.last_fill
                )

            # Check for convergence in the thermal capacity of the tank
            if cls.thermal_capacity_has_converged(tank, tank_states) or not thermal_capacity_convergence:
                if user_update:
                    cls._update(tank, tank_states, i)
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

    @classmethod
    def _update(cls, tank: FuelTank, tank_states: TankStates, thermal_capacity_iterations: int):
        print(
            f"{tank} done with analysis.",
            f"Max pressure: {tank_states.max_pressure}.",
            f"Thermal capacity iterations: {thermal_capacity_iterations}",
        )


class SwitchPhaseDrainingAnalysis:

    @classmethod
    def perform_analysis(
        cls,
        tank: FuelTank,
        initial_state: InitialConditions,
        mission: Mission,
        operational_envelope: OperationalEnvelope,
        multistep_method: MultistepMethod,
        dynamic_model_factory: DynamicModelFactory,
        thermal_model: ThermodynamicModel,
        heat_flux_factor: float,
        max_changes: int = 1000,
    ) -> TankStates:

        # Define initial state of the tank
        last_state = initial_state
        tank_states = TankStates(list(), multistep_method.timestep)

        mission_section = mission.sections[0]

        # Define the stopping criteria
        stopping_criteria = [
            cls._empty_criterion(),
            cls._max_pressure_criterion(),
            cls._min_pressure_criterion(),
        ]

        
        # Set up iterations
        for _ in range(max_changes):
            
            tank_states += MissionSectionAnalysis().analyse_section(
                tank,
                cls._define_initial_conditions(last_state),
                mission_section,
                stopping_criteria,
                operational_envelope,
                multistep_method,
                dynamic_model_factory,
                thermal_model,
                heat_flux_factor
            )

            if cls._empty_criterion().is_met(tank_states.last_state, operational_envelope):
                return tank_states

            mission_section.fuel_flows[0].phase = "gas" if mission_section.fuel_flows[0].phase == "liquid" else "liquid"
            stopping_criteria = cls._get_stopping_criteria(
                mission_section.fuel_flows[0].phase
            )

            last_state = tank_states.last_state
        
        raise ValueError(
            "Exceeded maximum iterations is switch drain analysis..."
        )
    
    @staticmethod
    def _define_initial_conditions(tank_state: TankState) -> InitialConditions:
        return InitialConditions(
            tank_state.pressure, tank_state.temperature, tank_state.fill
        )
    
    @classmethod
    def _get_stopping_criteria(cls, fuel_phase_flow) -> list[StoppingCriterion]:
        if fuel_phase_flow == "liquid":
            return [cls._empty_criterion(), cls._max_pressure_criterion()]
        if fuel_phase_flow == "gas":
            return [cls._empty_criterion(), cls._min_pressure_criterion()]

    @staticmethod
    def _empty_criterion() -> TankIsEmpty:
        return TankIsEmpty()

    @staticmethod
    def _max_pressure_criterion():
        return MaxPressureReached()
    
    @staticmethod
    def _min_pressure_criterion():
        return LowerPressureReached()
    
    
def main():
    pass


if __name__ == "__main__":
    main()


# End
