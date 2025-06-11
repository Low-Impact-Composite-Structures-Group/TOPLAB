

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Union, List

from src.dynamics.stopping_criteria import NoFuelMass, TankIsEmpty
from src.mission.mission import Mission, MissionSection
from src.thermodynamics.tank_states import (InitialState, TankState,
                                            TankStates, TargetState)

# The thermal capacity of the tank depends on the mass of the tank, as
# such this needs to be iterated as the operating pressure of the
# tank is refined. Here the maximum amount of iterations are defined
# and the percentage change in capacity of the tank
MAX_THERMAL_CAPACITY_ITERATIONS = 5
THERMAL_CAPACITY_THRESHOLD = 1              # This is as a percentage
LOWER_MASS_LIMIT = 500

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
        target_conditions: TargetState
    ) -> DynamicModel:
        ...


class ConvectiveMedium:
    ...


class StoppingCriterion(Protocol):

    @abstractmethod
    def is_met(
        self,
        tank_state: TankState,
        target_state: TargetState
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


class OutFlow(Protocol):
    mass_flow: float
    phase: str

@dataclass
class FuelFlow:
    mass_flow: Union[float, List[float]]
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
            tank,
            initial_state.temperature,
            initial_state.pressure,
            initial_state.compute_fuel_mass(tank.volume)
        )
        return TankStates([initial_state], timestep)

    @staticmethod
    def define_fuel_flows(
        fuel_flows: List[Union[FuelFlow, OutFlow]], tank_state: TankState, section_iter: int, steps: int
    ) -> List[FuelFlow]:
        flows = []
        for fuel_flow in fuel_flows:
            if hasattr(fuel_flow, "phase"):  # OutFlow
                flows.append(
                    FuelFlow(
                        MissionSectionAnalysis.interpolate_mass_flows(fuel_flow.mass_flow, section_iter, steps)
                        if isinstance(fuel_flow.mass_flow, list) else fuel_flow.mass_flow,
                        tank_state.hydrogen.get_phase(fuel_flow.phase)
                    )
                )
            elif hasattr(fuel_flow, "hydrogen"):  # InFlow
                flows.append(
                    FuelFlow(
                        fuel_flow.mass_flow,
                        fuel_flow.hydrogen
                    )
                )
            else:
                raise AttributeError("Unknown fuel_flow type")
        return flows

    @staticmethod
    def interpolate_mass_flows(mass_flows: list[float], section_iter: int, steps: int) -> float:
        if len(mass_flows) != 2:
            raise ValueError("Only two mass flows are supported)")
        start, end = mass_flows
        return start + (end - start) * section_iter / steps

    @classmethod
    def compute_state_derivatives(
        cls,
        tank: FuelTank,
        thermal_model: ThermodynamicModel,
        tank_state: TankState,
        mission_section: MissionSection,
        dynamic_model_factory: DynamicModelFactory,
        target_conditions: TargetState,
        heat_flux_factor: float,
        section_iter: int = None,
        steps: int = None
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
                tank_state,
                section_iter,
                steps
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

        for section_iter in range(steps):

            # print(f"Section iteration: {section_iter}")
            cls.compute_state_derivatives(
                tank,
                thermal_model,
                tank_states.last_state,
                mission_section,
                dynamic_model_factory,
                target_conditions,
                heat_flux_factor,
                section_iter,
                steps
            )

            tank_states.add_tank_state(
                TankState(
                    tank,
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

        # Iterate till the thermal capacity has converged
        for i in range(MAX_THERMAL_CAPACITY_ITERATIONS):

            # Define initial state of the tank
            initial = initial_state
            tank_states = TankStates(list(), multistep_method.timestep)

            for mission_section in mission.sections:
                section_string = mission_section.fuel_flow_key  # Access the key associated with the fuel flow
                if section_string == None:
                    print(f"Now calculating singular mission section, thermal iteration index = {i}")
                else:
                    print(f"Now calculating mission section {section_string}, thermal iteration index = {i}")
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
                print(f"Thermal capacity has converged with {i+1} iterations")
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


class SwitchMissionAnalysis:

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

        return tank_states


class DrainingAnalysis:

    @classmethod
    def perform_analysis(
        cls,
        tank: FuelTank,
        fuel_mass_flow: float,
        fuel_flow_state: float,
        initial_state: InitialState,
        min_pressure: float,
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
        stopping_criteria = [NoFuelMass(), TankIsEmpty()]

        # Define the target sate
        max_pressure = None
        temperature = None
        fill = None
        mass = LOWER_MASS_LIMIT
        target_conditions = TargetState(
            max_pressure,
            min_pressure,
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

 # TODO: add refuelling class here


def main():
    pass


if __name__ == "__main__":
    main()


# End
