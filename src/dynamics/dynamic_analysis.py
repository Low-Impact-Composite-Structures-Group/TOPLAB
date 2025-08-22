from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Union, List, Optional

from src.dynamics.stopping_criteria import NoFuelMass, TankIsEmpty
from src.dynamics.cryopump_model import CryoPumpModel
from src.fluids.hydrogen_retrievers import SinglePhaseRequester, TwoPhaseRequester, HydrogenRetriever
from src.mission.mission import Mission, MissionSection, InFlow

# Use the compute_pump_outlet_hydrogen from CryoPumpModel
compute_pump_outlet_hydrogen = CryoPumpModel.compute_pump_outlet_hydrogen
from src.mission.mission_sections import InFlow as ConcreteInFlow
from src.thermodynamics.tank_states import (InitialState, TankState,
                                            TankStates, TargetState)

# The thermal capacity of the tank depends on the mass of the tank, as
# such this needs to be iterated as the operating pressure of the
# tank is refined. Here the maximum amount of iterations are defined
# and the percentage change in capacity of the tank
MAX_THERMAL_CAPACITY_ITERATIONS = 10
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
        fuel_mass = initial_state.compute_fuel_mass(tank.volume)

        initial_state = TankState(
            tank,
            initial_state.temperature,
            initial_state.pressure,
            fuel_mass,
            multi_flow=getattr(initial_state, "multi_flow", False)
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
                # Update hydrogen properties using our simplified cryopump model
                hydrogen, new_pressure = compute_pump_outlet_hydrogen(tank_state.pressure, tank_state.temperature)

                # Update tank pressure with the pump outlet pressure
                tank_state.pressure = new_pressure

                flows.append(
                    FuelFlow(
                        fuel_flow.mass_flow,
                        hydrogen
                    )
                )
            else:
                raise AttributeError("Unknown fuel_flow type")
        return flows

    @staticmethod
    def interpolate_mass_flows(mass_flows: list[float], section_iter: int, steps: int) -> float:
        if len(mass_flows) != 2:
            raise ValueError("Can only interpolate between 2 values")
        start, end = mass_flows
        return start + (end - start) * section_iter / steps

    @staticmethod
    def update_inflow_hydrogen(flow, tank_pressure: float, tank_temperature: float = None) -> Optional[ConcreteInFlow]:
        """Update hydrogen properties for an InFlow based on current tank pressure.
        This uses our simplified cryopump model which calculates pump outlet conditions.

        Args:
            flow: The flow to check
            tank_pressure: The current tank pressure (Pa)
            tank_temperature: The current tank temperature (K)

        Returns:
            The updated InFlow if flow is an InFlow, None otherwise and new pressure
        """
        if isinstance(flow, ConcreteInFlow):
            # Update hydrogen properties using our simplified cryopump model
            if tank_temperature is None:
                # Use a default temperature if none provided (this is just for backward compatibility)
                tank_temperature = 50.0  # Default K

            # Get hydrogen properties and new pressure from the pump model
            flow.hydrogen, new_pressure = compute_pump_outlet_hydrogen(tank_pressure, tank_temperature)
            return flow, new_pressure
        return None, tank_pressure

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

        # Check if this is a multi-flow scenario
        if mission_section.has_multiple_flows() and getattr(tank_state, "multi_flow", False):
            # Multi-flow case
            inflows = mission_section.get_inflows()
            outflows = mission_section.get_outflows()

            # Process flows for interpolation if needed
            processed_inflows = cls.process_flows(inflows, tank_state, section_iter, steps)
            processed_outflows = cls.process_flows(outflows, tank_state, section_iter, steps)

            tank_state.compute_state_derivatives(
                dynamic_model,
                processed_inflows,
                processed_outflows,
                heat_flux * heat_flux_factor,
                tank.compute_thermal_capacity(temperatures[0])
            )
        else:
            # Single flow case (existing logic)
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

    @classmethod
    def process_flows(cls, flows: list[FuelFlow], tank_state: TankState,
                     section_iter: int, steps: int) -> list[FuelFlow]:
        """Process flows for interpolation, similar to define_fuel_flows"""
        from src.mission.mission_sections import InFlow as ConcreteInFlow
        from src.mission.mission_sections import OutFlow as ConcreteOutFlow

        processed = []
        for flow in flows:
            # Update hydrogen properties for InFlow using our simplified cryopump model
            if hasattr(flow, 'hydrogen') and isinstance(flow, ConcreteInFlow):
                # Update hydrogen properties to match current tank pressure and get new pressure
                flow.hydrogen, new_pressure = compute_pump_outlet_hydrogen(tank_state.pressure, tank_state.temperature)
                # Update tank pressure with the pump outlet pressure
                tank_state.pressure = new_pressure

            if isinstance(flow.mass_flow, list):
                # Interpolate
                interpolated_flow = cls.interpolate_mass_flows(
                    flow.mass_flow, section_iter, steps
                )
                # Create new flow with interpolated value
                if hasattr(flow, 'hydrogen'):  # It's an InFlow
                    processed.append(ConcreteInFlow(interpolated_flow, flow.hydrogen))
                else:  # It's an OutFlow
                    processed.append(ConcreteOutFlow(interpolated_flow, flow.phase))
            else:
                processed.append(flow)
        return processed

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

        # Debug mass changes
        if abs(new_mass - tank_states.last_state.fuel_mass) > 0.1:
            print(f"WARNING: Large mass change: {new_mass - tank_states.last_state.fuel_mass:.3f}kg")
            print(f"  Current: {tank_states.last_state.fuel_mass:.3f}kg → New: {new_mass:.3f}kg")
            print(f"  Mass derivatives: {tank_states.last_state.derivatives.liquid_mass + tank_states.last_state.derivatives.gas_mass:.3f}kg/s")
            print(f"  Timestep: {timestep}s")

        return new_mass
    @classmethod
    def stopping_criterion_is_met(
        cls,
        stopping_criteria: list,
        tank_state: TankState,
        target_conditions: TargetState
    ) -> bool:
        """Check if any stopping criterion is met"""
        for criterion in stopping_criteria:
            if criterion.is_met(tank_state, target_conditions):
                return True
        return False

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

        # Initial update of refuel flow properties using our simplified cryopump model
        for flow in mission_section.fuel_flows:
            if isinstance(flow, ConcreteInFlow):
                # Update using our simplified cryopump model
                flow.hydrogen, new_pressure = compute_pump_outlet_hydrogen(tank_states.last_state.pressure, tank_states.last_state.temperature)
                # Update tank pressure with the pump outlet pressure
                tank_states.last_state.pressure = new_pressure
                print(f"Initial refuel properties: T={flow.hydrogen.temperature:.2f}K, P={flow.hydrogen.pressure/1e5:.2f}bar, tank pressure updated to {new_pressure/1e5:.2f}bar")

        for section_iter in range(steps):
            # Update hydrogen supply properties based on current tank pressure
            # This simulates refueling where supply properties change with tank pressure
            for flow in mission_section.fuel_flows:
                if isinstance(flow, ConcreteInFlow) and section_iter > 0:
                    # Update using our simplified cryopump model
                    flow.hydrogen, new_pressure = compute_pump_outlet_hydrogen(tank_states.last_state.pressure, tank_states.last_state.temperature)
                    # Update tank pressure with the pump outlet pressure
                    tank_states.last_state.pressure = new_pressure

                    # Only print every 100 steps to avoid excessive output
                    if section_iter % 100 == 0:
                        print(f"Updated refuel: T={flow.hydrogen.temperature:.2f}K, P={flow.hydrogen.pressure/1e5:.2f}bar, tank pressure set to {new_pressure/1e5:.2f}bar at step {section_iter}")

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
                    cls.compute_new_temperature(multistep_method, tank_states),
                    cls.compute_new_pressure(multistep_method, tank_states),
                    cls.compute_new_mass(multistep_method.timestep, tank_states),
                    multi_flow=getattr(tank_states.last_state, "multi_flow", False)
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
                    tank_states.last_fill,
                    multi_flow=getattr(initial, "multi_flow", False)
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
                tank_states.last_fill,
                multi_flow=getattr(initial, "multi_flow", False)
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
