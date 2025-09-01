from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Union, List, Optional

from src.dynamics.stopping_criteria import NoFuelMass, TankIsEmpty
from src.dynamics.cryopump_model import CryoPumpModel
from src.fluids.hydrogen_retrievers import SinglePhaseRequester, TwoPhaseRequester, HydrogenRetriever
from src.mission.mission import Mission, MissionSection, InFlow

# Use the compute_pump_outlet methods from CryoPumpModel directly
# No need to assign to a variable anymore
from src.mission.mission_sections import InFlow as ConcreteInFlow
from src.thermodynamics.tank_states import (InitialState, TankState,
                                            TankStates, TargetState)

# The thermal capacity of the tank depends on the mass of the tank, as
# Note: With NIST temperature-dependent materials, thermal capacity iteration
# is no longer needed - materials provide direct temperature-dependent specific heat
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
    inlet_enthalpy: float = None  # Optional raw enthalpy value for inflows


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
                # Get enthalpy from our simplified cryopump model
                enthalpy = MissionSectionAnalysis.calculate_inflow_enthalpy(
                    tank_state.pressure, tank_state.temperature
                )

                print(f"Using raw enthalpy value: {enthalpy:.0f} J/kg for inflow")

                # For InFlow, we pass both the hydrogen object (for compatibility)
                # and the raw enthalpy value
                flows.append(
                    FuelFlow(
                        fuel_flow.mass_flow,
                        fuel_flow.hydrogen,
                        inlet_enthalpy=enthalpy
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
    def calculate_inflow_enthalpy(tank_pressure: float, tank_temperature: float = None) -> float:
        """Calculate the pump outlet enthalpy for an inflow based on current tank pressure.
        This uses our simplified cryopump model to get the raw enthalpy value.

        Args:
            tank_pressure: The current tank pressure (Pa)
            tank_temperature: The current tank temperature (K)

        Returns:
            The calculated enthalpy value (J/kg)
        """
        # Get enthalpy directly from our cryopump model
        enthalpy = CryoPumpModel.compute_pump_outlet_hydrogen(tank_pressure, tank_temperature)
        return enthalpy

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

            # Process flows for interpolation if needed and add raw enthalpy values
            processed_inflows = cls.process_flows(inflows, tank_state, section_iter, steps)
            processed_outflows = cls.process_flows(outflows, tank_state, section_iter, steps)

            # For debug purposes, check if any inflows have raw enthalpy values
            for flow in processed_inflows:
                if hasattr(flow, 'inlet_enthalpy') and flow.inlet_enthalpy is not None:
                    print(f"Raw enthalpy for inflow: {flow.inlet_enthalpy:.0f} J/kg")

            # Add a flag to the tank state to indicate that inlet enthalpy is available
            setattr(tank_state, "use_raw_enthalpy", True)

            tank_state.compute_state_derivatives(
                dynamic_model,
                processed_inflows,
                processed_outflows,
                heat_flux * heat_flux_factor,
                tank.compute_thermal_capacity(temperatures[0])
            )
        else:
            # Single flow case (existing logic)
            flows = cls.define_fuel_flows(
                mission_section.fuel_flows,
                tank_state,
                section_iter,
                steps
            )

            # For debug purposes, check if any flows have raw enthalpy values
            for flow in flows:
                if hasattr(flow, 'inlet_enthalpy') and flow.inlet_enthalpy is not None:
                    print(f"Raw enthalpy for flow: {flow.inlet_enthalpy:.0f} J/kg")

            # Add a flag to the tank state to indicate that inlet enthalpy is available
            setattr(tank_state, "use_raw_enthalpy", True)

            tank_state.compute_state_derivatives(
                dynamic_model,
                flows,
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
            # Calculate raw enthalpy for inflows
            if hasattr(flow, 'hydrogen') and isinstance(flow, ConcreteInFlow):
                # Get enthalpy directly from our cryopump model
                enthalpy = cls.calculate_inflow_enthalpy(tank_state.pressure, tank_state.temperature)

                # We'll track this as a separate property instead of modifying the hydrogen object
                flow_enthalpy = enthalpy
            else:
                flow_enthalpy = None

            if isinstance(flow.mass_flow, list):
                # Interpolate
                interpolated_flow = cls.interpolate_mass_flows(
                    flow.mass_flow, section_iter, steps
                )
                # Create new flow with interpolated value
                if hasattr(flow, 'hydrogen'):  # It's an InFlow
                    # Create a FuelFlow with the raw enthalpy value
                    processed.append(
                        FuelFlow(
                            interpolated_flow,
                            flow.hydrogen,
                            inlet_enthalpy=flow_enthalpy
                        )
                    )
                else:  # It's an OutFlow
                    processed.append(
                        FuelFlow(
                            interpolated_flow,
                            tank_state.hydrogen.get_phase(flow.phase)
                        )
                    )
            else:
                if isinstance(flow, ConcreteInFlow):
                    # Update with the raw enthalpy
                    processed.append(
                        FuelFlow(
                            flow.mass_flow,
                            flow.hydrogen,
                            inlet_enthalpy=flow_enthalpy
                        )
                    )
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

        # Initial calculation of refuel enthalpy using our simplified cryopump model
        enthalpy = cls.calculate_inflow_enthalpy(tank_states.last_state.pressure, tank_states.last_state.temperature)
        print(f"Initial refuel enthalpy calculated: {enthalpy:.0f} J/kg at tank pressure {tank_states.last_state.pressure/1e5:.2f}bar")

        for section_iter in range(steps):
            # We don't need to update flow objects here anymore - we'll calculate the enthalpy
            # when we process the flows in compute_state_derivatives

            # Check for phase transitions every time, not just in refueling scenario
            current_time = section_iter * multistep_method.timestep
            # Call the phase transition check method if it exists
            if hasattr(tank_states.last_state, 'check_phase_transition'):
                # print(f"DEBUG: Calling check_phase_transition at t={current_time:.1f}s, P={tank_states.last_state.pressure/1e5:.1f}bar, T={tank_states.last_state.temperature:.1f}K")
                tank_states.last_state.check_phase_transition(current_time)
            else:
                print(f"DEBUG: tank_states.last_state does not have check_phase_transition method, type: {type(tank_states.last_state)}")

            # After phase transition check, update hydrogen object if needed
            if hasattr(tank_states.last_state, '_forced_phase') and tank_states.last_state._forced_phase:
                if tank_states.last_state._forced_phase == "liquid" and getattr(tank_states.last_state, '_in_transition', False):
                    # We just detected a transition to liquid, update the hydrogen object
                    try:
                        from src.fluids.hydrogen_retrievers import SinglePhaseRequester
                        new_hydrogen = SinglePhaseRequester().get_hydrogen_properties(
                            tank_states.last_state.pressure, tank_states.last_state.temperature
                        )
                        tank_states.last_state.hydrogen = new_hydrogen
                        tank_states.last_state._in_transition = False
                        print(f"Updated hydrogen object to SinglePhase for liquid phase transition")
                    except Exception as e:
                        print(f"Could not update hydrogen object during transition: {e}")

            # Print enthalpy updates occasionally for debugging
            if section_iter > 0 and section_iter % 100 == 0:
                enthalpy = cls.calculate_inflow_enthalpy(tank_states.last_state.pressure, tank_states.last_state.temperature)
                print(f"Updated refuel enthalpy: {enthalpy:.0f} J/kg at tank pressure {tank_states.last_state.pressure/1e5:.2f}bar at step {section_iter}")

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

            # Create the new tank state
            new_state = TankState(
                tank,
                cls.compute_new_temperature(multistep_method, tank_states),
                cls.compute_new_pressure(multistep_method, tank_states),
                cls.compute_new_mass(multistep_method.timestep, tank_states),
                multi_flow=getattr(tank_states.last_state, "multi_flow", False)
            )

            # Copy any phase transition information from previous state BEFORE __post_init__ can overwrite
            if hasattr(tank_states.last_state, '_forced_phase'):
                new_state._forced_phase = tank_states.last_state._forced_phase
            if hasattr(tank_states.last_state, '_in_transition'):
                new_state._in_transition = tank_states.last_state._in_transition

            # If we have a forced phase, we need to preserve the hydrogen object from the previous state
            # or immediately recreate it with the correct phase
            if hasattr(tank_states.last_state, '_forced_phase') and tank_states.last_state._forced_phase:
                if tank_states.last_state._forced_phase == "liquid" and not getattr(tank_states.last_state, '_in_transition', True):
                    # Previous state had successfully completed transition to liquid
                    if hasattr(tank_states.last_state.hydrogen, 'dRho_dP'):  # It's already a single-phase hydrogen
                        new_state.hydrogen = tank_states.last_state.hydrogen
                        print(f"Preserved liquid hydrogen object from previous state")
                    else:
                        # Need to create a new liquid hydrogen object
                        try:
                            from src.fluids.hydrogen_retrievers import SinglePhaseRequester
                            new_hydrogen = SinglePhaseRequester().get_hydrogen_properties(
                                new_state.pressure, new_state.temperature
                            )
                            new_state.hydrogen = new_hydrogen
                            new_state._in_transition = False
                            print(f"Created new liquid hydrogen object for new state")
                        except Exception as e:
                            print(f"Could not create liquid hydrogen for new state: {e}")

            # Add the new state to our collection
            tank_states.add_tank_state(new_state)
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

        # With NIST materials, thermal capacity is directly temperature-dependent
        # No iteration needed - just run the analysis once
        print("Using NIST materials - no thermal capacity iteration required")

        # Define initial state of the tank
        initial = initial_state
        tank_states = TankStates(list(), multistep_method.timestep)

        for mission_section in mission.sections:
            section_string = mission_section.fuel_flow_key  # Access the key associated with the fuel flow
            if section_string == None:
                print(f"Now calculating singular mission section")
            else:
                print(f"Now calculating mission section {section_string}")
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

        # Set the final operating pressure for thermal capacity calculation
        tank.set_operating_pressure(tank_states.max_pressure)

        return tank_states


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
