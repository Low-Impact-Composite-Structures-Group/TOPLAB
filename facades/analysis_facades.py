from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Union
from itertools import cycle

from src.dynamics.dynamic_analysis import (MissionAnalysis,
                                           SwitchMissionAnalysis)
from src.dynamics.dynamic_model_factories import (DynamicModelFactory,
                                                  SwitchCaseFactory)
from src.dynamics.stopping_criteria import (EMPTY_LIMIT, LowerPressureReached,
                                            MaxPressureReached, NoFuelMass,
                                            StoppingCriterion, TankIsEmpty,
                                            TargetFillReached,
                                            TargetMassReached, TargetDensityReached)
from src.efficiencies.tank_performance import TankPerformance
from src.mission.mission import Mission
from src.mission.mission_sections import MissionSection, InFlow, OutFlow
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import Tank, TankFactory
from src.thermodynamics.external_models import ForcedConvectionModel, NaturalConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.tank_states import (InitialState, TankStates,
                                            TargetState, TankState)
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.rules.interaction_rules import (
    InteractionRule, MissionBasedFlow, TimeBasedFlow, ConditionalFlow,
    pressure_above, pressure_below, time_after, mission_based_flow
)
import numpy as np


# The lower mass limit is to be used for draining analysis og gas tanks
LOWER_MASS_LIMIT = 1 # TODO: find suitable lower limit


# Factories and constants to be used in the analysis
#TODO: expose timestep to driver programs
# to pass the configuration to the classes and functions that need it
TIMESTEP = 5
MULTISTEP_METHOD = EulerMethod(TIMESTEP)
DYNAMIC_MODEL_FACTORY = DynamicModelFactory()
INTERNAL_MODEL = SingleZoneModel()
EXTERNAL_MODEL = NaturalConvectionModel()
HEAT_FLUX_FACTOR = 1


class Material(Protocol):
    ...


class Insulation(Protocol):
    ...


@dataclass
class TankDimensions:
    radius: float
    body_length: float

@dataclass
class GenericTankDimensions(TankDimensions):
    a: float
    b: float

@dataclass
class OperatingEnvelope:
    max_pressure: float
    min_pressure: float
    min_temperature: float


@dataclass
class InitialConditions:
    pressure: float
    temperature: float
    fill: float
    multi_flow: bool = False  # Used for dual tank analysis

@dataclass
class TargetConditions:
    fuel_mass: float
    fill: float
    density: float = None


class AnalysisFacade(Protocol):

    @classmethod
    def analyse(cls) -> TankPerformance:
        ...

    @classmethod
    def _define_tank(
        cls,
        tank_dimensions: Union[TankDimensions, GenericTankDimensions],
        material: Material,
        target_state: OperatingEnvelope,
        initial_state: InitialState
    ) -> Tank:
        # TODO: this is sloppy... find a better way to do this
        if hasattr(tank_dimensions, 'a') and hasattr(tank_dimensions, 'b'):
            return TankFactory.create_tank(
                tank_dimensions.radius,
                tank_dimensions.body_length,
                material,
                cls._define_operating_pressure(target_state, initial_state),
                tank_dimensions.a,
                tank_dimensions.b
            )
        else:
            return TankFactory.create_tank(
                tank_dimensions.radius,
                tank_dimensions.body_length,
                material,
                cls._define_operating_pressure(target_state, initial_state)
            )

    @staticmethod
    def _define_operating_pressure(
        target_state: TargetState,
        initial_state: InitialState
    ) -> float:
        if target_state.max_pressure is not None:
            return target_state.max_pressure
        return initial_state.pressure

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope
    ) -> TargetState:
        fill = None
        target_conditions = TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            fill,
            LOWER_MASS_LIMIT
        )
        return target_conditions

    @staticmethod
    def _define_thermal_model(
        insulation: Insulation,
        constant_heat_flux: float = None  # Optional constant heat flux
    ) -> ThermodynamicModel:
        return ThermodynamicModel(
            INTERNAL_MODEL, EXTERNAL_MODEL, insulation, constant_heat_flux=constant_heat_flux
        )

    @staticmethod
    def _define_initial_state(
        initial_conditions: InitialConditions,
        tank_volume: float = None  # Optional parameter for tank volume
    ) -> InitialState:

        state = InitialState(
            initial_conditions.pressure,
            initial_conditions.temperature,
            initial_conditions.fill,
            multi_flow=initial_conditions.multi_flow
        )

        return state


class DrainingAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        fuel_mass_flow: float,
        fuel_flow_state: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope
    ) -> TankPerformance:
        print(tank_dimensions, datetime.now().strftime("%H:%M:%S"))
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            cls._define_mission(fuel_mass_flow, fuel_flow_state),
            cls._define_stopping_criteria(),
            cls._define_target_conditions(operating_envelope),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @staticmethod
    def _define_mission(
        fuel_mass_flow: float,
        fuel_flow_state: float
    ) -> Mission:
        mission_sections = [
            MissionSection.draining(
                fuel_mass_flow, fuel_flow_state
            )
        ]
        return Mission(mission_sections)

    @staticmethod
    def _define_stopping_criteria() -> list[StoppingCriterion]:
        return [NoFuelMass(), TankIsEmpty()]


@dataclass
class ParallelDrainingAnalysis(AnalysisFacade):
    radius: float
    body_length: float
    material: Material
    insulation: Insulation
    fuel_mass_flow: float
    fuel_phase_flow: str
    initial_state: InitialConditions
    operating_envelope: OperatingEnvelope

    def __post_init__(self):
        self.tank = self._define_tank(
            TankDimensions(self.radius, self.body_length),
            self.material,
            self.operating_envelope,
            self.initial_state
        )


    def analyse(self):
        tank_states = MissionAnalysis.perform_analysis(
            self.tank,
            self.initial_state,
            self._define_mission(
                self.fuel_mass_flow, self.fuel_phase_flow
            ),
            self._define_stopping_criteria(),
            self._define_target_conditions(self.operating_envelope),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            self._define_thermal_model(self.insulation),
            HEAT_FLUX_FACTOR
        )

        return tank_states.max_pressure

    @staticmethod
    def _define_mission(
        fuel_mass_flow: float,
        fuel_flow_state: float
    ) -> Mission:
        mission_sections = [
            MissionSection.draining(
                fuel_mass_flow, fuel_flow_state
            )
        ]
        return Mission(mission_sections)

    @staticmethod
    def _define_stopping_criteria() -> list[StoppingCriterion]:
        return [NoFuelMass(), TankIsEmpty()]


class MissionAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        constant_heat_flux: float = None
    ) -> TankPerformance:
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            cls._define_target_conditions(operating_envelope),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation, constant_heat_flux),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return list()


class FillingAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions,
    ) -> TankPerformance:
        MULTISTEP_METHOD.timestep = 10
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
        tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            cls._define_target_conditions(
                operating_envelope, target_conditions
            ),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return [TargetFillReached(), TargetMassReached()]

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TargetState:
        target_conditions = TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            target_conditions.fill,
            target_conditions.fuel_mass
        )
        return target_conditions

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        constant_heat_flux: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions,
        timestep: float = None,
    ) -> TankPerformance:
        # Set timestep if provided
        if timestep is not None:
            MULTISTEP_METHOD.timestep = timestep
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")

        # Force multi_flow to True for refuelling analysis
        initial_conditions.multi_flow = True

        # Create tank with correct volume
        temp_initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(tank_dimensions, material, operating_envelope, temp_initial_state)
        print(f"Tank volume: {tank.volume} m³")

        # Create proper initial state with volume information and multi_flow=True
        initial_state = cls._define_initial_state(initial_conditions, tank.volume)
        initial_state.multi_flow = True

        # Create thermal model
        thermal_model = cls._define_thermal_model(insulation, constant_heat_flux)

        # Define target conditions
        target_state = cls._define_target_conditions(operating_envelope, target_conditions)

        # Use MissionAnalysis to perform the simulation
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            target_state,
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            thermal_model,
            HEAT_FLUX_FACTOR
        )

        print(f"Current state: m={tank_states.last_state.fuel_mass:.1f}kg, P={tank_states.last_state.pressure/1e5:.1f}bar, T = {tank_states.last_state.temperature:.1f}K")
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        criteria = [TargetFillReached(), TargetMassReached(), MaxPressureReached()]
        print(f"Using stopping criteria: {[c.__class__.__name__ for c in criteria]}")
        return criteria

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TargetState:
        target_state = TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            target_conditions.fill,
            target_conditions.fuel_mass
        )
        return target_state
class SwitchPhaseDrainingAnalysis(DrainingAnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        fuel_mass_flow: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope
    ) -> TankPerformance:

        # Define the initial state and the fuel tank
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )

        # Set up iterations
        tank_states = TankStates(list(), MULTISTEP_METHOD.timestep)
        max_changes = 100
        for _ in range(max_changes):

            # Compute new tank states
            tank_states += SwitchMissionAnalysis.perform_analysis(
                tank,
                initial_state,
                cls._define_mission(
                    fuel_mass_flow,
                    cls._define_flow_state(
                        operating_envelope, tank_states
                    )
                ),
                cls._define_stopping_criteria(),
                cls._define_target_conditions(operating_envelope),
                MULTISTEP_METHOD,
                SwitchCaseFactory(),
                cls._define_thermal_model(insulation),
                HEAT_FLUX_FACTOR
            )

            # Verify if tank has been drained
            if cls._tank_is_drained(tank_states):
                return TankPerformance(tank, insulation, tank_states)

            # Update the initial state for the new iteration
            initial_state = cls._define_initial_state(
                tank_states.last_state
            )

        raise ValueError(
            "Exceeded maximum iterations is switch drain analysis..."
        )

    @classmethod
    def _tank_is_drained(
        cls, tank_states: TankStates
    ) -> bool:
        return (
            tank_states.last_state.fuel_mass <= LOWER_MASS_LIMIT
            or tank_states.last_fill < EMPTY_LIMIT
        )

    @classmethod
    def _define_flow_state(
        cls,
        operating_envelope: OperatingEnvelope,
        tank_states: TankStates
    ) -> str:
        if len(tank_states.states) == 0:
            return "liquid"
        if tank_states.last_pressure < operating_envelope.min_pressure:
            return "liquid"
        elif tank_states.last_pressure > operating_envelope.max_pressure:
            return "gas"
        ValueError("Tank state out of bound for operating envelope...")

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return [
            NoFuelMass(),
            TankIsEmpty(),
            MaxPressureReached(),
            LowerPressureReached()
        ]

class DensityAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TankPerformance:
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            cls._define_target_conditions(operating_envelope, target_conditions),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return [TargetDensityReached()]

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TargetState:
        return TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            target_conditions.fill,
            target_conditions.fuel_mass,
            target_conditions.density
        )



class InOutTankAnalysisFacade(AnalysisFacade):
    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        constant_heat_flux: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions,
    ) -> TankPerformance:
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")
        # Ensure multi_flow is True for this analysis
        initial_state = InitialState(
            initial_conditions.pressure,
            initial_conditions.temperature,
            initial_conditions.fill,
            multi_flow=True
        )
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )

        tank_states = TankStates([], MULTISTEP_METHOD.timestep)

        # Iterate through all mission sections
        for section in mission.sections:
            # 1. Solve tank for this section with both inflow and outflow
            tank_states_section = MissionAnalysis.perform_analysis(
                tank,
                initial_state,
                Mission([section]),
                cls._define_stopping_criteria(),
                cls._define_target_conditions(operating_envelope),
                MULTISTEP_METHOD,
                DYNAMIC_MODEL_FACTORY,
                cls._define_thermal_model(insulation, constant_heat_flux),
                HEAT_FLUX_FACTOR
            )

            tank_states += tank_states_section
            # Propagate multi_flow=True
            initial_state = InitialState(
                tank_states.last_pressure,
                tank_states.last_temperature,
                tank_states.last_fill,
                multi_flow=True
            )

        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return list()


class MultiTankAnalysisFacade(AnalysisFacade):
    @classmethod
    def analyse(
        cls,
        tank_configurations: list,
        mission: Mission,
        interaction_rules: dict,
        initial_conditions: list,
        operating_envelopes: list,
        target_conditions: list,
    ) -> list[TankPerformance]:
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")

        # Setup tanks and initial states
        tanks = []
        tank_states = []
        for i, config in enumerate(tank_configurations):
            # Calculate tank volume
            dimensions = config["dimensions"]
            tank_volume = None
            if dimensions.body_length == 0:  # Spherical tank
                tank_volume = (4/3) * np.pi * dimensions.radius**3
            else:  # Cylindrical tank with hemispherical ends
                cylinder_volume = np.pi * dimensions.radius**2 * dimensions.body_length
                hemispheres_volume = (4/3) * np.pi * dimensions.radius**3
                tank_volume = cylinder_volume + hemispheres_volume

            # Create initial state with tank volume for mass fraction adjustment
            initial_state = cls._define_initial_state(initial_conditions[i], tank_volume)

            # Create tank
            tank = cls._define_tank(
                dimensions,
                config["material"],
                operating_envelopes[i],
                initial_state
            )
            tanks.append(tank)
            tank_states.append(TankStates([], MULTISTEP_METHOD.timestep))

        # Setup flow tracking
        inflow_1, outflow_1, inflow_2, outflow_2, time_points = [], [], [], [], []
        timestep = MULTISTEP_METHOD.timestep

        # Initialize absolute minimum mass thresholds
        MIN_ABSOLUTE_MASS = 2.0  # kg - absolute minimum to prevent extreme conditions
        initial_mass_1 = None
        initial_mass_2 = None
        min_mass_1 = MIN_ABSOLUTE_MASS
        min_mass_2 = MIN_ABSOLUTE_MASS

        # Define maximum allowable temperature
        MAX_TEMPERATURE = 800.0  # Kelvin - prevent unrealistic temperatures

        # Create the flow rule from configuration
        flow_rule = cls._create_flow_rule(interaction_rules)

        # Loop over mission sections
        time_offset = 0
        early_stop = False

        for section in mission.sections:
            steps = int(section.duration / timestep)
            for step in range(steps):
                t = time_offset + step * timestep
                time_points.append(t)

                # Interpolate mission outflow for tank 2
                mission_outflow = 0.0
                refuel_inflow = 0.0
                for flow in section.fuel_flows:
                    if isinstance(flow, OutFlow):
                        if isinstance(flow.mass_flow, list):
                            from src.dynamics.dynamic_analysis import MissionSectionAnalysis
                            flow_value = MissionSectionAnalysis.interpolate_mass_flows(
                                flow.mass_flow, step, steps
                            )
                        else:
                            flow_value = flow.mass_flow

                        # Check if this is refueling (negative outflow in a Refueling section)
                        if flow_value < 0 and hasattr(section, "fuel_flow_key") and section.fuel_flow_key == "Refuelling":
                            refuel_inflow += abs(flow_value)  # Make positive for inflow to Tank 1
                        else:
                            mission_outflow += flow_value  # Regular outflow for Tank 2

                # Evaluate the flow rule
                last_state_1 = tank_states[0].last_state if tank_states[0].states else cls._define_initial_state(initial_conditions[0])
                last_state_2 = tank_states[1].last_state if tank_states[1].states else cls._define_initial_state(initial_conditions[1])

                # Add step info to mission data for interpolation
                section.current_step = step
                section.total_steps = steps

                # Evaluate the flow rule
                transfer_flow = flow_rule.evaluate(
                    [last_state_1, last_state_2],
                    t,
                    section
                )

                # Net flows for each tank
                # Tank 1: only outflow (should be negative)
                tank1_net_flow = refuel_inflow -transfer_flow
                # Tank 2: inflow (positive) + mission outflow (negative)
                tank2_net_flow = transfer_flow + mission_outflow

                # Get last states
                last_state_1 = tank_states[0].last_state if tank_states[0].states else cls._define_initial_state(initial_conditions[0])
                last_state_2 = tank_states[1].last_state if tank_states[1].states else cls._define_initial_state(initial_conditions[1])

                # Convert InitialState to TankState because we need to access certain attributes
                if isinstance(last_state_1, InitialState):
                    # Check if we have an overridden mass
                    if hasattr(last_state_1, 'override_mass'):
                        # Use the overridden mass value
                        override_mass = last_state_1.override_mass
                    else:
                        # Calculate normally
                        override_mass = last_state_1.compute_fuel_mass(tanks[0].volume)

                    last_state_1 = TankState(
                        tanks[0],
                        last_state_1.temperature,
                        last_state_1.pressure,
                        override_mass,  # Use the override mass here
                        multi_flow=last_state_1.multi_flow
                    )

                if isinstance(last_state_2, InitialState):
                    # Check if we have an overridden mass
                    if hasattr(last_state_2, 'override_mass'):
                        # Use the overridden mass value
                        override_mass = last_state_2.override_mass
                    else:
                        # Calculate normally
                        override_mass = last_state_2.compute_fuel_mass(tanks[1].volume)

                    last_state_2 = TankState(
                        tanks[1],
                        last_state_2.temperature,
                        last_state_2.pressure,
                        override_mass,  # Use the override mass here
                        multi_flow=last_state_2.multi_flow
                    )

                # Get current masses
                current_mass_1 = last_state_1.fuel_mass if hasattr(last_state_1, "fuel_mass") else last_state_1.compute_fuel_mass(tanks[0].volume)
                current_mass_2 = last_state_2.fuel_mass if hasattr(last_state_2, "fuel_mass") else last_state_2.compute_fuel_mass(tanks[1].volume)
                # current_mass_2 = 220
                # Update initial masses if first step
                if step == 0 and initial_mass_1 is None:
                    initial_mass_1 = current_mass_1
                    initial_mass_2 = current_mass_2
                    min_mass_1 = max(MIN_ABSOLUTE_MASS, initial_mass_1 * 0.01)  # 1% of initial
                    min_mass_2 = max(MIN_ABSOLUTE_MASS, initial_mass_2 * 0.01)  # 1% of initial
                    print(f"Initial masses: Tank 1 = {initial_mass_1:.2f}kg, Tank 2 = {initial_mass_2:.2f}kg")

                    # Add this new code block to handle Tank 1's mass_fraction
                    if hasattr(initial_conditions[0], 'mass_fraction') and initial_conditions[0].mass_fraction == 0.0:
                        initial_mass_1 = 0.0
                        current_mass_1 = 0.0  # Also update current_mass_1
                        print(f"Forced Tank 1 initial mass to 0.0 kg (mass_fraction={initial_conditions[0].mass_fraction})")

                    # Existing code for Tank 2
                    if hasattr(initial_conditions[1], 'mass_fraction') and initial_conditions[1].mass_fraction < 1.0:
                        initial_mass_2 = initial_mass_2 * initial_conditions[1].mass_fraction
                        current_mass_2 = initial_mass_2  # Also update current_mass_2
                        print(f"Adjusted Tank 2 initial mass to {initial_mass_2:.2f} kg (mass_fraction={initial_conditions[1].mass_fraction})")

                #TODO: make this feature optional. It's a good feature for analysis, but not for development
                # # Early stop if either tank is approaching the minimum mass
                # if current_mass_1 < min_mass_1 + 0.5:  # Add 0.5kg buffer
                #     print(f"\nTank 1 approaching minimum mass ({current_mass_1:.2f}kg < {min_mass_1+0.5:.2f}kg).")
                #     print(f"Limiting outflow to prevent extreme conditions.")
                #     tank1_net_flow = 0.0
                #     transfer_flow = 0.0

                # if current_mass_2 < min_mass_2 + 0.5:  # Add 0.5kg buffer
                #     print(f"\nTank 2 approaching minimum mass ({current_mass_2:.2f}kg < {min_mass_2+0.5:.2f}kg).")
                #     print(f"Limiting outflow to prevent extreme conditions.")
                #     mission_outflow = 0.0
                #     tank2_net_flow = transfer_flow  # Only inflow, no outflow

                # 12. Build the thermal and dynamic models for each tank
                thermal_model_1 = ThermodynamicModel(INTERNAL_MODEL, EXTERNAL_MODEL, tank_configurations[0]["insulation"], constant_heat_flux=tank_configurations[0].get("heat_flux"))
                dynamic_model_1 = DYNAMIC_MODEL_FACTORY.get_dynamic_model(last_state_1, operating_envelopes[0])
                thermal_model_2 = ThermodynamicModel(INTERNAL_MODEL, EXTERNAL_MODEL, tank_configurations[1]["insulation"], constant_heat_flux=tank_configurations[1].get("heat_flux"))
                dynamic_model_2 = DYNAMIC_MODEL_FACTORY.get_dynamic_model(last_state_2, operating_envelopes[1])

                # Build flow lists for each tank
                # Tank 1: only outflow (always negative)
                fuel_flows_in_1 = []
                fuel_flows_out_1 = [OutFlow(abs(tank1_net_flow), "gas")] if tank1_net_flow < 0 else []

                # Tank 2: inflow (positive) and outflow (mission) -> net should be negative as well
                fuel_flows_in_2 = [InFlow(transfer_flow, last_state_2.hydrogen)] if transfer_flow > 0 else []
                fuel_flows_out_2 = [OutFlow(abs(mission_outflow), "gas")] if mission_outflow < 0 else []

                # Compute heat ingress and tank thermal capacity
                heat_flux_1 = tank_configurations[0].get("heat_flux")
                heat_flux_2 = tank_configurations[1].get("heat_flux")
                tank_thermal_capacity_1 = tanks[0].compute_thermal_capacity(last_state_1.temperature)
                tank_thermal_capacity_2 = tanks[1].compute_thermal_capacity(last_state_2.temperature)

                # Compute state derivatives: we need these to compute the next state values
                derivs_1 = last_state_1.compute_state_derivatives(
                    dynamic_model_1,
                    fuel_flows_in_1,
                    fuel_flows_out_1,
                    heat_flux_1,
                    tank_thermal_capacity_1
                )
                dTdt_1 = derivs_1.temperature
                dPdt_1 = derivs_1.pressure

                derivs_2 = last_state_2.compute_state_derivatives(
                    dynamic_model_2,
                    fuel_flows_in_2,
                    fuel_flows_out_2,
                    heat_flux_2,
                    tank_thermal_capacity_2
                )
                dTdt_2 = derivs_2.temperature
                dPdt_2 = derivs_2.pressure

                # Update states with timestep (essentially a forward Euler)
                new_temp_1 = last_state_1.temperature + dTdt_1 * timestep
                new_pressure_1 = last_state_1.pressure + dPdt_1 * timestep
                new_mass_1 = current_mass_1 + tank1_net_flow * timestep

                new_temp_2 = last_state_2.temperature + dTdt_2 * timestep
                new_pressure_2 = last_state_2.pressure + dPdt_2 * timestep
                new_mass_2 = current_mass_2 + tank2_net_flow * timestep


                # Create new TankState objects
                new_state_1 = TankState(
                    tanks[0],
                    new_temp_1,
                    new_pressure_1,
                    new_mass_1,
                    multi_flow=True
                )

                new_state_2 = TankState(
                    tanks[1],
                    new_temp_2,
                    new_pressure_2,
                    new_mass_2,
                    multi_flow=True
                )

                # Check stopping criteria
                target_state_1 = TargetState(
                    operating_envelopes[0].max_pressure,
                    operating_envelopes[0].min_pressure,
                    operating_envelopes[0].min_temperature,
                    0.0, # fill
                    min_mass_1,  # Stop when mass drops below this
                    None  # density
                )

                target_state_2 = TargetState(
                    operating_envelopes[1].max_pressure,
                    operating_envelopes[1].min_pressure,
                    operating_envelopes[1].min_temperature,
                    0.0,
                    min_mass_2,
                    None
                )

                # Check for stopping conditions
                if new_mass_1 <= min_mass_1 and refuel_inflow <= 0:  # Skip this check when refueling
                    print(f"\nStopping simulation: Tank 1 reached minimum mass threshold of {min_mass_1:.2f} kg")
                    # Add states to history
                    tank_states[0].add_tank_state(new_state_1)
                    tank_states[1].add_tank_state(new_state_2)
                    early_stop = True
                    break

                if new_mass_2 <= min_mass_2 and refuel_inflow <= 0:
                    print(f"\nStopping simulation: Tank 2 reached minimum mass threshold of {min_mass_2:.2f} kg")
                    # Add states to history
                    tank_states[0].add_tank_state(new_state_1)
                    tank_states[1].add_tank_state(new_state_2)
                    early_stop = True
                    break

                if new_pressure_1 >= operating_envelopes[0].max_pressure:
                    print(f"\nStopping simulation: Tank 1 reached maximum pressure of {new_pressure_1/1e5:.2f} bar")
                    # Add states to history
                    tank_states[0].add_tank_state(new_state_1)
                    tank_states[1].add_tank_state(new_state_2)
                    early_stop = True
                    break

                if new_pressure_2 >= operating_envelopes[1].max_pressure:
                    print(f"\nStopping simulation: Tank 2 reached maximum pressure of {new_pressure_2/1e5:.2f} bar")
                    # Add states to history
                    tank_states[0].add_tank_state(new_state_1)
                    tank_states[1].add_tank_state(new_state_2)
                    early_stop = True
                    break

                # Add states to history
                tank_states[0].add_tank_state(new_state_1)
                tank_states[1].add_tank_state(new_state_2)

                # Break out if stopping condition met
                if early_stop:
                    break

                # Print iteration details (can be commented out later)
                if step % 20 == 0:  # Print every 20 steps
                    print(f"Time: {t:.1f}s | Tank 1: m={new_mass_1:.1f}kg, P={new_pressure_1/1e5:.1f}bar, T = {new_temp_1:.1f}K  | "
                        f"Tank 2: m={new_mass_2:.1f}kg, P={new_pressure_2/1e5:.1f}bar, T = {new_temp_2:.1f}K | "
                        f"Flow T1→T2: {transfer_flow:.3f}kg/s, Mission: {mission_outflow:.3f}kg/s")

                # Track flows for plotting
                inflow_1.append(refuel_inflow)
                outflow_1.append(-transfer_flow) # manually making this negative for plotting purposes
                inflow_2.append(transfer_flow)
                outflow_2.append(mission_outflow)

            time_offset += section.duration

        # Package results
        performances = []
        for i, tank in enumerate(tanks):
            performances.append(TankPerformance(
                tank,
                tank_configurations[i]["insulation"],
                tank_states[i]
            ))

        # Ensure all arrays have the same length
        min_length = min(len(time_points), len(inflow_1), len(outflow_1), len(inflow_2), len(outflow_2))
        if min_length < len(time_points):
            print(f"Warning: Truncating arrays from {len(time_points)} to {min_length} points for plotting")
            time_points = time_points[:min_length]
            inflow_1 = inflow_1[:min_length]
            outflow_1 = outflow_1[:min_length]
            inflow_2 = inflow_2[:min_length]
            outflow_2 = outflow_2[:min_length]

        # Store flows for plotting
        cls.flow_data = {
            'time': time_points,
            'tank1_inflow': inflow_1,
            'tank1_outflow': outflow_1,
            'tank2_inflow': inflow_2,
            'tank2_outflow': outflow_2
        }

        return performances

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        """Define stopping criteria for multi-tank analysis."""
        return [NoFuelMass(), MaxPressureReached()]

    @classmethod
    def _create_flow_rule(cls, rule_config: dict) -> InteractionRule:
        """Create an InteractionRule from configuration dictionary"""
        rule_type = rule_config.get("type", "mission_based")
        max_flow_rate = rule_config.get("max_flow_rate")

        if rule_type == "mission_based":
            return MissionBasedFlow(
                safety_factor=rule_config.get("safety_factor", 0.8),
                max_flow_rate=max_flow_rate,
                active=rule_config.get("active_at_start", True)
            )

        elif rule_type == "time_based":
            return TimeBasedFlow(
                rule_config["time_flow_pairs"],
                max_flow_rate=max_flow_rate,
                active=rule_config.get("active_at_start", True)
            )

        elif rule_type == "conditional":
            rule = ConditionalFlow(
                max_flow_rate=max_flow_rate,
                active=rule_config.get("active_at_start", True)
            )

            # Add conditions from config
            for condition in rule_config.get("conditions", []):
                cond_type = condition.get("type")

                if cond_type == "pressure_above":
                    cond_func = pressure_above(
                        condition.get("tank_idx", 1),
                        condition.get("threshold")
                    )
                elif cond_type == "pressure_below":
                    cond_func = pressure_below(
                        condition.get("tank_idx", 1),
                        condition.get("threshold")
                    )
                elif cond_type == "time_after":
                    cond_func = time_after(condition.get("threshold"))
                else:
                    raise ValueError(f"Unknown condition type: {cond_type}")

                # Handle flow value or calculator
                flow_value = condition.get("flow_value")
                if condition.get("use_mission_flow", False):
                    flow_calc = mission_based_flow(
                        condition.get("safety_factor", 0.8)
                    )
                    rule.add_condition(cond_func, flow_calc)
                else:
                    rule.add_condition(cond_func, flow_value)

            # Set default flow
            if "default_flow" in rule_config:
                rule.set_default_flow(rule_config["default_flow"])

            return rule

        else:
            raise ValueError(f"Unsupported rule type: {rule_type}")


def main():
    pass


if __name__ == "__main__":
    main()


# End
