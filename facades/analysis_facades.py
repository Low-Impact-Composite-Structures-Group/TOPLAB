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
                                            TargetState)
from src.thermodynamics.thermodynamic_models import ThermodynamicModel

# The lower mass limit is to be used for draining analysis og gas tanks
LOWER_MASS_LIMIT = 1 # TODO: find suitable lower limit


# Factories and constants to be used in the analysis
#TODO: refactor to use yaml input for these constants using dependency injection
# to pass the configuration to the classes and functions that need it
TIMESTEP = 10
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
        initial_conditions: InitialConditions
    ) -> InitialState:
        return InitialState(
            initial_conditions.pressure,
            initial_conditions.temperature,
            initial_conditions.fill,
            multi_flow=initial_conditions.multi_flow
        )


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


def main():
    pass


if __name__ == "__main__":
    main()


# End
