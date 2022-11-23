
from dataclasses import dataclass
from typing import Protocol

from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import (NoFuelMass, StoppingCriterion,
                                            TankIsEmpty)
from src.efficiencies.tank_performance import TankPerformance
from src.mission.mission import Mission
from src.mission.mission_sections import MissionSection
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import Tank, TankFactory
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.tank_states import InitialState, TargetState
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


# The lower mass limit is to be used for draining analysis og gas tanks
LOWER_MASS_LIMIT = 500


# Factories and constants to be used in the analysis
TIMESTEP = 60
MULTISTEP_METHOD = EulerMethod(TIMESTEP)
DYNAMIC_MODEL_FACTORY = DynamicModelFactory()
INTERNAL_MODEL = SingleZoneModel()
EXTERNAL_MODEL = ForcedConvectionModel()
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
class OperatingEnvelope:
    max_pressure: float
    min_pressure: float
    min_temperature: float


class AnalysisFacade(Protocol):

    @classmethod
    def analyse(cls) -> TankPerformance:
        ...


class DrainingAnalysisFacade(AnalysisFacade):
    
    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        fuel_mass_flow: float,
        fuel_flow_state: float,
        initial_state: InitialState,
        operating_envelope: OperatingEnvelope
    ) -> TankPerformance:
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
    
    @classmethod
    def _define_tank(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        target_state: OperatingEnvelope,
        initial_state: InitialState
    ) -> Tank:
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
    def _define_stopping_criteria() -> list[StoppingCriterion]:
        return [NoFuelMass(), TankIsEmpty()]

    @staticmethod
    def _define_thermal_model(
        insulation: Insulation
    ) -> ThermodynamicModel:
        return ThermodynamicModel(
            INTERNAL_MODEL, EXTERNAL_MODEL, insulation
        )


def main():
    pass


if __name__ == "__main__":
    main()
        

# End
