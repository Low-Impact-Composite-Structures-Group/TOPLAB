
from src.dynamics.dynamic_models.protocols import DynamicModel
from src.dynamics.dynamic_models.ahluwalia import (SinglePhaseLimitLowerPressureModel,
                                         SinglePhaseModel,
                                         TwoPhaseLimitLowerPressureModel,
                                         TwoPhaseModel)


class TankState:
    pressure: float
    phase: str


class OperatingEnvelope:
    max_pressure: float
    min_pressure: float


class DynamicModelFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        if tank_state.phase == "twophase":
            return TwoPhaseFactory().get_dynamic_model(
                tank_state, target_conditions
            )
        return SinglePhaseFactory().get_dynamic_model(
            tank_state, target_conditions
        )


class TwoPhaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        operating_envelope: OperatingEnvelope
    ) -> DynamicModel:
        if (
            operating_envelope.min_pressure is None
            and operating_envelope.max_pressure is None
        ):
            return TwoPhaseModel
        if operating_envelope.min_pressure is not None:
            if tank_state.pressure <= operating_envelope.min_pressure:
                return TwoPhaseLimitLowerPressureModel
        return TwoPhaseModel


class SinglePhaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        if (
            target_conditions.min_pressure is None
            and target_conditions.max_pressure is None
        ):
            return SinglePhaseModel
        if (
            target_conditions.min_pressure is not None
            and target_conditions.min_pressure >= tank_state.pressure
        ):
            return SinglePhaseLimitLowerPressureModel
        return SinglePhaseModel


class SwitchCaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        return TwoPhaseModel


def main():
    pass


if __name__ == "__main__":
    main()


# End
