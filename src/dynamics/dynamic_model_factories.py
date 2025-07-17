from src.dynamics.dynamic_models import (DynamicModel,
                                         SinglePhaseLimitLowerPressureModel,
                                         SinglePhaseModel,
                                         TwoPhaseLimitLowerPressureModel,
                                         TwoPhaseModel, SinglePhaseInOutModel, SinglePhaseLimitLowerPressureInOutModel, TwoPhaseInOutModel, TwoPhaseLimitLowerPressureInOutModel)


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
        # Debug print to see what phase is being detected
        # print(f"DEBUG: Tank state phase = {tank_state.phase}")

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
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        # Check for multi-flow first
        if getattr(tank_state, "multi_flow", False):
            # Multi-flow models have different signatures
            if (target_conditions.min_pressure is not None and
                target_conditions.min_pressure >= tank_state.pressure):
                return TwoPhaseLimitLowerPressureInOutModel
            return TwoPhaseInOutModel

        # Original single-flow logic
        if (target_conditions.min_pressure is None and
            target_conditions.max_pressure is None):
            return TwoPhaseModel
        if (target_conditions.min_pressure is not None and
            target_conditions.min_pressure >= tank_state.pressure):
            return TwoPhaseLimitLowerPressureModel
        return TwoPhaseModel


class SinglePhaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        # Check for multi-flow first
        if getattr(tank_state, "multi_flow", False):
            # Multi-flow models have different signatures
            if (target_conditions.min_pressure is not None and
                target_conditions.min_pressure >= tank_state.pressure):
                return SinglePhaseLimitLowerPressureInOutModel
            return SinglePhaseInOutModel

        # Single-flow models (backward compatible)
        if (target_conditions.min_pressure is None and
            target_conditions.max_pressure is None):
            return SinglePhaseModel
        if (target_conditions.min_pressure is not None and
            target_conditions.min_pressure >= tank_state.pressure):
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
