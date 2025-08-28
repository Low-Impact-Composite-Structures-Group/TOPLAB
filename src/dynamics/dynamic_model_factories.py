from src.dynamics.dynamic_models import (DynamicModel,
                                         SinglePhaseLimitLowerPressureModel,
                                         SinglePhaseModel,
                                         TwoPhaseLimitLowerPressureModel,
                                         TwoPhaseModel, SinglePhaseInOutModel,
                                         SinglePhaseLimitLowerPressureInOutModel,
                                         TwoPhaseInOutModel, TwoPhaseRefuelModel,TwoPhaseLimitLowerPressureInOutModel,
                                         SinglePhaseLimitUpperPressureModel,
                                         TwoPhaseLimitUpperPressureModel)


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

        # Check for forced phase transition first
        if hasattr(tank_state, '_forced_phase') and tank_state._forced_phase:
            if tank_state._forced_phase == "liquid":
                print("\n==== SELECTING SinglePhase MODEL DUE TO FORCED PHASE ====")
                print("Enforcing transition from two-phase to liquid phase!")
                print("=====================================\n")
                return SinglePhaseFactory().get_dynamic_model(tank_state, target_conditions)
            elif tank_state._forced_phase == "twophase":
                print("\n==== SELECTING TwoPhase MODEL DUE TO FORCED PHASE ====")
                print("Enforcing transition from gas to two-phase!")
                print("=====================================\n")
                return TwoPhaseFactory().get_dynamic_model(tank_state, target_conditions)

        # Normal phase detection continues as before
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
            # Check for upper pressure limit first (highest priority)
            if (target_conditions.max_pressure is not None and
                target_conditions.max_pressure <= tank_state.pressure):
                return TwoPhaseLimitUpperPressureModel

            # Check for lower pressure limit
            if (target_conditions.min_pressure is not None and
                target_conditions.min_pressure >= tank_state.pressure):
                return TwoPhaseLimitLowerPressureInOutModel

            # For refueling scenario testing, use the new TwoPhaseRefuelModel instead
            print("\n==== USING TWOPHASEREFUELMODEL ====")
            print("Using TwoPhaseRefuelModel instead of TwoPhaseInOutModel for testing")
            print("====================================\n")
            # Mark the tank_state to indicate it's using the refuel model
            setattr(tank_state, "using_refuel_model", True)
            return TwoPhaseRefuelModel

        # Original single-flow logic
        # Check upper pressure limit first
        if (target_conditions.max_pressure is not None and
            target_conditions.max_pressure <= tank_state.pressure):
            return TwoPhaseLimitUpperPressureModel

        # Check for no limits
        if (target_conditions.min_pressure is None and
            target_conditions.max_pressure is None):
            return TwoPhaseModel

        # Check lower pressure limit
        if (target_conditions.min_pressure is not None and
            target_conditions.min_pressure >= tank_state.pressure):
            return TwoPhaseLimitLowerPressureModel

        # Normal two-phase model
        return TwoPhaseModel


class SinglePhaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        # Check for multi-flow first
        if getattr(tank_state, "multi_flow", False):
            # Check for upper pressure limit first (highest priority)
            if (target_conditions.max_pressure is not None and
                target_conditions.max_pressure <= tank_state.pressure):
                return SinglePhaseLimitUpperPressureModel

            # Check for lower pressure limit
            if (target_conditions.min_pressure is not None and
                target_conditions.min_pressure >= tank_state.pressure):
                return SinglePhaseLimitLowerPressureInOutModel

            # Normal multi-flow
            return SinglePhaseInOutModel

        # Single-flow models (backward compatible)
        # Check upper pressure limit first
        if (target_conditions.max_pressure is not None and
            target_conditions.max_pressure <= tank_state.pressure):
            return SinglePhaseLimitUpperPressureModel

        # Check for no limits
        if (target_conditions.min_pressure is None and
            target_conditions.max_pressure is None):
            return SinglePhaseModel

        # Check lower pressure limit
        if (target_conditions.min_pressure is not None and
            target_conditions.min_pressure >= tank_state.pressure):
            return SinglePhaseLimitLowerPressureModel

        # Normal single-phase model
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
