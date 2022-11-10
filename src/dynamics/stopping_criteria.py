from abc import abstractmethod
from typing import Protocol

EMPTY_LIMIT = 0.01  # A lower limit to define when the tank is empty


class InitialState(Protocol):
    pressure: float
    temperature: float
    fill: float


class TargetState(Protocol):
    max_pressure: float
    min_pressure: float
    min_temperature: float
    fill: float
    fuel_mass: float


class FuelTankState(Protocol):
    pressure: float
    fill: float


class StoppingCriterion(Protocol):

    @abstractmethod
    def is_met(
        self,
        fuel_tank_state: FuelTankState,
        target_state: TargetState
    ) -> bool:
        ...


class MaxPressure(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.pressure >= target_state.max_pressure


class IsEmpty(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, _: TargetState
    ) -> bool:
        return fuel_tank_state.fill <= EMPTY_LIMIT


class IsFull(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= 1


class TargetFillIsReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= target_state.fill


def main():
    pass


if __name__ == "__main__":
    main()


# End
