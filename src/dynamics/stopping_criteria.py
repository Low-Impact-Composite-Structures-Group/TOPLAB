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
    mass: float


class FuelTankState(Protocol):
    pressure: float
    fill: float
    fuel_mass: float


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


class TankIsEmpty(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, _: TargetState
    ) -> bool:
        return fuel_tank_state.fill <= EMPTY_LIMIT


class TankIsFull(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= 1


class TargetFillReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= target_state.fill


class TargetMassReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fuel_mass >= target_state.mass


def main():
    pass


if __name__ == "__main__":
    main()


# End
