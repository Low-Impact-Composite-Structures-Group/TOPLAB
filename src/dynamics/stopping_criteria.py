from abc import abstractmethod
from typing import Protocol


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


class TargetState(Protocol):
    pressure: float


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
        return fuel_tank_state.pressure >= target_state.pressure


def main():
    pass


if __name__ == "__main__":
    main()


# End
