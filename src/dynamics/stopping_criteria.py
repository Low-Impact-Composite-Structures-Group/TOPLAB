from abc import abstractmethod
from typing import Protocol

EMPTY_LIMIT = 0.01  # A lower limit to define when the tank is empty


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
    phase: str


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
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        
        return (
            fuel_tank_state.fill <= EMPTY_LIMIT 
            and fuel_tank_state.phase == "twophase"
        )


class NoFuelMass(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fuel_mass <= target_state.mass


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


class LowerPressureReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.pressure <= target_state.min_pressure


class MaxPressureReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.pressure >= target_state.max_pressure


def main():
    pass


if __name__ == "__main__":
    main()


# End
