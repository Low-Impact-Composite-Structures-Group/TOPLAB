from abc import abstractmethod
from typing import Protocol

EMPTY_LIMIT = 0.01  # A lower limit to define when the tank is empty


class TargetState(Protocol):
    max_pressure: float
    min_pressure: float
    min_temperature: float
    min_fill: float
    min_mass: float
    target_fill: float
    target_mass: float


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


class TankIsEmpty(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:

        return (
            fuel_tank_state.fill <= target_state.min_fill
            and fuel_tank_state.phase == "twophase"
        )


class NoFuelMass(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fuel_mass <= target_state.min_mass


class TankIsFull(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= 1


class TargetFillReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= target_state.target_fill


class TargetMassReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fuel_mass >= target_state.target_mass


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


class StoppingCriteriaFactory:

    _criteria = {
        "tank_is_empty": TankIsEmpty(),
        "no_fuel_mass": NoFuelMass(),
        "tank_is_full": TankIsFull(),
        "target_fill_reached": TargetFillReached(),
        "target_mass_reached": TargetMassReached(),
        "lower_pressure_reached": LowerPressureReached(),
        "upper_pressure_reached": MaxPressureReached(),
    }

    @property
    def _available(self):
        return ", ".join(self._criteria.keys())

    def create_criterion(self, type: str) -> StoppingCriterion:
        criterion = self._criteria.get(type)

        if criterion is not None:
            return criterion

        raise ValueError(
            f"'{type}' is not a valid stopping criterion.\n"
            f"Available criteria are: {self._available}"
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
