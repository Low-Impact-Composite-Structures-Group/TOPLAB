from abc import abstractmethod
from typing import Protocol

EMPTY_LIMIT = 0.01  # A lower limit to define when the tank is empty


class TargetState(Protocol):
    max_pressure: float
    min_pressure: float
    min_temperature: float
    # Newer configs use explicit min/target fields; legacy tests use
    # `fill` and `mass`.
    min_fill: float
    min_mass: float
    target_fill: float
    target_mass: float
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


class TankIsEmpty(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        min_fill = getattr(target_state, "min_fill", EMPTY_LIMIT)
        return (
            fuel_tank_state.fill <= min_fill
            and fuel_tank_state.phase == "twophase"
        )


class NoFuelMass(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        min_mass = getattr(target_state, "min_mass", getattr(target_state, "mass", 0.0))
        return fuel_tank_state.fuel_mass <= min_mass


class TankIsFull(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        return fuel_tank_state.fill >= 1


class TargetFillReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        target_fill = getattr(target_state, "target_fill", getattr(target_state, "fill", 1.0))
        return fuel_tank_state.fill >= target_fill


class TargetMassReached(StoppingCriterion):

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        target_mass = getattr(target_state, "target_mass", getattr(target_state, "mass", 0.0))
        return fuel_tank_state.fuel_mass >= target_mass


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


# Backward-compatible alias (older code/tests import MaxPressure).
class MaxPressure(MaxPressureReached):
    pass


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
