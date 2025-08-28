from abc import abstractmethod
from typing import Protocol
from CoolProp.CoolProp import PropsSI

EMPTY_LIMIT = 0.01  # A lower limit to define when the tank is empty


class TargetState(Protocol):
    max_pressure: float
    min_pressure: float
    min_temperature: float
    fill: float
    mass: float
    density: float = None


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

class TargetDensityReached(StoppingCriterion):
    def __init__(self, target_density=None):
        self.target_density = target_density

    def is_met(
        self, fuel_tank_state: FuelTankState, target_state: TargetState
    ) -> bool:
        # If target density is explicitly provided to this instance, use it
        target = self.target_density

        # Otherwise fall back to the target state's density
        if target is None:
            target = target_state.density

        # If no density target is available, we can't meet this criterion
        if target is None:
            return False

        # Calculate density based on phase
        phase = fuel_tank_state.hydrogen.phase
        if phase == 'twophase':
            # get quality of mixture from coolprop PropSI call
            Q = PropsSI("Q", "P", fuel_tank_state.pressure, "T", fuel_tank_state.temperature, "H2")
            # print(f"DEBUG: Two-phase quality Q={Q:.3f}")
            # use lever rule
            # density = 1 / (
            #     (1 - Q) / fuel_tank_state.hydrogen.liquid.density +
            #     Q / fuel_tank_state.hydrogen.gas.density
            # )
            density = fuel_tank_state.hydrogen.gas.density

        elif phase in ["gas", "supercritical"]:
            density = fuel_tank_state.hydrogen.gas.density
        elif phase in ["liquid", "supercritical_liquid"]:
            density = fuel_tank_state.hydrogen.liquid.density
        else:
            # Default calculation if we can't determine phase
            if hasattr(fuel_tank_state, 'fuel_mass') and hasattr(fuel_tank_state, 'tank'):
                if hasattr(fuel_tank_state.tank, 'volume'):
                    density = fuel_tank_state.fuel_mass / fuel_tank_state.tank.volume
                else:
                    return False  # Can't calculate density
            else:
                return False  # Can't calculate density

        # Print debug info
        if density >= target:
            print(f"STOPPING: Target density reached ({density:.1f} kg/m³)")

        return density >= target

def main():
    pass


if __name__ == "__main__":
    main()


# End
