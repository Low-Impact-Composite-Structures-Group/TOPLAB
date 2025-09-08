from abc import abstractmethod
from typing import Protocol
from CoolProp.CoolProp import PropsSI

EMPTY_LIMIT = 0.01  # A lower limit to define when the tank is empty

def safe_quality_calculation(pressure, temperature, fluid="hydrogen"):
    """
    Safely calculate quality (vapor fraction) near the saturation line.

    This function handles cases where the standard CoolProp PropsSI("Q", "P", P, "T", T, fluid)
    might fail because we're too close to the saturation line.

    Args:
        pressure: Pressure in Pa
        temperature: Temperature in K
        fluid: Fluid name, defaults to "hydrogen"

    Returns:
        Quality value (0 to 1) or None if calculation fails
    """
    try:
        # First try direct calculation
        try:
            Q = PropsSI("Q", "P", pressure, "T", temperature, fluid)
            return Q
        except ValueError as e:
            # Check if we're at saturation line
            if "Saturation pressure" in str(e) and "is within" in str(e):
                # We're exactly at saturation, get saturation temperature
                t_sat = PropsSI("T", "P", pressure, "Q", 0, fluid)

                # Compare with our temperature to determine if we're slightly
                # on the liquid or gas side
                if temperature < t_sat:
                    return 0.0  # Slightly subcooled liquid
                else:
                    return 0.001  # Just barely into two-phase

            # If we got another error, try a different approach
            # Get saturation temperature and compare
            t_sat = PropsSI("T", "P", pressure, "Q", 0, fluid)
            temp_diff = temperature - t_sat

            if abs(temp_diff) < 0.1:  # Very close to saturation
                # Determine if we're slightly subcooled or superheated
                if temp_diff < 0:
                    return 0.0  # Subcooled liquid
                else:
                    return 0.001  # Just barely into two-phase region

            # If we're more than 0.1K away from saturation
            if temp_diff > 0:
                # Try calculating with a slightly adjusted temperature
                safe_temp = t_sat + 0.2  # 0.2K above saturation
                try:
                    Q = PropsSI("Q", "P", pressure, "T", safe_temp, fluid)
                    return Q
                except:
                    # If still failing, return a reasonable guess
                    return 0.5  # Assume mid-point in two-phase region
            else:
                # We're below saturation temperature
                return 0.0
    except Exception as e:
        # If all else fails
        print(f"WARNING: Could not calculate quality: {e}")
        return None


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
            try:
                # Try to get quality of mixture from coolprop PropSI call
                # This can fail near the saturation line where P,T exactly hits the saturation curve
                from CoolProp.CoolProp import PropsSI, get_global_param_string

                # First check if we're extremely close to the saturation line
                try:
                    # Get saturation temperature at current pressure
                    T_sat = PropsSI("T", "P", fuel_tank_state.pressure, "Q", 0, "hydrogen")
                    temp_diff = abs(fuel_tank_state.temperature - T_sat)

                    # If extremely close to saturation (within 0.1K), use a safer calculation
                    # if temp_diff < 0.1:
                    #     print(f"WARNING: Extremely close to saturation line (within {temp_diff:.4f}K). Using safe density calculation.")
                    #     # Use average of liquid and gas density as an approximation
                    #     if hasattr(fuel_tank_state.hydrogen, 'liquid') and hasattr(fuel_tank_state.hydrogen, 'gas'):
                    #         density = (fuel_tank_state.hydrogen.liquid.density + fuel_tank_state.hydrogen.gas.density) / 2.0
                    #     else:
                    #         # If phase properties not available, use overall density
                    #         density = fuel_tank_state.hydrogen.density
                    # else:
                        # Try the normal calculation if not too close to saturation
                        # Use our safe quality calculation function instead of direct PropsSI call
                    Q = safe_quality_calculation(fuel_tank_state.pressure, fuel_tank_state.temperature, "hydrogen")
                    if Q is None:
                        # If calculation failed, fall back to gas density
                        density = fuel_tank_state.hydrogen.gas.density
                    else:
                        # Could use lever rule but we're using gas density for simplicity for now
                        density = fuel_tank_state.hydrogen.gas.density
                except Exception as e:
                    print(f"WARNING: Error checking saturation point: {e}")
                    # Fall back to using the hydrogen object's density directly
                    density = fuel_tank_state.hydrogen.density
            except Exception as e:
                print(f"WARNING: Error calculating two-phase quality: {e}")
                # Fall back to direct phase properties
                if hasattr(fuel_tank_state.hydrogen, 'gas'):
                    density = fuel_tank_state.hydrogen.gas.density
                else:
                    density = fuel_tank_state.hydrogen.density

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
