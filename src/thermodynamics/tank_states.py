from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from src.fluids.hydrogen_retrievers import HydrogenRetriever

SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5


class Hydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen
    density: float


class StateDerivatives(Protocol):
    pressure: float
    temperature: float
    gas_mass: float
    liquid_mass: float
    venting_mass: float
    heat_flux: float


class FuelFlow(Protocol):
    ...


class Tank(Protocol):
    volume: float

    @abstractmethod
    def compute_fuel_height(self, fuel_volume: float):
        ...


class DynamicModel(Protocol):
    @abstractmethod
    def compute_state_derivatives(
        self, tank_state: TankState, *args
    ) -> StateDerivatives:
        ...


@dataclass
class InitialState:
    pressure: float
    temperature: float
    fill: float
    multi_flow: bool = False

    def __post_init__(self):
        self.hydrogen = self.get_hydrogen_properties()

    def get_hydrogen_properties(self) -> Hydrogen:
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        return self.hydrogen

    def compute_fuel_mass(self, tank_volume: float) -> float:
        """Compute the initial fuel mass based on tank volume and fill ratio."""
        try:
            # Special cases
            if self.fill == 0.0:
                return tank_volume * self.hydrogen.gas.density
            elif self.fill == 1.0:
                return tank_volume * self.hydrogen.liquid.density

            # Try using both phases if available
            return tank_volume * (
                self.fill * self.hydrogen.liquid.density
                + (1 - self.fill) * self.hydrogen.gas.density
            )
        except (ValueError, AttributeError) as e:
            # If phases aren't available as expected, use a fallback approach
            print(f"Warning: Issue accessing hydrogen phases: {e}")

            # Try to get density from the available phase
            try:
                # If it's a two-phase hydrogen with one phase available
                if hasattr(self.hydrogen, 'liquid'):
                    print("Using liquid phase density for initial mass calculation")
                    return tank_volume * self.hydrogen.liquid.density
            except ValueError:
                pass

            try:
                if hasattr(self.hydrogen, 'gas'):
                    print("Using gas phase density for initial mass calculation")
                    return tank_volume * self.hydrogen.gas.density
            except ValueError:
                pass

            # Last resort: use the density directly if it's a single-phase hydrogen
            print("Using base hydrogen density for initial mass calculation")
            return tank_volume * self.hydrogen.density


@dataclass
class TargetState:
    max_pressure: float
    min_pressure: float
    min_temperature: float
    fill: float
    mass: float
    density: float = None



@dataclass
class TankState:
    tank: Tank
    temperature: float
    pressure: float
    fuel_mass: float
    multi_flow: bool = False

    @property
    def volume(self):
        return self.tank.volume

    @property
    def liquid_mass(self) -> float:
        if self.fill == 0:
            return 0
        return self.volume * self.fill * self.hydrogen.liquid.density

    @property
    def gas_mass(self) -> float:
        if self.fill == 1:
            return 0
        return (
            self.volume
            * (1 - self.fill)
            * self.hydrogen.gas.density
        )

    @property
    def fill(self):
        if self.phase == "gas":
            return 0.0
        if self.phase == "liquid":
            return 1.0

        # Ensure that divide by zero is not possible
        if self.volume == 0:
            raise ValueError("Volume cannot be zero")

        # Ensure densities are valid
        if self.hydrogen.liquid.density <= self.hydrogen.gas.density:
            raise ValueError("Liquid density must be greater than gas density")

        fill_value = (
            (self.fuel_mass / self.volume - self.hydrogen.gas.density)
            / (self.hydrogen.liquid.density - self.hydrogen.gas.density)
        )
         # Ensure fill value is not negative
        if fill_value < 0:
            fill_value = 0

        return fill_value

    @property
    def fuel_volume(self):
        return self.fill * self.volume

    @property
    def fuel_height(self):
        if self.fuel_volume <= 0:
            return 0
        return self.tank.compute_fuel_height(self.fuel_volume)

    @property
    def is_full(self):
        return self.fill >= 1

    @property
    def is_empty(self):
        return self.fill == 0 or self.fuel_height == 0

    @property
    def phase(self) -> str:
        """Determine the phase of the tank state without causing recursion."""
        # Direct class check to avoid triggering property accessors
        hydrogen_class_name = self.hydrogen.__class__.__name__

        if 'TwoPhase' in hydrogen_class_name:
            return "twophase"

        # Direct attribute checks instead of using properties
        # that might trigger recursion
        if hasattr(self, '_fill_level'):
            fill_level = self._fill_level  # Access the backing field directly
        elif hasattr(self, 'fuel_mass') and hasattr(self, 'tank') and hasattr(self.tank, 'volume'):
            # Calculate fill directly without using properties
            try:
                max_liquid_mass = self.tank.volume * self.hydrogen.density
                fill_level = self.fuel_mass / max_liquid_mass if max_liquid_mass > 0 else 0
            except:
                # If calculation fails, use a fallback value
                fill_level = 0.5
        else:
            # Default value if we can't determine
            fill_level = 0.5

        # Determine phase based on fill level directly
        if fill_level < 0.01:  # Almost empty
            return "gas"
        elif fill_level > 0.99:  # Almost full
            return "liquid"
        else:
            # Default to gas for single-phase if we can't determine otherwise
            return "gas"

    def __post_init__(self) -> None:
        self.get_hydrogen_properties()
        self.complete_state_properties()

    def complete_state_properties(self):
        if self.pressure is None:
            self.pressure = self.hydrogen.pressure
        if self.temperature is None:
            self.temperature = self.hydrogen.temperature

    def get_hydrogen_properties(self) -> Hydrogen:
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        return self.hydrogen

    def compute_state_derivatives(
        self,
        dynamic_model: DynamicModel,
        *args
    ) -> StateDerivatives:
        self.heat_flux = args[-2]  # Second to last argument
        self.tank_thermal_capacity = args[-1]  # Last argument

        if self.multi_flow:
            # Multi-flow case needs to handle different argument counts
            if len(args) == 3:
                # Only one list of flows provided with multi_flow=True
                fuel_flows, heat_flux, tank_thermal_capacity = args
                # Check if model requires separate in/out flows
                if hasattr(dynamic_model, 'compute_state_derivatives') and 'fuel_flow_out' in dynamic_model.compute_state_derivatives.__code__.co_varnames:
                    # Model expects separate in/out flows - use empty list for inflows
                    # If SinglePhaseInOutModel, need to handle empty lists differently
                    if "SinglePhaseInOutModel" in str(type(dynamic_model)):
                        # Create a dummy flow for empty lists to avoid index errors
                        from src.mission.mission_sections import InFlow, OutFlow
                        from src.fluids.hydrogen_retrievers import SinglePhaseRequester

                        # For Tank 1 (empty inflow list)
                        if not fuel_flows:
                            # Both lists empty, create dummy flows
                            dummy_props = SinglePhaseRequester().get_hydrogen_properties(self.pressure, self.temperature)
                            dummy_inflow = InFlow(0.0, dummy_props)
                            dummy_outflow = OutFlow(0.0, "gas")
                            self.derivatives = dynamic_model.compute_state_derivatives(
                                self, [dummy_inflow], [dummy_outflow]
                            )
                        else:
                            # Only inflow list is empty
                            dummy_props = SinglePhaseRequester().get_hydrogen_properties(self.pressure, self.temperature)
                            dummy_inflow = InFlow(0.0, dummy_props)
                            self.derivatives = dynamic_model.compute_state_derivatives(
                                self, [dummy_inflow], fuel_flows
                            )
                    else:
                        self.derivatives = dynamic_model.compute_state_derivatives(
                            self, [], fuel_flows
                        )
                else:
                    # Model can handle single list of flows
                    self.derivatives = dynamic_model.compute_state_derivatives(
                        self, fuel_flows
                    )
            else:
                # Full multi-flow case with separate in/out flow lists
                fuel_flows_in, fuel_flows_out, heat_flux, tank_thermal_capacity = args

                # Handle empty flow lists for SinglePhaseInOutModel
                if "SinglePhaseInOutModel" in str(type(dynamic_model)):
                    from src.mission.mission_sections import InFlow, OutFlow
                    from src.fluids.hydrogen_retrievers import SinglePhaseRequester

                    if not fuel_flows_in:
                        dummy_props = SinglePhaseRequester().get_hydrogen_properties(self.pressure, self.temperature)
                        dummy_inflow = InFlow(0.0, dummy_props)
                        fuel_flows_in = [dummy_inflow]

                    if not fuel_flows_out:
                        dummy_outflow = OutFlow(0.0, "gas")
                        fuel_flows_out = [dummy_outflow]

                self.derivatives = dynamic_model.compute_state_derivatives(
                    self, fuel_flows_in, fuel_flows_out
                )
        else:
            # Single flow case: (fuel_flows, heat_flux, tank_thermal_capacity)
            fuel_flows, heat_flux, tank_thermal_capacity = args
            self.derivatives = dynamic_model.compute_state_derivatives(
                self, fuel_flows
            )

        return self.derivatives


@dataclass
class TankStates:
    states: list[TankState]
    timestep: float

    def __add__(self, other: TankStates) -> TankStates:
        if len(self.states) == 0:
            self.states = other.states
            return self
        if self.states[-1] == other.states[0]:
            self.states += other.states[1:]
            return self
        self.states += other.states
        return self

    def add_tank_state(self, tank_state: TankState) -> list[TankState]:
        self.states.append(tank_state)
        return self.states

    @property
    def timesteps_in_hours(self):
        return [
            i * self.timestep * SECONDS_TO_HOURS
            for i, _ in enumerate(self.pressures)
        ]

    @property
    def last_state(self):
        return self.states[-1]

    @property
    def first_state(self):
        return self.states[0]

    @property
    def pressures_in_bar(self):
        return [
            pressure * PASCAL_TO_BAR for pressure in self.pressures
        ]

    @property
    def pressures(self):
        return [state.pressure for state in self.states]

    @property
    def temperatures(self):
        return [state.temperature for state in self.states]

    @property
    def pressure_derivatives(self):
        return [state.derivatives.pressure for state in self.states]

    @property
    def temperature_derivatives(self):
        return [state.derivatives.temperature for state in self.states]

    @property
    def initial_temperature(self) -> float:
        return self.states[0].temperature

    @property
    def last_pressure(self):
        return self.last_state.pressure

    @property
    def last_temperature(self):
        return self.last_state.temperature

    @property
    def last_fill(self):
        return self.last_state.fill

    @property
    def max_pressure(self):
        return max(self.pressures)

    @property
    def average_temperature(self):
        return mean(self.temperatures)

    @property
    def min_temperature(self):
        return min(self.temperatures)

    @property
    def hydrogens(self) -> list[Hydrogen]:
        return [state.hydrogen for state in self.states]

    @property
    def fills(self) -> list[float]:
        return [state.fill for state in self.states]

    @property
    def volumes(self) -> list[float]:
        return [state.volume for state in self.states]

    @property
    def liquid_masses(self) -> list[float]:
        masses = [
            fill * volume * hydrogen.liquid.density
            if fill != 0 else 0
            for fill, volume, hydrogen in zip(
                self.fills, self.volumes, self.hydrogens
            )
        ]
        for mass, fill, volume, hydrogen in zip(masses, self.fills, self.volumes, self.hydrogens):
            if mass < 0:
                raise ValueError(f"Negative liquid mass detected: mass={mass}, volume={volume}, fill={fill}, density={hydrogen.liquid.density}")
        return masses

    @property
    def gas_masses(self) -> list[float]:
        masses = [
            (1 - fill) * volume * hydrogen.gas.density
            if fill < 1 else 0
            for fill, volume, hydrogen in zip(
                self.fills, self.volumes, self.hydrogens
            )
        ]
        for mass, fill, volume, hydrogen in zip(masses, self.fills, self.volumes, self.hydrogens):
            if mass < 0:
                raise ValueError(f"Negative gas mass detected: mass={mass}, volume={volume}, fill={fill}, density={hydrogen.gas.density}")
        return masses

    @property
    def total_masses(self) -> list[float]:
        masses = [
            liquid_mass + gas_mass
            for liquid_mass, gas_mass in zip(self.liquid_masses, self.gas_masses)
        ]
        for mass, liquid_mass, gas_mass, fill, volume, hydrogen in zip(masses, self.liquid_masses, self.gas_masses, self.fills, self.volumes, self.hydrogens):
            if mass < 0:
                raise ValueError(f"Negative total mass detected: mass={mass}, liquid_mass={liquid_mass}, gas_mass={gas_mass}, volume={volume}, fill={fill}, liquid_density={hydrogen.liquid.density}, gas_density={hydrogen.gas.density}")
        return masses

    @property
    def state_derivatives(self):
        return [
            state.derivatives
            if hasattr(state, "derivatives")
            else self.states[i-1].derivatives
            for i, state in enumerate(self.states[:-1])
        ]

    @property
    def required_fluxes(self):
        return [
            derivative.heat_flux
            for derivative in self.state_derivatives
        ]


def main():
    pass


if __name__ == "__main__":
    main()

# End
