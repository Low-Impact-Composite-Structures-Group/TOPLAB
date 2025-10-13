from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol

from src.fluids.hydrogen_retrievers import HydrogenRetriever, IsochoricHydrogenRequester
from src.fluids.convective_mediums import IsochoricHydrogen

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
        # Check if a phase transition just occurred - if so, don't override it
        if hasattr(self, '_recent_phase_transition') and self._recent_phase_transition:
            # Keep the hydrogen object from the phase transition for a short time
            if hasattr(self, '_transition_time'):
                # Use the simulation time if available, otherwise fall back to system time
                current_time = getattr(self, '_last_check_time', self._transition_time + 5)
                if current_time - self._transition_time < 5.0:  # Preserve for 5 seconds
                    print(f"Preserving phase transition - not overriding hydrogen properties")
                    return self.hydrogen
                else:
                    # Transition is old enough, we can update normally
                    self._recent_phase_transition = False

        original_phase = getattr(self.hydrogen, 'phase', 'unknown') if hasattr(self, 'hydrogen') else 'none'

        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )

        new_phase = getattr(self.hydrogen, 'phase', 'unknown')
        if original_phase != 'none' and original_phase != new_phase:
            print(f"WARNING: get_hydrogen_properties() changed phase from {original_phase} to {new_phase}")

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

    # Flow rates (positive = inflow, negative = outflow)
    inflow_rate: float = 0.0  # kg/s
    outflow_rate: float = 0.0  # kg/s
    vent_rate: float = 0.0  # kg/s
    coupling_inflow_rate: float = 0.0  # kg/s (from other tanks)
    coupling_outflow_rate: float = 0.0  # kg/s (to other tanks)

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

        # For supercritical phase, treat as single-phase gas (fill = 0)
        # because supercritical fluids behave more like dense gases
        if hasattr(self.hydrogen, 'phase') and 'supercritical' in self.hydrogen.phase:
            return 0.0

        # Ensure that divide by zero is not possible
        if self.volume == 0:
            raise ValueError("Volume cannot be zero")

        # Two-phase case: need to safely access densities
        try:
            # Ensure densities are valid
            if self.hydrogen.liquid.density <= self.hydrogen.gas.density:
                raise ValueError("Liquid density must be greater than gas density")

            fill_value = (
                (self.fuel_mass / self.volume - self.hydrogen.gas.density)
                / (self.hydrogen.liquid.density - self.hydrogen.gas.density)
            )
        except (ValueError, AttributeError):
            # If we can't access both phases, estimate based on fuel mass and available density
            if hasattr(self.hydrogen, 'density'):
                # Single phase - calculate fill based on total density
                total_density = self.fuel_mass / self.volume
                reference_density = self.hydrogen.density
                if total_density >= reference_density * 0.8:  # High density = more liquid-like
                    return min(1.0, total_density / reference_density)
                else:  # Low density = more gas-like
                    return 0.0
            else:
                # Last resort: use fuel mass to estimate
                return min(1.0, self.fuel_mass / (self.volume * 70))  # 70 kg/m³ ~ liquid H2 density

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
        # Direct phase check for safety
        if self.phase == "liquid":
            return True

        # Safe check that doesn't require accessing phase-specific properties
        try:
            return self.fill >= 1
        except ValueError:
            # If error occurs accessing fill, use alternative check
            # Check if we have high density relative to critical density of hydrogen
            try:
                density = self.fuel_mass / self.volume
                # Hydrogen critical density is ~31 kg/m³, liquid is ~70 kg/m³
                return density > 65  # Close to liquid density
            except:
                return False

    @property
    def is_empty(self):
        # Safe check that doesn't try to access phase-specific properties
        if hasattr(self, 'fuel_mass') and self.fuel_mass <= 0:
            return True

        # For single-phase tanks (gas or liquid), check based on fuel mass or volume
        if self.phase in ["gas", "liquid"]:
            # Tank can have gas phase and still have significant mass
            # Only consider empty if fuel mass is very small
            return self.fuel_mass <= 1e-6  # Very small threshold for numerical precision

        # For two-phase, use fill-based check
        try:
            return self.fill == 0 or self.fuel_height == 0
        except ValueError:
            # If error occurs accessing fill, fall back to mass-based check
            return self.fuel_mass <= 1e-6

    @property
    def phase(self) -> str:
        """Determine the phase of the tank state based on hydrogen properties."""
        # Direct class check to avoid triggering property accessors
        hydrogen_class_name = self.hydrogen.__class__.__name__

        if 'TwoPhase' in hydrogen_class_name:
            return "twophase"

        # Check if hydrogen object has phase attribute
        if hasattr(self.hydrogen, 'phase'):
            phase = self.hydrogen.phase
            # Map supercritical phases to simpler categories for thermal calculations
            if 'supercritical' in phase:
                return "gas"  # Treat supercritical as gas for thermal resistance
            return phase

        # Check phase based on available properties
        try:
            # Try to access gas properties - if this works, it's gas or supercritical
            _ = self.hydrogen.gas
            return "gas"
        except (ValueError, AttributeError):
            # If gas access fails, check if liquid properties work
            try:
                _ = self.hydrogen.liquid
                return "liquid"
            except (ValueError, AttributeError):
                # If both fail, default to gas for CCH2 analysis
                return "gas"

    def check_phase_transition(self, current_time):
        """
        Check if phase transition should occur based on natural thermodynamic conditions.
        Uses the HydrogenRetriever's natural phase detection instead of artificial triggers.
        """
        # Only check every few seconds to avoid overhead
        if hasattr(self, '_last_check_time') and current_time - self._last_check_time < 5.0:
            return

        self._last_check_time = current_time

        # Get natural phase from hydrogen retriever
        try:
            from src.fluids.hydrogen_retrievers import PhaseRequester
            phase_requester = PhaseRequester()
            natural_phase = phase_requester.get_fluid_phase(self.temperature, self.pressure)

            # Debug: Always print the phase comparison
            print(f"PHASE CHECK: current={self.phase}, natural={natural_phase}, P={self.pressure/1e5:.2f}bar, T={self.temperature:.2f}K")

            # Only transition if the natural phase is different from current phase
            if natural_phase != self.phase:
                print(f"NATURAL PHASE TRANSITION: {self.phase} → {natural_phase}")
                print(f"  Conditions: P={self.pressure/1e5:.2f}bar, T={self.temperature:.2f}K")

                # Update hydrogen properties to match the new phase
                from src.fluids.hydrogen_retrievers import HydrogenRetriever
                retriever = HydrogenRetriever()

                try:
                    new_hydrogen = retriever.get_hydrogen_properties(self.pressure, self.temperature)
                    self.hydrogen = new_hydrogen
                    # Mark that a phase transition just occurred
                    self._recent_phase_transition = True
                    self._transition_time = current_time
                    print(f"Successfully transitioned to {natural_phase} phase")

                    # If transitioning to two-phase, initialize gas/liquid masses properly
                    if natural_phase == "twophase" and hasattr(self, 'fuel_mass'):
                        # Estimate initial gas/liquid split based on conditions
                        # This is a reasonable approximation that can be refined by the dynamics
                        total_mass = self.fuel_mass
                        if hasattr(new_hydrogen, 'gas') and hasattr(new_hydrogen, 'liquid'):
                            # Use density ratio to estimate initial split
                            rho_gas = new_hydrogen.gas.density
                            rho_liquid = new_hydrogen.liquid.density
                            # Start with 50/50 volume split as initial guess
                            vol_fraction_gas = 0.5
                            vol_fraction_liquid = 0.5

                            # Calculate masses based on volume fractions
                            total_volume = self.volume
                            mass_gas = rho_gas * vol_fraction_gas * total_volume
                            mass_liquid = rho_liquid * vol_fraction_liquid * total_volume

                            # Normalize to match total fuel mass
                            mass_total_calc = mass_gas + mass_liquid
                            if mass_total_calc > 0:
                                self.gas_mass = mass_gas * (total_mass / mass_total_calc)
                                self.liquid_mass = mass_liquid * (total_mass / mass_total_calc)
                            else:
                                self.gas_mass = total_mass * 0.5
                                self.liquid_mass = total_mass * 0.5

                            print(f"Initialized two-phase: gas={self.gas_mass:.3f}kg, liquid={self.liquid_mass:.3f}kg")
                        else:
                            # Fallback if hydrogen object doesn't have gas/liquid components
                            self.gas_mass = total_mass * 0.5
                            self.liquid_mass = total_mass * 0.5
                            print(f"Fallback two-phase initialization: gas={self.gas_mass:.3f}kg, liquid={self.liquid_mass:.3f}kg")

                except Exception as e:
                    print(f"Could not complete phase transition to {natural_phase}: {e}")
                    # Keep the current phase but log the issue

        except Exception as e:
            print(f"Error in natural phase detection: {e}")
            # Continue with current phase - no artificial forcing

    def __post_init__(self) -> None:
        # No forced phase transitions - rely on natural thermodynamic detection
        self.get_hydrogen_properties()
        self.complete_state_properties()

    def complete_state_properties(self):
        if self.pressure is None:
            self.pressure = self.hydrogen.pressure
        if self.temperature is None:
            self.temperature = self.hydrogen.temperature

    def get_hydrogen_properties(self) -> Hydrogen:
        # Check if a phase transition just occurred - if so, don't override it
        if hasattr(self, '_recent_phase_transition') and self._recent_phase_transition:
            # Keep the hydrogen object from the phase transition for a short time
            if hasattr(self, '_transition_time'):
                # Use the simulation time if available, otherwise fall back to system time
                current_time = getattr(self, '_last_check_time', self._transition_time + 5)
                if current_time - self._transition_time < 5.0:  # Preserve for 5 seconds
                    print(f"Preserving phase transition - not overriding hydrogen properties")
                    return self.hydrogen
                else:
                    # Transition is old enough, we can update normally
                    self._recent_phase_transition = False

        original_phase = getattr(self.hydrogen, 'phase', 'unknown') if hasattr(self, 'hydrogen') else 'none'

        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )

        new_phase = getattr(self.hydrogen, 'phase', 'unknown')
        if original_phase != 'none' and original_phase != new_phase:
            print(f"WARNING: get_hydrogen_properties() changed phase from {original_phase} to {new_phase}")

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
                    # Sort by attributes rather than class type
                    inflows = []
                    outflows = []

                    # Sort by checking for specific attributes that distinguish flow types
                    for flow in fuel_flows:
                        # Check if it has hydrogen attribute (InFlow) or phase attribute (OutFlow)
                        if hasattr(flow, 'hydrogen'):
                            inflows.append(flow)
                        elif hasattr(flow, 'phase'):
                            outflows.append(flow)
                        else:
                            # If we can't determine type from attributes, check if mass_flow is negative
                            if hasattr(flow, 'mass_flow'):
                                flow_rate = flow.mass_flow
                                if isinstance(flow_rate, (int, float)) and flow_rate < 0:
                                    # Convert this to a proper inflow

                                    dummy_props = SinglePhaseRequester().get_hydrogen_properties(self.pressure, self.temperature)
                                    # Create positive inflow
                                    converted_inflow = InFlow(abs(flow_rate), dummy_props)
                                    inflows.append(converted_inflow)
                                else:
                                    # Default to gas phase if not specified
                                    converted_outflow = OutFlow(flow_rate, "gas")
                                    outflows.append(converted_outflow)

                    # If SinglePhaseInOutModel, ensure no empty lists
                    if "SinglePhaseInOutModel" in str(type(dynamic_model)):
                        # Create dummy flows if needed
                        from src.fluids.hydrogen_retrievers import SinglePhaseRequester
                        from src.mission.mission_sections import InFlow, OutFlow

                        if not inflows:
                            dummy_props = SinglePhaseRequester().get_hydrogen_properties(self.pressure, self.temperature)
                            dummy_inflow = InFlow(0.0, dummy_props)
                            inflows = [dummy_inflow]

                        if not outflows:
                            dummy_outflow = OutFlow(0.0, "gas")
                            outflows = [dummy_outflow]

                    # Call with properly sorted flows
                    self.derivatives = dynamic_model.compute_state_derivatives(
                        self, inflows, outflows
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


@dataclass
class IsochoricTankState:
    """
    IsochoricTankState represents the tank state for the stops_model approach.

    This state class handles the [m, T, Ts] state vector from the stops_model:
    - m: Total fuel mass [kg]
    - T: Fluid temperature [K]
    - Ts: Solid temperature [K] (tank structure temperature)

    Unlike the standard TankState, this class:
    - Uses IsochoricHydrogen for thermodynamic properties
    - Tracks solid temperature evolution
    - Supports configuration-dependent behavior (A/B/C configurations)
    - Handles near-saturation conditions with isochoric assumptions
    """
    tank: Tank
    fuel_mass: float  # m in stops_model state vector
    temperature: float  # T in stops_model state vector
    solid_temperature: float  # Ts in stops_model state vector
    pressure: float = None  # Computed from EOS
    configuration: str = "A"  # Configuration A, B, or C from stops_model
    scenario: str = "DISCHARGE"  # DISCHARGE, REFUEL, or DORMANCY

    # Additional properties for isochoric analysis
    hydrogen: 'IsochoricHydrogen' = None
    derivatives: 'IsochoricStateDerivatives' = None

    # Flow rates (positive = inflow, negative = outflow)
    inflow_rate: float = 0.0  # kg/s
    outflow_rate: float = 0.0  # kg/s
    vent_rate: float = 0.0  # kg/s
    coupling_inflow_rate: float = 0.0  # kg/s (from other tanks)
    coupling_outflow_rate: float = 0.0  # kg/s (to other tanks)

    @property
    def volume(self):
        return self.tank.volume

    @property
    def density(self):
        """Fuel density [kg/m³]"""
        return self.fuel_mass / self.volume

    @property
    def state_vector(self):
        """Returns the [m, T, Ts] state vector for ODE integration"""
        return [self.fuel_mass, self.temperature, self.solid_temperature]

    @classmethod
    def from_state_vector(cls, tank: Tank, state_vector: list, **kwargs):
        """Create IsochoricTankState from [m, T, Ts] state vector"""
        m, T, Ts = state_vector
        return cls(
            tank=tank,
            fuel_mass=m,
            temperature=T,
            solid_temperature=Ts,
            **kwargs
        )

    def __post_init__(self):
        """Initialize hydrogen properties and compute pressure"""
        self.get_hydrogen_properties()
        self.compute_pressure()

    def get_hydrogen_properties(self):
        """Get IsochoricHydrogen properties for current state"""
        if self.hydrogen is None or self._needs_hydrogen_update():
            requester = IsochoricHydrogenRequester()

            # If pressure is not set, estimate it first
            if self.pressure is None:
                self.compute_pressure()

            self.hydrogen = requester.get_hydrogen_properties(
                self.pressure, self.temperature, self.density
            )

    def compute_pressure(self):
        """Compute pressure from equation of state"""
        if self.pressure is None:
            from CoolProp.CoolProp import PropsSI
            try:
                # Check for valid temperature range
                if self.temperature <= 0:
                    print(f"⚠️ Invalid temperature {self.temperature:.2f} K detected - using fallback pressure")
                    self.pressure = 1e5  # Default to 1 bar
                    return

                self.pressure = PropsSI("P", "T", self.temperature, "Dmass", self.density, "hydrogen")
            except Exception as e:
                # Fallback for extreme conditions
                print(f"⚠️ CoolProp error for T={self.temperature:.2f}K, ρ={self.density:.2f}kg/m³: {e}")
                self.pressure = 1e5  # Default to 1 bar

    def _needs_hydrogen_update(self) -> bool:
        """Check if hydrogen properties need to be updated"""
        if self.hydrogen is None:
            return True

        # Check if temperature or density changed significantly
        temp_change = abs(self.temperature - self.hydrogen.temperature) / self.hydrogen.temperature
        density_change = abs(self.density - self.hydrogen.density) / self.hydrogen.density

        return temp_change > 0.01 or density_change > 0.01  # 1% threshold

    def update_from_state_vector(self, state_vector: list):
        """Update state from [m, T, Ts] vector (for ODE integration)"""
        self.fuel_mass, self.temperature, self.solid_temperature = state_vector

        # Recompute derived properties
        self.compute_pressure()
        self.get_hydrogen_properties()

    def is_configuration_B(self, p_min: float) -> bool:
        """Check if pressure is below minimum threshold (Configuration B)"""
        return self.pressure <= p_min

    def is_configuration_C(self, p_vent: float) -> bool:
        """Check if pressure is above venting threshold (Configuration C)"""
        return self.pressure >= p_vent

    def determine_configuration(self, p_min: float, p_vent: float) -> str:
        """Determine current configuration based on pressure thresholds"""
        if self.is_configuration_C(p_vent):
            return "C"
        elif self.is_configuration_B(p_min):
            return "B"
        else:
            return "A"

    def get_effective_cv(self) -> float:
        """Get effective specific heat for isochoric process"""
        if self.hydrogen is not None:
            return self.hydrogen.get_effective_cv()
        else:
            # Fallback
            from CoolProp.CoolProp import PropsSI
            return PropsSI("Cvmass", "T", self.temperature, "Dmass", self.density, "hydrogen")


@dataclass
class IsochoricStateDerivatives:
    """
    State derivatives for the isochoric ODE system.

    Represents d/dt[m, T, Ts] from the stops_model:
    - fuel_mass_derivative: dm/dt [kg/s]
    - temperature_derivative: dT/dt [K/s]
    - solid_temperature_derivative: dTs/dt [K/s]
    """
    fuel_mass_derivative: float  # dm/dt
    temperature_derivative: float  # dT/dt
    solid_temperature_derivative: float  # dTs/dt

    # Additional information for analysis
    heat_flux: float = 0.0  # Heat flux from solid to fluid [W]
    discharge_heat_flux: float = 0.0  # Heat flux for discharge [W]
    alpha_s: float = 0.0  # Heat transfer coefficient [W/m²-K]

    @property
    def state_derivative_vector(self):
        """Returns [dm/dt, dT/dt, dTs/dt] for ODE integration"""
        return [
            self.fuel_mass_derivative,
            self.temperature_derivative,
            self.solid_temperature_derivative
        ]


@dataclass
class IsochoricInitialState:
    """
    Initial state for isochoric analysis.

    Similar to InitialState but for the stops_model approach with solid temperature.
    """
    fuel_mass: float  # Initial fuel mass [kg]
    temperature: float  # Initial fluid temperature [K]
    solid_temperature: float  # Initial solid temperature [K]
    pressure: float = None  # Computed from EOS if not provided
    scenario: str = "DISCHARGE"  # Scenario name

    def to_isochoric_tank_state(self, tank: Tank) -> IsochoricTankState:
        """Convert to IsochoricTankState"""
        return IsochoricTankState(
            tank=tank,
            fuel_mass=self.fuel_mass,
            temperature=self.temperature,
            solid_temperature=self.solid_temperature,
            pressure=self.pressure,
            scenario=self.scenario
        )

    def get_state_vector(self):
        """Get [m, T, Ts] initial state vector"""
        return [self.fuel_mass, self.temperature, self.solid_temperature]


@dataclass
class IsochoricTankStates:
    """
    Collection of IsochoricTankState objects for time series analysis.

    Similar to TankStates but for the isochoric approach.
    """
    states: list[IsochoricTankState]
    timestep: float

    def __add__(self, other: 'IsochoricTankStates') -> 'IsochoricTankStates':
        if len(self.states) == 0:
            self.states = other.states
            return self
        if len(other.states) > 0:
            self.states += other.states
        return self

    def add_state(self, state: IsochoricTankState):
        """Add a new state to the collection"""
        self.states.append(state)

    @property
    def last_state(self) -> IsochoricTankState:
        return self.states[-1]

    @property
    def first_state(self) -> IsochoricTankState:
        return self.states[0]

    @property
    def times(self) -> list[float]:
        """Time values [s]"""
        return [i * self.timestep for i in range(len(self.states))]

    @property
    def fuel_masses(self) -> list[float]:
        return [state.fuel_mass for state in self.states]

    @property
    def temperatures(self) -> list[float]:
        return [state.temperature for state in self.states]

    @property
    def solid_temperatures(self) -> list[float]:
        return [state.solid_temperature for state in self.states]

    @property
    def pressures(self) -> list[float]:
        return [state.pressure for state in self.states]

    @property
    def densities(self) -> list[float]:
        return [state.density for state in self.states]

    @property
    def configurations(self) -> list[str]:
        return [state.configuration for state in self.states]

    @property
    def max_pressure(self) -> float:
        return max(self.pressures)

    @property
    def min_temperature(self) -> float:
        return min(self.temperatures)

    @property
    def state_derivatives(self) -> list[IsochoricStateDerivatives]:
        return [state.derivatives for state in self.states if state.derivatives is not None]


def main():
    pass


if __name__ == "__main__":
    main()

# End
