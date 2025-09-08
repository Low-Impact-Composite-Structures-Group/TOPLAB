from __future__ import annotations

"""
Convective mediums module for handling heat transfer properties of different fluids.

This module defines classes for representing convective mediums (like air and hydrogen)
used in heat transfer calculations. For hydrogen specifically, it handles:

1. Phase representation: Hydrogen can exist in various phases (liquid, gas, supercritical)
2. Phase access: Methods for accessing properties specific to gas-like or liquid-like behavior
3. Two-phase representation: Special handling for when hydrogen exists in two phases

PHASE HANDLING LOGIC:
- For standard states (gas, liquid), the phase is straightforward
- For supercritical states, we allow access through both .gas and .liquid properties:
  * This is physically justified as supercritical fluids exist on a continuum between
    gas-like and liquid-like behavior
  * In heat transfer calculations, this enables consistent model handling regardless of phase
  * The exact phase description is still maintained in the .phase property
"""

from dataclasses import dataclass
import math


GAMMA = 1.4                         # [-]
MOLAR_MASS_AIR = 0.02897            # [kg/mol]
MOLAR_MASS_HYDROGEN = 2.00784e-3    # [kg/mol]
UNIVERSAL_GAS_CONSTANT = 8.31       # [J/mol]
PHASE_INDICES = {
    "0": "liquid",
    "1": "supercritical",
    "2": "supercritical_gas",
    "3": "supercritical_liquid",
    "5": "gas",
    "6": "twophase"
}


@dataclass
class ConvectiveMedium:
    """Convective medium is use in the convective heat transfer modes.
    The base class is used for air convection, defining the kep
    properties required for the computations. In a child class hydrogen
    is defined, which requires additional properties for the convective
    heat transfer modes.
    """
    temperature: float
    pressure: float
    density: float
    dynamic_viscosity: float
    specific_heat_constant_pressure: float
    thermal_conductivity: float

    @property
    def thermal_expansion_coefficient(self):
        return 1 / self.temperature

    @property
    def kinematic_viscosity(self):
        return self.dynamic_viscosity / self.density

    @property
    def prantl_number(self):
        return (
            self.dynamic_viscosity
            * self.specific_heat_constant_pressure
            / self.thermal_conductivity
        )

    @property
    def speed_of_sound(self):
        return math.sqrt(
            GAMMA * UNIVERSAL_GAS_CONSTANT * self.temperature
            / MOLAR_MASS_AIR
        )


@dataclass
class Hydrogen(ConvectiveMedium):
    """Hydrogen class is used to define single phase hydrogen properties
    to be used in heat transfer modes. In a later class two phase
    hydrogen can be defined.

    Note the properties of hydrogen are obtained with CoolProp, to be
    used with the HydrogenRetrievers classes.
    """
    enthalpy: float
    internal_energy: float
    speed_of_sound_database: float
    dRho_dP: float
    dRho_dT: float
    dH_dP: float
    dH_dT: float
    dP_dT: float
    state: str

    @property
    def speed_of_sound(self):
        return self.speed_of_sound_database

    @property
    def phase(self):
        """
        Get the phase of hydrogen based on the state value.

        The state code is mapped to a descriptive phase name using PHASE_INDICES.

        Returns:
            str: Phase name (liquid, gas, supercritical, etc.)
        """
        try:
            state_code = str(int(self.state))
            phase = PHASE_INDICES.get(state_code)
            if not phase:
                print(f"Warning: Unknown phase state code: {state_code}")
                return "unknown"
            return phase
        except (ValueError, TypeError) as e:
            print(f"Error determining phase from state value: {self.state}")
            return "unknown"

    @property
    def gas(self) -> Hydrogen:
        """
        Get hydrogen properties for gas-like behavior.

        For heat transfer calculations, we treat supercritical fluid as gas-like
        regardless of whether it's 'supercritical_liquid' or 'supercritical_gas',
        as the properties are continuous across the pseudo-critical line.
        """
        if self.phase not in [
            "supercritical", "supercritical_gas", "gas", "supercritical_liquid"
        ]:
            raise ValueError(f"Cannot access gas properties: hydrogen is in {self.phase} phase")
        return self

    @property
    def liquid(self) -> Hydrogen:
        """
        Get hydrogen properties for liquid-like behavior.

        For consistency with the gas property, we also allow supercritical_liquid
        to be accessed as liquid, since this represents the region where the
        supercritical fluid has more liquid-like properties.
        """
        if self.phase not in ["liquid", "supercritical_liquid", "twophase"]:
            raise ValueError(f"Cannot access liquid properties: hydrogen is in {self.phase} phase")
        return self

    def get_phase(self, phase: str) -> Hydrogen:
        """
        Get hydrogen properties for a specific phase.

        Args:
            phase: The requested phase ("gas" or "liquid")

        Returns:
            The hydrogen object with properties for the requested phase

        Raises:
            ValueError: If the requested phase is not supported
        """
        if phase == "gas":
            return self.gas
        elif phase == "liquid":
            return self.liquid
        else:
            raise ValueError(f"Unsupported phase request: {phase}")
        return self


@dataclass
class TwoPhaseHydrogen:
    """Two phase hydrogen, which encapsulates the liquid and the gas
    phases in separate attributes. The class also enables to compute
    average densities and other properties based on the fuel quality,
    which are generally required with the dynamic models.
    """
    liquid: Hydrogen
    gas: Hydrogen
    dP_dT: float

    @property
    def pressure(self):
        return self.liquid.pressure

    @property
    def temperature(self):
        return self.liquid.temperature

    @property
    def phase(self):
        return "twophase"

    @property
    def heat_of_evaporation(self):
        return self.gas.enthalpy - self.liquid.enthalpy

    def compute_homogeneous_density(self):
        """
        Compute homogeneous two-phase mixture density.

        For a homogeneous mixture, we assume the hydrogen is completely mixed
        throughout the tank. This uses the harmonic mean of densities which
        provides a good approximation for a well-mixed two-phase system.
        """
        if self.liquid.density is None or self.gas.density is None:
            raise ValueError("Cannot compute two-phase density: missing phase densities")
        if self.liquid.density <= 0 or self.gas.density <= 0:
            raise ValueError("Cannot compute two-phase density: non-positive phase densities")

        # For homogeneous mixture, use harmonic mean
        # This represents a well-mixed two-phase system
        rho_l = self.liquid.density
        rho_g = self.gas.density

        # Harmonic mean provides good approximation for homogeneous mixture
        homogeneous_density = 2 * rho_l * rho_g / (rho_l + rho_g)

        return homogeneous_density

    @property
    def density(self):
        """
        Get the homogeneous mixture density for two-phase hydrogen.
        """
        return self.compute_homogeneous_density()

    def get_phase(self, phase: str) -> Hydrogen:
        if phase == "gas":
            return self.gas
        if phase == "liquid":
            return self.liquid
        if phase == "supercritical":
            # For supercritical conditions, use gas phase as a fallback
            print(f"SUPERCRITICAL PHASE REQUESTED - Using gas phase as fallback")
            return self.gas

        # Default to gas phase as a fallback with warning
        print(f"WARNING: Unsupported phase '{phase}' requested - falling back to gas phase")
        return self.gas


def main():
    pass


if __name__ == "__main__":
    main()


# End
