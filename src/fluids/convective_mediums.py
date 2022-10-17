from __future__ import annotations

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
        return PHASE_INDICES.get(str(int(self.state)))

    @property
    def gas(self) -> Hydrogen:
        if self.phase not in [
            "supercritical", "supercritical_gas", "gas"
        ]:
            raise ValueError("Hydrogen not in gas phase")
        return self

    @property
    def liquid(self) -> Hydrogen:
        if "liquid" not in self.phase:
            raise ValueError("Hydrogen not in liquid phase")
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
    def heat_of_evaporation(self):
        return self.gas.enthalpy - self.liquid.enthalpy


def main():
    pass


if __name__ == "__main__":
    main()


# End
