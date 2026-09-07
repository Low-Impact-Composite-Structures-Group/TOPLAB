from __future__ import annotations

from dataclasses import dataclass
import math


GAMMA = 1.4
MOLAR_MASS_AIR = 0.02897
UNIVERSAL_GAS_CONSTANT = 8.31
PHASE_INDICES = {
    "0": "liquid",
    "1": "supercritical",
    "2": "supercritical_gas",
    "3": "supercritical_liquid",
    "5": "gas",
    "6": "twophase",
}


@dataclass
class ConvectiveMedium:
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
            GAMMA * UNIVERSAL_GAS_CONSTANT * self.temperature / MOLAR_MASS_AIR
        )


@dataclass
class Hydrogen(ConvectiveMedium):
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
        if isinstance(self.state, str):
            state = self.state.strip().lower()
            if state in {"two-phase", "two phase"}:
                return "twophase"
            if state in {
                "liquid",
                "gas",
                "twophase",
                "supercritical",
                "supercritical_gas",
                "supercritical_liquid",
            }:
                return state
        try:
            return PHASE_INDICES.get(str(int(self.state)), "unknown")
        except (TypeError, ValueError):
            return "unknown"

    @property
    def gas(self) -> "PARAHYD":
        if self.phase not in [
            "supercritical",
            "supercritical_gas",
            "gas",
            "supercritical_liquid",
        ]:
            raise ValueError("Hydrogen not in gas phase")
        return self

    @property
    def liquid(self) -> "PARAHYD":
        if self.phase not in ["liquid", "supercritical_liquid", "twophase"]:
            raise ValueError("Hydrogen not in liquid phase")
        return self

    def get_phase(self, phase: str) -> "PARAHYD":
        if phase in (None, ""):
            return self
        if phase == "gas":
            return self.gas
        if phase == "liquid":
            return self.liquid
        raise ValueError(f"Unsupported phase request: {phase}")


@dataclass
class TwoPhaseHydrogen:
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
        if self.liquid.density is None or self.gas.density is None:
            raise ValueError("Cannot compute two-phase density: missing phase densities")
        if self.liquid.density <= 0 or self.gas.density <= 0:
            raise ValueError("Cannot compute two-phase density: non-positive phase densities")
        rho_l = self.liquid.density
        rho_g = self.gas.density
        return 2 * rho_l * rho_g / (rho_l + rho_g)

    @property
    def density(self):
        return self.compute_homogeneous_density()

    def get_phase(self, phase: str) -> Hydrogen:
        if phase == "gas":
            return self.gas
        if phase == "liquid":
            return self.liquid
        raise ValueError(f"'{phase}' is an unsupported phase...")


@dataclass
class IsochoricHydrogen(ConvectiveMedium):
    enthalpy: float
    internal_energy: float
    speed_of_sound_database: float
    dRho_dP: float
    dRho_dT: float
    dH_dP: float
    dH_dT: float
    dP_dT: float
    state: str
    is_near_saturation: bool = False
    saturation_pressure: float = None
    vapor_fraction: float = None

    @property
    def speed_of_sound(self):
        return self.speed_of_sound_database

    @property
    def phase(self):
        if self.is_near_saturation:
            return "isochoric_twophase"
        try:
            return PHASE_INDICES.get(str(int(self.state)), "unknown")
        except (ValueError, TypeError):
            return "unknown"

    @property
    def effective_density(self):
        return self.density

    def get_effective_cv(self):
        try:
            return self.specific_heat_constant_pressure - self.pressure / (
                self.temperature * self.density
            )
        except Exception:
            return self.specific_heat_constant_pressure