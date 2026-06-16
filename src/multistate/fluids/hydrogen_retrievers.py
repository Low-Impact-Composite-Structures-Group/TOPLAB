from __future__ import annotations

import os
from abc import abstractmethod
from typing import Protocol, Union

from CoolProp.CoolProp import PhaseSI, PropsSI
import CoolProp.CoolProp as CP

from src.multistate.fluids.convective_mediums import Hydrogen, IsochoricHydrogen, TwoPhaseHydrogen

path = os.getcwd() + "/src/fluids/refprop/"
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, path)

HYDROGEN_FLUID = "hydrogen"


class HydrogenRequester(Protocol):
    @abstractmethod
    def get_property(self, pressure: float, temperature: float, property: float) -> float:
        ...

    @abstractmethod
    def get_hydrogen_properties(
        self, pressure: float, temperature: float
    ) -> Union[Hydrogen, TwoPhaseHydrogen]:
        ...


class SinglePhaseRequester(HydrogenRequester):
    fluid = HYDROGEN_FLUID
    properties = [
        "T", "P", "D", "V", "C", "L", "H", "U", "A", "d(D)/d(P)|T",
        "d(D)/d(T)|P", "d(H)/d(P)|T", "d(H)/d(T)|P", "d(P)/d(T)|D", "Phase",
    ]

    def get_property(self, pressure: float, temperature: float, property: float) -> float:
        try:
            return PropsSI(property, "P", pressure, "T", temperature, self.fluid)
        except ValueError as error:
            error_keywords = [
                "Saturation pressure", "ptriple", "PQ_flash", "Brent", "bracket",
                "molar density", "below the minimum", "Tmin",
            ]
            if not any(keyword in str(error) for keyword in error_keywords):
                raise error
            adjusted_pressure = max(pressure, 15000.0)
            adjusted_temperature = max(temperature, 20.0)
            if any(keyword in str(error) for keyword in ["within 1e-4", "Saturation pressure"]):
                adjusted_pressure *= 1.002
            return PropsSI(property, "P", adjusted_pressure, "T", adjusted_temperature, self.fluid)

    def get_hydrogen_properties(self, pressure: float, temperature: float) -> Hydrogen:
        return Hydrogen(*[
            self.get_property(pressure, temperature, property_name)
            for property_name in self.properties
        ])


class TwoPhaseRequester(SinglePhaseRequester):
    def get_property(self, pressure: float, property: str, state):
        state_code = {"gas": 1, "liquid": 0}
        return PropsSI(property, "P", pressure, "Q", state_code.get(state), self.fluid)

    def get_hydrogen_properties(self, pressure: float, temperature: float) -> TwoPhaseHydrogen:
        if pressure is None:
            pressure = PropsSI("P", "T", temperature, "Q", 0, self.fluid)
        gas = Hydrogen(*[
            self.get_property(pressure, property_name, "gas")
            for property_name in self.properties
        ])
        liquid = Hydrogen(*[
            self.get_property(pressure, property_name, "liquid")
            for property_name in self.properties
        ])
        return TwoPhaseHydrogen(liquid, gas, self.compute_pressure_derivative(liquid.temperature))

    def compute_pressure_derivative(self, temperature: float) -> float:
        temperature_factor = 1.0001
        pressure = PropsSI("P", "T", temperature, "Q", 0, self.fluid)
        new_temperature = temperature * temperature_factor
        new_pressure = PropsSI("P", "T", new_temperature, "Q", 0, self.fluid)
        return (new_pressure - pressure) / (new_temperature - temperature)


class IsochoricHydrogenRequester(SinglePhaseRequester):
    properties = [
        "T", "P", "D", "V", "C", "L", "H", "U", "A", "d(D)/d(P)|T",
        "d(D)/d(T)|P", "d(H)/d(P)|T", "d(H)/d(T)|P", "d(P)/d(T)|D", "Phase",
    ]

    def __init__(self, saturation_tolerance: float = 1e-3):
        self.saturation_tolerance = saturation_tolerance

    def is_near_saturation(self, temperature: float, pressure: float) -> bool:
        try:
            saturation_pressure = PropsSI("P", "T", temperature, "Q", 0, self.fluid)
            return abs(pressure - saturation_pressure) / saturation_pressure < self.saturation_tolerance
        except Exception:
            return False

    def compute_vapor_fraction(self, temperature: float, density: float) -> float:
        try:
            rho_l = PropsSI("D", "T", temperature, "Q", 0, self.fluid)
            rho_v = PropsSI("D", "T", temperature, "Q", 1, self.fluid)
            if abs(rho_l - rho_v) <= 1e-6:
                return 0.0
            vapor_fraction = (1.0 / density - 1.0 / rho_l) / (1.0 / rho_v - 1.0 / rho_l)
            return max(0.0, min(1.0, vapor_fraction))
        except Exception:
            return 0.0

    def get_hydrogen_properties(
        self,
        pressure: float,
        temperature: float,
        density: float = None,
    ) -> IsochoricHydrogen:
        base_properties = [
            self.get_property(pressure, temperature, property_name)
            for property_name in self.properties
        ]
        near_saturation = self.is_near_saturation(temperature, pressure)
        if density is None:
            density = base_properties[2]

        saturation_pressure = None
        vapor_fraction = None
        if near_saturation:
            try:
                saturation_pressure = PropsSI("P", "T", temperature, "Q", 0, self.fluid)
                vapor_fraction = self.compute_vapor_fraction(temperature, density)
            except Exception:
                near_saturation = False

        return IsochoricHydrogen(
            *base_properties,
            is_near_saturation=near_saturation,
            saturation_pressure=saturation_pressure,
            vapor_fraction=vapor_fraction,
        )

    def get_property_at_saturation(self, pressure: float, property: str, phase: str = "liquid") -> float:
        state_code = {"gas": 1, "liquid": 0}
        return PropsSI("P", pressure, "Q", state_code.get(phase, 0), self.fluid)


class HydrogenRequesterFactory:
    @staticmethod
    def get_hydrogen_retriever(hydrogen_phase: str) -> HydrogenRequester:
        if hydrogen_phase == "twophase":
            return TwoPhaseRequester()
        if hydrogen_phase == "isochoric":
            return IsochoricHydrogenRequester()
        if hydrogen_phase in ["gas", "liquid"]:
            return SinglePhaseRequester()
        raise ValueError(f"'{hydrogen_phase}' is an unsupported phase for the hydrogen factory.")


class PhaseRequester:
    fluid = HYDROGEN_FLUID

    def get_fluid_phase(self, temperature: float, pressure: float) -> str:
        critical_temperature = PropsSI("Tcrit", "", 0, "", 0, self.fluid)
        critical_pressure = PropsSI("Pcrit", "", 0, "", 0, self.fluid)
        if pressure > critical_pressure and temperature > critical_temperature:
            return "gas"
        pressure_ratio = pressure / critical_pressure
        if pressure_ratio > 0.75:
            try:
                saturation_temperature = PropsSI("T", "P", pressure, "Q", 0, self.fluid)
                if temperature > saturation_temperature * 1.01:
                    return "gas"
            except Exception:
                if temperature > 0.7 * critical_temperature:
                    return "gas"

        if temperature <= critical_temperature and pressure <= critical_pressure:
            try:
                reference_temperature = PropsSI("T", "P", pressure, "Q", 0, self.fluid)
                if abs(reference_temperature - temperature) / temperature < 1e-2:
                    return "twophase"
            except Exception:
                pass

        phase = PhaseSI("P", pressure, "T", temperature, self.fluid)
        if phase == "supercritical":
            return "gas"
        return phase.split("_")[-1]


class HydrogenRetriever:
    def define_requester(self, pressure: float, temperature: float) -> HydrogenRequester:
        if pressure is None and temperature is None:
            raise ValueError("Not pressure nor temperature have been provided")
        if pressure is None or temperature is None:
            return TwoPhaseRequester()
        return HydrogenRequesterFactory.get_hydrogen_retriever(
            PhaseRequester().get_fluid_phase(temperature, pressure)
        )

    def get_hydrogen_properties(
        self,
        pressure: float,
        temperature: float,
    ) -> Union[Hydrogen, TwoPhaseHydrogen]:
        requester = self.define_requester(pressure, temperature)
        return requester.get_hydrogen_properties(pressure, temperature)