"""Hydrogen retrievers can be used to request hydrogen properties.
There are two types of retrievers, namely single phase and two phase.
For single phase retrievers, both temperature and pressure are required.
The two phase requester only requires pressure, where the provided
temperature is discarded in the request.

Fuel Tank - Hydrogen Retrievers
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""

import os
from abc import abstractmethod
from typing import Protocol, Union

from CoolProp.CoolProp import PropsSI, PhaseSI
import CoolProp.CoolProp as CP

from src.fluids.convective_mediums import Hydrogen, TwoPhaseHydrogen

path = os.getcwd() + "/src/fluids/refprop/"
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, path)


HYDROGEN_FLUID = "REFPROP::PARAHYD" # To be use with Refprop (i84 chip)
HYDROGEN_FLUID = "hydrogen"         # To be used with Coolprop (M1 chip)


class HydrogenRequester(Protocol):

    @abstractmethod
    def get_property(
        self, pressure: float, temperature: float, property: float
    ) -> float:
        """Method to get a property of the hydrogen from the database.

        Args:
            pressure (float): Pressure of the hydrogen.
            temperature (float): Temperature of the hydrogen.
            property (float): Desired property of the hydrogen, these 
            have to be in the form accepted by CoolProp database.

        Returns:
            float: Value of the requested property
        """
        ...
    
    @abstractmethod
    def get_hydrogen_properties(
        self, pressure: float, temperature: float
    ) -> Union[Hydrogen, TwoPhaseHydrogen]:
        """Method to retrieve the properties of hydrogen.

        The properties should entail all the properties required for 
        the thermodynamic and dynamic models.

        Args:
            pressure (float): Pressure of the hydrogen.
            temperature (float): Temperature of the hydrogen.

        Returns:
            Union[Hydrogen, TwoPhaseHydrogen]: Returns a Single or 
            Two Phase Hydrogen dataclass object, depending on the 
            requester type.
        """
        ...


class SinglePhaseRequester(HydrogenRequester):
    """Single phase requester is used to request the properties of 
    hydrogen for a single phase. See the list of properties in the class
    attribute to see which properties are requested form the database.

    Note the list with attributes is in line with the properties that
    have to be passed in the Hydrogen dataclass.
    """

    fluid = HYDROGEN_FLUID

    properties = [
        "T", "P", "D", "V", "C", "L", "H", "U", "A", "d(D)/d(P)|T", 
        "d(D)/d(T)|P", "d(H)/d(P)|T", "d(H)/d(T)|P", "d(P)/d(T)|D",
        "Phase"
    ]

    def get_property(
        self, pressure: float, temperature: float, property: float
    ) -> float:
        """Method to get a property of the hydrogen from the database.

        Args:
            pressure (float): Pressure of the hydrogen.
            temperature (float): Temperature of the hydrogen.
            property (float): Desired property of the hydrogen, these 
            have to be in the form accepted by CoolProp database.

        Returns:
            float: Value of the requested property
        """
        return PropsSI(
            property,
            "P", pressure,
            "T", temperature,
            self.fluid
        )


    def get_hydrogen_properties(
        self, pressure: float, temperature: float
    ) -> Hydrogen:
        return Hydrogen(*[
            self.get_property(pressure, temperature, property)
            for property in self.properties
        ])


class TwoPhaseRequester(SinglePhaseRequester):
    
    def get_property(
        self, pressure: float, property: str, state
    ) -> float:
        state_code = {
            "gas": 1,
            "liquid": 0
        }
        return PropsSI(
            property,
            "P", pressure,
            "Q", state_code.get(state),
            self.fluid
        )

    def get_hydrogen_properties(
        self, pressure: float, temperature: float
    ) -> TwoPhaseHydrogen:
        if pressure is None:
            pressure = PropsSI(
                "P",
                "T", temperature,
                "Q", 0,
                self.fluid
            )
        gas = Hydrogen(*[
            self.get_property(pressure, property, "gas")
            for property in self.properties
        ])
        liquid = Hydrogen(*[
            self.get_property(pressure, property, "liquid")
            for property in self.properties
        ])
        return TwoPhaseHydrogen(
            liquid,
            gas,
            self.compute_pressure_derivative(liquid.temperature)
        )
    
    def compute_pressure_derivative(
        self, temperature: float
    ) -> float:
        """Method to compute the saturated pressure derivative with 
        respect to temperature.

        Args:
            temperature (float): Temperature at which the derivative is 
            to be determined.

        Returns:
            float: Pressure derivative with respect to temperature.
        """
        # Small factor to compute a temperature delta
        temperature_factor = 1.0001
        pressure = PropsSI(
            "P",
            "T", temperature,
            "Q", 0,
            self.fluid
        )
        new_temp = temperature * temperature_factor
        new_pressure = PropsSI(
            "P",
            "T", new_temp,
            "Q", 0,
            self.fluid
        )
        return (new_pressure - pressure) / (new_temp - temperature)


class HydrogenRequesterFactory():

    @staticmethod
    def get_hydrogen_retriever(hydrogen_phase: str) -> HydrogenRequester:
        if hydrogen_phase == "twophase":
            return TwoPhaseRequester()
        if hydrogen_phase in ["gas", "liquid"]:
            return SinglePhaseRequester()
        raise ValueError(
            f"'{hydrogen_phase}' is an unsupported phase for the "\
            "hydrogen factory."
        )


class PhaseRequester():
    """The phase requester is to be used to retrieve the phase of
    hydrogen. To do this the get_fluid_phase method can be used.
    """

    fluid = HYDROGEN_FLUID

    def get_fluid_phase(
        self, temperature: float, pressure: float
    ) -> str:
        """Get the phase of hydrogen for the provided temperature and
        pressure.

        Note that the states are simplified to twophase, liquid and gas.
        As such the supercritical states are simplified, where the 
        actual supercritical state is simplified to gas.

        Args:
            temperature (float): Temperature of the hydrogen.
            pressure (float): Pressure of the hydrogen.

        Returns:
            str: State of the hydrogen, thus twophase, liquid or gas.
        """
        # When the temperature and pressure are close to the 
        # saturated properties, twophase is to be returned
        two_phase_temperature_limit = 1e-2
        if (
            temperature <= PropsSI("Tcrit", "", 0, "", 0, self.fluid)
            and pressure <= PropsSI("Pcrit", "", 0, "", 0, self.fluid) 
        ):
            ref_temperature = PropsSI(
                "T", "P", pressure, "Q", 0, self.fluid
            )
            if (
                abs(ref_temperature - temperature) / temperature
                < two_phase_temperature_limit
            ):
                return "twophase"
        phase: str = PhaseSI(
            'P',pressure,
            'T',temperature,
            self.fluid
        )
        # The supercritical phase is simplified to the gas phase
        if phase == "supercritical":
            return "gas"
        # Should the fluid be in a supercritical gas or liquid state,
        # simply gas and liquid are returned respectively
        return phase.split("_")[-1]


class PropertyRetriever(Protocol):
    
    @abstractmethod
    def get_hydrogen_properties(
        self, pressure: float, temperature: float
    ) -> Union[Hydrogen, TwoPhaseHydrogen]:
        ...


class HydrogenRetriever(PropertyRetriever):

    def define_requester(
        self, pressure: float, temperature: float
    ) -> HydrogenRequester:
        if pressure is None and temperature is None:
            raise ValueError(
                "Not pressure nor temperature have been provided"
            )
        if pressure is None or temperature is None:
            return TwoPhaseRequester()
        return HydrogenRequesterFactory.get_hydrogen_retriever(
            PhaseRequester().get_fluid_phase(temperature, pressure)
        )

    def get_hydrogen_properties(
        self, pressure: float, temperature: float
    ) -> Union[Hydrogen, TwoPhaseHydrogen]:
        requester = self.define_requester(pressure, temperature)
        return requester.get_hydrogen_properties(pressure, temperature)


def main():
    pass


if __name__ == "__main__":
    main()


# End
