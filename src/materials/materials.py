


import math
from dataclasses import dataclass

import scipy.integrate as integrate

UNIVERSAL_GAS_CONSTANT = 8.31434    # [J/(mol*K)]


@dataclass
class Material:
    failure_stress: float
    density: float
    characteristic_temperature: float
    molecular_weight: float

    @property
    def specific_gas_constant(self):
        return UNIVERSAL_GAS_CONSTANT * self.molecular_weight

    def determine_specific_heat(
        self, temperature: float
    ) -> float:
        """Compute the specific heat of the solid for given temperature.

        Args:
            temperature (float): temperature of the solid [K].

        Returns:
            float: specific heat of the solid at given temperature
            [J/(kg*K)].
        """
        # The computation is based on the method suggested in Cryogenic
        # Systems (Barron 1985), page 24. The method is also used by
        # Ahluwalia in paper of 2008.

        # Define the integral part of the function, which later can be
        # solved for the given temperature
        def integral_function(x: float) -> float:
            return (x ** 4 * math.exp(x)) / (math.exp(x) - 1) ** 2

        # Solve the integral for the given temperature interval
        integral, error_estimate = integrate.quad(
            lambda x: integral_function(x),
            0, self.characteristic_temperature / temperature
        )

        # Verify if the estimated error of the integral is not too large
        if abs(error_estimate) > 1e-5:
            raise StopIteration(
                "Error estimate in integral above 1e-5."
                + f"\nError: {error_estimate}"
            )

        # Return the actual equation (2.6 in Cryogenic Systems)
        return (
            9 * self.specific_gas_constant
            * (temperature / self.characteristic_temperature) ** 3
            * integral
        )

    def determine_thermal_capacity(
        self, temperature: float, mass: float
    ) -> float:
        """Compute the thermal capacity of the solid, for given mass and
        temperature.

        Args:
            temperature (float): temperature of the solid [K].
            mass (float): mass of the solid [kg]

        Returns:
            float: thermal capacity of the solid [J/K].
        """
        return (
            self.determine_specific_heat(temperature)
            * mass
        )

    def determine_thermal_conductivity(self, temperature: float) -> float:
        """Compute the thermal conductivity of the material for given temperature.

        Args:
            temperature (float): temperature of the material [K].

        Returns:
            float: thermal conductivity of the material [W/(m·K)].
        """
        # Default implementation - should be overridden in subclasses
        # This is a fallback for basic materials
        return 50.0  # W/(m·K)


@dataclass
class Metal(Material):

    def __post_init__(self):
        self.type = "metal"

    def determine_thermal_conductivity(self, temperature: float) -> float:
        """Compute the thermal conductivity of the metal for given temperature.

        Args:
            temperature (float): temperature of the metal [K].

        Returns:
            float: thermal conductivity of the metal [W/(m·K)].
        """
        # For basic aluminum, use a temperature-dependent approximation
        # This will be overridden in NIST materials for better accuracy
        if abs(self.density - 2700) < 100:  # Aluminum
            # Aluminum thermal conductivity decreases with decreasing temperature
            if temperature < 100:  # Cryogenic temperatures
                return 180.0  # W/(m·K)
            elif temperature < 200:
                return 200.0  # W/(m·K)
            else:
                return 220.0  # W/(m·K)
        else:
            # Default for other metals
            return 50.0  # W/(m·K)

    @classmethod
    def aluminum(cls):
        failure_stress = 450e6
        density = 2700
        specific_temperature = 389.4
        molecular_weight = 26.981539
        return cls(
            failure_stress,
            density,
            specific_temperature,
            molecular_weight
        )


@dataclass
class Composite(Material):
    winding_angle: float

    def __post_init__(self):
        self.type = "composite"

    def determine_thermal_conductivity(self, temperature: float) -> float:
        """Compute the thermal conductivity of the composite for given temperature.

        Args:
            temperature (float): temperature of the composite [K].

        Returns:
            float: thermal conductivity of the composite [W/(m·K)].
        """
        # For basic G10 composite, use a temperature-dependent approximation
        # This will be overridden in NIST materials for better accuracy
        if temperature < 50:  # Very low temperatures
            return 0.1  # W/(m·K)
        elif temperature < 100:
            return 0.15  # W/(m·K)
        elif temperature < 200:
            return 0.2  # W/(m·K)
        else:
            return 0.25  # W/(m·K)

    @classmethod
    def carbon(cls, winding_angle: float):
        # Values from internet and master thesis
        failure_stress = 2560E6
        density = 1580
        specific_temperature = 1500
        molecular_weight = 12.01
        return cls(
            failure_stress,
            density,
            specific_temperature,
            molecular_weight,
            winding_angle
        )




def main():
    pass


if __name__ == "__main__":
    main()


# End
