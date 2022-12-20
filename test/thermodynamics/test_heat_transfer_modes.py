from __future__ import annotations
import math

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from src.thermodynamics.heat_transfer_modes import (GRAVITATIONAL_ACCELERATION,
                                                    STEPHAN_BOLTZMANN_CONSTANT, ChurchillNaturalConvection,
                                                    ForcedConvection, FujiiNaturalConvection,
                                                    GasPhaseConvection,
                                                    LiquidPhaseConvection,
                                                    NaturalConvection, NaturalCylinderConvection, NaturalSphereConvection,
                                                    Radiation, RohsenowNaturalConvection)


@dataclass
class ConvectiveMedium:
    thermal_conductivity: float
    dynamic_viscosity: float
    specific_heat_constant_pressure: float
    density: float
    prantl_number: float
    kinematic_viscosity: float
    thermal_expansion_coefficient: float
    temperature: float

    @classmethod
    def test_medium(cls) -> ConvectiveMedium:
        return cls(
            thermal_conductivity=5,
            dynamic_viscosity=15,
            specific_heat_constant_pressure=25,
            density=35,
            prantl_number=15,
            kinematic_viscosity=13,
            thermal_expansion_coefficient=11,
            temperature=250
        )


class TestRadiation(unittest.TestCase):

    def test_heat_transfer_coefficient(self):

        medium = ConvectiveMedium.test_medium()
        ambient_temp = medium.temperature
        skin_temp = 200
        emittance = 0.95
        expected_value = (
            STEPHAN_BOLTZMANN_CONSTANT
            * emittance
            * (
                skin_temp ** 2 + ambient_temp ** 2
            ) * (
                skin_temp + ambient_temp
            )
        )
        radiation = Radiation(skin_temp, ambient_temp)
        actual_value = radiation.heat_transfer_coefficient
        self.assertEqual(expected_value, actual_value)


class TestNaturalConvection(unittest.TestCase):

    def setUp(self) -> None:

        # Normal test case
        dummy = 10
        self.dummy_nussult = dummy
        self.convective_length = 0.5
        self.surface_temperature = 300
        self.medium = ConvectiveMedium.test_medium()

        class ConcreteNaturalConvection(NaturalConvection):

            @property
            def nussult_number(self) -> float:
                return dummy
        
        self.convection = ConcreteNaturalConvection(
            self.medium,
            self.convective_length,
            self.surface_temperature
        )

    def test_heat_transfer_coefficient(self):

        # Normal test case
        actual_value = self.convection.heat_transfer_coefficient
        expected_value = (
            self.dummy_nussult
            * ConvectiveMedium.test_medium().thermal_conductivity
            / self.convective_length
        )
        self.assertEqual(expected_value, actual_value)

        # Case where the the convective length is null
        self.convection.characteristic_dimension = 0.0
        actual_value = self.convection.heat_transfer_coefficient
        expected_value = float("inf")
        self.assertEqual(expected_value, actual_value)

    def test_rayleigh_number(self):

        # Normal example
        expected_value = (
            GRAVITATIONAL_ACCELERATION
            * self.medium.thermal_expansion_coefficient
            * self.convection.temperature_delta
            * self.convective_length ** 3
            * self.medium.prantl_number
            / self.medium.kinematic_viscosity
        )
        actual_value = self.convection.rayleigh_number
        self.assertEqual(expected_value, actual_value)

        # Example where temperature delta is null, so a dummy value
        # should be return to initialise temperature iterations
        self.convection.surface_temperature = self.medium.temperature
        expected_value = 1e-13
        actual_value = self.convection.rayleigh_number
        self.assertEqual(expected_value, actual_value)


class TestForcedConvection(unittest.TestCase):

    def setUp(self) -> None:
        self.medium = ConvectiveMedium.test_medium()
        self.dimension = 5
        self.surface_temperature = 300
        self.velocity = 3
        self.convection = ForcedConvection(
            self.medium,
            self.dimension,
            self.surface_temperature,
            self.velocity
        )

    def test_reynolds_number(self):
        expected_value = (
            self.medium.density
            * self.velocity
            * self.dimension
            / self.medium.dynamic_viscosity
        )
        actual_value = self.convection.reynolds_number
        self.assertEqual(expected_value, actual_value)

    @patch(
        "src.thermodynamics.heat_transfer_modes." \
        "ForcedConvection.reynolds_number",
        5
    )
    def test_nussult_number(self):
        expected_value = (
            0.03625
            * (self.medium.prantl_number) ** (0.43)
            * (5) ** (0.8)
        )
        actual_value = self.convection.nussult_number
        self.assertEqual(expected_value, actual_value)


class TestLiquidPhaseConvection(unittest.TestCase):

    @patch(
        "src.thermodynamics.heat_transfer_modes." \
        "LiquidPhaseConvection.rayleigh_number",
        5
    )
    def test_nussult_number(self):

        convection = LiquidPhaseConvection(
            ConvectiveMedium.test_medium(), None, None
        )
        expected_value = 0.10345354477
        actual_value = convection.nussult_number
        self.assertAlmostEqual(expected_value, actual_value)


class TestGasPhaseConvection(unittest.TestCase):

    def test_nussult_number(self):

        convection = GasPhaseConvection(
            None, None, None
        )
        expected_value = 17
        actual_value = convection.nussult_number
        self.assertAlmostEqual(expected_value, actual_value)


class TestNaturalSphereConvection(unittest.TestCase):

    def test_nussult_number(self):
        medium = ConvectiveMedium.test_medium()
        dimension = 0.5
        temperature = 333
        convection = NaturalSphereConvection(
                    medium, dimension, temperature
                )
        expected_value = (
            2 
            + (
                0.589 * convection.rayleigh_number ** (1/4)
                / (
                    1 + (
                        0.469 / medium.prantl_number
                    ) ** (9/16)
                ) ** (4/9)
            )
        )
        actual_value = convection.nussult_number
        self.assertEqual(expected_value, actual_value)


class TestNaturalCylinderConvection(unittest.TestCase):

    def test_nussult_number(self):
        medium = ConvectiveMedium.test_medium()
        dimension = 0.5
        temperature = 333
        convection = NaturalCylinderConvection(
                    medium, dimension, temperature
                )
        expected_value = (
            0.6 
            + (
                0.387 * convection.rayleigh_number ** (1/6)
                / (
                    1 + (
                        0.559 / medium.prantl_number
                    ) ** (9/16)
                ) ** (8/27)
            )
        ) ** 2
        actual_value = convection.nussult_number
        self.assertEqual(expected_value, actual_value)


class TestRohsenowNaturalConvection(unittest.TestCase):

    def setUp(self) -> None:
        self.medium = ConvectiveMedium.test_medium()
        self.dimension = 0.5
        self.temperature = 333
        self.convection = RohsenowNaturalConvection(
            self.medium, self.dimension, self.temperature
        )

    def test_c_barred_l(self):
        expected_value = (
            0.671 / (
                1 + (
                    0.492 / self.medium.prantl_number
                ) ** (9 / 16)
            ) ** (4 / 9)
        )
        actual_value = self.convection.c_barred_l
        self.assertEqual(expected_value, actual_value)

    def test_nussult_T(self):
        expected_value = (
            0.772
            * self.convection.c_barred_l
            * self.convection.rayleigh_number ** (1 / 4)
        )
        actual_value = self.convection.nussult_T
        self.assertEqual(expected_value, actual_value)

    def test_f(self):
        expected_value = (
            1 - (0.13 / self.convection.nussult_T ** (0.16))
        )
        actual_value = self.convection.f
        self.assertEqual(expected_value, actual_value)

    def test_nussult_l(self):
        expected_value = (
            2 * self.convection.f
            / math.log(
                1 + 2 * self.convection.f / self.convection.nussult_T
            )
        )
        actual_value = self.convection.nussult_l
        self.assertEqual(expected_value, actual_value)

    def test_nussult_t(self):
        expected_value = (
            self.convection.c_barred_l
            * self.convection.rayleigh_number ** (1 / 3)
        )
        actual_value = self.convection.nussult_t
        self.assertEqual(expected_value, actual_value)

    def test_nussult_number(self):
        m = 10
        expected_value = (
            self.convection.nussult_l ** m 
            + self.convection.nussult_t ** m
        ) ** (1 / m)
        actual_value = self.convection.nussult_number
        self.assertEqual(expected_value, actual_value)


class TestChurchillNaturalConvection(unittest.TestCase):

    def test_nussult_number(self):

        medium = ConvectiveMedium.test_medium()
        dimension = 0.5
        temperature = 333
        convection = ChurchillNaturalConvection(
                    medium, dimension, temperature
                )
        expected_value = math.sqrt(
            0.825 + (
                (
                    0.387 * convection.rayleigh_number ** (1 / 6)
                ) / (
                    1 + (
                        0.437 / medium.prantl_number
                    ) ** (9 / 16)
                ) ** (8 / 27)
            )
        )
        actual_value = convection.nussult_number
        self.assertEqual(expected_value, actual_value)


class TestFujiiNaturalConvection(unittest.TestCase):

    def test_nussult_number(self):

        medium = ConvectiveMedium.test_medium()
        dimension = 0.5
        temperature = 333
        convection = FujiiNaturalConvection(
                    medium, dimension, temperature
                )
        expected_value = 0.56 * convection.rayleigh_number ** (1 / 4)
        actual_value = convection.nussult_number
        self.assertEqual(expected_value, actual_value)



if __name__ == "__main__":
    unittest.main()


# End
