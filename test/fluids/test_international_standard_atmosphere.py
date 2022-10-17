

import unittest

from src.fluids.convective_mediums import ConvectiveMedium
from src.fluids.international_standard_atmosphere import (
    ISA, get_ISA_air_properties)


class TestISA(unittest.TestCase):

    def setUp(self) -> None:
        self.troposphere = ISA(250)
        self.tropopause = ISA(12e3)
        self.stratosphere = ISA(22e3)

    def test_specific_heat_constant_pressure(self):
        self.assertEqual(
            self.troposphere.specific_heat_constant_pressure, 1005
        )
    
    def test_thermal_conductivity(self):
        self.assertEqual(self.troposphere.thermal_conductivity, 0.025)

    def test_temperature(self):

        self.assertAlmostEqual(
            self.troposphere.temperature, 286.525, places=0
        )

        self.assertAlmostEqual(
            self.tropopause.temperature, 216.650, places=0
        )

        with self.assertRaises(ValueError) as context:
            self.stratosphere.temperature
        self.assertTrue(
            "Altitude out of bound..." in str(context.exception)
        )

    def test_pressure(self):

        self.assertAlmostEqual(
            self.troposphere.pressure, 98358, places=-2
        )

        self.assertAlmostEqual(
            self.tropopause.pressure, 19330.4, places=-2
        )
    
    def test_density(self):

        self.assertAlmostEqual(
            self.troposphere.density, 1.19587, places=4
        )
        
    def test_dynamic_viscosity(self):
        
        self.assertAlmostEqual(
            ISA(0).dynamic_viscosity, 0.00001812, places=6
        )

    def test_thermal_expansion_coefficient(self):

        self.assertEqual(
            1 / self.troposphere.temperature,
            self.troposphere.thermal_expansion_coefficient
        )

    def test_tropopause(self):

        self.assertEqual(
            ISA.tropopause(), ISA(11e3)
        )


class Test_get_ISA_air_properties(unittest.TestCase):

    def test_without_temperature(self):

        medium = ConvectiveMedium(
            temperature=288.15,
            pressure=101300.0,
            density=1.2249808415492711,
            dynamic_viscosity=1.820971015782525e-05,
            specific_heat_constant_pressure=1005,
            thermal_conductivity=0.025
        )

        self.assertDictEqual(
            medium.__dict__, get_ISA_air_properties(0).__dict__
        )

    def test_with_temperature(self):

        medium = ConvectiveMedium(
            temperature=300,
            pressure=101300.0,
            density=1.1765940983080747,
            dynamic_viscosity=1.8760792897522253e-05,
            specific_heat_constant_pressure=1005,
            thermal_conductivity=0.025
        )

        self.assertDictEqual(
            medium.__dict__,
            get_ISA_air_properties(0, temperature=300).__dict__
        )

if __name__ == "__main__":
    unittest.main()


# End
