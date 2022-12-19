import unittest

from src.materials.materials import UNIVERSAL_GAS_CONSTANT, Composite, Material, Metal

import numpy as np


class TestMaterial(unittest.TestCase):

    def setUp(self) -> None:
        self.aluminum = Metal.aluminum()
        winding_angle = np.deg2rad(55)
        self.carbon = Composite.carbon(winding_angle)

    def test_specific_gas_constant(self):
        expected_value = (
            UNIVERSAL_GAS_CONSTANT * self.aluminum.molecular_weight
        )
        actual_value = self.aluminum.specific_gas_constant
        self.assertEqual(expected_value, actual_value)

    def test_determine_specific_heat(self):
        temperature = 200
        masses = [12, 49]
        materials: list[Material] = [self.aluminum, self.carbon]
        capacities = [
            mass * material.determine_specific_heat(200) * temperature
            for material, mass in zip(materials, masses)
        ]
        expected_value = 1.8e6
        actual_value = sum(capacities)
        self.assertAlmostEqual(expected_value, actual_value, places=-5)

    def test_determine_thermal_capacity(self):
        temperature = 200
        masses = [12, 49]
        materials: list[Material] = [self.aluminum, self.carbon]
        capacities = [
            material.determine_thermal_capacity(temperature, mass)
            for material, mass in zip(materials, masses)
        ]
        expected_value = 1.8e6
        actual_value = sum(capacities)
        self.assertAlmostEqual(expected_value, actual_value, places=-5) 



if __name__ == "__main__":
    unittest.main()


# End
