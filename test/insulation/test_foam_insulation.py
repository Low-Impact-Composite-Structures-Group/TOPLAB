import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from src.insulation.foam_insulations import (ConstantFoamInsulation,
                                             FoamInsulation,
                                             VariableFoamInsulation)


@dataclass
class FuelTank:
    radius: float
    surface_area: float


class TestFoamInsulation(unittest.TestCase):


    @patch.multiple(
        FoamInsulation, __abstractmethods__=set()
    )
    def setUp(self) -> None:
        self.thickness = 4e-2
        self.insulation = FoamInsulation(self.thickness)

    def test_compute_heat_transfer_coefficient(self):
        
        inner_radius = 2.5
        thermal_conductivity = 5
        outer_radius = inner_radius + self.thickness
        expected_value = 314.993386132
        actual_value = self.insulation.compute_heat_transfer_coefficient(
            thermal_conductivity, outer_radius, inner_radius
        )
        self.assertAlmostEqual(expected_value, actual_value)


class TestConstantFoamInsulation(unittest.TestCase):

    def setUp(self) -> None:
        self.thickness = 4e-2
        self.thermal_conductivity = 0.0046
        self.insulation = ConstantFoamInsulation(
            self.thickness, self.thermal_conductivity
        )
    
    def test_compute_thermal_conductivity(self):
        expected_value = self.thermal_conductivity
        actual_value = self.insulation.compute_thermal_conductivity(
            1e10, 85
        )
        self.assertEqual(expected_value, actual_value)

    def test_compute_thermal_resistances(self):
        
        # Define the fuel tank for for the test
        radius = 2.5
        surface_area = 85
        tank = FuelTank(radius, surface_area)

        # Define the list of temperatures
        temperatures = [33, 300.15]
        heat_transfer_coefficient = self.insulation.compute_heat_transfer_coefficient(
            self.insulation.thermal_conductivity,
            radius + self.insulation.thickness,
            radius
        )
        thermal_resistance = (
            1 / (
                heat_transfer_coefficient * tank.surface_area
            )
        )
        expected_value = [thermal_resistance]
        actual_value = self.insulation.compute_thermal_resistances(
            temperatures, tank
        )
        self.assertEqual(expected_value, actual_value)



    def test_polyvinylchloride(self):
        expected_value = 0.0046
        insulation = ConstantFoamInsulation.polyvinylchloride(
            self.thickness
        )
        actual_value = insulation.thermal_conductivity
        self.assertEqual(expected_value, actual_value)

    def test_rohacell(self):
        expected_value = 0.015
        insulation = ConstantFoamInsulation.rohacell(
            self.thickness
        )
        actual_value = insulation.thermal_conductivity
        self.assertEqual(expected_value, actual_value)


class TestVariableFoamInsulation(unittest.TestCase):

    def setUp(self) -> None:
        self.thickness = 4e-2
        name = "rohacell"
        self.foam = VariableFoamInsulation(self.thickness, name)
    
    def test_load_foam_data(self):

        foam_data = self.foam.load_foam_data()
        self.assertIsInstance(
            foam_data, dict
        )

    def test_compute_average_temperature(self):

        hot_temp = 300
        cold_temp = 200
        self.assertEqual(
            self.foam.compute_average_temperature(hot_temp, cold_temp),
            250
        )
    
    def test_compute_thermal_conductivity(self):

        # Case where temperature is too cold
        cold_temp, hot_temp = 0, 0
        with self.assertRaises(ValueError) as context:
            self.foam.compute_thermal_conductivity(
                hot_temp, cold_temp
            )
        self.assertTrue(
            "Temperature too cold for foam data..."
            in str(context.exception)
        )

        # Exact match with data value
        cold_temp, hot_temp = 29.7865, 29.7865
        expected_value = 0.00505
        self.assertEqual(
            expected_value,
            self.foam.compute_thermal_conductivity(
                hot_temp, cold_temp
            )
        )

        # Almost match with data value
        cold_temp, hot_temp = 29.9, 29.9
        expected_value = 0.00505
        self.assertEqual(
            expected_value,
            self.foam.compute_thermal_conductivity(
                hot_temp, cold_temp
            )
        )

        # Case where temperature is too hot
        cold_temp, hot_temp = 330, 330
        with self.assertRaises(ValueError) as context:
            self.foam.compute_thermal_conductivity(
                hot_temp, cold_temp
            )
        self.assertTrue(
            "Temperature too hot for foam data..."
            in str(context.exception)
        )

    def test_rohacell(self):

        VariableFoamInsulation.rohacell(self.thickness)

    def test_polyurethane(self):

        VariableFoamInsulation.polyurethane(self.thickness)


if __name__ == "__main__":
    unittest.main()


# End
