import unittest

from src.thermodynamics.thermal_resistances import (ParallelResistances,
                                                    SeriesResistances,
                                                    ThermalResistance)


class TestThermalResistance(unittest.TestCase):
    
    def test_resistance(self):
            
        # Normal case
        heat_transfer_coefficient = 3
        surface_area = 5
        resistance = ThermalResistance(
            heat_transfer_coefficient, surface_area
        )
        actual_value = resistance.value
        expected_value = 1 / 15
        self.assertEqual(expected_value, actual_value)

        # Null surface area case
        heat_transfer_coefficient = 1
        surface_area = 0
        resistance = ThermalResistance(
            heat_transfer_coefficient, surface_area
        )
        actual_value = resistance.value
        expected_value = float("inf")
        self.assertEqual(expected_value, actual_value)

        # Null heat transfer coefficient area case
        heat_transfer_coefficient = 0
        surface_area = 1
        resistance = ThermalResistance(
            heat_transfer_coefficient, surface_area
        )
        actual_value = resistance.value
        expected_value = float("inf")
        self.assertEqual(expected_value, actual_value)


class TestParallelResistances(unittest.TestCase):

    def test_compute_equivalent_resistance(self):

        coupling_method = ParallelResistances()

        # Normal case
        actual_value = (
            coupling_method.compute_equivalent_resistance(
                [1, 2, 3, 4]
            )
        )
        expected_value = 0.48
        self.assertAlmostEqual(
            actual_value, expected_value
        )

        # Case where one of the resistance is null
        actual_value = (
            coupling_method.compute_equivalent_resistance(
                [1, 2, 3, 0]
            )
        )
        expected_value = 0.0
        self.assertAlmostEqual(
            actual_value, expected_value
        )

        # Case where one of the resistance is infinite
        actual_value = (
            coupling_method.compute_equivalent_resistance(
                [5.0, 5.0, float("inf")]
            )
        )
        expected_value = 2.5
        self.assertAlmostEqual(
            actual_value, expected_value
        )


class TestSeriesResistances(unittest.TestCase):

    def test_compute_equivalent_resistance(self):

        coupling_method = SeriesResistances()

        # Normal case
        actual_value = (
            coupling_method.compute_equivalent_resistance(
                [1, 2, 3, 4]
            )
        )
        expected_value = 10.0
        self.assertAlmostEqual(
            actual_value, expected_value
        )

        # Case where one of the resistance is null
        actual_value = (
            coupling_method.compute_equivalent_resistance(
                [1, 2, 3, 0]
            )
        )
        expected_value = 6.0
        self.assertAlmostEqual(
            actual_value, expected_value
        )

        # Case where one of the resistance is infinite
        actual_value = (
            coupling_method.compute_equivalent_resistance(
                [5.0, 5.0, float("inf")]
            )
        )
        expected_value = float("inf")
        self.assertAlmostEqual(
            actual_value, expected_value
        )
 

if __name__ == "__main__":
    unittest.main()

# End
