from dataclasses import dataclass
import unittest
import numpy as np
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


@dataclass
class MissionSection:
    temperature: float


@dataclass
class TankState:
    temperature: float


class FuelTank:
    ...


class Insulation:
    
    def compute_thermal_resistances(
        self,
        temperatures: list[float],
        tank: FuelTank
    ) -> list[float]:
        return [8.5] * 12


class InternalModel:

    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> float:
        return 11.11


class ExternalModel():

    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> float:
        return 11.89


class TestThermodynamicModel(unittest.TestCase):

    def test_construct_a_matrix(self):

        # Single layer case
        expected_value = np.array([
            [1, 0], [-1, 1], [0, -1]
        ])
        a = ThermodynamicModel.construct_a_matrix(1)
        np.testing.assert_array_equal(a, expected_value)

        # Multiple layer case, which should be applicable to multiple
        expected_value = [
            [1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]
        ]
        a = ThermodynamicModel.construct_a_matrix(2)
        np.testing.assert_array_equal(a, expected_value)

    def test_construct_y_vector(self):

        resistances = [3, 2, 6]
        heat_flux = 11
        fuel_temperature = 20
        ambient_temperature = 222

        y_vector = ThermodynamicModel.construct_y_vector(
            resistances,
            fuel_temperature,
            ambient_temperature,
            heat_flux
        )
        expected_value = np.array([[53], [22], [66-222]])
        np.testing.assert_array_equal(y_vector, expected_value)

    def test_compute_total_tank_heat_flux(self):

        ambient_temperature = 300.15
        fuel_temperature = 33
        total_resistance = 85
        expected_value = (
            (ambient_temperature - fuel_temperature)
            / total_resistance
        )
        actual_value = ThermodynamicModel.compute_total_tank_heat_flux(
            ambient_temperature, fuel_temperature, total_resistance
        )
        self.assertEqual(expected_value, actual_value)

    def test_temperatures_have_converged(self):
        temps_1 = [0, 1, 2]
        temps_2 = [0, 1, 2]
        self.assertTrue(
            ThermodynamicModel.temperatures_have_converged(
                temps_1, temps_2
            )
        )

        temps_1 = [0, 1, 5]
        temps_2 = [0, 1, 2]
        self.assertFalse(
            ThermodynamicModel.temperatures_have_converged(
                temps_1, temps_2
            )
        )

        temps_1 = [0, 1, 2.09]
        temps_2 = [0, 1, 2]
        self.assertTrue(
            ThermodynamicModel.temperatures_have_converged(
                temps_1, temps_2
            )
        )

        temps_1 = [0, 1, 2.2]
        temps_2 = [0, 1, 2]
        self.assertTrue(
            ThermodynamicModel.temperatures_have_converged(
                temps_1, temps_2
            )
        )

    def test_compute_new_temperatures(self):

        a_matrix = np.array([
            [1, 0], [-1, 1], [0, -1]
        ])
        y_vector = np.array([[53], [22], [66-222]])
        expected_value = [80, 129]
        new_temperatures = ThermodynamicModel.compute_new_temperatures(
            a_matrix, y_vector
        )
        np.testing.assert_array_equal(
            new_temperatures, expected_value
        )

    def test_define_initial_temperatures(self):

        hot_temperature = 300
        cold_temperature = 15

        # Single layer case
        model = ThermodynamicModel(
            None, None, None, insulation_layers=1
        )
        expected_value = [cold_temperature, hot_temperature]
        actual_value = model.define_initial_temperatures(
            cold_temperature, hot_temperature
        )
        self.assertEqual(expected_value, actual_value)

        # Test with default of 12 layers
        model = ThermodynamicModel(None, None, None)
        expected_value = [
            15,
            38.75,
            62.5,
            86.25,
            110,
            133.75,
            157.5,
            181.25,
            205,
            228.75,
            252.5,
            276.25,
            300
        ]
        actual_value = model.define_initial_temperatures(
            cold_temperature, hot_temperature
        )
    
        # Test with 30 layers
        model = ThermodynamicModel(
            None, None, None, insulation_layers=30
        )
        expected_value = [
            15,
            24.5,
            34,
            43.5,
            53,
            62.5,
            72,
            81.5,
            91,
            100.5,
            110,
            119.5,
            129,
            138.5,
            148,
            157.5,
            167,
            176.5,
            186,
            195.5,
            205,
            214.5,
            224,
            233.5,
            243,
            252.5,
            262,
            271.5,
            281,
            290.5,
            300
        ]
        actual_value = model.define_initial_temperatures(
            cold_temperature, hot_temperature
        )
        for expected, actual in zip(expected_value, actual_value):
            self.assertAlmostEqual(expected, actual)

    def test_compute_heat_flux(self):
        
        tank_temperature = 25.0
        ambient_temperature = 300.15
        model = ThermodynamicModel(
            InternalModel(), ExternalModel(), Insulation()
        )
        expected_value = 2.2011999999999996
        actual_value, _ = model.compute_heat_flux(
            FuelTank(),
            TankState(tank_temperature),
            MissionSection(ambient_temperature)
        )
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End