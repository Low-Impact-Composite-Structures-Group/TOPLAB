import unittest
from src.fluids.hydrogen_retrievers import HydrogenRetriever

from src.thermodynamics.tank_states import TankState


class TestTankState(unittest.TestCase):

    def setUp(self) -> None:

        self.temperature = None
        self.pressure = 1.4e5
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        self.fill = 0.5
        self.fuel_height = 2.5
        self.volume = 110
        self.state = TankState(
            self.temperature,
            self.pressure,
            self.fill,
            self.fuel_height,
            self.volume,
        )

    def test__post_init__(self):

        # Test version where the two phase hydrogen is defined with 
        # pressure
        expected_temperature = self.hydrogen.temperature
        actual_temperature = self.state.temperature
        self.assertEqual(expected_temperature, actual_temperature)

        # Test case where the hydrogen is defined with temperature
        temperature = self.hydrogen.temperature
        fill = 0.5
        fuel_height = 2.5
        volume = 110
        self.state = TankState(
            temperature,
            None,
            fill,
            fuel_height,
            volume,
        )
        expected_value = 1.4e5
        actual_value = self.state.pressure
        self.assertAlmostEqual(expected_value, actual_value, places=0)

    def test_liquid_mass(self):
        
        expected_value = (
            self.hydrogen.liquid.density
            * self.volume
            * self.fill
        )
        actual_value = self.state.liquid_mass
        self.assertEqual(expected_value, actual_value)

        self.state.fill = 0
        expected_value = 0
        actual_value = self.state.liquid_mass
        self.assertEqual(expected_value, actual_value)

        # Case where the hydrogen is fully filled and thus sole liquid
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            100e5, 15
        )
        self.state.hydrogen = hydrogen
        self.state.fill = 1
        expected_value = (
            hydrogen.density * self.volume
        )
        actual_value = self.state.liquid_mass
        self.assertEqual(expected_value, actual_value)

    def test_gas_mass(self):
        
        expected_value = (
            self.hydrogen.gas.density
            * self.volume
            * (1 - self.fill)
        )
        actual_value = self.state.gas_mass
        self.assertEqual(expected_value, actual_value)

        self.state.fill = 1
        expected_value = 0
        actual_value = self.state.gas_mass
        self.assertEqual(expected_value, actual_value)

        # Case where the hydrogen is fully filled and thus sole liquid
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            300e5, 70
        )
        self.state.hydrogen = hydrogen
        self.state.fill = 0
        expected_value = (
            hydrogen.density * self.volume
        )
        actual_value = self.state.gas_mass
        self.assertEqual(expected_value, actual_value)

    def test_fuel_mass(self):

        liquid_mass = (
            self.hydrogen.liquid.density
            * self.volume
            * self.fill
        )
        gas_mass = (
            self.hydrogen.gas.density
            * self.volume
            * (1 - self.fill)
        )
        expected_value = liquid_mass + gas_mass
        actual_value = self.state.fuel_mass
        self.assertEqual(expected_value, actual_value)

    def test_set_state_derivatives(self):

        expected_value = "Test"
        actual_value = self.state.set_state_derivatives(expected_value)
        self.assertEqual(expected_value, actual_value)

    def test_set_thermal_capacity(self):

        expected_value = "Test"
        actual_value = self.state.set_thermal_capacity(expected_value)
        self.assertEqual(expected_value, actual_value)

    def test_set_heat_flux(self):

        expected_value = "Test"
        actual_value = self.state.set_heat_flux(expected_value)
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
