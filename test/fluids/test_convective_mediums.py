import unittest

from src.fluids.convective_mediums import (ConvectiveMedium, Hydrogen,
                                           TwoPhaseHydrogen)
from src.fluids.hydrogen_retrievers import HydrogenRetriever


class TestConvectiveMedium(unittest.TestCase):

    def setUp(self) -> None:
        self.temperature = 288.15                       # [K]
        self.pressure = 101.325e3                       # [Pa]
        self.density = 1.225                            # [kg/m^3]
        self.dynamic_viscosity = 1.81e-5                # [kg/(m*s)]
        self.specific_heat_constant_pressure = 1.005e3  # [kJ/(kg*K)]
        self.thermal_conductivity = 0.025               # [W/(m*K)]

        self.medium = ConvectiveMedium(
            self.temperature,
            self.pressure,
            self.density,
            self.dynamic_viscosity,
            self.specific_heat_constant_pressure,
            self.thermal_conductivity
        )

    def test_thermal_expansion_coefficient(self):
        expected_value = 1 / self.temperature
        actual_value = self.medium.thermal_expansion_coefficient
        self.assertEqual(expected_value, actual_value)

    def test_kinematic_viscosity(self):

        expected_value = (
            self.medium.dynamic_viscosity
            / self.medium.density
        )
        actual_value = self.medium.kinematic_viscosity
        self.assertEqual(expected_value, actual_value)

    def test_compute_prantl_number(self):

        expected_value = (
            self.medium.specific_heat_constant_pressure
            * self.medium.dynamic_viscosity
            / self.medium.thermal_conductivity
        )
        actual_value = self.medium.prantl_number
        self.assertEqual(expected_value, actual_value)

    def test_speed_of_sound(self):

        expected_value = 340
        actual_value = self.medium.speed_of_sound
        self.assertAlmostEqual(expected_value, actual_value, places=0)


class TestHydrogen(unittest.TestCase):

    def setUp(self) -> None:
        self.temperature = 288.15                       # [K]
        self.pressure = 101.325e3                       # [Pa]
        self.density = 1.225                            # [kg/m^3]
        self.dynamic_viscosity = 1.81e-5                # [kg/(m*s)]
        self.specific_heat_constant_pressure = 1.005e3  # [kJ/(kg*K)]
        self.thermal_conductivity = 0.025               # [W/(m*K)]
        self.enthalpy = None
        self.internal_energy = None
        self.speed_of_sound_database = 85.85            # [m/s]
        self.dRho_dP = None
        self.dRho_dT = None
        self.dH_dP = None
        self.dH_dT = None
        self.dP_dT = None
        self.state = None

        self.medium = Hydrogen(
            self.temperature,
            self.pressure,
            self.density,
            self.dynamic_viscosity,
            self.specific_heat_constant_pressure,
            self.thermal_conductivity,
            self.enthalpy,
            self.internal_energy,
            self.speed_of_sound_database,
            self.dRho_dP,
            self.dRho_dT,
            self.dH_dP,
            self.dH_dT,
            self.dP_dT,
            self.state
        )
    
    def test_speed_of_sound(self):
        
        expected_value = self.speed_of_sound_database
        actual_value = self.medium.speed_of_sound
        self.assertEqual(expected_value, actual_value)

    def test_gas(self):

        # Supercritical
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            300e5, 70
        )
        self.assertEqual(hydrogen, hydrogen.gas)

        # Supercritical gas
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            1e5, 300
        )
        self.assertEqual(hydrogen, hydrogen.gas)

        # Gas
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            1e5, 30
        )
        self.assertEqual(hydrogen, hydrogen.gas)

        # Not valid
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            1e5, 15
        )
        with self.assertRaises(ValueError) as context:
            hydrogen.gas
        message = "Hydrogen not in gas phase"
        self.assertTrue(message in str(context.exception))

    def test_liquid(self):

        # Liquid
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            1e5, 15
        )
        self.assertEqual(hydrogen, hydrogen.liquid)

        # Not valid
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            1e5, 30
        )
        with self.assertRaises(ValueError) as context:
            hydrogen.liquid
        message = "Hydrogen not in liquid phase"
        self.assertTrue(message in str(context.exception))


class TestTwoPhaseHydrogen(unittest.TestCase):

    def setUp(self) -> None:
        self.temperature = 288.15                       # [K]
        self.pressure = 101.325e3                       # [Pa]
        self.density = 1.225                            # [kg/m^3]
        self.dynamic_viscosity = 1.81e-5                # [kg/(m*s)]
        self.specific_heat_constant_pressure = 1.005e3  # [J/(kg*K)]
        self.thermal_conductivity = 0.025               # [W/(m*K)]
        self.enthalpy = 399.4e3                         # [J/kg]
        self.internal_energy = None
        self.speed_of_sound_database = 85.85            # [m/s]
        self.dRho_dP = None
        self.dRho_dT = None
        self.dH_dP = None
        self.dH_dT = None
        self.dP_dT = None
        self.state = None

        self.medium = Hydrogen(
            self.temperature,
            self.pressure,
            self.density,
            self.dynamic_viscosity,
            self.specific_heat_constant_pressure,
            self.thermal_conductivity,
            self.enthalpy,
            self.internal_energy,
            self.speed_of_sound_database,
            self.dRho_dP,
            self.dRho_dT,
            self.dH_dP,
            self.dH_dT,
            self.dP_dT,
            self.state
        )
        self.two_phase_medium = TwoPhaseHydrogen(
            self.medium, self.medium, self.dP_dT
        )

    def test_pressure(self):

        expected_value = self.pressure
        actual_value = self.two_phase_medium.pressure
        self.assertEqual(expected_value, actual_value)

    def test_temperature(self):

        expected_value = self.temperature
        actual_value = self.two_phase_medium.temperature
        self.assertEqual(expected_value, actual_value)

    def test_heat_of_evaporation(self):

        expected_value = 0 
        actual_value = self.two_phase_medium.heat_of_evaporation
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
