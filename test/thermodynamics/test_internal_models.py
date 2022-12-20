from dataclasses import dataclass
import unittest

from src.thermodynamics.internal_models import InternalModel, SingleZoneModel, ThreeZoneModel
from src.thermodynamics.tank_states import TankState
from src.thermodynamics.thermal_resistances import ParallelResistances
import numpy as np

@dataclass
class FuelTank:
    volume: float
    characteristic_height: float

    @classmethod
    def test_tank(cls):
        characteristic_height = 5.5
        volume = 85.85

        return cls(volume, characteristic_height)

    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        return 33.3

    def compute_gas_wetted_surface(self, fuel_height: float) -> float:
        return 25.5

    def compute_zone_1_length(self, fuel_height: float) -> float:
        return 5
    
    def compute_zone_2_length(self, fuel_height: float) -> float:
        return 2
    
    def compute_zone_3_length(self, fuel_height: float) -> float:
        return 3
    
    def compute_zone_1_area(self, fuel_height: float) -> float:
        return 55
    
    def compute_zone_2_area(self, fuel_height: float) -> float:
        return 99
    
    def compute_zone_3_area(self, fuel_height: float) -> float:
        return 31


@dataclass
class ConvectiveMedium:
    # density: float
    prantl_number: float
    # dynamic_viscosity: float
    thermal_conductivity: float
    temperature: float
    thermal_expansion_coefficient: float
    kinematic_viscosity: float

    @classmethod
    def test_medium(cls):
        # density = 1.225
        prantl_number = 0.5
        # dynamic_viscosity = 12.3
        thermal_conductivity = 12.21
        temperature = 20.15
        thermal_expansion_coefficient = 1.225
        kinematic_viscosity = 0.333
        
        return cls(
            # density,
            prantl_number,
            # dynamic_viscosity,
            thermal_conductivity,
            temperature,
            thermal_expansion_coefficient,
            kinematic_viscosity
        )


@dataclass
class Hydrogen:
    liquid: ConvectiveMedium
    gas: ConvectiveMedium
    
    @classmethod
    def test_hydrogen(cls):
        liquid = ConvectiveMedium.test_medium()
        gas = ConvectiveMedium.test_medium()

        return cls(liquid, gas)


@dataclass
class TankState:
    fuel_height: float
    hydrogen: Hydrogen
    is_full: bool
    is_empty: bool

    @classmethod
    def test_state(cls):
        fuel_height = 4
        hydrogen = Hydrogen.test_hydrogen()
        is_full = False
        is_empty = False

        return cls(fuel_height, hydrogen, is_full, is_empty)


class TestInternalModel(unittest.TestCase):

    ...


class TestSingleZoneModel(unittest.TestCase):

    def setUp(self) -> None:

        self.tank = FuelTank.test_tank()
        self.state = TankState.test_state()
        self.model = SingleZoneModel()
        self.surface_temperature = 30

        self.expected_liquid_resistance = 0.007230400634421978
        self.expected_gas_resistance = 0.0002833912868514944

    def test_create_liquid_resistance(self):

        resistances = self.model.create_liquid_resistance(
            self.tank, self.state, self.surface_temperature
        )
        actual_value = resistances[0].value
        expected_value = self.expected_liquid_resistance
        self.assertEqual(expected_value, actual_value)

    def test_create_gas_resistance(self):

        resistance = self.model.create_gas_resistance(
            self.tank, self.state, self.surface_temperature
        )
        actual_value = resistance.value
        expected_value = self.expected_gas_resistance
        self.assertEqual(expected_value, actual_value)

    def test_create_two_phase_thermal_resistances(self):
        resistances = self.model.create_two_phase_thermal_resistances(
            self.tank, self.state, self.surface_temperature
        )
        actual_value = [
            resistance.value
            for resistance in resistances
        ]
        expected_value = [
            self.expected_liquid_resistance,
            self.expected_gas_resistance
        ]
        self.assertEqual(expected_value, actual_value)

    def test_get_thermal_resistances(self):

        # Two phase case
        resistances = self.model.get_thermal_resistances(
            self.tank, self.state, self.surface_temperature
        )
        actual_value = [
            resistance.value
            for resistance in resistances
        ]
        expected_value = [
            self.expected_liquid_resistance,
            self.expected_gas_resistance
        ]
        self.assertEqual(expected_value, actual_value)

        # Gas case
        self.state.is_empty = True
        expected_value = [0.0002833912868514944]
        resistances = self.model.get_thermal_resistances(
            self.tank, self.state, self.surface_temperature
        )
        actual_value = [
            resistance.value
            for resistance in resistances
        ]
        self.assertEqual(expected_value, actual_value)

        # Liquid case
        self.state.is_full = True
        self.state.is_empty = False
        self.state.fuel_height = self.tank.characteristic_height
        expected_value = [0.00723040063442198]
        resistances = self.model.get_thermal_resistances(
            self.tank, self.state, self.surface_temperature
        )
        actual_value = [
            resistance.value
            for resistance in resistances
        ]
        self.assertEqual(expected_value, actual_value)

    def test_compute_equivalent_resistances(self):

        resistances = [
            self.expected_liquid_resistance,
            self.expected_gas_resistance
        ]
        expected_value = np.product(resistances) / sum(resistances)
        actual_value = self.model.compute_equivalent_resistance(
            self.tank, self.state, self.surface_temperature
        )
        self.assertEqual(expected_value, actual_value)


class TestThreeZoneModel(unittest.TestCase):

    def setUp(self) -> None:

        self.tank = FuelTank.test_tank()
        self.state = TankState.test_state()
        self.model = ThreeZoneModel()
        self.surface_temperature = 30

    def test_create_liquid_resistance(self):

        resistances = self.model.create_liquid_resistance(
            self.tank, self.state, self.surface_temperature
        )
        resistance_values = [
            resistance.value
            for resistance in resistances
        ]
        actual_to_test = (
            np.prod(resistance_values) / sum(resistance_values)
        )
        expected_to_test = 8.502898167721188e-08
        self.assertEqual(expected_to_test, actual_to_test)


if __name__ == "__main__":
    unittest.main()


# End
