from dataclasses import dataclass
import unittest
from main import FuelTank

from src.thermodynamics.internal_models import InternalModel, SingleZoneModel
from src.thermodynamics.tank_states import TankState
from src.thermodynamics.thermal_resistances import ParallelResistances


@dataclass
class FuelTank:
    volume: float
    characteristic_height: float

    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        return 33.3

    def compute_gas_wetted_surface(self, fuel_height: float) -> float:
        return 25.5


class TestInternalModel(unittest.TestCase):

    def test_compute_equivalent_resistances(self):

        @dataclass
        class ThermalResistance:
            value: float

        class InternalModelTester(InternalModel):

            def get_thermal_resistances(
                self,
                tank: FuelTank,
                tank_state: TankState,
                surface_temperature: float
            ) -> list[ThermalResistance]:
                return resistances

        resistances = [
            ThermalResistance(1),
            ThermalResistance(2),
            ThermalResistance(3)
        ]
        expected_value = (
            ParallelResistances().compute_equivalent_resistance(
                [resistance.value for resistance in resistances]
            )
        )

        internal_model = InternalModelTester()
        actual_value = internal_model.compute_equivalent_resistance(
            None, None, None
        )
        self.assertEqual(expected_value, actual_value)


class TestSingleZoneModel(unittest.TestCase):

    def setUp(self) -> None:
        self.temperature = None
        self.pressure = 1.4e5
        self.fill = 0.5
        self.fuel_height = 1.25
        self.volume = 85.85
        self.tank_state = TankState(
            self.temperature,
            self.pressure,
            self.fill,
            self.fuel_height,
            self.volume
        )

        self.characteristic_height = 5.5
        self.tank = FuelTank(self.volume, self.characteristic_height)

        self.surface_temperature = 30

        self.model = SingleZoneModel()

        
        self.expected_liquid_resistance = 0.01593310022491519
        self.expected_gas_resistance = 0.51766588713377

    def test_create_liquid_resistance(self):

        resistance = self.model.create_liquid_resistance(
            self.tank, self.tank_state, self.surface_temperature
        )
        actual_value = resistance.value
        self.assertEqual(self.expected_liquid_resistance, actual_value)

    def test_create_gas_resistance(self):

        resistance = self.model.create_gas_resistance(
            self.tank, self.tank_state, self.surface_temperature
        )
        actual_value = resistance.value
        self.assertEqual(self.expected_gas_resistance, actual_value)

    def test_create_two_phase_thermal_resistances(self):
        resistances = self.model.create_two_phase_thermal_resistances(
            self.tank, self.tank_state, self.surface_temperature
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
            self.tank, self.tank_state, self.surface_temperature
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
        self.tank_state.fuel_height = 0
        expected_value = [0.669920559820173]
        resistances = self.model.get_thermal_resistances(
            self.tank, self.tank_state, self.surface_temperature
        )
        actual_value = [
            resistance.value
            for resistance in resistances
        ]
        self.assertEqual(expected_value, actual_value)

        # Liquid case
        self.tank_state.fuel_height = self.characteristic_height
        expected_value = [0.01593310022491519]
        resistances = self.model.get_thermal_resistances(
            self.tank, self.tank_state, self.surface_temperature
        )
        actual_value = [
            resistance.value
            for resistance in resistances
        ]
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
