import unittest
from dataclasses import dataclass

from src.thermodynamics.external_models import (ExternalModel,
                                                ForcedConvectionModel)
from src.thermodynamics.tank_states import TankState
from src.thermodynamics.thermal_resistances import SeriesResistances


@dataclass
class FuelTank:
    characteristic_length: float
    exposed_surface: float


@dataclass
class Ambient:
    density: float = 1.225
    prantl_number: float = 0.5
    dynamic_viscosity: float = 12.3
    thermal_conductivity: float = 12.21


@dataclass
class MissionSection:
    ambient: Ambient
    flight_speed: float


class TestExternalModel(unittest.TestCase):

    def test_compute_equivalent_resistance(self):
        
        @dataclass
        class ThermalResistance:
            value: float

        class ExternalModelTester(ExternalModel):

            def get_convective_motions(
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
            SeriesResistances().compute_equivalent_resistance(
                [resistance.value for resistance in resistances]
            )
        )

        internal_model = ExternalModelTester()
        actual_value = internal_model.compute_equivalent_resistance(
            None, None, None
        )
        self.assertEqual(expected_value, actual_value)


class TestForcedExternalConvection(unittest.TestCase):

    def test_get_thermal_resistances(self):

        tank = FuelTank(0.5, 22.2)
        
        ambient = Ambient()
        mission_section = MissionSection(ambient, 85.5)

        surface_temperature = 333.3

        model = ForcedConvectionModel()

        expected_value = [0.021513079201850624]
        thermal_resistances = model.get_convective_motions(
            tank, mission_section, surface_temperature
        )
        actual_values = [
            resistance.value
            for resistance in thermal_resistances
        ]
        self.assertEqual(expected_value, actual_values)


if __name__ == "__main__":
    unittest.main()


# End
