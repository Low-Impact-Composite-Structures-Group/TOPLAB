import unittest
from dataclasses import dataclass
import numpy as np
from src.thermodynamics.external_models import (ExternalModel,
                                                ForcedConvectionModel, NaturalConvectionModel)
from src.thermodynamics.tank_states import TankState
from src.thermodynamics.thermal_resistances import SeriesResistances


@dataclass
class FuelTank:
    characteristic_length: float
    exposed_surface: float
    characteristic_height: float
    surface_area: float

    @classmethod
    def test_tank(cls):
        characteristic_length = 0.5
        exposed_surface = 22.2
        characteristic_height = 12.3
        surface_area = 44.1
        
        return cls(
            characteristic_length,
            exposed_surface,
            characteristic_height,
            surface_area
        )


@dataclass
class Ambient:
    density: float
    prantl_number: float
    dynamic_viscosity: float
    thermal_conductivity: float
    temperature: float
    thermal_expansion_coefficient: float
    kinematic_viscosity: float

    @classmethod
    def test_ambient(cls):
        density = 1.225
        prantl_number = 0.5
        dynamic_viscosity = 12.3
        thermal_conductivity = 12.21
        temperature = 20.15
        thermal_expansion_coefficient = 1.225
        kinematic_viscosity = 0.333
        
        return cls(
            density,
            prantl_number,
            dynamic_viscosity,
            thermal_conductivity,
            temperature,
            thermal_expansion_coefficient,
            kinematic_viscosity
        )


@dataclass
class MissionSection:
    ambient: Ambient
    flight_speed: float

    @classmethod
    def test_section(cls):
        ambient = Ambient.test_ambient()
        flight_speed = 0.85

        return cls(ambient, flight_speed)


class TestExternalModel(unittest.TestCase):

    ...


class TestForcedExternalConvection(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = FuelTank.test_tank()
        self.mission_section = MissionSection.test_section()
        self.model = ForcedConvectionModel()
        self.surface_temperature = 333.3

    def test_get_convective_motions(self):

        thermal_resistances = self.model.get_convective_motions(
            self.tank, self.mission_section, self.surface_temperature
        )
        actual_values = [
            resistance.value
            for resistance in thermal_resistances
        ]
        expected_value = [0.8604791020161002, 0.0017135709448913136]
        self.assertEqual(expected_value, actual_values)


class TestNaturalConvectionModel(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = FuelTank.test_tank()
        self.mission_section = MissionSection.test_section()
        self.model = NaturalConvectionModel()
        self.surface_temperature = 333.3

    def test_get_convective_motions(self):

        thermal_resistances = self.model.get_convective_motions(
            self.tank, self.mission_section, self.surface_temperature
        )
        actual_values = [
            resistance.value
            for resistance in thermal_resistances
        ]
        expected_value = [0.0016648054377864617, 0.0017135709448913136]
        self.assertEqual(expected_value, actual_values)

    def test_define_radiation_resistance(self):

        expected_value = 0.010682470675914857
        actual_value = self.model.define_radiation_resistance(
            self.tank, self.mission_section, self.surface_temperature
        )
        self.assertEqual(expected_value, actual_value)

    def test_equivalent_convection_resistance(self):
        resistances = [0.0016648054377864617, 0.0017135709448913136]
        expected_value = (
            np.product(resistances) / sum(resistances)
        )
        actual_value = self.model.equivalent_convection_resistance(
            self.tank, self.mission_section, self.surface_temperature
        )
        self.assertEqual(expected_value, actual_value)

    def test_compute_equivalent_resistance(self):
        resistances = [0.0008444181180389327, 0.010682470675914857]
        expected_value = sum(resistances)
        actual_value = self.model.compute_equivalent_resistance(
            self.tank, self.mission_section, self.surface_temperature
        )
        self.assertEqual(expected_value, actual_value)

        




if __name__ == "__main__":
    unittest.main()


# End
