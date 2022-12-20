import unittest
from dataclasses import dataclass

from src.efficiencies.tank_performance import TankPerformance


@dataclass
class Insulation:
    thickness: float
    density: float

    def compute_volume(self, surface_area: float) -> float:
        return self.thickness * surface_area

    def compute_mass(self, surface_area: float) -> float:
        return self.compute_volume(surface_area) * self.density

    @classmethod
    def test_insulation(cls):
        thickness = 4e-2
        density = 50.0

        return cls(thickness, density)


@dataclass
class InitialState:
    fuel_mass: float
    fuel_volume: float

    @classmethod
    def test_initial_state(cls):
        fuel_mass = 85.85
        fuel_volume = 58.58

        return cls(fuel_mass, fuel_volume)


@dataclass
class TankStates:
    first_state: InitialState

    @classmethod
    def test_tank_states(cls):
        return cls(InitialState.test_initial_state())


@dataclass
class Tank:
    surface_area: float
    structural_mass: float
    volume: float
    structural_volume: float
    radius: float
    thickness: float
    total_length: float

    @classmethod
    def test_tank(cls):
        surface_area = 125.5
        structural_mass = 33.3
        volume = 22.22
        structural_volume = 123.4
        radius = 22.23
        thickness = 0.33
        total_length = 8.5

        return cls(
            surface_area,
            structural_mass,
            volume, 
            structural_volume,
            radius,
            thickness,
            total_length
        )


class TestTankPerformance(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = Tank.test_tank()
        self.insulation = Insulation.test_insulation()
        self.tank_states = TankStates.test_tank_states()
        self.performance = TankPerformance(
            self.tank, self.insulation, self.tank_states
        )

    def test_volumetric_efficiency(self):
        expected_value = 0.14750398300584172
        actual_value = self.performance.volumetric_efficiency
        self.assertEqual(expected_value, actual_value)

    def test_gravimetric_efficiency(self):
        expected_value = 0.23193300013508036
        actual_value = self.performance.gravimetric_efficiency
        self.assertEqual(expected_value, actual_value)

    def test_volume(self):
        expected_value = self.tank.volume
        actual_value = self.performance.volume
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
