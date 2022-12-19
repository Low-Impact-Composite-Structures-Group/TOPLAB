import math
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from src.efficiencies.efficiency_computers import Efficiency, GravimetricEfficiency, HexagonVolumetricEfficiency, SquareVolumetricEfficiency, VolumetricEfficiency


@dataclass
class Insulation:
    thickness: float
    density: float

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


class TestEfficiency(unittest.TestCase):

    @patch.multiple(
        "src.efficiencies.efficiency_computers.Efficiency",
        __abstractmethods__=set()
    )
    def setUp(self):

        @dataclass
        class Computer(Efficiency):
            ...
        
        self.computer = Computer(
            Tank.test_tank(), Insulation.test_insulation()
        )

    def test_insulation_volume(self):
        tank = Tank.test_tank()
        insulation = Insulation.test_insulation()

        expected_value = tank.surface_area * insulation.thickness
        actual_value = self.computer.insulation_volume
        self.assertEqual(expected_value, actual_value)


class TestGravimetricEfficiency(unittest.TestCase):

    @patch.multiple(
        "src.efficiencies.efficiency_computers.Efficiency",
        __abstractmethods__=set()
    )
    def setUp(self):

        self.tank = Tank.test_tank()
        self.insulation = Insulation.test_insulation()
        self.initial_state = InitialState.test_initial_state()
        
        self.computer = GravimetricEfficiency(
            self.tank,
            self.insulation,
            self.initial_state
        )

    def test_insulation_mass(self):
        insulation_volume = (
            self.tank.surface_area * self.insulation.thickness
        )
        expected_value = insulation_volume * self.insulation.density
        actual_value = self.computer.insulation_mass
        self.assertEqual(expected_value, actual_value)

    def test_system_mass(self):
        expected_value = (
            self.tank.structural_mass + (
                self.insulation.density
                * self.tank.surface_area
                * self.insulation.thickness
            )
        )
        actual_value = self.computer.system_mass
        self.assertEqual(expected_value, actual_value)

    def test_fuel_mass(self):
        expected_value = self.initial_state.fuel_mass
        actual_value = self.computer.fuel_mass
        self.assertEqual(expected_value, actual_value)

    def test_efficiency(self):
        system_mass = (
            self.tank.structural_mass + (
                self.insulation.density
                * self.tank.surface_area
                * self.insulation.thickness
            )
        )
        expected_value = (
            self.initial_state.fuel_mass
            / (system_mass + self.initial_state.fuel_mass)
        )
        actual_value = self.computer.efficiency
        self.assertEqual(expected_value, actual_value)


class TestVolumetricEfficiency(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = Tank.test_tank()
        self.insulation = Insulation.test_insulation()
        self.initial_state = InitialState.test_initial_state()
        
        self.computer = VolumetricEfficiency(
            self.tank,
            self.insulation
        )

    def test_system_volume(self):

        expected_value = (
            self.tank.volume
            + self.tank.surface_area * self.insulation.thickness
            + self.tank.structural_volume
        )
        actual_value = self.computer.system_volume
        self.assertEqual(expected_value, actual_value)

    def test_fuel_volume(self):
        expected_value = self.tank.volume
        actual_value = self.computer.fuel_volume
        self.assertEqual(expected_value, actual_value)

    def test_efficiency(self):
        system_volume = (
            self.tank.volume
            + self.tank.surface_area * self.insulation.thickness
            + self.tank.structural_volume
        )
        expected_value = self.tank.volume / system_volume
        actual_value = self.computer.efficiency
        self.assertEqual(expected_value, actual_value)


class TestSquareVolumetricEfficiency(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = Tank.test_tank()
        self.insulation = Insulation.test_insulation()
        self.initial_state = InitialState.test_initial_state()
        
        self.computer = SquareVolumetricEfficiency(
            self.tank,
            self.insulation
        )

    def test_effective_radius(self):
        expected_value = (
            self.tank.thickness
            + self.tank.radius
            + self.insulation.thickness
        )
        actual_value = self.computer.effective_radius
        self.assertEqual(expected_value, actual_value)

    def test_effective_diameter(self):
        expected_value = (
            self.tank.thickness
            + self.tank.radius
            + self.insulation.thickness
        ) * 2
        actual_value = self.computer.effective_diameter
        self.assertEqual(expected_value, actual_value)

    def test_effective_area(self):
        expected_value = (
            (
                self.tank.thickness
                + self.tank.radius
                + self.insulation.thickness
            ) * 2
        ) ** 2
        actual_value = self.computer.effective_area
        self.assertEqual(expected_value, actual_value)

    def test_system_volume(self):
        effective_area = (
            (
                self.tank.thickness
                + self.tank.radius
                + self.insulation.thickness
            ) * 2
        ) ** 2
        expected_value = (effective_area * self.tank.total_length)
        actual_value = self.computer.system_volume
        self.assertEqual(expected_value, actual_value)


class TestHexagonVolumetricEfficiency(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = Tank.test_tank()
        self.insulation = Insulation.test_insulation()
        self.initial_state = InitialState.test_initial_state()
        
        self.computer = HexagonVolumetricEfficiency(
            self.tank,
            self.insulation
        )

    def test_effective_area(self):
        effective_radius = (
            self.tank.thickness
            + self.tank.radius
            + self.insulation.thickness
        )
        a = 2 * effective_radius * math.tan(math.pi / 12)
        expected_value = 3 * math.sqrt(3) / 2 * a ** 2
        actual_value = self.computer.effective_area
        self.assertEqual(expected_value, actual_value)






if __name__ == "__main__":
    unittest.main()


# End
