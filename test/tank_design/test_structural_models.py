from __future__ import annotations


from dataclasses import dataclass
import math
import unittest
from unittest.mock import patch

from src.tank_design.structural_models import CompositeCylinder, CompositeSphericalEndCap, CompositeModel, MetalCylinder, MetalSphericalEndCap, StructuralModelFactory


@dataclass
class Material:
    failure_stress: float
    winding_angle: float
    type: str

    @classmethod
    def test_material(cls) -> Material:
        failure_stress = 859.546e6
        orientation = math.radians(19.5)
        type = "metal"
        
        return cls(failure_stress, orientation, type)


@dataclass
class TankSection:
    radius: float
    type: str
    material: Material

    @classmethod
    def test_section(cls) -> TankSection:
        material = Material.test_material()
        radius = 0.3 / 2
        type = "cylinder"

        return cls(radius, type, material)


class TestMetalSphericalEndCap(unittest.TestCase):

    def test_compute_thickness(self):

        section = TankSection.test_section()
        model = MetalSphericalEndCap()
        pressure = 500e6
        expected_value = (
            pressure * section.radius / section.material.failure_stress
        )
        actual_value = model.compute_thickness(section, pressure)
        self.assertEqual(expected_value, actual_value)


class TestMetalCylinder(unittest.TestCase):

    def test_compute_thickness(self):

        section = TankSection.test_section()
        model = MetalCylinder()
        pressure = 500e6
        expected_value = (
            pressure * section.radius / section.material.failure_stress
        )
        actual_value = model.compute_thickness(section, pressure)
        self.assertEqual(expected_value, actual_value)


class TestCompositeModel(unittest.TestCase):

    @patch(
        "src.tank_design.structural_models." \
            "CompositeModel.__abstractmethods__",
        set()
    )
    def setUp(self) -> None:

        self.section = TankSection.test_section()
        self.model = CompositeModel()
        self.pressure = 22e6

    def test_hoop_stress(self):

        expected_value = 3300e3
        actual_value = self.model.hoop_stress(
            self.pressure, self.section.radius
        )
        self.assertEqual(expected_value, actual_value)

    def test_meridional_stress(self):

        expected_value = 1650e3
        actual_value = self.model.meridional_stress(
            self.pressure, self.section.radius
        )
        self.assertEqual(expected_value, actual_value)

    def test_helical_thickness(self):
        
        expected_value = 2.16e-3
        actual_value = self.model.helical_thickness(
            self.pressure, self.section.radius, self.section.material
        )
        self.assertAlmostEqual(expected_value, actual_value, places=5)


class TestCompositeEndCap(unittest.TestCase):

    def setUp(self) -> None:

        self.section = TankSection.test_section()
        self.model = CompositeSphericalEndCap()
        self.pressure = 22e6

    def test_compute_thickness(self):

        expected_value = 2.16e-3
        actual_value = self.model.compute_thickness(
            self.section, self.pressure
        )
        self.assertAlmostEqual(expected_value, actual_value, places=5)


class TestCompositeCylinder(unittest.TestCase):

    def setUp(self) -> None:

        self.section = TankSection.test_section()
        self.model = CompositeCylinder()
        self.pressure = 22e6

    def test_hoop_thickness(self):

        expected_value = 3.6e-3
        actual_value = self.model.hoop_thickness(
            self.pressure, self.section.radius, self.section.material
        )
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_compute_thickness(self):

        expected_value = 5.758e-3
        actual_value = self.model.compute_thickness(
            self.section, self.pressure
        )
        self.assertAlmostEqual(expected_value, actual_value, places=5)


class TestStructuralModel(unittest.TestCase):

    def test_get_structural_model(self):
        radius = 8.5
        failure_stress = 850e6
        metal = Material(failure_stress, None, "metal")
        composite = Material(failure_stress, math.pi/4, "composite")
        cylindrical = "cylinder"
        spherical_end_cap = "spherical_end_cap"
        factory = StructuralModelFactory()

        # Test no material
        section = TankSection(radius, cylindrical, None)
        with self.assertRaises(ValueError) as context:
            factory.get_structural_model(section)
        self.assertTrue(
            "Tank section has no material" in str(context.exception)
        )

        # Test metal material with cylinder
        section = TankSection(radius, cylindrical, metal)
        model = factory.get_structural_model(section)
        self.assertIsInstance(model, MetalCylinder)

        # Test metal material with sphere
        section = TankSection(radius, spherical_end_cap, metal)
        model = factory.get_structural_model(section)
        self.assertIsInstance(model, MetalSphericalEndCap)

        # Test metal material with cylinder
        section = TankSection(radius, cylindrical, composite)
        model = factory.get_structural_model(section)
        self.assertIsInstance(model, CompositeCylinder)

        # Test metal material with cylinder
        section = TankSection(radius, spherical_end_cap, composite)
        model = factory.get_structural_model(section)
        self.assertIsInstance(model, CompositeSphericalEndCap)

        # Test no material
        section = TankSection(radius, "test", metal)
        with self.assertRaises(ValueError) as context:
            factory.get_structural_model(section)
        self.assertTrue(
            f"{section.material.type} and {section.type}" \
                "not supported in StructuralModelFactory"
            in str(context.exception)
        )
            

if __name__ == "__main__":
    unittest.main()


# End
