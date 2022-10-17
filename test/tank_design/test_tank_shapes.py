import math
import unittest

from src.tank_design.tank_shapes import CylindricalBody, CylindricalTankSphericalCaps, SphericalEndCap, SphericalTank, Tank, bisection_method


class TestCylindricalBody(unittest.TestCase):

    def setUp(self) -> None:
        self.cylinder = CylindricalBody(
            radius=2.5,
            length=6.5
        )

    def test_volume(self):

        expected_value = 127.627201552
        self.assertAlmostEqual(
            self.cylinder.volume, expected_value
        )

    def test_surface_area(self):

        expected_value = 102.101761242
        self.assertAlmostEqual(
            self.cylinder.surface_area, expected_value
        )

    def test_compute_volume_section(self):

        fuel_height = 1.5
        expected_value = 32.2024
        self.assertAlmostEqual(
            self.cylinder.compute_volume_section(fuel_height),
            expected_value,
            places=4
        )

        self.assertAlmostEqual(
            self.cylinder.compute_volume_section(
                self.cylinder.radius * 2
            ),
            self.cylinder.volume
        )

        self.assertAlmostEqual(
            self.cylinder.compute_volume_section(0),
            0
        )

        with self.assertRaises(ValueError) as context:
            self.cylinder.compute_volume_section(-1)
        self.assertTrue(
            "Negative fuel height..." in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            self.cylinder.compute_volume_section(
                1.1 * self.cylinder.radius * 2
            )
        self.assertTrue(
            "Fuel height higher than tank..." in str(context.exception)
        )
    
    def test_compute_fuel_amplitude_angle(self):

        # No fuel present
        fuel_height = 0
        expected_value = 0
        self.assertAlmostEqual(
            self.cylinder.compute_fuel_amplitude_angle(fuel_height),
            expected_value
        )

        # Fuel lower than half the tank
        fuel_height = 1.5
        expected_value = 2.31855896
        self.assertAlmostEqual(
            self.cylinder.compute_fuel_amplitude_angle(fuel_height),
            expected_value
        )

        # Fuel at half the tank
        fuel_height = self.cylinder.radius
        expected_value = math.pi
        self.assertAlmostEqual(
            self.cylinder.compute_fuel_amplitude_angle(fuel_height),
            expected_value
        )

        # Fuel above half the tank
        fuel_height = 4.5
        expected_value = 4.99618308918
        self.assertAlmostEqual(
            self.cylinder.compute_fuel_amplitude_angle(fuel_height),
            expected_value
        )

        with self.assertRaises(ValueError) as context:
            self.cylinder.compute_fuel_amplitude_angle(
                1.1 * self.cylinder.radius * 2
            )
        self.assertTrue(
            "Fuel height larger than diameter..."
            in str(context.exception)
        )

    def test_compute_wetted_area(self):

        fuel_height = 1.5
        # https://www.omnicalculator.com/math/arc-length
        arc_length = 5.7964
        expected_value = arc_length * self.cylinder.length
        self.assertAlmostEqual(
            self.cylinder.compute_wetted_surface(fuel_height),
            expected_value,
            places=4
        )


class TestSphericalEndCap(unittest.TestCase):

    def setUp(self) -> None:
        self.end_cap = SphericalEndCap(radius=2.5)
    
    def test_surface_area(self):

        expected_value = 39.26991
        self.assertAlmostEqual(
            self.end_cap.surface_area, expected_value, places=5
        )
    
    def test_volume(self):

        expected_value = 32.72492
        self.assertAlmostEqual(
            self.end_cap.volume, expected_value, places=5
        )

    def test_compute_volume_section(self):

        expected_value = 7.06858347058
        fuel_height = 1.5
        self.assertAlmostEqual(
            self.end_cap.compute_volume_section(fuel_height),
            expected_value
        )

        self.assertAlmostEqual(
            self.end_cap.compute_volume_section(self.end_cap.radius),
            self.end_cap.volume / 2
        )

        self.assertAlmostEqual(
            self.end_cap.compute_volume_section(
                self.end_cap.radius * 2
            ),
            self.end_cap.volume
        )

        self.assertAlmostEqual(
            self.end_cap.compute_volume_section(0), 0
        )

        with self.assertRaises(ValueError) as context:
            self.end_cap.compute_volume_section(-1)
        self.assertTrue(
            "Negative fuel height..." in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            self.end_cap.compute_volume_section(
                1.1 * self.end_cap.radius * 2
            )
        self.assertTrue(
            "Fuel height higher than diameter..."
            in str(context.exception)
        )  

    def test_compute_wetted_surface(self):

        self.assertAlmostEqual(
            self.end_cap.compute_wetted_surface( 0),
            0
        )

        self.assertAlmostEqual(
            self.end_cap.compute_wetted_surface(self.end_cap.radius * 2),
            self.end_cap.surface_area
        )

        self.assertAlmostEqual(
            self.end_cap.compute_wetted_surface(self.end_cap.radius),
            self.end_cap.surface_area / 2
        )

        expected_value = 11.780972451
        fuel_height = 1.5
        self.assertAlmostEqual(
            self.end_cap.compute_wetted_surface(fuel_height),
            expected_value
        )


class TestTank(unittest.TestCase):

    def setUp(self) -> None:
        radius = 2.5
        length = 6.5
        self.end_cap = SphericalEndCap(radius)
        self.body = CylindricalBody(radius, length)
        tank_sections = [self.end_cap, self.body, self.end_cap]
        self.tank = Tank()
        self.tank.set_sections(tank_sections)

    def test_volume(self):
        cap_volume = 32.72492
        body_volume = 127.627201552
        expected_value = 2 * cap_volume + body_volume
        self.assertAlmostEqual(
            self.tank.volume, expected_value, places=4
        )

    def test_surface_area(self):
        cap_area = 39.26991
        body_area = 102.101761242
        expected_value = 2 * cap_area + body_area
        self.assertAlmostEqual(
            self.tank.surface_area, expected_value, places=4
        )

    def test_compute_fuel_volume(self):

        fuel_height = 1.5
        cap_volume = self.end_cap.compute_volume_section(fuel_height)
        body_volume = self.body.compute_volume_section(fuel_height)
        fuel_volume = 2 * cap_volume + body_volume
        self.assertAlmostEqual(
            self.tank.compute_fuel_volume(fuel_height),
            fuel_volume
        )

    def test_compute_fuel_wetted_area(self):

        fuel_height = 1.5
        cap_area = self.end_cap.compute_wetted_surface(fuel_height)
        body_area = self.body.compute_wetted_surface(fuel_height)
        wetted_surface = 2 * cap_area + body_area
        self.assertAlmostEqual(
            self.tank.compute_fuel_wetted_surface(fuel_height),
            wetted_surface
        )

    def test_compute_gas_wetted_area(self):
        fuel_height = 1.5
        cap_area = self.end_cap.compute_wetted_surface(fuel_height)
        body_area = self.body.compute_wetted_surface(fuel_height)
        fuel_wetted_surface = 2 * cap_area + body_area
        expected_value = (
            self.end_cap.surface_area * 2 + self.body.surface_area
            - fuel_wetted_surface
        )
        self.assertAlmostEqual(
            self.tank.compute_gas_wetted_surface(fuel_height),
            expected_value
        )


class TestCylindricalTankSphericalCaps(unittest.TestCase):

    def setUp(self) -> None:
        self.radius = 2.5
        body_length = 6.5
        total_length = body_length + 2 * self.radius
        self.tank = CylindricalTankSphericalCaps(
            self.radius, total_length
        )
    
    def test_compute_fuel_height(self):

        fuel_volume = self.tank.volume / 2
        self.assertAlmostEqual(
            self.tank.compute_fuel_height(fuel_volume),
            self.tank.radius
        )

        self.assertAlmostEqual(
            self.tank.compute_fuel_height(0), 0, places=4
        )

        self.assertAlmostEqual(
            self.tank.compute_fuel_height(self.tank.volume),
            self.tank.radius * 2, places=4
        )

    def test_diameter(self):

        self.assertEqual(
            self.tank.diameter, self.radius * 2
        )

    def test_characteristic_dimension(self):

        self.assertEqual(
            self.tank.characteristic_height, self.radius * 2
        )

    def test_outer_segment(self):

        outer_segment = (1 - math.cos(math.pi / 4)) * self.radius
        self.assertEqual(
            outer_segment, self.tank.outer_segment
        )

    def test_compute_zone_1_length(self):

        outer_segment = (1 - math.cos(math.pi / 4)) * self.radius

        # Test fuel not anymore in zone 1
        fuel_height = 0.9 * (self.radius * 2 - outer_segment)
        self.assertEqual(
            0, self.tank.compute_zone_1_length(fuel_height)
        )

        # Compute normal case of fuel height somewhere in in zone 1
        lower_limit = 2 * self.radius - outer_segment
        fuel_height = lower_limit + 0.5 * outer_segment
        self.assertEqual(
            0.5 * outer_segment,
            self.tank.compute_zone_1_length(fuel_height)
        )

    def test_compute_zone_2_length(self):

        self.assertEqual(
            self.tank.outer_segment,
            self.tank.compute_zone_2_length(self.radius)
        )

        self.assertEqual(
            self.tank.outer_segment / 2,
            self.tank.compute_zone_2_length(self.tank.outer_segment / 2)
        )

    def test_compute_zone_3_length(self):

        # Case where the fuel is only in zone 2
        self.assertEqual(
            0,
            self.tank.compute_zone_3_length(self.tank.outer_segment / 2)
        )

        # Case where the fuel is also in zone 1, thus full in zone 3
        self.assertEqual(
            self.radius * 2 * math.cos(math.pi / 4),
            self.tank.compute_zone_3_length(2 * self.radius)
        )

        # Case where the fuel is somewhere halfway the zone 3
        self.assertEqual(
            self.tank.compute_zone_3_length(self.radius),
            self.radius - self.tank.outer_segment
        )

    def test_compute_convective_lengths(self):

        fuel_height = 0.85 * 2 * self.radius
        lengths = [
            self.radius * 2 - fuel_height,
            self.tank.compute_zone_1_length(fuel_height),
            self.tank.compute_zone_2_length(fuel_height),
            self.tank.compute_zone_3_length(fuel_height)
        ]
        self.assertEqual(
            self.tank.compute_convective_lengths(fuel_height), lengths
        )

    def test_compute_zone_1_area(self):
        
        # Test for normal case where the fuel is in zone 1
        fuel_height = self.radius * 2 - self.tank.outer_segment / 2
        cap_area = self.tank.end_cap.compute_wetted_surface(
            self.tank.outer_segment
        )
        body_area = self.tank.body.compute_wetted_surface(
            self.tank.outer_segment
        )
        remove_area = self.tank.compute_fuel_wetted_surface(
            self.tank.diameter - fuel_height
        )
        self.assertEqual(
            self.tank.compute_zone_1_area(fuel_height),
            2 * cap_area + body_area - remove_area
        )

        # Test for the case where the fuel is not in zone 1
        self.assertEqual(
            self.tank.compute_zone_1_area(self.radius), 0
        )

    def test_compute_zone_2_area(self):

        # Case where the fuel is still in zone 3
        self.assertEqual(
            self.tank.compute_zone_2_area(self.radius),
            self.tank.compute_fuel_wetted_surface(
                self.tank.outer_segment
            )
        )

        # Case where the fuel is only left in zone 3
        fuel_height = self.tank.outer_segment / 2
        self.assertEqual(
            self.tank.compute_fuel_wetted_surface(fuel_height),
            self.tank.compute_zone_2_area(fuel_height)
        )

        # Limit case when the fuel tank is empty
        self.assertEqual(
            self.tank.compute_zone_2_area(0), 0
        )

    def test_compute_zone_3_area(self):

        # Case where the fuel is still in zone 1
        fuel_height = 2 * self.radius
        expected_area = (
            self.tank.surface_area
            - self.tank.compute_zone_1_area(fuel_height)
            - self.tank.compute_zone_2_area(fuel_height)
        )
        self.assertEqual(
            self.tank.compute_zone_3_area(fuel_height), expected_area
        )

        # Case where the fuel is still in zone 3
        fuel_height = self.radius
        expected_area = (
            self.tank.compute_fuel_wetted_surface(fuel_height)
            - self.tank.compute_zone_2_area(fuel_height)
        )
        self.assertEqual(
            self.tank.compute_zone_3_area(fuel_height), expected_area
        )

        # Case where the fuel is only in zone 2 
        self.assertEqual(
            self.tank.compute_zone_3_area(
                self.tank.outer_segment
            ),
            0
        )

    def test_compute_fuel_area_zones(self):

        fuel_height = self.radius
        areas = [
            self.tank.surface_area \
                - self.tank.compute_fuel_wetted_surface(fuel_height),
            self.tank.compute_zone_1_area(fuel_height),
            self.tank.compute_zone_2_area(fuel_height),
            self.tank.compute_zone_3_area(fuel_height)
        ]
        self.assertEqual(
            areas, self.tank.compute_fuel_area_zones(fuel_height)
        )

    def test_characteristic_length(self):

        self.assertEqual(
            self.tank.body_length,
            self.tank.characteristic_length
        )

    def test_exposed_surface(self):

        expected_value = self.tank.body.surface_area
        actual_value =self.tank.exposed_surface
        self.assertEqual(expected_value, actual_value)

    def test_rompokos(self):

        CylindricalTankSphericalCaps.rompokos()

    def test_ahluwalia(self):

        CylindricalTankSphericalCaps.ahluwalia()

    def test_example(self):

        CylindricalTankSphericalCaps.example()


class TestSphericalTank(unittest.TestCase):

    def test_compute_fuel_height(self):

        radius = 2.5
        tank = SphericalTank(radius)
        fuel_volume = tank.volume / 2
        expected_value = radius
        actual_value = tank.compute_fuel_height(fuel_volume)
        self.assertEqual(expected_value, actual_value)

    def test_lin(self):

        SphericalTank.lin()


class TestFindValue(unittest.TestCase):

    def test_find_value(self):

        func = (lambda x: 5 * x - 100)
        low = -100
        high = 100
        target = 0
        expected_value = 20
        self.assertAlmostEqual(
            bisection_method(high, low, target, func),
            expected_value,
            places=5
        )


if __name__ == "__main__":
    unittest.main()


# End
