import unittest

from src.mission.mission_sections import MissionSection, OutFlow


class TestOutFlow(unittest.TestCase):

    def test_rompokos_cruise(self):

        expected_value = -0.3841045513916 * 0.2
        actual_value = OutFlow.rompokos_cruise("test").mass_flow
        self.assertEqual(expected_value, actual_value)

    def test_SMR_cruise(self):

        expected_value = -0.21
        actual_value = OutFlow.SMR_cruise("test").mass_flow
        self.assertEqual(expected_value, actual_value)


class TestMissionSection(unittest.TestCase):

    def setUp(self) -> None:

        self.duration = 60 ** 2
        self.altitude = 0
        self.mach_number = 0.85
        self.fuel_flows = None
        self.section = MissionSection(
            self.duration,
            self.fuel_flows,
            self.altitude,
            self.mach_number
        )

    def test_temperature(self):

        sea_level_temperature = 288.15      # [K]
        expected_value = sea_level_temperature
        actual_value = self.section.temperature
        self.assertEqual(expected_value, actual_value)

    def test_flight_speed(self):

        sea_level_speed_of_sound = 340
        expected_value = sea_level_speed_of_sound * self.mach_number
        actual_value = self.section.flight_speed
        self.assertAlmostEqual(expected_value, actual_value, places=0)

    def test_number_of_timesteps(self):

        timestep = 60
        expected_value = 60
        actual_value = self.section.number_of_timesteps(timestep)
        self.assertEqual(expected_value, actual_value)

        # Case where the timestep is not allowed
        timestep = 76
        with self.assertRaises(ValueError) as context:
            self.section.number_of_timesteps(timestep)
        message = (
            "Invalid timestep and duration combination\n" \
            "Ensure that the duration is a multiple of the step."
        )
        self.assertTrue(message in str(context.exception))

    def test_draining(self):
        expected_value = 0.8
        mission_section = MissionSection.draining(88.8, "liquid")
        actual_value = mission_section.mach_number
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()

# End
