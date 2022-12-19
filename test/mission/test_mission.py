import unittest

from src.fluids.international_standard_atmosphere import get_ISA_air_properties
from src.mission.mission import Mission


class TestMission(unittest.TestCase):

    def test_duration_from_speed_and_range(self):
        cruise_speed = 0.85
        flight_range = 4000e3
        altitude = 10e3
        ambient = get_ISA_air_properties(altitude)
        flight_speed = ambient.speed_of_sound * cruise_speed
        assumed_timestep = 60
        time = flight_range / flight_speed
        expected_value = time // assumed_timestep * assumed_timestep
        actual_value = Mission.duration_from_speed_and_range(
            cruise_speed, flight_range, altitude
        )
        self.assertEqual(expected_value, actual_value)

    def test_rompokos(self):

        mission = Mission.rompokos()

    def test_regional(self):

        mission = Mission.regional("liquid")

    def test_small_medium_range(self):

        mission = Mission.small_medium_range("liquid")

    def test_large_passenger_aircraft(self):

        mission = Mission.large_passenger_aircraft("liquid")

    def test_required_fuel(self):

        actual_value = Mission.rompokos().required_fuel
        expected_value = 4182.898564654527
        self.assertEqual(expected_value, actual_value)



if __name__ == "__main__":
    unittest.main()


# End
