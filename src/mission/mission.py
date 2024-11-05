from __future__ import annotations

from dataclasses import dataclass

from src.fluids.international_standard_atmosphere import get_ISA_air_properties
from src.mission.mission_sections import MissionSection, OutFlow

ASSUMED_TIMESTEP = 60


MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = MINUTES_TO_SECONDS * 60

# The take and hold times are assumed to be of 20 minutes
CLIMB_TIME = 5 * MINUTES_TO_SECONDS
TAKE_OFF_TIME = 20 * MINUTES_TO_SECONDS
HOLD_TIME = 20 * MINUTES_TO_SECONDS
DESCENT_TIME = 20 * MINUTES_TO_SECONDS
TIME_TO_ALTERNATE_CRUISE = 10 * MINUTES_TO_SECONDS
ALTERNATE_DESCENT_TIME = 5 * MINUTES_TO_SECONDS
LANDING_TIME = 5 * MINUTES_TO_SECONDS

# Assumptions for ground speed
average_ground_wind_speed = 4       # [m/s]
AVERAGE_GROUND_MACH_NUMBER = (
    average_ground_wind_speed * get_ISA_air_properties(0).speed_of_sound
)

@dataclass
class Mission:
    sections: list[MissionSection]

    @property
    def required_fuel(self) -> float:
        return sum([
            sum([
                abs(flow.mass_flow) * section.duration
                for flow in section.fuel_flows
            ])
            for section in self.sections
        ])

    @classmethod
    def rompokos(cls):

        # Definition of the mission particulars
        durations = [2, 0.3, 14.5]
        altitudes = [0, 5e3, 8e3]
        temperatures = [273.15, 273.15, 273.15]
        mach_numbers = [0.02, 0.5, 0.85]
        # Full fuel flow is estimated in an old file
        full_fuel_flow = 0.38410455139160027    # Estimated from paper
        throttles = [0, 0.9, 0.19]              # Estimated from paper

        mission_sections = [
            MissionSection(
                duration * HOURS_TO_SECONDS,
                [
                    OutFlow(
                        - throttle * full_fuel_flow, "liquid"
                    )
                ],
                altitude,
                mach_number,
                ground_temperature
            )
            for duration,
                throttle,
                altitude,
                mach_number,
                ground_temperature
            in zip(
                durations,
                throttles,
                altitudes,
                mach_numbers,
                temperatures
            )
        ]

        return cls(mission_sections)

    @classmethod
    def aircraft_mission(
        cls,
        fuel_flow_state: str,
        fuel_flows: list[float],
        cruise_altitude: float,
        cruise_range: float,
        diversion_range: float,
        cruise_mach_number: float
    ) -> Mission:
        loiter_time = 30 * MINUTES_TO_SECONDS   # Based on Onorato

        mission_sections = [
            "hold_period",
            "take_off",
            "climb",
            "cruise",
            "descent",
            "alternate_climb",
            "alternate_cruise",
            "alternate_descent",
            "loiter",
            "descent_2",
            "landing"
        ]

        durations = cls.define_durations(
            cruise_altitude,
            cruise_range,
            diversion_range,
            cruise_mach_number,
            loiter_time
        )
        mach_numbers = cls.define_mach_numbers(cruise_mach_number)
        altitudes = cls.define_altitudes(cruise_altitude)

        mission_sections = [
            MissionSection(
                durations[section],
                [OutFlow(fuel_flows[section], fuel_flow_state)],
                altitudes[section],
                mach_numbers[section]
            )
            for section in mission_sections
        ]

        return cls(mission_sections)

    @staticmethod
    def define_altitudes(cruise_altitude):
        altitudes = {
            "hold_period": 0,
            "take_off": cruise_altitude / 2,
            "climb": cruise_altitude / 2,
            "cruise": cruise_altitude,
            "descent": cruise_altitude / 2,
            "alternate_climb": cruise_altitude / 2,
            "alternate_cruise": cruise_altitude,
            "alternate_descent": cruise_altitude / 2,
            "loiter": cruise_altitude / 2,
            "descent_2": cruise_altitude / 2,
            "landing": cruise_altitude / 2
        }

        return altitudes

    @staticmethod
    def define_mach_numbers(cruise_speed):
        mach_numbers = {
            "hold_period": AVERAGE_GROUND_MACH_NUMBER,
            "take_off": cruise_speed / 2,
            "climb": cruise_speed / 2,
            "cruise": cruise_speed,
            "descent": cruise_speed / 2,
            "alternate_climb": cruise_speed / 2,
            "alternate_cruise": cruise_speed,
            "alternate_descent": cruise_speed / 2,
            "loiter": cruise_speed / 2,
            "descent_2": cruise_speed / 2,
            "landing": cruise_speed / 2
        }

        return mach_numbers

    @classmethod
    def define_durations(
        cls,
        cruise_altitude,
        cruise_range,
        diversion_range,
        cruise_speed,
        loiter_time
    ):
        durations = {
            "hold_period": HOLD_TIME,
            "take_off": TAKE_OFF_TIME,
            "climb": CLIMB_TIME,
            "cruise": cls.duration_from_speed_and_range(
                cruise_speed, cruise_range, cruise_altitude / 2
            ),
            "descent": DESCENT_TIME,
            "alternate_climb": TIME_TO_ALTERNATE_CRUISE,
            "alternate_cruise": cls.duration_from_speed_and_range(
                cruise_speed, diversion_range, cruise_altitude / 2
            ),
            "alternate_descent": ALTERNATE_DESCENT_TIME,
            "loiter": loiter_time,
            "descent_2": ALTERNATE_DESCENT_TIME,
            "landing": LANDING_TIME
        }

        return durations

    @staticmethod
    def duration_from_speed_and_range(
        cruise_speed: float,
        range: float,
        altitude: float
    ) -> float:
        ambient = get_ISA_air_properties(altitude)
        speed = ambient.speed_of_sound * cruise_speed
        time = range / speed
        return time // ASSUMED_TIMESTEP * ASSUMED_TIMESTEP

    @classmethod
    def regional(cls, fuel_flow_state) -> Mission:
        fuel_flows = {
            "hold_period": -0,
            "take_off": -0.1132,
            "climb": -0.0966,
            "cruise": -0.0512,
            "descent": -0.004,
            "alternate_climb": -0.1102,
            "alternate_cruise": -0.0724,
            "alternate_descent": -0.004,
            "loiter": -0.0286,
            "descent_2": -0.004,
            "landing": -0.0818
        }
        cruise_altitude = 5200
        cruise_range = 926e3
        diversion_range = 160e3
        cruise_mach_number = 0.44
        return cls.aircraft_mission(
            fuel_flow_state,
            fuel_flows,
            cruise_altitude,
            cruise_range,
            diversion_range,
            cruise_mach_number
        )

    @classmethod
    def small_medium_range(cls, fuel_flow_state) -> Mission:
        fuel_flows = {
            "hold_period": -0,
            "take_off": -0.9976,
            "climb": -0.4148,
            "cruise": -0.2082,
            "descent": -0.0514,
            "alternate_climb": -0.5994,
            "alternate_cruise": -0.3788,
            "alternate_descent": -0.0362,
            "loiter": -0.1862,
            "descent_2": -0.1036,
            "landing": -0.0436
        }
        cruise_altitude = 11e3
        cruise_range = 4560e3
        diversion_range = 370e3
        cruise_mach_number = 0.78
        return cls.aircraft_mission(
            fuel_flow_state,
            fuel_flows,
            cruise_altitude,
            cruise_range,
            diversion_range,
            cruise_mach_number
        )

    @classmethod
    def large_passenger_aircraft(cls, fuel_flow_state) -> Mission:
        fuel_flows = {
            "hold_period": -0,
            "take_off": -3.5176,
            "climb": -1.4648,
            "cruise": -0.6544,
            "descent": -0.1678,
            "alternate_climb": -2.1122,
            "alternate_cruise": -1.266,
            "alternate_descent": -0.1266,
            "loiter": -0.5856,
            "descent_2": -0.3274,
            "landing": -0.1344
        }
        cruise_altitude = 12e3
        cruise_range = 7674e3
        diversion_range = 370e3
        cruise_mach_number = 0.82
        return cls.aircraft_mission(
            fuel_flow_state,
            fuel_flows,
            cruise_altitude,
            cruise_range,
            diversion_range,
            cruise_mach_number
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
