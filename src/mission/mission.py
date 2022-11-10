
from dataclasses import dataclass

from src.mission.mission_sections import MissionSection, OutFlow


MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = MINUTES_TO_SECONDS * 60


@dataclass
class Mission:
    sections: list[MissionSection]

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


def main():
    pass


if __name__ == "__main__":
    main()


# End
