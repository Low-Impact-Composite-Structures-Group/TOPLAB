from __future__ import annotations

from dataclasses import dataclass

from src.fluids.international_standard_atmosphere import get_ISA_air_properties
from src.mission.mission_sections import MissionSection, OutFlow, InFlow

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

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        cruise_altitude: float = 7010.0,
        standard_temperature: float = 273.15,
        mach_number: float = 0.0,
        phase: str = "gas",
        ambient_temperature: float = 288.15,
    ) -> "Mission":
        """Create a Mission from a CSV time series of mass flow.

        Expected columns (case-insensitive):
        - "Time [hr]" (hours)
        - "Mass flow rate [kg/s]" (kg/s, positive for consumption)

        The returned mission is built as piecewise-linear segments between
        successive sample points.
        """

        import csv
        from pathlib import Path

        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        def _norm(header: str) -> str:
            return "".join(ch for ch in header.strip().lower() if ch.isalnum())

        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError(f"CSV is empty: {csv_path}")

            header_map = {_norm(h): idx for idx, h in enumerate(headers)}
            time_idx = header_map.get(_norm("Time [hr]"))
            flow_idx = header_map.get(_norm("Mass flow rate [kg/s]"))

            if time_idx is None or flow_idx is None:
                raise ValueError(
                    "CSV must contain columns 'Time [hr]' and 'Mass flow rate [kg/s]'. "
                    f"Found headers: {headers}"
                )

            times_hr: list[float] = []
            flow_rates: list[float] = []

            for row in reader:
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                times_hr.append(float(row[time_idx]))
                flow_rates.append(float(row[flow_idx]))

        if len(times_hr) < 2:
            raise ValueError(f"CSV must contain at least 2 rows of data: {csv_path}")

        mission_sections: list[MissionSection] = []
        for i in range(len(times_hr) - 1):
            dt_hr = times_hr[i + 1] - times_hr[i]
            if dt_hr <= 0:
                continue
            duration = dt_hr * HOURS_TO_SECONDS

            flow_start = abs(flow_rates[i])
            flow_end = abs(flow_rates[i + 1])

            flow_tolerance = 1e-12
            if abs(flow_end - flow_start) < flow_tolerance:
                fuel_flow = OutFlow(-flow_start, phase)
            else:
                fuel_flow = OutFlow([-flow_start, -flow_end], phase)

            mission_sections.append(
                MissionSection(
                    duration=duration,
                    fuel_flows=[fuel_flow],
                    altitude=cruise_altitude,
                    mach_number=mach_number,
                    fuel_flow_key=None,
                    ground_temperature=standard_temperature,
                )
            )

        return cls(mission_sections)

    @property
    def required_fuel(self) -> float:
        total_fuel = 0.0
        for section in self.sections:
            for flow in section.fuel_flows:
                if isinstance(flow.mass_flow, list):
                    # Correct trapezoidal integration for linearly varying flow
                    start_rate = abs(flow.mass_flow[0])
                    end_rate = abs(flow.mass_flow[-1])
                    base = section.duration
                    total_fuel += 0.5 * (start_rate + end_rate) * base
                else:
                    total_fuel += abs(flow.mass_flow) * section.duration
        return total_fuel

    @classmethod
    def discharge_section(cls, duration: float, altitude: float, fuel_flow: float, throttle: float, phase: str, mach_number: float) -> MissionSection:
        return MissionSection(
            duration * HOURS_TO_SECONDS,
            [
                OutFlow(
                    - throttle * fuel_flow, phase
                )
            ],
            altitude,
            mach_number,
            "Discharge"
        )

    # @classmethod
    # def refuel_section(cls, duration: float, altitude: float, fuel_flow: float, throttle: float, mach_number: float) -> MissionSection:
    #     # For refueling, we use negative OutFlow to represent inflow (the opposite of consumption)
    #     return MissionSection(
    #             duration*HOURS_TO_SECONDS,
    #             [
    #                 OutFlow(
    #                     -1 * throttle * fuel_flow,  # Negative flow means adding fuel
    #                     "gas"                      # Using gas phase for refueling
    #                 )
    #             ],
    #             altitude,
    #             mach_number,
    #             "Refueling"
    #         )

    @classmethod
    def dormancy_section(cls, duration: float, altitude: float, fuel_flow: float, throttle: float, phase: str, mach_number: float) -> MissionSection:
        """Create a mission section for dormancy with zero fuel flow"""
        # Force zero fuel flow for dormancy
        fuel_flow = 0.0
        throttle = 0.0
        return MissionSection(
                duration*HOURS_TO_SECONDS,
                [],  # Empty list - no fuel flows during dormancy
                altitude,
                mach_number,
                "Dormancy"
            )

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


class MissionFactory:
    def create_mission_from_list(self, mission_sections: list[dict]) -> Mission:
        sections: list[MissionSection] = []
        for section in mission_sections:
            fuel_flow_key = section.get("name", section.get("fuel_flow_key"))

            flows: list[OutFlow | InFlow] = []
            for flow in section.get("fuel_flows", []):
                if "phase" in flow:
                    flows.append(OutFlow(flow["mass_flow"], flow["phase"]))
                elif "hydrogen" in flow:
                    flows.append(InFlow(flow["mass_flow"], flow["hydrogen"]))
                else:
                    raise ValueError(
                        "Invalid fuel flow config; expected either 'phase' (OutFlow) or 'hydrogen' (InFlow)."
                    )

            sections.append(
                MissionSection(
                    duration=section["duration"],
                    fuel_flows=flows,
                    altitude=section["altitude"],
                    mach_number=section["mach_number"],
                    fuel_flow_key=fuel_flow_key,
                    ground_temperature=section.get("ground_temperature"),
                )
            )

        return Mission(sections)

    def create_mission_from_file(
        self,
        file_name: str | None = None,
        fuel_flow_phase: str | None = None,
        **kwargs,
    ) -> Mission:
        mission_name = file_name or kwargs.pop("file", None)
        if not mission_name:
            raise ValueError("Missing mission name; expected 'file_name' (or legacy 'file').")

        try:
            mission_builder = getattr(Mission, mission_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown mission '{mission_name}'.") from exc

        if kwargs:
            return mission_builder(**kwargs)

        if fuel_flow_phase is None:
            return mission_builder()

        try:
            return mission_builder(fuel_flow_phase)
        except TypeError:
            return mission_builder()

    @classmethod
    def atr72(cls):

        # Definition of the mission particulars
        cruise_altitude = 7010 # [m]
        standard_temperature = 273.15 # [K]
        durations = [0.008333333, 0.009464785, 0.251716247, 0.446224256, 0.008899059, 0.101703534, 0.002542588, 0.035596237, 0.044495296,0.00817315, 0.10751462] # durations in hours
        # In the atr72() method:
        altitudes = [0, cruise_altitude/2, cruise_altitude/2, cruise_altitude, cruise_altitude/2, cruise_altitude/2, cruise_altitude/2,
             cruise_altitude/2, cruise_altitude/2, cruise_altitude/2, cruise_altitude/2]  # Now 11 elements to match durations
        temperatures = [standard_temperature] * len(durations)  # assume constant standard temperature
        mach_number = 0.0
        fuel_flows = [[0.0, 0.098061674], 0.098061674, [0.098061674, 0.060528634], 0.060528634, [0.060528634, 0.026167401], [0.026167401, 0.01215859], [0.01215859, 0.03753304], [0.03753304, 0.054185022], [0.054185022, 0.035154185], [0.035154185, 0.007665198], [0.007665198, 0.0]] # [kg/s] values from H2 mass and flow estimation excel sheet
        throttles = [1.0] * len(durations) # default to 1 for now
        fuel_flow_keys = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven']

        mission_sections = [
            MissionSection(
                duration * HOURS_TO_SECONDS,
                [OutFlow([-throttle * flow for flow in fuel_flow], "gas") if isinstance(fuel_flow, list) else OutFlow(-throttle * fuel_flow, "gas")],
                altitude,
                mach_number,
                fuel_flow_key,
                temperature
            )
            for duration, altitude, throttle, fuel_flow, temperature, fuel_flow_key in zip(durations, altitudes, throttles, fuel_flows, temperatures, fuel_flow_keys)
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
                mach_numbers[section],
                section
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

    @classmethod
    def from_csv(cls,
                 csv_path: str,
                 cruise_altitude: float = 7010.0,
                 standard_temperature: float = 273.15,
                 mach_number: float = 0.0,
                 phase: str = "gas",
                 ambient_temperature: float = 288.15):
        """
        Create a Mission from a CSV file containing time and mass flow rate data.

        Args:
            csv_path: Path to CSV file with 'Time [hr]' and 'Mass flow rate [kg/s]' columns
            cruise_altitude: Constant altitude for all sections [m]
            standard_temperature: Constant temperature for all sections [K]
            mach_number: Constant Mach number for all sections [-]
            phase: Phase identifier for OutFlow ('gas' or 'liquid')
            ambient_temperature: Ambient/ground temperature [K]

        Returns:
            Mission object with sections derived from CSV data
        """
        import pandas as pd
        import numpy as np
        from pathlib import Path

        # Read CSV file
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_file)

        # Validate columns
        expected_cols = ['Time [hr]', 'Mass flow rate [kg/s]']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {df.columns.tolist()}")

        # Extract time and flow rate data
        times_hr = df['Time [hr]'].values
        flow_rates = df['Mass flow rate [kg/s]'].values

        # Convert time to seconds
        times_s = times_hr * HOURS_TO_SECONDS

        # Calculate section durations from time deltas
        # Each consecutive pair of points defines a section
        mission_sections = []

        for i in range(len(times_s) - 1):
            duration = times_s[i + 1] - times_s[i]

            # Skip zero-duration sections
            if duration <= 0:
                continue

            # Get flow rates at start and end of section
            flow_start = flow_rates[i]
            flow_end = flow_rates[i + 1]

            # Determine if flow is constant or varying
            flow_tolerance = 1e-9  # kg/s
            is_constant_flow = abs(flow_end - flow_start) < flow_tolerance

            # Create OutFlow object (negate flow for discharge convention)
            if is_constant_flow:
                # Constant flow
                fuel_flow = OutFlow(-abs(flow_start), phase)
            else:
                # Linearly varying flow (list format)
                fuel_flow = OutFlow([-abs(flow_start), -abs(flow_end)], phase)

            # Create mission section
            # Note: fuel_flow_key is set to None (no longer required)
            section = MissionSection(
                duration,
                [fuel_flow],
                cruise_altitude,
                mach_number,
                None,  # fuel_flow_key - not needed for CSV missions
                standard_temperature
            )

            mission_sections.append(section)

        print(f"Loaded CSV mission from {csv_file.name}")
        print(f"   Created {len(mission_sections)} mission sections")
        print(f"   Total duration: {sum(s.duration for s in mission_sections) / HOURS_TO_SECONDS:.2f} hours")

        return cls(mission_sections)


def main():
    pass


if __name__ == "__main__":
    main()


# End
