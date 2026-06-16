from __future__ import annotations

from dataclasses import dataclass

from src.fluids.international_standard_atmosphere import get_ISA_air_properties

from src.missions.mission_sections import InFlow, MissionSection, OutFlow

ASSUMED_TIMESTEP = 60
MINUTES_TO_SECONDS = 60
HOURS_TO_SECONDS = MINUTES_TO_SECONDS * 60
CLIMB_TIME = 5 * MINUTES_TO_SECONDS
TAKE_OFF_TIME = 20 * MINUTES_TO_SECONDS
HOLD_TIME = 20 * MINUTES_TO_SECONDS
DESCENT_TIME = 20 * MINUTES_TO_SECONDS
TIME_TO_ALTERNATE_CRUISE = 10 * MINUTES_TO_SECONDS
ALTERNATE_DESCENT_TIME = 5 * MINUTES_TO_SECONDS
LANDING_TIME = 5 * MINUTES_TO_SECONDS

average_ground_wind_speed = 4
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
        import csv
        from pathlib import Path

        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        def _norm(header: str) -> str:
            return "".join(ch for ch in header.strip().lower() if ch.isalnum())

        with csv_file.open("r", newline="") as file_obj:
            reader = csv.reader(file_obj)
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV is empty: {csv_path}") from exc

            header_map = {_norm(header): idx for idx, header in enumerate(headers)}
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
            if abs(flow_end - flow_start) < 1e-12:
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
                    start_rate = abs(flow.mass_flow[0])
                    end_rate = abs(flow.mass_flow[-1])
                    total_fuel += 0.5 * (start_rate + end_rate) * section.duration
                else:
                    total_fuel += abs(flow.mass_flow) * section.duration
        return total_fuel

    @classmethod
    def dormancy_section(
        cls,
        duration: float,
        altitude: float,
        fuel_flow: float,
        throttle: float,
        phase: str,
        mach_number: float,
    ) -> MissionSection:
        return MissionSection(
            duration * HOURS_TO_SECONDS,
            [],
            altitude,
            mach_number,
            "Dormancy",
        )

    @classmethod
    def rompokos(cls):
        durations = [2, 0.3, 14.5]
        altitudes = [0, 5e3, 8e3]
        temperatures = [273.15, 273.15, 273.15]
        mach_numbers = [0.02, 0.5, 0.85]
        full_fuel_flow = 0.38410455139160027
        throttles = [0, 0.9, 0.19]

        mission_sections = [
            MissionSection(
                duration * HOURS_TO_SECONDS,
                [OutFlow(-throttle * full_fuel_flow, "liquid")],
                altitude,
                mach_number,
                ground_temperature,
            )
            for duration, throttle, altitude, mach_number, ground_temperature in zip(
                durations,
                throttles,
                altitudes,
                mach_numbers,
                temperatures,
            )
        ]
        return cls(mission_sections)

    @classmethod
    def atr72(cls):
        cruise_altitude = 7010
        standard_temperature = 273.15
        durations = [0.008333333, 0.009464785, 0.251716247, 0.446224256, 0.008899059, 0.101703534, 0.002542588, 0.035596237, 0.044495296, 0.00817315, 0.10751462]
        altitudes = [0, cruise_altitude / 2, cruise_altitude / 2, cruise_altitude, cruise_altitude / 2, cruise_altitude / 2, cruise_altitude / 2, cruise_altitude / 2, cruise_altitude / 2, cruise_altitude / 2, cruise_altitude / 2]
        temperatures = [standard_temperature] * len(durations)
        mach_number = 0.0
        fuel_flows = [[0.0, 0.098061674], 0.098061674, [0.098061674, 0.060528634], 0.060528634, [0.060528634, 0.026167401], [0.026167401, 0.01215859], [0.01215859, 0.03753304], [0.03753304, 0.054185022], [0.054185022, 0.035154185], [0.035154185, 0.007665198], [0.007665198, 0.0]]
        throttles = [1.0] * len(durations)
        fuel_flow_keys = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]

        mission_sections = [
            MissionSection(
                duration * HOURS_TO_SECONDS,
                [OutFlow([-throttle * flow for flow in fuel_flow], "gas") if isinstance(fuel_flow, list) else OutFlow(-throttle * fuel_flow, "gas")],
                altitude,
                mach_number,
                fuel_flow_key,
                temperature,
            )
            for duration, altitude, throttle, fuel_flow, temperature, fuel_flow_key in zip(
                durations,
                altitudes,
                throttles,
                fuel_flows,
                temperatures,
                fuel_flow_keys,
            )
        ]
        return cls(mission_sections)