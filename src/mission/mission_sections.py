
from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import Protocol
from src.fluids.international_standard_atmosphere import get_ISA_air_properties


class Hydrogen(Protocol):
    ...


@dataclass
class FuelFlow:
    mass_flow: float


@dataclass
class OutFlow(FuelFlow):
    phase: str

    @classmethod
    def rompokos_cruise(cls, phase: str):
        full_fuel_flow = - 0.384104551391600    # Estimated from paper
        throttle = 0.2                          # Estimated from paper
        fuel_flow = full_fuel_flow * throttle
        return cls(fuel_flow, phase)

    @classmethod
    def SMR_cruise(cls, phase: str):
        fuel_flow = -0.21
        return cls(fuel_flow, phase)
    
    @classmethod
    def LPA_cruise(cls, phase: str):
        fuel_flow = -0.654
        return cls(fuel_flow, phase)



@dataclass
class InFlow(FuelFlow):
    hydrogen: Hydrogen


@dataclass
class MissionSection:
    duration: float
    fuel_flows: list[FuelFlow]
    altitude: float
    mach_number: float

    ground_temperature: float = None
    name: str = None

    def __post_init__(self) -> None:
        self.ambient = get_ISA_air_properties(
            self.altitude, temperature=self.ground_temperature
        )

    @property
    def temperature(self) -> float:
        return self.ambient.temperature

    @property
    def flight_speed(self) -> float:
        return self.ambient.speed_of_sound * self.mach_number
    
    def number_of_timesteps(self, timestep: float) -> float:
        modulus = self.duration % timestep
        if modulus != 0:
            warnings.warn(
                f"Selected timestep ({timestep}) leaves remainder in mission duration.\n"
                + f"Remaining time is: {modulus}.\n"
                + "It is up to the user to see if this is desired..."
            )
        return int(self.duration // timestep)

    @classmethod
    def draining(
        cls, fuel_mass_flow: float, fuel_flow_state: str
    ) -> MissionSection:
        duration = 60e10
        altitude = 10e3
        mach_number = 0.8
        fuel_flows = [OutFlow(fuel_mass_flow, fuel_flow_state)]
        return cls(
            duration, fuel_flows, altitude, mach_number
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
