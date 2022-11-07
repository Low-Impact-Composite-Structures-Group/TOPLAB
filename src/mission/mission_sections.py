

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


class InFlow(FuelFlow):
    hydrogen: Hydrogen


@dataclass
class MissionSection:
    duration: float
    fuel_flows: list[FuelFlow]
    altitude: float
    mach_number: float

    ground_temperature: float = None

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
        if self.duration % timestep != 0:
            raise ValueError(
                "Invalid timestep and duration combination\n" \
                "Ensure that the duration is a multiple of the step."
            )
        return self.duration // timestep


def main():
    pass


if __name__ == "__main__":
    main()


# End
