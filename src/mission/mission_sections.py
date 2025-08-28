from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union
from src.fluids.international_standard_atmosphere import get_ISA_air_properties


class Hydrogen(Protocol):
    ...


@dataclass
class FuelFlow:
    mass_flow:  Union[float, list[float]]


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

@dataclass
class InFlow(FuelFlow):
    hydrogen: Hydrogen

@dataclass
class InFlowWithDirectEnthalpy(FuelFlow):
    """
    InFlow variant that carries direct enthalpy values instead of hydrogen objects.
    This is used specifically for the TwoPhaseRefuelModel to avoid issues with
    crossing the saturation line during refueling.
    """
    hydrogen: Hydrogen  # Still needed for compatibility with existing code
    direct_enthalpy: float = None  # Direct enthalpy value in J/kg


@dataclass
class MissionSection:
    duration: float
    fuel_flows: list[FuelFlow]
    altitude: float
    mach_number: float
    fuel_flow_key: str=None

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
        # if self.duration % timestep != 0:
        #     raise ValueError(
        #         "Invalid timestep and duration combination\n" \
        #         "Ensure that the duration is a multiple of the step."
        #     )
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

    def has_multiple_flows(self) -> bool:
        """Check if this section has multiple fuel flows"""
        return len(self.fuel_flows) > 1

    def get_inflows(self) -> list[InFlow]:
        """Get all inflows for this section"""
        return [ff for ff in self.fuel_flows if isinstance(ff, InFlow)]

    def get_outflows(self) -> list[OutFlow]:
        """Get all outflows for this section"""
        return [ff for ff in self.fuel_flows if isinstance(ff, OutFlow)]

    def get_single_flow(self) -> FuelFlow:
        """Get single flow (for backward compatibility)"""
        if len(self.fuel_flows) != 1:
            raise ValueError("Section has multiple flows, use get_inflows/get_outflows")
        return self.fuel_flows[0]


def main():
    pass


if __name__ == "__main__":
    main()


# End
