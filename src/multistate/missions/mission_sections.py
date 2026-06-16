from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union
import math

from src.multistate.fluids.international_standard_atmosphere import get_ISA_air_properties


class Hydrogen(Protocol):
    ...


@dataclass
class FuelFlow:
    mass_flow: Union[float, list[float]]


@dataclass
class OutFlow(FuelFlow):
    phase: str


@dataclass
class InFlow(FuelFlow):
    hydrogen: Hydrogen


@dataclass
class InFlowWithDirectEnthalpy(FuelFlow):
    hydrogen: Hydrogen
    direct_enthalpy: float = None


@dataclass
class MissionSection:
    duration: float
    fuel_flows: list[FuelFlow]
    altitude: float
    mach_number: float
    fuel_flow_key: str = None
    ground_temperature: float = None

    def __post_init__(self) -> None:
        self.ambient = get_ISA_air_properties(
            self.altitude,
            temperature=self.ground_temperature,
        )

    @property
    def temperature(self) -> float:
        return self.ambient.temperature

    @property
    def flight_speed(self) -> float:
        return self.ambient.speed_of_sound * self.mach_number

    def number_of_timesteps(self, timestep: float) -> float:
        if timestep <= 0:
            raise ValueError("Timestep must be > 0")

        quotient = self.duration / timestep
        nearest = round(quotient)
        if not math.isclose(quotient, nearest, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "Invalid timestep and duration combination\n"
                "Ensure that the duration is a multiple of the step."
            )
        return int(nearest)

    def has_multiple_flows(self) -> bool:
        return len(self.fuel_flows) > 1

    def get_inflows(self) -> list[InFlow]:
        return [flow for flow in self.fuel_flows if isinstance(flow, InFlow)]

    def get_outflows(self) -> list[OutFlow]:
        return [flow for flow in self.fuel_flows if isinstance(flow, OutFlow)]

    def get_single_flow(self) -> FuelFlow:
        if len(self.fuel_flows) != 1:
            raise ValueError("Section has multiple flows, use get_inflows/get_outflows")
        return self.fuel_flows[0]