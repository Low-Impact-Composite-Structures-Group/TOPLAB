from dataclasses import dataclass
from typing import Protocol


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