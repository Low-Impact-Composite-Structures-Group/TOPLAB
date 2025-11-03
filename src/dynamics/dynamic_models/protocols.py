from dataclasses import dataclass
from typing import Protocol, Union
from abc import abstractmethod


class OperatingEnvelope(Protocol):
    min_pressure: float
    max_pressure: float


class Hydrogen(Protocol):
    density: float
    dRho_dP: float
    dRho_dT: float
    dH_dT: float
    dH_dP: float
    enthalpy: float


class TwoPhaseHydrogen(Protocol):
    dP_dT: float
    liquid: Hydrogen
    gas: Hydrogen
    heat_of_evaporation: float


class FuelFlow(Protocol):
    hydrogen: Hydrogen
    mass_flow: float


class TankState(Protocol):
    fill: float
    heat_flux: float
    volume: float
    tank_temperature: float
    hydrogen: Union[Hydrogen, TwoPhaseHydrogen]
    gas_mass: float
    liquid_mass: float
    fuel_mass: float
    tank_thermal_capacity: float
    phase: str
    pressure: float


@dataclass
class StateDerivatives:
    """Dataclass for the state derivatives with respect to time for the 
    fuel tank state.
    """
    pressure: float
    temperature: float
    gas_mass: float
    liquid_mass: float
    venting_mass: float
    heat_flux: float



class DynamicModel(Protocol):
	
    @abstractmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        ...


