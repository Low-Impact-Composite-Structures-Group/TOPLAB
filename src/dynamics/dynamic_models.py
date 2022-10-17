"""Dynamic models can be used to compute the pressure and temperature 
changes in the fuel tank. These entail the models of Lin and Ahluwalia.

Fuel Tank - Dynamic Models
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Union

import numpy as np
import numpy.typing as npt


class TargetConditions(Protocol):
    ...


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


class SinglePhaseModel(DynamicModel):
    ...


class TwoPhaseModel(DynamicModel):

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        a = cls.define_a_matrix(tank_state)
        b = cls.define_b_vector(tank_state, fuel_flows)
        x = np.linalg.solve(a, b)
        return StateDerivatives(
            x[0][0],
            x[1][0],
            x[2][0],
            x[3][0],
            cls.venting_mass,
            cls.added_heat_flux
        )

    @staticmethod
    def a12(
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        return - hydrogen.dP_dT

    @staticmethod
    def a21(
        gas_mass: float,
        liquid_mass: float,
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        term1 = (
            gas_mass
            * hydrogen.gas.dRho_dP
            / hydrogen.gas.density ** 2
        )
        term2 = (
            liquid_mass
            * hydrogen.liquid.dRho_dP
            / hydrogen.liquid.density ** 2
        )
        return - (term1 + term2)

    @staticmethod
    def a22(
        gas_mass: float,
        liquid_mass: float,
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        term1 = (
            gas_mass
            * hydrogen.gas.dRho_dT
            / hydrogen.gas.density ** 2
        )
        term2 = (
            liquid_mass
            * hydrogen.liquid.dRho_dT
            / hydrogen.liquid.density ** 2
        )
        return - (term1 + term2)
    
    @staticmethod
    def a23(
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        return (
            1 / hydrogen.gas.density
            - 1 / hydrogen.liquid.density
        )

    @staticmethod
    def a42(
        tank_thermal_capacity: float,
        tank_volume: float,
        gas_mass: float,
        liquid_mass: float,
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        term4 = (
            liquid_mass * hydrogen.liquid.dH_dP
            + gas_mass * hydrogen.gas.dH_dP
            - tank_volume
        ) * hydrogen.dP_dT
        return (
            tank_thermal_capacity
            + liquid_mass * hydrogen.liquid.dH_dT
            + gas_mass * hydrogen.gas.dH_dT
            + term4
        )

    @staticmethod
    def a43(
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        return - (
            hydrogen.liquid.enthalpy
            - hydrogen.gas.enthalpy
        )

    @classmethod
    def define_a_matrix(
        cls,
        tank_state: TankState
    ) -> npt.ArrayLike:
        a12 = cls.a12(
            tank_state.hydrogen
        )
        a21 = cls.a21(
            tank_state.gas_mass,
            tank_state.liquid_mass,
            tank_state.hydrogen
        )
        a22 = cls.a22(
            tank_state.gas_mass,
            tank_state.liquid_mass,
            tank_state.hydrogen
        )
        a23 = cls.a23(
            tank_state.hydrogen
        )
        a42 = cls.a42(
            tank_state.tank_thermal_capacity,
            tank_state.volume,
            tank_state.gas_mass,
            tank_state.liquid_mass,
            tank_state.hydrogen
        )
        a43 = cls.a43(
            tank_state.hydrogen
        )
        a = [
            [1, a12, 0, 0],
            [a21, a22, a23, 0],
            [0, 0, 1, 1],
            [0, a42, a43, 0]
        ]
        
        return a

    @staticmethod
    def y2(
        fuel_flows: list[FuelFlow],
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        return sum([
            fuel_flow.mass_flow / hydrogen.liquid.density
            for fuel_flow in fuel_flows
        ])

    @staticmethod
    def y3(
        fuel_flows: list[FuelFlow]
    ) -> float:
        return sum([
            fuel_flow.mass_flow
            for fuel_flow in fuel_flows
        ])

    @staticmethod
    def y4(
        fuel_flows: list[FuelFlow],
        hydrogen: TwoPhaseHydrogen,
        heat_flux: float
    ) -> float:
        term2 = sum([
            fuel_flow.mass_flow
            * (
                fuel_flow.hydrogen.enthalpy
                - hydrogen.liquid.enthalpy
            )
            for fuel_flow in fuel_flows
        ])
        return heat_flux + term2

    @classmethod
    def define_b_vector(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> npt.ArrayLike:
        y2 = cls.y2(
            fuel_flows,
            tank_state.hydrogen
        )
        y3 = cls.y3(
            fuel_flows
        )
        y4 = cls.y4(
            fuel_flows,
            tank_state.hydrogen,
            tank_state.heat_flux
        )
        b = [[0], [y2], [y3], [y4]]
        return b

    @property
    def added_heat_flux(self) -> float:
        return 0

    @property
    def venting_mass(self) -> float:
        return 0


class DynamicModelFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: TargetConditions
    ) -> DynamicModel:
        # The logic for the definition of the dynamic model is still to 
        # be implemented
        return TwoPhaseModel


def main():
    pass


if __name__ == "__main__":
    main()


# End
