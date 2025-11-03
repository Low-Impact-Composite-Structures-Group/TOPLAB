import numpy as np
import numpy.typing as npt

from .protocols import TankState, DynamicModel, FuelFlow, StateDerivatives, TwoPhaseHydrogen, Hydrogen, OperatingEnvelope

class SinglePhaseModel(DynamicModel):
    
    @classmethod
    def compute_state_derivatives(
        cls, tank_state: TankState, fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        dP_dt, dT_dt = cls.solve_state_equations(tank_state, fuel_flows[0], tank_state.heat_flux)
        dMg_dt, dMl_dt = cls.define_liquid_and_mass_derivatives(
            tank_state.phase, fuel_flows[0].mass_flow
        )
        return StateDerivatives(
            dP_dt,
            dT_dt,
            dMg_dt,
            dMl_dt,
            cls.compute_venting_mass(),
            cls.compute_added_heat_flux()
        )

    @classmethod
    def solve_state_equations(
        cls,
        tank_state: TankState,
        fuel_flow: FuelFlow,
        heat_flux: float
    ) -> list[float]:
        a = [
            [
                cls.a11(tank_state.hydrogen),
                cls.a12(tank_state.hydrogen)
            ], [
                cls.a21(tank_state.hydrogen, tank_state.volume),
                cls.a22(
                    tank_state.hydrogen,
                    tank_state.volume,
                    tank_state.tank_thermal_capacity
                )
            ]
        ]
        b = [
            [
                cls.y1(
                    tank_state.fuel_mass,
                    tank_state.hydrogen,
                    fuel_flow.mass_flow
                )
            ], [
                cls.y2(
                    tank_state.hydrogen,
                    fuel_flow,
                    heat_flux
                )
            ]
        ]
        x = np.linalg.solve(a, b)
        return x[0][0], x[1][0]

    @staticmethod
    def a11(hydrogen: Hydrogen) -> float:
        return hydrogen.dRho_dP

    @staticmethod
    def a12(hydrogen: Hydrogen) -> float:
        return hydrogen.dRho_dT

    @staticmethod
    def a21(
        hydrogen: Hydrogen, tank_volume: float
    ) -> float:
        return (
            tank_volume
            * hydrogen.density
            * hydrogen.dH_dP
            - tank_volume
        )

    @staticmethod
    def a22(
        hydrogen: Hydrogen,
        tank_volume: float,
        tank_thermal_capacity: float
    ) -> float:
        return (
            tank_thermal_capacity
            + tank_volume * hydrogen.density * hydrogen.dH_dT
        )

    @staticmethod
    def y1(
        fuel_mass: float,
        hydrogen: Hydrogen,
        fuel_mass_flow: float
    ) -> float:
        return hydrogen.density * fuel_mass_flow / fuel_mass

    @staticmethod
    def y2(
        tank_hydrogen: Hydrogen,
        fuel_flow: FuelFlow,
        heat_flux: float
    ) -> float:
        return (
            heat_flux
            + fuel_flow.mass_flow * (
                fuel_flow.hydrogen.enthalpy
                - tank_hydrogen.enthalpy
            )
        )

    @classmethod
    def compute_venting_mass(cls):
        return 0

    @classmethod
    def compute_added_heat_flux(cls):
        return 0

    @staticmethod
    def define_liquid_and_mass_derivatives(tank_phase, fuel_mass_flow):
        if tank_phase == "gas":
            return fuel_mass_flow, 0
        if tank_phase == "liquid":
            return 0, fuel_mass_flow
        raise ValueError(
            f"{tank_phase} not supported in single phase model"
        )


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
            cls.venting_mass(),
            cls.added_heat_flux()
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
            - fuel_flow.mass_flow / hydrogen.liquid.density
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

    @classmethod
    def added_heat_flux(cls) -> float:
        return 0

    @classmethod
    def venting_mass(cls) -> float:
        return 0


class TwoPhaseLimitLowerPressureModel(DynamicModel):

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        return StateDerivatives(
            cls.compute_pressure_derivative(),
            cls.compute_temperature_derivative(),
            cls.compute_gas_mass_derivative(
                tank_state.hydrogen, fuel_flows
            ),
            cls.compute_liquid_mass_derivative(
                tank_state.hydrogen, fuel_flows
            ),
            cls.compute_venting_mass(),
            cls.compute_required_heat_flux(
                tank_state.hydrogen, fuel_flows, tank_state.heat_flux
            )
        )

    @classmethod
    def compute_required_heat_flux(
        cls,
        hydrogen: TwoPhaseHydrogen,
        fuel_flows: list[FuelFlow],
        heat_flux: float
    ) -> float:
        t1 = hydrogen.liquid.enthalpy * cls.compute_liquid_mass_derivative(
            hydrogen, fuel_flows
        )
        t2 = hydrogen.gas.enthalpy * cls.compute_gas_mass_derivative(
            hydrogen, fuel_flows
        )
        t3 = sum([
            - flow.mass_flow * flow.hydrogen.enthalpy
            for flow in fuel_flows
        ])
        return - (t1 + t2 + t3 - heat_flux)

    @staticmethod
    def compute_liquid_mass_derivative(
        hydrogen: TwoPhaseHydrogen, fuel_flows: list[FuelFlow]
    ) -> float:
        return (
            sum([flow.mass_flow for flow in fuel_flows]) / (
                1 - hydrogen.gas.density / hydrogen.liquid.density
            )
        )

    @staticmethod
    def compute_gas_mass_derivative(
        hydrogen: TwoPhaseHydrogen, fuel_flows: list[FuelFlow]
    ) -> float:
        return (
            sum([flow.mass_flow for flow in fuel_flows]) / (
                1 - hydrogen.liquid.density / hydrogen.gas.density
            )
        )

    @staticmethod
    def compute_pressure_derivative() -> float:
        return 0

    @staticmethod
    def compute_temperature_derivative() -> float:
        return 0

    @staticmethod
    def compute_venting_mass() -> float:
        return 0


class SinglePhaseLimitLowerPressureModel(DynamicModel):

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        dMg_dt, dMl_dt = cls.define_liquid_and_mass_derivatives(
            tank_state.phase, fuel_flows
        )
        dT_dt = cls.compute_temperature_derivative(tank_state, fuel_flows)
        return StateDerivatives(
            cls.compute_pressure_derivative(),
            dT_dt,
            dMg_dt,
            dMl_dt,
            cls.compute_venting_mass(),
            cls.compute_required_heat_flux(tank_state, dT_dt)
        )

    @staticmethod
    def compute_pressure_derivative():
        return 0

    @staticmethod
    def compute_temperature_derivative(
        tank_state: TankState, fuel_flows: list[FuelFlow]
    ) -> float:
        num = (
            tank_state.hydrogen.density
            * sum([flow.mass_flow for flow in fuel_flows])
            / tank_state.fuel_mass
        )
        den = tank_state.hydrogen.dRho_dT
        return num / den

    @staticmethod
    def define_liquid_and_mass_derivatives(
        tank_phase: str, fuel_flows: list[FuelFlow]
    ) -> tuple[float, float]:
        if tank_phase == "gas":
            return sum([flow.mass_flow for flow in fuel_flows]), 0
        if tank_phase == "liquid":
            return 0, sum([flow.mass_flow for flow in fuel_flows])
        raise ValueError(
            f"{tank_phase} not supported in single phase model"
        )

    @classmethod
    def compute_venting_mass(cls):
        return 0

    @classmethod
    def compute_required_heat_flux(
        cls, tank_state: TankState, temperature_derivative: float
    ) -> float:
        fac1 = (
            tank_state.tank_thermal_capacity
            + tank_state.fuel_mass * tank_state.hydrogen.dH_dT
        )
        return - (
            fac1 * temperature_derivative - tank_state.heat_flux
        )


class DynamicModelFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        if tank_state.phase == "twophase":
            return TwoPhaseFactory().get_dynamic_model(
                tank_state, target_conditions
            )
        return SinglePhaseFactory().get_dynamic_model(
            tank_state, target_conditions
        )


class TwoPhaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        # if (
        #     target_conditions.min_pressure is None
        #     and target_conditions.max_pressure is None
        # ):
        #     return TwoPhaseModel
        if target_conditions.min_pressure is not None:
            if tank_state.pressure <= target_conditions.min_pressure:
                return TwoPhaseLimitLowerPressureModel
        if target_conditions.max_pressure is not None:
            if tank_state.pressure >= target_conditions.max_pressure:
                return TwoPhaseLimitLowerPressureModel
        return TwoPhaseModel


class SinglePhaseFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        if (
            target_conditions.min_pressure is not None
            and target_conditions.min_pressure >= tank_state.pressure
        ):
            return SinglePhaseLimitLowerPressureModel
        if (
            target_conditions.max_pressure is not None
            and tank_state.pressure >= target_conditions.max_pressure
        ):
            print("Model")
            return SinglePhaseLimitLowerPressureModel
        return SinglePhaseModel
    

class AlwaysTwoPhaseFactory:
    # Used for switch draining analysis
    def get_dynamic_model(self, *args, **kwargs):
        return TwoPhaseModel
    
