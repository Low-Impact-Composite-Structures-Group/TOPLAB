"""Dynamic models can be used to compute the pressure and temperature
changes in the fuel tank. These entail the models of Lin and Ahluwalia.

Fuel Tank - Dynamic Models
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Union

import numpy as np
import numpy.typing as npt

from src.fluids.energy_derivative_computer import EnergyDerivativeComputer
from src.mission.mission_sections import OutFlow
from CoolProp.CoolProp import PropsSI

TEST_CP = 7000
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
        fuel_flows: list[FuelFlow] | None = None, # inflow
        fuel_flows_out: list[FuelFlow] | None = None # outflow (optional, for backward compatibility)
    ) -> StateDerivatives:
        """
        Compute state derivatives for the tank.
        If only one fuel flow list is provided, it is treated as outflow (for backward compatibility).
        If both are provided, inflow and outflow are handled separately.
        """
        ...


class SinglePhaseModelBase(ABC):
    """Abstract base class for all single-phase dynamic models"""

    @classmethod
    @abstractmethod
    def compute_state_derivatives(cls, tank_state, *args) -> 'StateDerivatives':
        """
        Compute state derivatives for single-phase models
        Args can be:
        - Single flow: (fuel_flows,)
        - Multi flow: (inflows, outflows)
        """
        pass


class TwoPhaseModelBase(ABC):
    """Abstract base class for all two-phase dynamic models"""

    @classmethod
    @abstractmethod
    def compute_state_derivatives(cls, tank_state, *args) -> 'StateDerivatives':
        """
        Compute state derivatives for two-phase models
        Args can be:
        - Single flow: (fuel_flows,)
        - Multi flow: (inflows, outflows)
        """
        pass


class SinglePhaseModel(SinglePhaseModelBase):

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
        """
        The first coefficient in the state equation for pressure rate.
        """
        # Avoid division by zero when mass is zero
        if fuel_mass <= 0:
            # When starting with zero mass, return a safe default
            # (If adding mass, rate should be positive; if removing, should be zero)
            return 0 if fuel_mass_flow >= 0 else 0
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


class TwoPhaseModel(DynamicModel, TwoPhaseModelBase):

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
        print(f"\n==== TwoPhaseModel.y4 Debug ====")
        print(f"Number of fuel flows: {len(fuel_flows)}")

        term2 = 0
        for i, fuel_flow in enumerate(fuel_flows):
            print(f"  Flow {i} mass_flow: {fuel_flow.mass_flow}")
            print(f"  Flow {i} hydrogen type: {type(fuel_flow.hydrogen).__name__}")

            try:
                if hasattr(fuel_flow.hydrogen, "enthalpy"):
                    inflow_enthalpy = fuel_flow.hydrogen.enthalpy
                    print(f"  Flow {i} hydrogen enthalpy: {inflow_enthalpy:.2f} J/kg")
                else:
                    # Try to get enthalpy from gas/liquid phase if it's TwoPhaseHydrogen
                    if hasattr(fuel_flow.hydrogen, "gas") and hasattr(fuel_flow.hydrogen, "liquid"):
                        # Use average of gas and liquid enthalpy as fallback
                        inflow_enthalpy = (fuel_flow.hydrogen.gas.enthalpy + fuel_flow.hydrogen.liquid.enthalpy) / 2
                        print(f"  Flow {i} using average of gas/liquid enthalpies: {inflow_enthalpy:.2f} J/kg")
                    elif hasattr(fuel_flow.hydrogen, "gas"):
                        inflow_enthalpy = fuel_flow.hydrogen.gas.enthalpy
                        print(f"  Flow {i} using gas enthalpy: {inflow_enthalpy:.2f} J/kg")
                    elif hasattr(fuel_flow.hydrogen, "liquid"):
                        inflow_enthalpy = fuel_flow.hydrogen.liquid.enthalpy
                        print(f"  Flow {i} using liquid enthalpy: {inflow_enthalpy:.2f} J/kg")
                    else:
                        # Create a fallback value
                        print(f"  Flow {i} ERROR: hydrogen has no enthalpy attribute!")
                        from CoolProp.CoolProp import PropsSI
                        # Use CoolProp to get enthalpy at standard conditions
                        P = 101325  # 1 atm
                        T = 293.15  # 20°C
                        inflow_enthalpy = PropsSI('H', 'P', P, 'T', T, 'hydrogen')
                        print(f"  Flow {i} using fallback enthalpy from CoolProp: {inflow_enthalpy:.2f} J/kg")

                # Use the same approach for the tank hydrogen
                if hasattr(hydrogen, "liquid"):
                    tank_enthalpy = hydrogen.liquid.enthalpy
                    print(f"  Tank liquid hydrogen enthalpy: {tank_enthalpy:.2f} J/kg")
                else:
                    print(f"  ERROR: Tank hydrogen has no liquid attribute!")
                    tank_enthalpy = 0

                # Calculate this flow's contribution to term2
                flow_contribution = fuel_flow.mass_flow * (inflow_enthalpy - tank_enthalpy)
                term2 += flow_contribution
                print(f"  Flow {i} contribution: {flow_contribution:.2f}")

            except Exception as e:
                print(f"  Error processing flow {i}: {str(e)}")
                import traceback
                traceback.print_exc()

        result = heat_flux + term2
        print(f"Heat flux: {heat_flux:.2f}, Term2: {term2:.2f}, Total y4: {result:.2f}")
        print(f"=====================================\n")
        return result

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


class SinglePhaseLimitLowerPressureModel(SinglePhaseModelBase):

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


class LinModel(DynamicModel):

    energy_derivative_computer = EnergyDerivativeComputer()

    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        dP_dt = cls.compute_pressure_derivative(tank_state, fuel_flows)
        dT_dt = dP_dt / tank_state.hydrogen.dP_dT
        return StateDerivatives(
            dP_dt,
            dT_dt,
            None,
            None,
            None,
            None
        )

    def compute_pressure_derivative(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> float:
        hydrogen: TwoPhaseHydrogen = tank_state.hydrogen
        energy_derivative = cls.compute_energy_derivative(
            hydrogen, tank_state.fill
        )
        factor1 = energy_derivative / tank_state.volume
        term2 = sum([
            fuel_flow.mass_flow * fuel_flow.hydrogen.enthalpy
            for fuel_flow in fuel_flows
        ])
        factor2 = (
            tank_state.heat_flux
            + term2
        )
        return factor1 * factor2

    def compute_energy_derivative(
        self, hydrogen: Hydrogen, fill: float
    ) -> float:
        return self.energy_derivative_computer.compute_energy_derivative(
            hydrogen, fill
        )


class SinglePhaseInOutModel(SinglePhaseModelBase):

    @classmethod
    def compute_state_derivatives(
        cls, tank_state: TankState, fuel_flow_in: list[FuelFlow], fuel_flow_out: list[FuelFlow]
    ) -> StateDerivatives:
        # Safety check - ensure both lists have at least one item
        if not fuel_flow_in:
            # Create a dummy inflow with zero rate
            from src.mission.mission_sections import InFlow
            dummy_inflow = InFlow(0.0, tank_state.hydrogen)
            fuel_flow_in = [dummy_inflow]

        if not fuel_flow_out:
            # Create a dummy outflow with zero rate
            from src.mission.mission_sections import OutFlow
            dummy_outflow = OutFlow(0.0, "gas")
            fuel_flow_out = [dummy_outflow]

        # Handle mass_flow as list if needed
        in_flow_rate = fuel_flow_in[0].mass_flow
        if isinstance(in_flow_rate, list):
            in_flow_rate = in_flow_rate[0]  # Use first value if it's a list

        out_flow_rate = fuel_flow_out[0].mass_flow
        if isinstance(out_flow_rate, list):
            out_flow_rate = out_flow_rate[0]  # Use first value if it's a list

        # Original code continues
        dP_dt, dT_dt = cls.solve_state_equations(tank_state, fuel_flow_in[0], fuel_flow_out[0], tank_state.heat_flux)
        dMg_dt, dMl_dt = cls.define_liquid_and_mass_derivatives(
            # Mass derivatives use the inflow convention for this model.
            tank_state.phase, in_flow_rate
        )

        # Apply safety limits to prevent non-physical values
        # Use much higher pressure change rate limit for refueling
        MAX_PRESSURE_CHANGE = 5e8  # Pa/s (5000 bar/s) for refueling, essentially unlimited

        # # Check if this is refueling or normal operation
        # # During refueling, the pressure should only increase
        # if dP_dt > 0:  # If pressure is increasing (refueling)
        #     # Apply a very high limit
        #     dP_dt = 2*min(dP_dt, MAX_PRESSURE_CHANGE)  # Only limit extreme values
        # else:  # Draining or other operations
        #     # Apply a more reasonable limit for draining
        #     dP_dt = max(dP_dt, -1e6)  # -10 bar/s max for pressure drop

        # # Limit temperature change rate (higher for refueling)
        # MAX_TEMP_CHANGE = 50.0  # K/s
        # dT_dt = np.clip(dT_dt, -MAX_TEMP_CHANGE, MAX_TEMP_CHANGE)


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
        fuel_flow_in: FuelFlow,
        fuel_flow_out: FuelFlow,
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
                    tank_state.tank_thermal_capacity,
                    tank_state.fuel_mass
                )
            ]
        ]

        # Handle mass_flow as list if needed
        in_flow_rate = fuel_flow_in.mass_flow
        if isinstance(in_flow_rate, list):
            in_flow_rate = in_flow_rate[0]  # Use first value if it's a list

        out_flow_rate = fuel_flow_out.mass_flow
        if isinstance(out_flow_rate, list):
            out_flow_rate = out_flow_rate[0]  # Use first value if it's a list

        b = [
            [
                cls.y1(
                    tank_state.fuel_mass,
                    tank_state.hydrogen,
                    in_flow_rate,
                    out_flow_rate
                )
            ], [
                cls.y2(
                    tank_state.hydrogen,
                    fuel_flow_in,
                    fuel_flow_out,
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
        tank_thermal_capacity: float,
        fuel_mass: float = None
    ) -> float:
        return (
            tank_thermal_capacity
            + tank_volume * hydrogen.density * hydrogen.dH_dT
        )

    @staticmethod
    def y1(
        fuel_mass: float,
        hydrogen: Hydrogen,
        fuel_flow_in: float,
        fuel_flow_out: float
    ) -> float:
        if fuel_mass == 0:
            return 0
        return hydrogen.density * (fuel_flow_in - fuel_flow_out) / fuel_mass

    @staticmethod
    def y2(
        tank_hydrogen: Hydrogen,
        fuel_flow_in: FuelFlow,
        fuel_flow_out: FuelFlow,
        heat_flux: float
    ) -> float:
        # Handle mass_flow as list if needed
        in_flow_rate = fuel_flow_in.mass_flow
        if isinstance(in_flow_rate, list):
            in_flow_rate = in_flow_rate[0]

        out_flow_rate = fuel_flow_out.mass_flow
        if isinstance(out_flow_rate, list):
            out_flow_rate = out_flow_rate[0]

        net_mass_flow = in_flow_rate - out_flow_rate

        if hasattr(fuel_flow_in, 'inlet_enthalpy') and fuel_flow_in.inlet_enthalpy is not None:
            h_in = fuel_flow_in.inlet_enthalpy
        else:
            h_in = fuel_flow_in.hydrogen.enthalpy

        h_tank = tank_hydrogen.enthalpy
        return heat_flux + net_mass_flow * (h_in - h_tank)

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


class SinglePhaseLimitLowerPressureInOutModel(SinglePhaseModelBase):
    """
    Single-phase dynamic model with lower pressure limit support that handles separate
    inflow and outflow fuel streams. Used when tank is at minimum pressure boundary.
    """

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> StateDerivatives:
        # Calculate net mass flow for derivatives (consistently subtracting outflow)
        inflow_sum = sum([flow.mass_flow for flow in fuel_flow_in])
        outflow_sum = sum([flow.mass_flow for flow in fuel_flow_out])

        # Use same convention as SinglePhaseInOutModel - subtract outflow
        net_mass_flow = inflow_sum - outflow_sum

        # Get mass derivatives based on phase
        dMg_dt, dMl_dt = cls.define_liquid_and_mass_derivatives(
            tank_state.phase, net_mass_flow
        )

        # Compute temperature derivative considering both flows
        dT_dt = cls.compute_temperature_derivative(tank_state, fuel_flow_in, fuel_flow_out)

        # Return state derivatives with required heat flux
        return StateDerivatives(
            cls.compute_pressure_derivative(),
            dT_dt,
            dMg_dt,
            dMl_dt,
            cls.compute_venting_mass(),
            cls.compute_required_heat_flux(tank_state, dT_dt, fuel_flow_in, fuel_flow_out)
        )

    @staticmethod
    def compute_pressure_derivative():
        return 0

    @staticmethod
    def compute_temperature_derivative(
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> float:
        # Calculate net mass flow (inflow - outflow)
        inflow_sum = sum([flow.mass_flow for flow in fuel_flow_in])
        outflow_sum = sum([flow.mass_flow for flow in fuel_flow_out])
        net_mass_flow = inflow_sum - outflow_sum

        num = (
            tank_state.hydrogen.density
            * net_mass_flow
            / tank_state.fuel_mass
        )
        den = tank_state.hydrogen.dRho_dT
        return num / den

    @staticmethod
    def define_liquid_and_mass_derivatives(
        tank_phase: str, net_mass_flow: float
    ) -> tuple[float, float]:
        if tank_phase == "gas":
            return net_mass_flow, 0
        if tank_phase == "liquid":
            return 0, net_mass_flow
        raise ValueError(
            f"{tank_phase} not supported in single phase model"
        )

    @classmethod
    def compute_venting_mass(cls):
        return 0

    @classmethod
    def compute_required_heat_flux(
        cls,
        tank_state: TankState,
        temperature_derivative: float,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> float:
        # thermal capacity term
        thermal_capacity_term = (
            tank_state.tank_thermal_capacity
            + tank_state.fuel_mass * tank_state.hydrogen.dH_dT
        )

        # Energy contribution from mass flows
        flow_energy = 0
        # Add inflow energy contribution
        for flow in fuel_flow_in:
            flow_energy += flow.mass_flow * (flow.hydrogen.enthalpy - tank_state.hydrogen.enthalpy)
        # add outflow energy contribution
        for flow in fuel_flow_out:
            flow_energy += flow.mass_flow * (tank_state.hydrogen.enthalpy - tank_state.hydrogen.enthalpy)

        # Required heat flux to maintain conditions
        return - (
            thermal_capacity_term * temperature_derivative
            - tank_state.heat_flux - flow_energy
        )


class TwoPhaseInOutModel(TwoPhaseModelBase):
    """
    Two-phase dynamic model that handles separate inflow and outflow fuel streams.
    """

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> StateDerivatives:
        # Debug information
        print(f"\n==== TwoPhaseInOutModel Debug ====")
        print(f"Tank state: P={tank_state.pressure/1e5:.2f}bar, T={tank_state.temperature:.2f}K")
        print(f"Tank phase: {tank_state.phase}")
        print(f"Tank hydrogen type: {type(tank_state.hydrogen).__name__}")

        # First, check if we're in a transition state (has _forced_phase = "twophase" but hydrogen.phase is still "gas")
        is_in_transition = False
        if hasattr(tank_state, '_forced_phase') and tank_state._forced_phase == "twophase":
            if not hasattr(tank_state.hydrogen, "liquid") or not hasattr(tank_state.hydrogen, "gas"):
                print(f"DETECTED TRANSITION STATE: Tank is marked as two-phase but hydrogen object is still single-phase")
                is_in_transition = True

                # Create a proper TwoPhaseHydrogen object to replace the current one
                from src.fluids.hydrogen_retrievers import TwoPhaseRequester
                # This will properly create liquid and gas phase objects
                try:
                    # First try creating with the current temperature/pressure
                    new_hydrogen = TwoPhaseRequester().get_hydrogen_properties(
                        tank_state.pressure, tank_state.temperature
                    )
                    print(f"Successfully created two-phase hydrogen at current conditions")
                    # Replace the hydrogen object
                    tank_state.hydrogen = new_hydrogen
                except Exception as e:
                    print(f"Error creating two-phase hydrogen at current conditions: {e}")
                    try:
                        # Try with a slightly adjusted temperature to hit saturation line
                        from CoolProp.CoolProp import PropsSI
                        # Get saturation temperature at current pressure
                        t_sat = PropsSI("T", "P", tank_state.pressure, "Q", 0, "hydrogen")
                        print(f"Using saturation temperature {t_sat:.2f}K instead of {tank_state.temperature:.2f}K")

                        # Create two-phase hydrogen at saturation
                        new_hydrogen = TwoPhaseRequester().get_hydrogen_properties(
                            tank_state.pressure, t_sat
                        )
                        print(f"Successfully created two-phase hydrogen at saturation conditions")
                        # Replace the hydrogen object
                        tank_state.hydrogen = new_hydrogen
                    except Exception as e2:
                        print(f"Failed to create two-phase hydrogen after adjustment: {e2}")
                        # We'll continue with the original hydrogen object

        # Debug hydrogen properties
        if hasattr(tank_state.hydrogen, "enthalpy"):
            print(f"Tank hydrogen has enthalpy attribute: {tank_state.hydrogen.enthalpy:.2f} J/kg")
        else:
            print(f"Tank hydrogen DOES NOT have enthalpy attribute")

        # Check for TwoPhaseHydrogen liquid/gas attributes
        if hasattr(tank_state.hydrogen, "liquid"):
            print(f"Tank hydrogen has liquid attribute with enthalpy: {tank_state.hydrogen.liquid.enthalpy:.2f} J/kg")
        if hasattr(tank_state.hydrogen, "gas"):
            print(f"Tank hydrogen has gas attribute with enthalpy: {tank_state.hydrogen.gas.enthalpy:.2f} J/kg")

        # Inflow details
        inflow_sum = sum([flow.mass_flow for flow in fuel_flow_in])
        print(f"Total inflow: {inflow_sum:.6f} kg/s from {len(fuel_flow_in)} flows")

        # Get critical properties for reference
        from CoolProp.CoolProp import PropsSI
        P_crit = PropsSI("Pcrit", "", 0, "", 0, "hydrogen")
        T_crit = PropsSI("Tcrit", "", 0, "", 0, "hydrogen")
        print(f"Hydrogen critical point: P_crit={P_crit/1e5:.2f}bar, T_crit={T_crit:.2f}K")
        print(f"Relative to critical: P/Pcrit={tank_state.pressure/P_crit:.3f}, T/Tcrit={tank_state.temperature/T_crit:.3f}")
        print(f"=====================================\n")

        # Ensure flows use appropriate hydrogen phase objects
        processed_inflows = []
        for flow in fuel_flow_in:
            flow_rate = flow.mass_flow
            if isinstance(flow_rate, list):
                flow_rate = flow_rate[0]
            processed_inflows.append(flow)

        combined_flows = []

        # Add inflows with positive mass flow
        for flow in processed_inflows:
            flow_rate = flow.mass_flow
            if isinstance(flow_rate, list):
                flow_rate = flow_rate[0]
            if flow_rate > 0:
                combined_flows.append(flow)

        # Add outflows with negated mass flow
        for flow in fuel_flow_out:
            flow_rate = flow.mass_flow
            if isinstance(flow_rate, list):
                flow_rate = flow_rate[0]
            if flow_rate > 0:
                # Create a flow with negative mass flow for outflows
                from src.mission.mission_sections import InFlow

                # Determine hydrogen phase for outflow
                if hasattr(flow, 'phase'):
                    # Create a new hydrogen object using safe methods
                    from src.fluids.hydrogen_retrievers import HydrogenRetriever
                    hydrogen_for_outflow = HydrogenRetriever().get_hydrogen_properties(
                        tank_state.pressure,
                        tank_state.temperature
                    )
                    print(f"Created hydrogen for outflow at P={tank_state.pressure/1e5:.2f}bar, T={tank_state.temperature:.2f}K")

                elif hasattr(flow, 'hydrogen'):
                    hydrogen_for_outflow = flow.hydrogen
                else:
                    # Default to creating a new hydrogen object at tank conditions
                    from src.fluids.hydrogen_retrievers import HydrogenRetriever
                    hydrogen_for_outflow = HydrogenRetriever().get_hydrogen_properties(
                        tank_state.pressure,
                        tank_state.temperature
                    )

                outflow = InFlow(-flow_rate, hydrogen_for_outflow)
                combined_flows.append(outflow)

        # Continue with matrix calculations as before
        try:
            # Make sure the hydrogen object is properly initialized for two-phase
            if is_in_transition or not hasattr(tank_state.hydrogen, "liquid") or not hasattr(tank_state.hydrogen, "gas"):
                print("Using simplified model for transition state...")
                # For transition states, use a simpler model that doesn't rely on matrix operations
                # Calculate estimated derivatives based on energy and mass balance

                # Temperature derivative based on heat input/output
                dT_dt = -0.5  # K/s (cooling effect from expansion)

                # Pressure derivative based on inflow/outflow
                dP_dt = 0  # Pa/s (at saturation line pressure is coupled to temperature)

                # Mass derivatives - all new mass goes to liquid phase during transition
                total_inflow = sum([flow.mass_flow for flow in fuel_flow_in])
                total_outflow = sum([flow.mass_flow for flow in fuel_flow_out])
                net_flow = total_inflow - total_outflow

                # For transition, initially all new mass becomes liquid
                dMl_dt = max(0.0, net_flow)  # Ensure non-negative liquid growth
                dMg_dt = net_flow - dMl_dt    # Rest goes to gas phase

                print(f"Transition state derivatives: dP_dt={dP_dt:.2f}, dT_dt={dT_dt:.2f}, dMl_dt={dMl_dt:.6f}, dMg_dt={dMg_dt:.6f}")

                return StateDerivatives(
                    dP_dt,
                    dT_dt,
                    dMg_dt,
                    dMl_dt,
                    cls.venting_mass(),
                    cls.added_heat_flux()
                )

            # Normal case - use the matrix approach
            import numpy as np
            a = cls.define_a_matrix(tank_state)
            b = cls.define_b_vector(tank_state, combined_flows)

            print("Attempting to solve matrix equation...")
            x = np.linalg.solve(a, b)

            # Use the raw values without limits
            dP_dt = x[0][0]
            dT_dt = x[1][0]
            dMg_dt = x[2][0]
            dMl_dt = x[3][0]

            print(f"Raw derivatives - dP_dt: {dP_dt:.2f} Pa/s, dT_dt: {dT_dt:.2f} K/s")
            print(f"Raw derivatives - dMg_dt: {dMg_dt:.6f} kg/s, dMl_dt: {dMl_dt:.6f} kg/s")

            # Apply reasonable limits to prevent instability
            # Limit pressure and temperature changes
            MAX_PRESSURE_RATE = 5e6  # Pa/s (50 bar/s)
            MAX_TEMPERATURE_RATE = 5.0  # K/s

            dP_dt = np.clip(dP_dt, -MAX_PRESSURE_RATE, MAX_PRESSURE_RATE)
            dT_dt = np.clip(dT_dt, -MAX_TEMPERATURE_RATE, MAX_TEMPERATURE_RATE)

            return StateDerivatives(
                dP_dt,
                dT_dt,
                dMg_dt,
                dMl_dt,
                cls.venting_mass(),
                cls.added_heat_flux()
            )
        except Exception as e:
            # In case of matrix solving errors, create a detailed debug report
            print(f"Error solving matrix equation: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Returning default state derivatives")

            # Create safe fallback values that don't cause instability
            # Use a small negative temperature derivative
            dT_dt = -0.1  # K/s (slight cooling)
            # Keep pressure constant
            dP_dt = 0.0  # Pa/s

            # Mass derivatives - distribute based on inflow/outflow
            total_inflow = sum([flow.mass_flow for flow in fuel_flow_in])
            total_outflow = sum([flow.mass_flow for flow in fuel_flow_out])
            net_flow = total_inflow - total_outflow

            # Split the flow between phases, assuming half goes to each
            dMl_dt = net_flow * 0.5
            dMg_dt = net_flow * 0.5

            return StateDerivatives(
                dP_dt,
                dT_dt,
                dMg_dt,
                dMl_dt,
                0,
                tank_state.heat_flux
            )

    @classmethod
    def define_a_matrix(
        cls,
        tank_state: TankState
    ) -> list:
        """Define the A matrix for the system of equations."""
        # Import numpy for calculations
        import numpy as np

        # Use TwoPhaseModel's static methods to calculate matrix components
        a12 = TwoPhaseModel.a12(
            tank_state.hydrogen
        )
        a21 = TwoPhaseModel.a21(
            tank_state.gas_mass,
            tank_state.liquid_mass,
            tank_state.hydrogen
        )
        a22 = TwoPhaseModel.a22(
            tank_state.gas_mass,
            tank_state.liquid_mass,
            tank_state.hydrogen
        )
        a23 = TwoPhaseModel.a23(
            tank_state.hydrogen
        )
        a42 = TwoPhaseModel.a42(
            tank_state.tank_thermal_capacity,
            tank_state.volume,
            tank_state.gas_mass,
            tank_state.liquid_mass,
            tank_state.hydrogen
        )
        a43 = TwoPhaseModel.a43(
            tank_state.hydrogen
        )
        a = [
            [1, a12, 0, 0],
            [a21, a22, a23, 0],
            [0, 0, 1, 1],
            [0, a42, a43, 0]
        ]

        return a

    @classmethod
    def define_b_vector(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> list:
        """Define the B vector for the system of equations."""
        # Use TwoPhaseModel's static methods to calculate vector components
        y2 = TwoPhaseModel.y2(
            fuel_flows,
            tank_state.hydrogen
        )
        y3 = TwoPhaseModel.y3(
            fuel_flows
        )
        y4 = TwoPhaseModel.y4(
            fuel_flows,
            tank_state.hydrogen,
            tank_state.heat_flux
        )
        b = [[0], [y2], [y3], [y4]]
        return b

    @classmethod
    def venting_mass(cls) -> float:
        return 0

    @classmethod
    def added_heat_flux(cls) -> float:
        return 0


class TwoPhaseLimitLowerPressureInOutModel(TwoPhaseModelBase):
    """
    Two-phase dynamic model with lower pressure limit support that handles separate
    inflow and outflow fuel streams. Used when tank is at minimum pressure boundary.
    """

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> StateDerivatives:
        # Calculate net mass flow (inflow - outflow)
        inflow_sum = sum([flow.mass_flow for flow in fuel_flow_in])
        outflow_sum = sum([flow.mass_flow for flow in fuel_flow_out])
        net_mass_flow = inflow_sum - outflow_sum

        # Create a combined flow list for compatibility with existing methods
        from src.mission.mission_sections import InFlow
        combined_flows = [InFlow(net_mass_flow, tank_state.hydrogen.liquid)]

        return StateDerivatives(
            cls.compute_pressure_derivative(),
            cls.compute_temperature_derivative(),
            cls.compute_gas_mass_derivative(tank_state.hydrogen, combined_flows),
            cls.compute_liquid_mass_derivative(tank_state.hydrogen, combined_flows),
            cls.compute_venting_mass(),
            cls.compute_required_heat_flux(
                tank_state.hydrogen,
                combined_flows,
                tank_state.heat_flux,
                fuel_flow_in,
                fuel_flow_out
            )
        )

    @staticmethod
    def compute_pressure_derivative() -> float:
        return 0

    @staticmethod
    def compute_temperature_derivative() -> float:
        return 0

    @staticmethod
    def compute_gas_mass_derivative(
        hydrogen: TwoPhaseHydrogen, combined_flows: list[FuelFlow]
    ) -> float:
        return (
            sum([flow.mass_flow for flow in combined_flows]) / (
                1 - hydrogen.liquid.density / hydrogen.gas.density
            )
        )

    @staticmethod
    def compute_liquid_mass_derivative(
        hydrogen: TwoPhaseHydrogen, combined_flows: list[FuelFlow]
    ) -> float:
        return (
            sum([flow.mass_flow for flow in combined_flows]) / (
                1 - hydrogen.gas.density / hydrogen.liquid.density
            )
        )

    @staticmethod
    def compute_venting_mass() -> float:
        return 0

    @classmethod
    def compute_required_heat_flux(
        cls,
        hydrogen: TwoPhaseHydrogen,
        combined_flows: list[FuelFlow],
        heat_flux: float,
        fuel_flow_in: list[FuelFlow] = None,
        fuel_flow_out: list[FuelFlow] = None
    ) -> float:
        # Calculate energy contributions from mass derivatives
        t1 = hydrogen.liquid.enthalpy * cls.compute_liquid_mass_derivative(
            hydrogen, combined_flows
        )
        t2 = hydrogen.gas.enthalpy * cls.compute_gas_mass_derivative(
            hydrogen, combined_flows
        )

        # Calculate energy contributions from flows
        flow_energy = 0
        # Add energy from inflows
        if fuel_flow_in:
            for flow in fuel_flow_in:
                flow_energy += flow.mass_flow * flow.hydrogen.enthalpy

        # Add energy from outflows
        if fuel_flow_out:
            for flow in fuel_flow_out:
                if hasattr(flow, 'hydrogen'):
                    flow_energy -= flow.mass_flow * flow.hydrogen.enthalpy
                else:
                    # If outflow has no hydrogen property, use tank phase
                    flow_energy -= flow.mass_flow * (
                        hydrogen.gas.enthalpy if flow.phase == "gas"
                        else hydrogen.liquid.enthalpy
                    )

        # Return required heat flux
        return - (t1 + t2 - flow_energy - heat_flux)


class SinglePhaseLimitUpperPressureModel(SinglePhaseModelBase):

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        # Calculate temperature derivative
        dT_dt = cls.compute_temperature_derivative(tank_state, fuel_flows)

        # Calculate required venting flow to maintain constant pressure
        venting_flow = cls.compute_required_venting_flow(tank_state, fuel_flows, dT_dt)

        # Add venting flow to original flows for mass derivative calculation
        total_flows = fuel_flows.copy()
        total_flows.append(OutFlow(venting_flow, "gas"))

        # Calculate mass derivatives with total flows (including venting)
        dMg_dt, dMl_dt = cls.define_liquid_and_mass_derivatives(
            tank_state.phase, total_flows
        )

        return StateDerivatives(
            cls.compute_pressure_derivative(),  # Always 0
            dT_dt,
            dMg_dt,
            dMl_dt,
            venting_flow,
            tank_state.heat_flux
        )

    @staticmethod
    def compute_pressure_derivative():
        return 0  # Constant pressure

    @staticmethod
    def compute_temperature_derivative(
        tank_state: TankState, fuel_flows: list[FuelFlow]
    ) -> float:
        # Implement equation: dT/dt = Q_in / [m_s*C_s + m_H2*(∂h_H2/∂T)_P]
        thermal_capacity = (
            tank_state.tank_thermal_capacity +
            tank_state.fuel_mass * tank_state.hydrogen.dH_dT
        )

        # Add a small epsilon to prevent division by zero
        return tank_state.heat_flux / (thermal_capacity + 1e-10)

    @classmethod
    def compute_required_venting_flow(
        cls, tank_state: TankState, fuel_flows: list[FuelFlow], dT_dt: float
    ) -> float:
        # Implement equation: m_H2_out = (m_H2/ρ_H2)*(∂ρ_H2/∂T)_P * dT/dt  (Ahluwalia and Peng)

        # Calculate required venting flow to maintain constant pressure
        venting_flow = (
            -tank_state.fuel_mass / tank_state.hydrogen.density *
            tank_state.hydrogen.dRho_dT * dT_dt
        )

        return venting_flow

    @staticmethod
    def define_liquid_and_mass_derivatives(
        tank_phase: str, fuel_flows: list[FuelFlow]
    ) -> tuple[float, float]:
        # Calculate net mass flow, accounting for OutFlow vs InFlow
        net_flow = 0
        for flow in fuel_flows:
            # Check if this is an OutFlow (should be subtracted) or InFlow (should be added)
            if hasattr(flow, '__class__') and flow.__class__.__name__ == 'OutFlow':
                net_flow -= flow.mass_flow  # Subtract outflows
            else:
                net_flow += flow.mass_flow  # Add inflows

        # Return appropriate derivatives based on phase
        if tank_phase == "gas":
            return net_flow, 0
        if tank_phase == "liquid":
            return 0, net_flow
        raise ValueError(
            f"{tank_phase} not supported in single phase model"
        )


class TwoPhaseLimitUpperPressureModel(DynamicModel):

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        # Temperature derivative is zero in two-phase at constant pressure (saturated conditions)
        dT_dt = cls.compute_temperature_derivative()  # Always 0 for two-phase at constant pressure

        # Calculate required venting flow using the heat input
        venting_flow = cls.compute_required_venting_flow(tank_state)

        # Calculate gas and liquid mass derivatives using the venting flow
        dMg_dt = cls.compute_gas_mass_derivative(tank_state.hydrogen, venting_flow)
        dMl_dt = cls.compute_liquid_mass_derivative(tank_state.hydrogen, venting_flow)

        return StateDerivatives(
            cls.compute_pressure_derivative(),  # Always 0
            dT_dt,
            dMg_dt,
            dMl_dt,
            venting_flow,  # Track venting flow separately
            tank_state.heat_flux  # No additional heat flux
        )

    @staticmethod
    def compute_pressure_derivative() -> float:
        return 0  # Constant pressure

    @staticmethod
    def compute_temperature_derivative() -> float:
        # dT/dt = (dP_s/dT)^(-1) * dP/dt = 0 since dP/dt = 0
        return 0  # Temperature is constant in two-phase at constant pressure

    @classmethod
    def compute_required_venting_flow(
        cls, tank_state: TankState
    ) -> float:
        # Using the equation: ṁ_H2^out = ((ρ_l - ρ_g) * Q̇_in^r) / (ρ_l * (h_g - h_l))
        hydrogen = tank_state.hydrogen

        # Get the properties
        rho_l = hydrogen.liquid.density
        rho_g = hydrogen.gas.density
        h_g = hydrogen.gas.enthalpy
        h_l = hydrogen.liquid.enthalpy
        heat_input = tank_state.heat_flux

        # Calculate venting flow rate
        # Ensure we don't divide by zero
        denominator = rho_l * (h_g - h_l)
        if abs(denominator) < 1e-10:
            return 0.0

        venting_flow = ((rho_l - rho_g) * heat_input) / denominator
        return venting_flow

    @staticmethod
    def compute_gas_mass_derivative(
        hydrogen: TwoPhaseHydrogen, venting_flow: float
    ) -> float:
        # Using the equation: dm_g/dt = -ṁ_H2^out / (1 - (ρ_l/ρ_g))
        rho_l = hydrogen.liquid.density
        rho_g = hydrogen.gas.density

        # Avoid division by zero
        denominator = 1.0 - (rho_l / rho_g)
        if abs(denominator) < 1e-10:
            return 0.0

        return -venting_flow / denominator

    @staticmethod
    def compute_liquid_mass_derivative(
        hydrogen: TwoPhaseHydrogen, venting_flow: float
    ) -> float:
        # Using the equation: dm_l/dt = -ṁ_H2^out / (1 - (ρ_g/ρ_l))
        rho_l = hydrogen.liquid.density
        rho_g = hydrogen.gas.density

        # Avoid division by zero
        denominator = 1.0 - (rho_g / rho_l)
        if abs(denominator) < 1e-10:
            return 0.0

        return -venting_flow / denominator

class TwoPhaseRefuelModel(TwoPhaseModelBase):
    """
    Two-phase refuelling model using the novel specific isochoric two-phase heat capacity (c_v2P)
    with relaxation-based saturation constraint.

    This model is based on the conventions of TwoPhaseModel but modifies the approach to handle
    the saturation constraint using a relaxation time constant rather than strict enforcement.

    Key differences from standard models:
    1. Uses cv2phase for energy equation (as described in generalized thermodynamic model paper)
    2. Implements relaxation approach: dP/dt = (P_sat(T) - P) / τ_relax + pressure_effects
    3. Allows gradual approach to thermodynamic equilibrium during rapid processes like refueling

    The relaxation approach is physically motivated by:
    - Non-equilibrium effects during rapid refueling
    - Finite heat and mass transfer rates
    - Local temperature and pressure gradients within the tank
    """

    # Class-level configuration for relaxation time constant
    RELAXATION_TIME_CONSTANT = 0.1  # seconds - increased for better stability

    @classmethod
    def set_relaxation_time_constant(cls, tau_relax: float) -> None:
        """
        Set the relaxation time constant for the model.

        Args:
            tau_relax: Relaxation time constant in seconds
                     - Smaller values (e.g., 0.01-0.1s): Faster equilibration, more responsive
                     - Larger values (e.g., 1-10s): Slower equilibration, more stable
        """
        cls.RELAXATION_TIME_CONSTANT = tau_relax
        print(f"TwoPhaseRefuelModel relaxation time constant set to {tau_relax:.3f} seconds")

    @classmethod
    def get_relaxation_time_constant(cls) -> float:
        """Get the current relaxation time constant."""
        return cls.RELAXATION_TIME_CONSTANT

    @classmethod
    def compute_cv2phase(cls, tank_state: TankState) -> float:
        """
        Compute the two-phase isochoric heat capacity.

        This includes a correction term for the isochoric path in two-phase region.
        """
        # Get hydrogen properties
        hydrogen = tank_state.hydrogen

        # Vapor mass fraction
        total_mass = tank_state.gas_mass + tank_state.liquid_mass
        if total_mass <= 0:
            return 0.0  # Avoid division by zero
        alpha = tank_state.gas_mass / total_mass

        # Use CoolProp to get single-phase cv at saturation states
        fluid = "hydrogen"  # Assuming hydrogen is the working fluid
        temperature = tank_state.temperature

        cv_g = PropsSI("CVMASS", "T", temperature, "Q", 1, fluid)
        cv_l = PropsSI("CVMASS", "T", temperature, "Q", 0, fluid)

        # Get saturated densities
        rho_g = hydrogen.gas.density
        rho_l = hydrogen.liquid.density

        # Get derivatives of density wrt T at saturation
        drho_g_dT = hydrogen.gas.dRho_dT
        drho_l_dT = hydrogen.liquid.dRho_dT

        # Get saturated enthalpies
        h_g = hydrogen.gas.enthalpy
        h_l = hydrogen.liquid.enthalpy

        # Calculate correction term for isochoric path
        denom = (1.0 / rho_g - 1.0 / rho_l)

        # Avoid division by zero
        if abs(denom) < 1e-10:
            return alpha * cv_g + (1.0 - alpha) * cv_l

        dvg_dT = -drho_g_dT / (rho_g**2)
        dvl_dT = -drho_l_dT / (rho_l**2)

        correction = (h_g - h_l) / denom * (dvg_dT - dvl_dT)

        return alpha * cv_g + (1.0 - alpha) * cv_l + correction

    @classmethod
    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow] = None
    ) -> StateDerivatives:
        """
        Compute state derivatives for the two-phase refuel model.

        This method handles both inflow and outflow fuel streams, with special focus on refueling.
        """
        if fuel_flow_out is None:
            fuel_flow_out = []

        print(f"\n==== TwoPhaseRefuelModel Debug ====")
        print(f"Tank state: P={tank_state.pressure/1e5:.2f}bar, T={tank_state.temperature:.2f}K")
        print(f"Tank phase: {tank_state.phase}")
        print(f"Inflow rate: {sum([flow.mass_flow for flow in fuel_flow_in]):.6f} kg/s")
        print(f"Outflow rate: {sum([flow.mass_flow for flow in fuel_flow_out]):.6f} kg/s")

        # Setup matrices and solve the system
        try:
            a = cls.define_a_matrix(tank_state)
            b = cls.define_b_vector(tank_state, fuel_flow_in, fuel_flow_out)

            print(f"A matrix shape: {a.shape}")
            print(f"b vector shape: {b.shape}")

            x = np.linalg.solve(a, b)

            # Extract derivatives from solution - NO SCALING UNDOING
            # The solution is already in proper units because scaling was applied to both A and b
            dP_dt = x[0][0]  # Already in Pa/s
            dT_dt = x[1][0]
            dMg_dt = x[2][0]
            dMl_dt = x[3][0]

            print(f"Solution - dP_dt: {dP_dt:.2f} Pa/s, dT_dt: {dT_dt:.2f} K/s")
            print(f"Solution - dMg_dt: {dMg_dt:.6f} kg/s, dMl_dt: {dMl_dt:.6f} kg/s")
            print(f"=====================================\n")

            return StateDerivatives(
                dP_dt,
                dT_dt,
                dMg_dt,
                dMl_dt,
                cls.venting_mass(),
                cls.added_heat_flux()
            )

        except Exception as e:
            print(f"Error in TwoPhaseRefuelModel: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return safe fallback values
            return StateDerivatives(0, 0, 0, 0, 0, tank_state.heat_flux)

    @staticmethod
    def a12(hydrogen: TwoPhaseHydrogen) -> float:
        """Calculate coefficient a12 in the A matrix (saturation pressure derivative)."""
        return hydrogen.dP_dT

    @staticmethod
    def a21(
        gas_mass: float,
        liquid_mass: float,
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        """Calculate coefficient a21 in the A matrix."""
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
        return -(term1 + term2)

    @staticmethod
    def a22(
        gas_mass: float,
        liquid_mass: float,
        hydrogen: TwoPhaseHydrogen
    ) -> float:
        """Calculate coefficient a22 in the A matrix."""
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
        return -(term1 + term2)

    @staticmethod
    def a23(hydrogen: TwoPhaseHydrogen) -> float:
        """Calculate coefficient a23 in the A matrix."""
        return (
            1.0 / hydrogen.gas.density
            - 1.0 / hydrogen.liquid.density
        )

    @classmethod
    def a42(
        cls,
        tank_state: TankState
    ) -> float:
        """
        Calculate coefficient a42 in the A matrix using cv2phase.

        This is the key difference from the standard TwoPhaseModel.
        Includes both structure thermal capacity and hydrogen cv2phase.
        """
        # Use two-phase isochoric heat capacity for hydrogen
        cv2p = cls.compute_cv2phase(tank_state)
        P_sat = PropsSI("P", "T", tank_state.temperature, "Q", 0.5, "hydrogen")

        # Include both structure thermal capacity and hydrogen thermal capacity
        # TEST_CP is the structure's heat capacity
        # return TEST_CP + tank_state.fuel_mass * cv2p

        return tank_state.tank_thermal_capacity + tank_state.gas_mass*tank_state.hydrogen.gas.dH_dT + tank_state.liquid_mass*tank_state.hydrogen.liquid.dH_dT + (tank_state.gas_mass*tank_state.hydrogen.gas.dH_dP + tank_state.liquid_mass*tank_state.hydrogen.liquid.dH_dP - tank_state.volume)*P_sat  # Approximate total fuel thermal capacity
        # return TEST_CP + tank_state.gas_mass*tank_state.hydrogen.gas.dH_dT + tank_state.liquid_mass*tank_state.hydrogen.liquid.dH_dT + (tank_state.gas_mass*tank_state.hydrogen.gas.dH_dP + tank_state.liquid_mass*tank_state.hydrogen.liquid.dH_dP - tank_state.volume)*P_sat  # Approximate total fuel thermal capacity
        # return tank_state.gas_mass*tank_state.hydrogen.gas.dH_dT + tank_state.liquid_mass*tank_state.hydrogen.liquid.dH_dT + (tank_state.gas_mass*tank_state.hydrogen.gas.dH_dP + tank_state.liquid_mass*tank_state.hydrogen.liquid.dH_dP - tank_state.volume)*P_sat

    @staticmethod
    def a43(hydrogen: TwoPhaseHydrogen) -> float:
        """Calculate coefficient a43 in the A matrix."""
        return -(
            hydrogen.liquid.enthalpy
            - hydrogen.gas.enthalpy
        )

    @classmethod
    def define_a_matrix(cls, tank_state: TankState) -> np.ndarray:
        """
        Define the A matrix for the system of equations using cv2phase approach.

        For refueling: Use a modified saturation constraint that allows for relaxation
        while maintaining proper thermodynamic coupling.
        """
        # Modified saturation constraint with relaxation:
        # Use balanced approach: dP/dt - α*dP_sat/dT * dT/dt = (1-α)*relaxation_term
        # where α is the coupling strength (0 = pure relaxation, 1 = strict saturation)

        hydrogen = tank_state.hydrogen

        # Use moderate coupling strength parameter to prevent ill-conditioning
        # This maintains proper matrix balance between A and b
        coupling_strength = 0.0  # Consistent with b vector definition

        a11 = 1.0  # Full coefficient for dP/dt
        a12 = -coupling_strength * cls.a12(hydrogen)  # Coupled to dT/dt with proper sign

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
        a23 = cls.a23(tank_state.hydrogen)
        a42 = cls.a42(tank_state)
        a43 = cls.a43(tank_state.hydrogen)

        # Apply scaling to improve conditioning
        pressure_scale = 1e-6  # Scale pressure derivatives to bar/s range

        # Matrix with balanced saturation constraint and improved scaling
        a = np.array([
            [a11 * pressure_scale, a12 * pressure_scale, 0.0, 0.0],  # Row 1: Balanced saturation constraint
            [a21, a22, a23, 0.0],  # Row 2: Volume constraint
            [0.0, 0.0, 1.0, 1.0],  # Row 3: Mass balance
            [0.0, a42, a43, 0.0]   # Row 4: Energy equation with cv2phase
        ])

        return a

    @staticmethod
    def y2(
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow] = None
    ) -> float:
        """Calculate the y2 element of the b vector."""
        hydrogen = tank_state.hydrogen
        return sum([
            -flow.mass_flow / hydrogen.liquid.density
            for flow in fuel_flow_in
        ])

    @staticmethod
    def y3(
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> float:
        """Calculate the y3 element of the b vector."""
        inflow_sum = sum([flow.mass_flow for flow in fuel_flow_in])
        outflow_sum = sum([flow.mass_flow for flow in fuel_flow_out])
        return inflow_sum - outflow_sum

    @staticmethod
    def y4(
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow] = None,
        hydrogen: TwoPhaseHydrogen = None,
        heat_flux: float = 0.0
    ) -> float:
        """Calculate the y4 element of the b vector."""
        print(f"\n==== TwoPhaseRefuelModel.y4 Debug ====")
        print(f"Tank state: P={hydrogen.gas.pressure/1e5:.2f}bar, T={hydrogen.gas.temperature:.2f}K")
        print(f"Tank phase: twophase")

        # Energy contribution from inflows
        inflow_energy = 0
        total_inflow = 0

        # Improved handling for refueling with adaptive reference enthalpy
        for flow in fuel_flow_in:
            total_inflow += flow.mass_flow

            # First check if this flow has inlet_enthalpy value from the cryopump
            if hasattr(flow, "inlet_enthalpy") and flow.inlet_enthalpy is not None:
                # Use the raw enthalpy directly without modifications
                h_in = flow.inlet_enthalpy
                print(f"Using raw inlet_enthalpy: {h_in:.2f} J/kg")
            # Then check if flow has direct_enthalpy (older approach)
            elif hasattr(flow, "direct_enthalpy"):
                # Get the raw enthalpy from the cryopump
                h_in = flow.direct_enthalpy
                print(f"Using direct_enthalpy: {h_in:.2f} J/kg")
            # Then try regular enthalpy access methods
            elif hasattr(flow.hydrogen, "enthalpy"):
                h_in = flow.hydrogen.enthalpy
                print(f"Using hydrogen.enthalpy: {h_in:.2f} J/kg")
            elif hasattr(flow.hydrogen, "liquid") and hasattr(flow.hydrogen.liquid, "enthalpy"):
                h_in = flow.hydrogen.liquid.enthalpy
                print(f"Using hydrogen.liquid.enthalpy: {h_in:.2f} J/kg")
            else:
                # Default to standard conditions if we can't get enthalpy
                h_in = PropsSI("H", "P", 101325, "T", 20+273.15, "hydrogen")
                print(f"Using default enthalpy: {h_in:.2f} J/kg")

# Use adaptive reference enthalpy that varies with gas mass fraction
            if hasattr(hydrogen, 'tank_state') and hasattr(hydrogen.tank_state, 'gas_mass') and \
               hasattr(hydrogen.tank_state, 'liquid_mass'):
                total_mass = max(hydrogen.tank_state.gas_mass + hydrogen.tank_state.liquid_mass, 1e-10)

                # Handle negative gas mass as indicator to use liquid reference
                if hydrogen.tank_state.gas_mass < 0:
                    print(f"WARNING: Negative gas mass {hydrogen.tank_state.gas_mass:.6f}kg detected in energy calculation")
                    alpha = 0.0  # Force liquid-only reference
                else:
                    alpha = hydrogen.tank_state.gas_mass / total_mass

                # Blend between liquid and gas reference enthalpy based on composition
                # When alpha is small (mostly liquid), use more liquid reference
                blend_factor = max(0, min(1, 1-alpha))  # More liquid = more liquid reference
                print(f"Gas mass fraction alpha: {alpha:.3f}, blend factor: {blend_factor:.3f}")

                # Calculate reference enthalpy as a blend of liquid and gas enthalpies
                try:
                    h_ref = blend_factor * hydrogen.liquid.enthalpy + (1-blend_factor) * hydrogen.gas.enthalpy
                    print(f"Using blended reference enthalpy: {h_ref:.2f} J/kg")
                    inflow_energy += flow.mass_flow * (h_in - h_ref)
                except Exception as e:
                    print(f"Error calculating blended enthalpy: {e}")
                    # Fallback to liquid reference enthalpy
                    inflow_energy += flow.mass_flow * (h_in - hydrogen.liquid.enthalpy)
            else:
                # For backward compatibility
                inflow_energy += flow.mass_flow * (h_in - hydrogen.liquid.enthalpy)

            print(f"Inflow: {flow.mass_flow:.6f} kg/s, h_in: {h_in:.2f} J/kg, energy contribution: {flow.mass_flow * (h_in - hydrogen.liquid.enthalpy):.2f} W")

        print(f"Inflow rate: {total_inflow:.6f} kg/s")

        # Energy contribution from outflows
        outflow_energy = 0
        total_outflow = 0
        if fuel_flow_out:  # Check if outflow list exists and is not empty
            for i, flow in enumerate(fuel_flow_out):
                total_outflow += flow.mass_flow

                # Try to determine the phase of outflow
                if hasattr(flow, "phase") and flow.phase == "liquid":
                    h_out = hydrogen.liquid.enthalpy
                    phase_str = "liquid"
                else:
                    # Default to gas phase for outflow
                    h_out = hydrogen.gas.enthalpy
                    phase_str = "gas"

                flow_energy = -flow.mass_flow * h_out
                outflow_energy += flow_energy
                print(f"Outflow {i}: {flow.mass_flow:.6f} kg/s, phase: {phase_str}, h_out: {h_out:.2f} J/kg, energy: {flow_energy:.2f} W")

        print(f"Total outflow rate: {total_outflow:.6f} kg/s, energy: {outflow_energy:.2f} W")
        # Calculate the total energy change
        total_energy = heat_flux + inflow_energy + outflow_energy
        print(f"Energy balance: heat_flux={heat_flux:.2f} W + inflow_energy={inflow_energy:.2f} W + outflow_energy={outflow_energy:.2f} W = {total_energy:.2f} W")
        print(f"=====================================\n")
        return total_energy

    @classmethod
    def compute_relaxation_pressure_rhs(
        cls,
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> float:
        """
        Compute the right-hand side of the relaxation pressure equation.

        dP/dt = (P_sat(T) - P) / τ_relax + pressure_rise_from_mass_addition

        Returns the relaxation term plus pressure effects from mass flows.
        """
        # Use class-level relaxation time constant
        tau_relax = cls.RELAXATION_TIME_CONSTANT

        # Get current saturation pressure at tank temperature
        from CoolProp.CoolProp import PropsSI
        try:
            P_sat = PropsSI("P", "T", tank_state.temperature, "Q", 0.5, "hydrogen")
        except Exception as e:
            # Fallback to current pressure if saturation pressure calculation fails
            print(f"Warning: Could not calculate P_sat, using current pressure. Error: {e}")
            P_sat = tank_state.pressure

        # Relaxation term: drives pressure toward saturation value
        relaxation_term = (P_sat - tank_state.pressure) / tau_relax

        # Pressure rise from mass addition (simplified)
        # This approximates the immediate pressure response to mass flows
        total_inflow = sum([flow.mass_flow for flow in fuel_flow_in])
        total_outflow = sum([flow.mass_flow for flow in fuel_flow_out])
        net_mass_flow = total_inflow - total_outflow

        # Estimate pressure rise from mass addition using bulk modulus approach
        # dP ≈ (ρ * dV/dt) / (V * compressibility)
        if tank_state.fuel_mass > 0 and tank_state.volume > 0:
            # Use hydrogen compressibility (inverse of bulk modulus)
            try:
                if hasattr(tank_state.hydrogen, 'liquid') and hasattr(tank_state.hydrogen.liquid, 'dRho_dP'):
                    compressibility = 1.0 / tank_state.hydrogen.liquid.dRho_dP
                else:
                    # Fallback compressibility for hydrogen
                    compressibility = 1e-9  # Pa⁻¹

                mass_effect = (net_mass_flow / tank_state.fuel_mass) / (compressibility * tank_state.volume)
            except Exception:
                mass_effect = 0.0
        else:
            mass_effect = 0.0

        total_pressure_rate = relaxation_term + mass_effect

        print(f"Relaxation: P_sat={P_sat/1e5:.2f}bar, P={tank_state.pressure/1e5:.2f}bar, "
              f"τ={tau_relax:.2f}s, relaxation_term={relaxation_term/1e5:.2f}bar/s, "
              f"mass_effect={mass_effect/1e5:.2f}bar/s, total={total_pressure_rate/1e5:.2f}bar/s")

        return total_pressure_rate

    @classmethod
    def define_b_vector(
        cls,
        tank_state: TankState,
        fuel_flow_in: list[FuelFlow],
        fuel_flow_out: list[FuelFlow]
    ) -> np.ndarray:
        """Define the b vector for the system of equations.

        Using balanced saturation constraint with proper relaxation term.
        """
        # For balanced saturation constraint:
        # dP/dt - α*dP_sat/dT * dT/dt = (1-α)*relaxation_pressure_term
        coupling_strength = 0.5  # Same as used in A matrix
        relaxation_pressure_term = cls.compute_relaxation_pressure_rhs(tank_state, fuel_flow_in, fuel_flow_out)

        # Apply the same pressure scaling as in A matrix
        pressure_scale = 1e-6  # Scale pressure derivatives to bar/s range

        # Right-hand side is the relaxation term weighted by (1 - coupling_strength)
        y1 = (1.0 - coupling_strength) * relaxation_pressure_term * pressure_scale

        # Volume constraint RHS
        y2 = cls.y2(tank_state, fuel_flow_in, fuel_flow_out)

        # Mass balance RHS
        y3 = cls.y3(fuel_flow_in, fuel_flow_out)

        # Energy equation RHS using cv2phase approach
        y4 = cls.y4(
            fuel_flow_in,
            fuel_flow_out,
            tank_state.hydrogen,
            tank_state.heat_flux
        )

        b = np.array([[y1], [y2], [y3], [y4]])
        return b

    @classmethod
    def venting_mass(cls) -> float:
        """Return the venting mass flow rate."""
        return 0.0

    @classmethod
    def added_heat_flux(cls) -> float:
        """Return any additional heat flux."""
        return 0.0

