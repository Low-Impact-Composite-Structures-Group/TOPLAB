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
        # Safety check - ensure we have a flow with hydrogen properties
        if not fuel_flows or not hasattr(fuel_flows[0], 'hydrogen'):
            # Create a dummy flow with hydrogen properties from tank
            from src.mission.mission_sections import OutFlow
            dummy_flow = OutFlow(0.0, "gas")
            dummy_flow.hydrogen = tank_state.hydrogen
            fuel_flows = [dummy_flow]

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
            tank_state.phase, in_flow_rate - out_flow_rate
        )

        # Apply safety limits to prevent non-physical values
        # Limit pressure change rate (max 1 bar/s)
        MAX_PRESSURE_CHANGE = 1e5  # Pa/s
        dP_dt = np.clip(dP_dt, -MAX_PRESSURE_CHANGE, MAX_PRESSURE_CHANGE)

        # Limit temperature change rate (max 1K/s)
        MAX_TEMP_CHANGE = 1.0  # K/s
        dT_dt = np.clip(dT_dt, -MAX_TEMP_CHANGE, MAX_TEMP_CHANGE)


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
                    tank_state.tank_thermal_capacity
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
        h_in = fuel_flow_in.hydrogen.enthalpy
        return heat_flux + net_mass_flow * (h_in - tank_hydrogen.enthalpy)

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
        # Safety check - ensure both lists have at least one item
        if not fuel_flow_in:
            # Create a dummy inflow with zero rate
            from src.mission.mission_sections import InFlow
            dummy_inflow = InFlow(0.0, tank_state.hydrogen.liquid)
            fuel_flow_in = [dummy_inflow]

        if not fuel_flow_out:
            # Create a dummy outflow with zero rate
            from src.mission.mission_sections import OutFlow
            dummy_outflow = OutFlow(0.0, "gas")
            fuel_flow_out = [dummy_outflow]

        # Ensure flows use appropriate hydrogen phase objects
        processed_inflows = []
        for flow in fuel_flow_in:
            flow_rate = flow.mass_flow
            if isinstance(flow_rate, list):
                flow_rate = flow_rate[0]

            # Make sure flow has the correct hydrogen type
            if not hasattr(flow.hydrogen, 'gas') and not hasattr(flow.hydrogen, 'liquid'):
                # If it's a single-phase hydrogen, convert to appropriate two-phase component
                from src.mission.mission_sections import InFlow
                processed_inflows.append(InFlow(flow_rate, tank_state.hydrogen.liquid))
            else:
                processed_inflows.append(flow)

        # Create combined flow list for matrix calculations
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

                # Determine hydrogen phase for outflow based on specified phase
                if hasattr(flow, 'phase'):
                    # Use appropriate phase from tank's hydrogen
                    hydrogen_for_outflow = (
                        tank_state.hydrogen.gas if flow.phase == "gas"
                        else tank_state.hydrogen.liquid
                    )
                elif hasattr(flow, 'hydrogen'):
                    hydrogen_for_outflow = flow.hydrogen
                else:
                    # Default to gas phase if not specified
                    hydrogen_for_outflow = tank_state.hydrogen.gas

                outflow = InFlow(-flow_rate, hydrogen_for_outflow)
                combined_flows.append(outflow)

        # Continue with matrix calculations as before
        a = cls.define_a_matrix(tank_state)
        b = cls.define_b_vector(tank_state, combined_flows)
        x = np.linalg.solve(a, b)

        # Apply safety limits to prevent non-physical values
        MAX_PRESSURE_CHANGE = 1e5  # Pa/s (max 1 bar/s)
        MAX_TEMP_CHANGE = 1.0  # K/s

        dP_dt = np.clip(x[0][0], -MAX_PRESSURE_CHANGE, MAX_PRESSURE_CHANGE)
        dT_dt = np.clip(x[1][0], -MAX_TEMP_CHANGE, MAX_TEMP_CHANGE)

        return StateDerivatives(
            dP_dt,
            dT_dt,
            x[2][0],
            x[3][0],
            cls.venting_mass(),
            cls.added_heat_flux()
        )

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
