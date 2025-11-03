from src.fluids.energy_derivative_computer import EnergyDerivativeComputer
from .protocols import TankState, DynamicModel, FuelFlow, StateDerivatives, TwoPhaseHydrogen, Hydrogen, OperatingEnvelope


class LinModel(DynamicModel):

    energy_derivative_computer = EnergyDerivativeComputer()

    def compute_state_derivatives(
        cls,
        tank_state: TankState,
        fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        dP_dt = cls.compute_pressure_derivative(tank_state, fuel_flows)
        dT_dt = dP_dt / tank_state.hydrogen.dP_dT

        liquid_derivative = 0
        gas_derivative = 0
        for fuel_flow in fuel_flows:
            if fuel_flow.hydrogen.state == 6.0:
                liquid_derivative += fuel_flow.mass_flow
            else:
                raise ValueError("Not implemented other phase than liquid draining in Lin")

        return StateDerivatives(
            dP_dt,
            dT_dt,
            gas_derivative,
            liquid_derivative,
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


class DynamicModelFactory:

    def get_dynamic_model(
        self,
        tank_state: TankState,
        target_conditions: OperatingEnvelope
    ) -> DynamicModel:
        if tank_state.phase == "twophase":
            return LinModel()
        raise ValueError("Single phase not supported by Lin")