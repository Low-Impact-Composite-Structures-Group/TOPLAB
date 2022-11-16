

from typing import Protocol

from src.dynamics.dynamic_analysis import DrainingAnalysis
from src.dynamics.dynamic_models import DynamicModelFactory
from src.efficiencies.tank_performance import TankPerformance
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.thermodynamic_models import ThermodynamicModel



class TankStates(Protocol):
    ...


class InitialState(Protocol):
    pressure: float


class Insulation(Protocol):
    ...


class AnalyseCylindricalTank:

    @classmethod
    def analyse_tank(
        cls,
        radius: float,
        body_length: float,
        tank_material: float,
        insulation: Insulation,
        fuel_mass_flow: float,
        fuel_phase_flow: float,
        initial_state: InitialState,
        timestep: float = 60,
    ) -> TankPerformance:
        tank = cls.create_tank(
            radius, body_length, tank_material, initial_state.pressure
        )
        tank_states = cls.compute_tank_states(
            tank,
            fuel_mass_flow,
            fuel_phase_flow,
            insulation,
            initial_state,
            timestep
        )
        
        return TankPerformance(tank, insulation, tank_states)

    @staticmethod
    def compute_tank_states(
        tank: CylindricalTankSphericalCaps,
        fuel_mass_flow: float,
        fuel_phase_flow: float,
        insulation: Insulation,
        initial_state: InitialState,
        timestep: float
    ) -> TankStates:

        # Define the timestep and multistep method
        multistep_method = EulerMethod(timestep)

        # Define insulation and thermal model
        thermal_model = ThermodynamicModel(
            SingleZoneModel(), ForcedConvectionModel(), insulation
        )

        # Define the dynamic model factory
        dynamic_model_factory = DynamicModelFactory()

        # Define the heat flux factor
        heat_flux_factor = 1

        return DrainingAnalysis.perform_analysis(
            tank,
            fuel_mass_flow,
            fuel_phase_flow,
            initial_state,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor
        )

    @staticmethod
    def create_tank(
        radius, body_length, material, operating_pressure
    ) -> CylindricalTankSphericalCaps:
        return CylindricalTankSphericalCaps(
            radius,
            body_length + 2 * radius,
            material,
            operating_pressure
        )



def main():
    pass


if __name__ == "__main__":
    main()


# End
