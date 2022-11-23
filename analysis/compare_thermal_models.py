

from typing import Protocol

from plotting.plot_tank_states import plot_tank_loads
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import TankIsEmpty
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Metal
from src.mission.mission import Mission
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.external_models import (ForcedConvectionModel,
                                                NaturalConvectionModel)
from src.thermodynamics.internal_models import SingleZoneModel, ThreeZoneModel
from src.thermodynamics.tank_states import InitialState, TargetState
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


class TankState(Protocol):
    pressure: float
    temperature: float
    fill: float


def perform_analysis():

    # Define the state of the fuel tank
    pressure = 1.4e5
    temperature = None
    fill = 0.95
    initial_conditions = InitialState(
        pressure, temperature, fill
    )

    # Define the fuel tank
    material  = Metal.aluminum()
    tank = CylindricalTankSphericalCaps.rompokos(material, pressure)

    # Define the stopping criteria for the fuel tank
    stopping_criteria = [TankIsEmpty()]

    # Define the target conditions
    target_conditions = TargetState(
        pressure=10e5,
        min_temperature=None,
        fill=0.0,
        mass=None
    )

    # Define insulation and thermodynamic model
    insulation_thickens = 8e-2
    insulation = ConstantFoamInsulation.polyvinylchloride(
        insulation_thickens
    )

    # Define the dynamic model factory
    dynamic_model_factory = DynamicModelFactory()

    # Define the heat flux factory, which is to be used to account for
    # extra losses
    heat_flux_factor = 1

    # Time integration and steps
    timestep = 60
    multistep_method = EulerMethod(timestep)

    thermal_models = [
        ThermodynamicModel(
            SingleZoneModel(),
            ForcedConvectionModel(),
            insulation
        ),
        ThermodynamicModel(
            ThreeZoneModel(),
            ForcedConvectionModel(),
            insulation
        ),
        ThermodynamicModel(
            SingleZoneModel(),
            NaturalConvectionModel(),
            insulation
        )
    ]

    mission = Mission.rompokos()
    data = [
        MissionAnalysis.perform_analysis(
            tank,
            initial_conditions,
            mission,
            stopping_criteria,
            target_conditions,
            multistep_method,
            dynamic_model_factory,
            model,
            heat_flux_factor
        )
        for model in thermal_models
    ]
        

    yticks = [i / 10 for i in range(14, 63, 2)]
    yticks = None
    xticks = [i for i in range(0, 25, 4)]
    fig = plot_tank_loads(
        data,
        ["Single Zone", "Three Zone", "Natural"],
        x_ticks=xticks,
        y_ticks=yticks
    )
    fig.show()
    

def main():
    pass


if __name__ == "__main__":
    main()


# End
