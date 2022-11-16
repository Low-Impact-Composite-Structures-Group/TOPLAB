
from typing import Protocol

import matplotlib.pyplot as plt

from plotting.plot_tank_states import plot_tank_fill, plot_thermo_mechanical_loading
from src.dynamics.dynamic_analysis import AnalyseMissionSection
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import TankIsEmpty
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Metal
from src.mission.mission import Mission
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.tank_states import InitialState, TankStates, TargetState
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
    tank = CylindricalTankSphericalCaps.rompokos(Metal.aluminum(), pressure)

    # Define the stopping criteria for the fuel tank
    stopping_criteria = [TankIsEmpty()]

    # Define the target conditions
    target_conditions = TargetState(
        pressure=10e5,
        temperature=None,
        fill=0.0,
        mass=None
    )

    # Define insulation and thermodynamic model
    insulation_thickens = 8e-2
    insulation = ConstantFoamInsulation.polyvinylchloride(
        insulation_thickens
    )
    thermodynamic_model = ThermodynamicModel(
        SingleZoneModel(),
        ForcedConvectionModel(),
        insulation
    )

    # Define the dynamic model factory
    dynamic_model_factory = DynamicModelFactory()

    # Define the heat flux factory, which is to be used to account for
    # extra losses
    heat_flux_factor = 1

    # Time integration and steps
    timestep = 60
    multistep_method = EulerMethod(timestep)

    # Define the mission
    mission = Mission.small_medium_range("liquid")

    tank_states = TankStates(list(), timestep)
    # The index is used to remove the first tank state of mission
    # section that are not the first, to avoid doubles
    i = 0  
    for mission_section in mission.sections:
        analysis = AnalyseMissionSection(
            tank,
            initial_conditions,
            mission_section,
            stopping_criteria,
            target_conditions,
            multistep_method,
            dynamic_model_factory,
            thermodynamic_model,
            heat_flux_factor
        )
        for state in analysis.perform_analysis().states[i:]:
            i = 1
            tank_states.add_tank_state(state)
        initial_conditions = InitialState(
            tank_states.last_pressure,
            tank_states.last_temperature,
            tank_states.last_state.fill 
        )
    y1ticks = [i / 10 for i in range(14, 23)]
    y2ticks = [i / 10 for i in range(210, 251, 5)]
    xticks = [i for i in range(0, 25, 4)]
    plot_thermo_mechanical_loading(
        tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    y1ticks = [i for i in range(0, int(10001), int(2e3))]
    y2ticks = [i for i in range(0, 101, 20)]
    plot_tank_fill(
        tank_states,
        xticks,
        y1ticks,
    )
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()


# End
