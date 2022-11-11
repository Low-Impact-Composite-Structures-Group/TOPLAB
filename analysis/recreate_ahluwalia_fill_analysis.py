

from typing import Protocol

import matplotlib.pyplot as plt

from plotting.plot_tank_states import (plot_tank_fill,
                                       plot_thermo_mechanical_loading)
from src.dynamics.dynamic_analysis import AnalyseMissionSection
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import TankIsFull
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.mission.mission_sections import InFlow, MissionSection
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.tank_states import InitialState, TargetState
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


class TankState(Protocol):
    pressure: float
    temperature: float
    fill: float


def perform_analysis():

    # Define the state of the fuel tank
    pressure = 8e5
    temperature = 50
    fill = 0.0
    initial_conditions = InitialState(
        pressure, temperature, fill
    )

    # Define the fuel tank
    tank = CylindricalTankSphericalCaps.ahluwalia()

    # Define the stopping criteria for the fuel tank
    stopping_criteria = [
        TankIsFull()
    ]

    # Define the target conditions
    target_conditions = TargetState(
        pressure=10e5,
        temperature=None,
        fill=1.0,
        mass=11.0
    )

    # Define insulation and thermodynamic model
    insulation_thickens = 4e-2
    insulation = ConstantFoamInsulation.rohacell(
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
    timestep = 1
    multistep_method = EulerMethod(timestep)

    duration = 1e4
    mass_flow = 0.01
    altitude = 0
    mach_number = 0.01
    mission_section = MissionSection(
            duration,
            [
                InFlow(
                    mass_flow,
                    SinglePhaseRequester().get_hydrogen_properties(
                        16e5, 22
                    )
                )
            ],
            altitude,
            mach_number
        )

    analysis = AnalyseMissionSection(
        tank,
        initial_conditions,
        mission_section,
        stopping_criteria,
        target_conditions,
        multistep_method,
        dynamic_model_factory,
        thermodynamic_model,
        heat_flux_factor=heat_flux_factor
    )

    tank_states = analysis.analyse_mission_section()

    initial_conditions = InitialState(
        tank_states[-1].pressure,
        tank_states[-1].temperature,
        tank_states[-1].fill 
    )

    y1ticks = None
    y2ticks = None
    xticks = None
    plot_thermo_mechanical_loading(
        tank_states,
        timestep,
        xticks,
        y1ticks,
        y2ticks
    )
    y1ticks = None
    y2ticks = None
    plot_tank_fill(
        tank_states,
        timestep,
        xticks,
        y1ticks,
        y2ticks
    )
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()


# End
