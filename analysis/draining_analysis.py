

from dataclasses import dataclass
from typing import Protocol

from src.dynamics.dynamic_analysis import AnalyseMissionSection
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import TankIsEmpty
from src.mission.mission_sections import MissionSection
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


class FuelTank(Protocol):
    ...


class InitialState(Protocol):
    ...


class Insulation(Protocol):
    ...


class FuelFlow(Protocol):
    ...



@dataclass
class TargetState:
    pressure: float
    temperature: float
    fill: float
    mass: float


def perform_draining_analysis(
    tank: FuelTank,
    initial_state: InitialState,
    fuel_flow: FuelFlow,
    insulation: Insulation,
    timestep: float = 60,
    heat_flux_factor: float = 1
):

    # Assumptions about the mission section
    duration = 60e6
    altitude = 10e3
    mach_number = 0.85
    mission_section = MissionSection(
        duration,
        [fuel_flow],
        altitude,
        mach_number
    )

    # Define the stopping criteria for the fuel tank
    stopping_criteria = [TankIsEmpty()]

    # Define the target conditions
    target_conditions = TargetState(
        pressure=None,
        temperature=None,
        fill=None,
        mass=None
    )

    analysis = AnalyseMissionSection(
        tank,
        initial_state,
        mission_section,
        stopping_criteria,
        target_conditions,
        EulerMethod(timestep),
        DynamicModelFactory(),
        ThermodynamicModel(
            SingleZoneModel(),
            ForcedConvectionModel(),
            insulation
        ),
        heat_flux_factor=heat_flux_factor
    )
    return analysis.analyse_mission_section()



def main():
    pass


if __name__ == "__main__":
    main()


# End
