

from dataclasses import dataclass
from typing import Protocol

from src.dynamics.dynamic_analysis import AnalyseMissionSection
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import LowerPressureReached, TankIsEmpty
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


class TankState(Protocol):
    ...

@dataclass
class TargetState:
    max_pressure: float
    min_pressure: float
    temperature: float
    fill: float
    mass: float


def liquid_draining_analysis(
    tank: FuelTank,
    initial_state: InitialState,
    fuel_flow: FuelFlow,
    insulation: Insulation,
    timestep: float = 60,
    heat_flux_factor: float = 1
) -> list[TankState]:

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
        max_pressure=None,
        min_pressure=None,
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


def gas_draining_analysis(
    tank: FuelTank,
    initial_state: InitialState,
    fuel_flow: FuelFlow,
    insulation: Insulation,
    min_pressure: float,
    timestep: float = 60,
    heat_flux_factor: float = 1
) -> list[TankState]:

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
    stopping_criteria = [LowerPressureReached()]

    target_conditions = TargetState(
        max_pressure=None,
        min_pressure=min_pressure,
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
