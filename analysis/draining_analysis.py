

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


@dataclass
class InitialState:
    pressure: float
    temperature: float
    fill: float


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


@dataclass
class LiquidDrainingAnalysis:
    tank: FuelTank
    fuel_flow: FuelFlow
    insulation: Insulation

    initial_pressure: float = 140e3
    initial_fill: float = 0.97
    timestep: float = 60
    heat_flux_factor: float = 1

    def __post_init__(self):
        self.stopping_criteria = [TankIsEmpty()]
        self.define_initial_state()
        self.define_mission_section()
        self.define_target_conditions()

    def define_initial_state(self) -> InitialState:
        initial_temperature = None
        self.initial_state = InitialState(
            self.initial_pressure,
            initial_temperature,
            self.initial_fill
        )

        return self.initial_state

    def define_mission_section(self):
        duration = 60e6
        altitude = 10e3
        mach_number = 0.85
        self.mission_section = MissionSection(
            duration,
            [self.fuel_flow],
            altitude,
            mach_number
        )

        return self.mission_section

    def define_target_conditions(self):
        self.target_conditions = TargetState(
            max_pressure=None,
            min_pressure=None,
            temperature=None,
            fill=None,
            mass=None
        )

    def drain_tank(self):
        analysis = AnalyseMissionSection(
            self.tank,
            self.initial_state,
            self.mission_section,
            self.stopping_criteria,
            self.target_conditions,
            EulerMethod(self.timestep),
            DynamicModelFactory(),
            ThermodynamicModel(
                SingleZoneModel(),
                ForcedConvectionModel(),
                self.insulation
            ),
            heat_flux_factor=self.heat_flux_factor
        )
        self.tank_states = analysis.perform_analysis()
        
        return self.tank_states


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
