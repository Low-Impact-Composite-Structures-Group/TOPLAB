from dataclasses import dataclass

import matplotlib.pyplot as plt

from src.dynamics.dynamic_analysis import (AnalyseMissionSection, InitialState,
                                           TankState, TargetState)
from src.dynamics.dynamic_models import DynamicModelFactory
from src.dynamics.stopping_criteria import MaxPressure
from src.fluids.hydrogen_retrievers import HydrogenRetriever
from src.fluids.international_standard_atmosphere import get_ISA_air_properties
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.mission.mission_sections import FuelFlow, MissionSection
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import (CylindricalTankSphericalCaps,
                                         SphericalTank)
from src.thermodynamics.external_models import ForcedConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.thermodynamic_models import ThermodynamicModel


@dataclass
class FuelTank:
    volume: float

    def compute_thermal_capacity(self, temperature: float) -> float:
        return 0


@dataclass
class FixedHeatFluxModel:
    heat_flux: float

    def compute_heat_flux(
        self, tank_state: TankState, ambient_state: MissionSection
    ) -> float:
        return self.heat_flux, [tank_state.temperature]


def test_lin():

    # Initial tank conditions
    initial_pressure = 101e3
    fill_percentage = 0.95
    initial_temperature = None
    initial_conditions = InitialState(
        initial_pressure, initial_temperature, fill_percentage
    )

    # Time integration and steps
    timestep = 60 * 60
    multistep_method = EulerMethod(timestep)

    # Define the fuel flows from the fuel tank
    fuel_mass_flow = 0
    hydrogen_fuel_flow = HydrogenRetriever().get_hydrogen_properties(
        initial_pressure, initial_temperature
    ).liquid
    fuel_flows = [
        FuelFlow(
            hydrogen=hydrogen_fuel_flow,
            mass_flow=fuel_mass_flow
        )
    ]
    mission_duration = 13 * 24 * 60 ** 2
    altitude = 0
    mach_number = 5 / 300
    mission_section = MissionSection(
        mission_duration, fuel_flows, altitude, mach_number
    )

    # Define the stopping criteria for the fuel tank
    stopping_criteria = [
        MaxPressure()
    ]

    # Define the target conditions
    target_conditions = TargetState(
        pressure=138e3,
        temperature=None,
        fill=1.0,
        mass=None
    )

    # Define the fuel tank
    fuel_tank = SphericalTank.lin()

    # Define the dynamic model factory
    model_factory = DynamicModelFactory()

    # Define the heat flux factory, which is to be used to account for
    # extra losses
    heat_flux_factor = 1

    # Define thermodynamic model
    heat_flux = 20
    thermodynamic_model = FixedHeatFluxModel(heat_flux)


    analysis = AnalyseMissionSection(
        fuel_tank,
        initial_conditions,
        mission_section,
        stopping_criteria,
        target_conditions,
        multistep_method,
        model_factory,
        thermodynamic_model,
        heat_flux_factor=heat_flux_factor
    )

    tank_states = analysis.analyse_mission_section()

    pressures = [
        state.pressure * 1e-3
        for state in tank_states
    ]
    # print(pressures)
    timesteps = [
        i * timestep / (24 * 80 ** 2)
        for i, _ in enumerate(pressures)
    ]

    plt.plot(timesteps, pressures)
    plt.show()


def test_thermodynamic_model():

    # Define the state of the fuel tank
    pressure = 1.4e5
    temperature = None
    fill = 0.95
    hydrogen = HydrogenRetriever().get_hydrogen_properties(
        pressure, temperature
    )

    # Define the fuel flows
    fuel_mass_flow = 0
    fuel_flows = [
        FuelFlow(
            hydrogen=hydrogen,
            mass_flow=fuel_mass_flow
        )
    ]

    # Define the fuel tank
    tank = CylindricalTankSphericalCaps.ahluwalia_tank()

    # Define the state of the tank
    tank_state = TankState(
        tank, temperature, pressure, fill, fuel_flows
    )

    # Define the ambient state
    mach_number = 0.01
    ground_temperature = 300
    altitude = 0
    ambient = get_ISA_air_properties(
        altitude, temperature=ground_temperature
    )
    duration = None
    mission_section = MissionSection(
        duration,
        fuel_flows,
        altitude,
        mach_number,
        ground_temperature=ground_temperature
    )

    # Define insulation
    insulation_thickens = 4e-2
    insulation = ConstantFoamInsulation.polyvinylchloride(insulation_thickens)

    thermodynamic_model = ThermodynamicModel(
        SingleZoneModel(),
        ForcedConvectionModel(),
        insulation
    )

    print(thermodynamic_model.compute_heat_flux(
        tank_state, mission_section
    ))


def test_dynamic_analysis():

    # Define the state of the fuel tank
    pressure = 1.4e5
    temperature = None
    fill = 0.95
    hydrogen = HydrogenRetriever().get_hydrogen_properties(
        pressure, temperature
    )

    initial_conditions = InitialState(
        pressure, temperature, fill
    )

    # Define the fuel flows
    internal_energy = 120e6
    required_energy = 10e6
    fuel_mass_flow = - required_energy / internal_energy
    fuel_flows = [
        FuelFlow(
            hydrogen=hydrogen.liquid,
            mass_flow=fuel_mass_flow
        )
    ]

    # Define the fuel tank
    tank = CylindricalTankSphericalCaps.rompokos()

    # Define the ambient state
    altitude = 0
    duration = 6 * 60 ** 2
    mach_number = 0.01
    mission_section = MissionSection(
        duration, fuel_flows, altitude, mach_number
    )

    # Define the stopping criteria for the fuel tank
    stopping_criteria = [
        MaxPressure()
    ]

    # Define the target conditions
    target_conditions = TargetState(
        pressure=3e5,
        temperature=None,
        fill=1.0,
        mass=None
    )

    # Define insulation
    insulation_thickens = 4e-2
    insulation = ConstantFoamInsulation.polyvinylchloride(insulation_thickens)

    thermodynamic_model = ThermodynamicModel(
        SingleZoneModel(),
        ForcedConvectionModel(),
        insulation
    )

    # Define the dynamic model factory
    model_factory = DynamicModelFactory()

    # Define the heat flux factory, which is to be used to account for
    # extra losses
    heat_flux_factor = 1

    # Time integration and steps
    timestep = 60
    multistep_method = EulerMethod(timestep)


    analysis = AnalyseMissionSection(
        tank,
        initial_conditions,
        mission_section,
        stopping_criteria,
        target_conditions,
        multistep_method,
        model_factory,
        thermodynamic_model,
        heat_flux_factor=heat_flux_factor
    )

    tank_states = analysis.analyse_mission_section()

    pressures = [
        state.pressure * 1e-3
        for state in tank_states
    ]
    # print(pressures)
    timesteps = [
        i * timestep / (60 ** 2)
        for i, _ in enumerate(pressures)
    ]

    plt.plot(timesteps, pressures)
    plt.show()


def main():

    test_dynamic_analysis()



if __name__ == "__main__":
    main()


# End
