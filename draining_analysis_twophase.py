

import matplotlib.pyplot as plt


from src.dynamics.stopping_criteria import MaxPressure
from src.fluids.hydrogen_retrievers import HydrogenRetriever
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.mission.mission_sections import FuelFlow, MissionSection
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.tank_states import InitialState, TargetState
from src.thermodynamics.thermodynamic_models import (ForcedConvectionFactory,
                                                     SingleZoneFactory,
                                                     ThermodynamicModel)
from src.dynamics.dynamic_models import DynamicModelFactory
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.dynamics.dynamic_analysis import AnalyseMissionSection


def main():

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
    fuel_internal_energy = 120e6
    power_required = 10e6
    fuel_mass_flow = - power_required / fuel_internal_energy
    fuel_flows = [
        FuelFlow(
            hydrogen=hydrogen.liquid,
            mass_flow=fuel_mass_flow
        )
    ]

    # Define the fuel tank
    tank = CylindricalTankSphericalCaps.rompokos_tank()

    # Define the ambient state
    altitude = 10e3
    duration = 13 * 60 ** 2
    mach_number = 0.85
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
    insulation_thickens = 8e-2
    insulation = ConstantFoamInsulation.rohacell(insulation_thickens)

    thermodynamic_model = ThermodynamicModel(
        SingleZoneFactory(),
        ForcedConvectionFactory(),
        insulation
    )

    # Define the dynamic model factory
    model_factory = DynamicModelFactory()

    # Define the heat flux factory, which is to be used to account for
    # extra losses
    heat_flux_factor = 1

    # Time integration and steps
    timestep = 60 * 60
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
        i * timestep
        for i, _ in enumerate(pressures)
    ]

    plt.plot(timesteps, pressures)
    plt.show()


if __name__ == "__main__":
    main()


# End
