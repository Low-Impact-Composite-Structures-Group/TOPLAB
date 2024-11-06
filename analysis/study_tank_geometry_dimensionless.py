from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures, plot_required_flux)
from facades.analysis_facades import (DrainingAnalysisFacade, InitialConditions,
                                          OperatingEnvelope, TankDimensions, GenericTankDimensions, MissionAnalysisFacade)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission_sections import OutFlow
from src.mission.mission import Mission
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import WinnefeldTank
import numpy as np
import time


def perform_analysis():

    # Record the start time
    start_time = time.time()

    # Define the initial state of the tank
    pressure = 140e3 # [Pa]
    temperature = None
    fill = 0.97
    initial_state = InitialState(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-2 # [m]
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    winding_angle = 55.0 # [degrees] placeholder value
    tank_material = Composite.carbon(winding_angle)

    # Define fuel flow
    # fuel_flow = OutFlow.fly_eco_cruise("liquid")
    fuel_flow = "liquid"

    mission = Mission.fly_eco_mission(fuel_flow)

    # Define the minimum pressure of the tank
    min_pressure = 1.3e5 # [Pa]
    min_temperature = None
    operating_window = OperatingEnvelope(
        max_pressure=None,
        min_pressure=min_pressure,
        min_temperature=min_temperature
    )

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.get_hydrogen_properties()
    fuel_volume = fuel_mass / initial_fuel.liquid.density
    VOLUME_MARGIN = 1.15

    # Define tank dimensions
    # See Winnefeld paper for a visual key corresponding to these dimensions
    # "Modelling and Designing Cryogenic Hydrogen Tanks for Future Aircraft Applications (2018)"

    body_length = 5 # [m] length of entire tank, corresponds to l_t in Winnefeld paper
    # radii = [i / 100 for i in range(25, 276, 25)] # [m] horizontal axis of elliptical shell cross-section, corresponds to a in Winnefeld paper

    min_radius = 0.2
    max_radius = 0.3
    min_b = 0.2
    max_b = 0.3
    min_body_length = 3.0
    max_body_length = 5.0
    radius_range = (min_radius, max_radius)
    b_range = (min_b, max_b)
    body_length_range = (min_body_length, max_body_length)
    num_samples = 1
    decimals = 4
    radii_samples = np.round(np.random.uniform(radius_range[0], radius_range[1], num_samples), decimals)
    b_samples = np.round(np.random.uniform(b_range[0], b_range[1], num_samples), decimals)
    body_length_samples = np.round(np.random.uniform(body_length_range[0], body_length_range[1], num_samples), decimals)

    # remainder of dimensions needed for Winnefeld analysis
    # l_s = 5 # [m]q     length of shell section. not presently used since the shell length
    # can be deduced from the body length and endcap lengths

    labels = [f'{radius} m' for radius in radii_samples]


    # Perform the analysis
    performances = [
        MissionAnalysisFacade.analyse(
            GenericTankDimensions(
                # NB: for now, quantity a is set to the same value as the radius (c)
                # This is because the EllipticCylinderBody does not yet support partial
                # volume calculations with elliptic cross sections
                radius,  WinnefeldTank.length_from_radius_b_and_volume(
                radius, VOLUME_MARGIN * fuel_volume, b), radius, radius),
            tank_material,
            insulation,
            mission,
            initial_state,
            operating_window
        )
            for radius, b, in zip(radii_samples, b_samples)
    ]
    # data = [performance.tank_states for performance in performances]

    # compute psi
    psi_values = [radius / b for radius, b in zip(radii_samples, b_samples)]

    # fig1 = plot_tank_loads(data, labels, None, None)
    # fig2 = plot_tank_temperatures(data, labels, None, None)
    # fig3 = plot_tank_fill(data[-1], None, None, None)
    # fig4 = plot_tank_efficiencies_scatter(performances, psi_values, "psi (c/b) [m/m]", None, None)

    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()

    for performance in performances:
        print("Gravimetric Efficiency\t:", performance.gravimetric_efficiency)
        print("Volumetric Efficiency\t:", performance.volumetric_efficiency)

    # Record the end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Plotting the data
    fig = plot_tank_loads(
        [row.tank_states for row in performances],
        labels
    )
    plot_tank_temperatures(
        [row.tank_states for row in performances],
        labels
    )
    plot_tank_fill(
        performances[-1].tank_states
    )
    plot_required_flux(
        [row.tank_states for row in performances],
        labels
    )

    # Autoscale the axes
    fig.ax[0].autoscale()

    # Show the figure
    fig.show()




def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
