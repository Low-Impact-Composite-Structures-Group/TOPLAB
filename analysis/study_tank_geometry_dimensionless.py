

from plotting.plot_tank_states import (plot_tank_efficiencies_scatter, plot_tank_fill,
                                       plot_tank_loads, plot_tank_temperatures)
from facades.analysis_facades import (DrainingAnalysisFacade, InitialConditions,
                                          OperatingEnvelope, TankDimensions, GenericTankDimensions)
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission_sections import OutFlow
import numpy as np


def perform_analysis():

    # Define the initial state of the tank
    pressure = 140e3 # [Pa]
    temperature = None
    fill = 0.97
    initial_state = InitialConditions(
        pressure, temperature, fill
    )

    # Define insulation and thermal model
    thickness = 4e-2 # [m]
    insulation = ConstantFoamInsulation.rohacell(thickness)

    # Define tank material
    winding_angle = 55.0 # [degrees] placeholder value
    tank_material = Composite.carbon(winding_angle)

    # Define fuel flow
    fuel_flow = OutFlow.fly_eco_cruise("liquid")

    # Define the minimum pressure of the tank
    min_pressure = 1.3e5 # [Pa]

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
    num_samples = 10
    decimals = 4
    radii_samples = np.round(np.random.uniform(radius_range[0], radius_range[1], num_samples), decimals)
    b_samples = np.round(np.random.uniform(b_range[0], b_range[1], num_samples), decimals)
    body_length_samples = np.round(np.random.uniform(body_length_range[0], body_length_range[1], num_samples), decimals)

    # remainder of dimensions needed for Winnelfeld analysis
    # l_s = 5 # [m] length of shell section. not presently used since the shell length
    # can be deduced from the body length and endcap lengths

    labels = [f'{radius} m' for radius in radii_samples]


    # Perform the analysis
    performances = [
        DrainingAnalysisFacade.analyse(
            GenericTankDimensions(
                # NB: for now, quantity a is set to the same value as the radius (c)
                # This is because the EllipticCylinderBody does not yet support partial
                # volume calculations with elliptic cross sections
                radius, body_length, radius, radius
            ),
            tank_material,
            insulation,
            fuel_flow.mass_flow,
            fuel_flow.phase,
            initial_state,
            OperatingEnvelope(
                None, # TODO: define max pressure relating to the tank material and geometry
                min_pressure,
                None
            )
        )
         for radius, b, body_length in zip(radii_samples, b_samples, body_length_samples)
    ]
    data = [performance.tank_states for performance in performances]

    # compute psi
    psi_values = [radius / b for radius, b in zip(radii_samples, b_samples)]

    fig1 = plot_tank_loads(data, labels, None, None)
    fig2 = plot_tank_temperatures(data, labels, None, None)
    fig3 = plot_tank_fill(data[-1], None, None, None)
    fig4 = plot_tank_efficiencies_scatter(performances, psi_values, "psi (c/b) [m/m]", None, None)

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
