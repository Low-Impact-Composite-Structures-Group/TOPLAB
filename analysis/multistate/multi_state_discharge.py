import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_single_tank_temperatures, plot_single_tank_loads
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt

def perform_discharge_analysis():
    # Define Tank 1 parameters (reservoir)
    p_init_1 = 5e+7  # Pa
    t_init_1 = 400  # K
    fill_1 = 0.0 # no liquid
    p_max_1 = 5.0e+8  # Pa
    p_min_1 = 1500000  # Pa
    ambient_heat_load_1 = 2.0  # W/m²
    mass_fraction_1 = 1.0 # analog to fill, but for gas wrt mass

    # Define Tank 2 parameters (consumer)
    p_init_2 = 4e+7  # Pa
    t_init_2 =  70 # K
    fill_2 = 0.0 # no liquid
    p_max_2 = 5.0e+8  # Pa
    p_min_2 = 1500000  # Pa
    ambient_heat_load_2 = 2.0  # W/m²
    mass_fraction_2 = 0.5

    # Get mission details
    mission = Mission.atr72()
    # Add after creating the mission
    print(f"Mission created: {mission.__class__.__name__}")
    print(f"Number of sections: {len(mission.sections)}")
    for i, section in enumerate(mission.sections):
        print(f"Section {i+1}: {section.duration/3600:.4f}h")
    print(f"Total duration: {sum(s.duration for s in mission.sections)/3600:.4f}h")
    total_fuel_mass = mission.required_fuel
    print(f"Total fuel mass required for mission: {total_fuel_mass:.2f} kg")

    # Calculate appropriate radius based on required mass
    hydrogen_requester = SinglePhaseRequester()
    hydrogen_props = hydrogen_requester.get_hydrogen_properties(p_init_1, t_init_1)
    VOLUME_MARGIN = 1.5 # make the tank 10% larger than the required volume
    required_volume = VOLUME_MARGIN*(mission.required_fuel / hydrogen_props.density)
    radius_1 = (3 * required_volume / (4 * np.pi))**(1/3)
    radius_2 = radius_1  # Same radius for both tanks
    print(f"Calculated tank radius: {radius_1:.2f} m")

    # Instantiate tank objects
    tank_material = Composite.carbon(np.radians(55))
    tank_dimensions_1 = TankDimensions(radius_1, 0.0)  # Spherical tank
    tank_dimensions_2 = TankDimensions(radius_2, 0.0)  # Spherical tank

    # Insulation for both tanks
    insulation_thickness = 0.05  # m
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Operating envelopes
    operating_window_1 = OperatingEnvelope(p_max_1, p_min_1, None)
    operating_window_2 = OperatingEnvelope(p_max_2, p_min_2, None)

    # Initial conditions
    initial_conditions_1 = InitialConditions(p_init_1, t_init_1, fill_1, multi_flow=True, mass_fraction=mass_fraction_1)
    initial_conditions_2 = InitialConditions(p_init_2, t_init_2, fill_2, multi_flow=True, mass_fraction=mass_fraction_2)

    # Define tank configurations for MultiTankAnalysisFacade
    tank_configs = [
        {
            "dimensions": tank_dimensions_1,
            "material": tank_material,
            "insulation": insulation,
            "heat_flux": ambient_heat_load_1
        },
        {
            "dimensions": tank_dimensions_2,
            "material": tank_material,
            "insulation": insulation,
            "heat_flux": ambient_heat_load_2
        }
    ]

    # Define interaction rules
    # interaction_rules = {
    # "type": "conditional",
    # "max_flow_rate": 0.1,  # kg/s - limit maximum flow between tanks
    # "active_at_start": True,
    # "conditions": [
    #     {
    #         "type": "time_after",
    #         "tank_idx": 1,        # Monitor Tank 2 (consumer)
    #         "threshold": 0.1*3600,
    #         "use_mission_flow": True,
    #         "safety_factor": 0.8  # Same as before
    #     }
    # ],
    # "default_flow": 0.0  # No flow until condition is met
    # }
    interaction_rules = {
        "type": "mission_based",
        "max_flow_rate": 0.1,  # kg/s - limit maximum flow between tanks
        "active_at_start": True,
        "safety_factor": 0.8  # Move this to top level
    }

    # Run multi-tank analysis
    print("Analyzing multi-tank system...")
    tank_performances = MultiTankAnalysisFacade.analyse(
        tank_configurations=tank_configs,
        mission=mission,
        interaction_rules=interaction_rules,
        initial_conditions=[initial_conditions_1, initial_conditions_2],
        operating_envelopes=[operating_window_1, operating_window_2],
        target_conditions=[
            TargetConditions(0.10 * total_fuel_mass, 0.0),
            TargetConditions(5.0, 0.0)
        ]
    )

    # Extract results
    tank_states_1 = tank_performances[0].tank_states
    tank_states_2 = tank_performances[1].tank_states

    # Access flow data for plotting
    flow_data = MultiTankAnalysisFacade.flow_data

    # Initialize the plotter at the beginning of the analysis
    plotter = SeabornPlotter(font="Cambria", palette="deep")

    # Plot the mass flows
    flow_fig = plotter.plot_tank_mass_flows(
        flow_data['time'],
        flow_data['tank1_inflow'],
        flow_data['tank1_outflow'],
        flow_data['tank2_inflow'],
        flow_data['tank2_outflow']
    )

    # Create comparative plots (all in one figure)
    comparative_fig = plotter.plot_comparative_tank_states(
        tank_states_1,
        tank_states_2,
        figsize=(15, 5),
        titles=[
            "Fuel Masses - Reservoir vs Consumer",
            "Tank Temperatures - Reservoir vs Consumer",
            "Tank Pressures - Reservoir vs Consumer"
        ]
    )

    plt.show()


def main():
    perform_discharge_analysis()

if __name__ == "__main__":
    main()
