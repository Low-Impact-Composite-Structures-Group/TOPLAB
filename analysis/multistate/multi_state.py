import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_single_tank_temperatures, plot_single_tank_loads
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt

def perform_analysis():
    # Define Tank 1 parameters (reservoir)
    # radius_1 = 1.0  # m
    p_init_1 = 4e+7  # Pa (400 bar)
    t_init_1 = 70  # K
    fill_1 = 0.0 # no liquid
    p_max_1 = 4.5e+7  # Pa (450 bar)
    p_min_1 = 1500000  # Pa (15 bar)
    ambient_heat_load_1 = 2.0  # W/m²

    # Define Tank 2 parameters (consumer)
    # radius_2 = 1.0  # m
    p_init_2 = 4e+7  # Pa (20 bar)
    t_init_2 = 70  # K
    fill_2 = 0.0 # no liquid
    p_max_2 = 4.5e+7  # Pa (40 bar)
    p_min_2 = 1500000  # Pa (10 bar)
    ambient_heat_load_2 = 2.0  # W/m²

    # Get mission details
    mission = Mission.triathlon()
    total_fuel_mass = mission.required_fuel
    print(f"Total fuel mass required for mission: {total_fuel_mass:.2f} kg")

    # Calculate appropriate radius based on required mass
    hydrogen_requester = SinglePhaseRequester()
    hydrogen_props = hydrogen_requester.get_hydrogen_properties(p_init_1, t_init_1)
    VOLUME_MARGIN = 1.1 # make the tank 10% larger than the required volume
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
    initial_conditions_1 = InitialConditions(p_init_1, t_init_1, fill_1, multi_flow=True)
    initial_conditions_2 = InitialConditions(p_init_2, t_init_2, fill_2, multi_flow=True)

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

    # Define interaction rules - Transfer fuel from Tank 1 to Tank 2
    # based on the mission requirements
    interaction_rules = {
        "interaction_type": "mission_based",
        "reservoir_tank_idx": 0,
        "consumer_tank_idx": 1,
        "safety_margin": 1.05,
        "max_flow_rate": 0.1  # kg/s - limit maximum flow between tanks
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

    # Plot the mass flows
    from plotting.plot_tank_states import plot_tank_mass_flows
    flow_fig = plot_tank_mass_flows(
        flow_data['time'],
        flow_data['tank1_inflow'],
        flow_data['tank1_outflow'],
        flow_data['tank2_inflow'],
        flow_data['tank2_outflow']
    )

    # Generate figures and plots
    fig_tank_fill_1 = plot_single_tank_fill(tank_states_1)
    fig_tank_fill_2 = plot_single_tank_fill(tank_states_2)
    fig_tank_temperatures_1 = plot_single_tank_temperatures(tank_states_1)
    fig_tank_temperatures_2 = plot_single_tank_temperatures(tank_states_2)
    fig_tank_pressures_1 = plot_single_tank_loads(tank_states_1)
    fig_tank_pressures_2 = plot_single_tank_loads(tank_states_2)


    # Create a combined figure with a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    # List of source axes and titles
    axes_list = [
        fig_tank_fill_1.ax[0],         # Row 1, Col 1: Tank 1 Fill Level
        fig_tank_temperatures_1.ax[0], # Row 1, Col 2: Tank 1 Temperature
        fig_tank_pressures_1.ax[0],    # Row 1, Col 3: Tank 1 Pressure
        fig_tank_fill_2.ax[0],         # Row 2, Col 1: Tank 2 Fill Level
        fig_tank_temperatures_2.ax[0], # Row 2, Col 2: Tank 2 Temperature
        fig_tank_pressures_2.ax[0],    # Row 2, Col 3: Tank 2 Pressure
    ]
    titles = [
        "Tank 1 (Reservoir) Fill Level",
        "Tank 1 (Reservoir) Temperature",
        "Tank 1 (Reservoir) Pressure",
        "Tank 2 (Consumer) Fill Level",
        "Tank 2 (Consumer) Temperature",
        "Tank 2 (Consumer) Pressure",
    ]

    # Copy data to combined figure
    for ax_target, ax_source, title in zip(axs, axes_list, titles):
        for line in ax_source.get_lines():
            ax_target.plot(
                line.get_xdata(), line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle()
            )
        ax_target.set_title(title)
        ax_target.set_xlabel(ax_source.get_xlabel())
        ax_target.set_ylabel(ax_source.get_ylabel())
        if ax_source.get_legend():
            ax_target.legend()

    # Clean up original figures
    plt.close(fig_tank_fill_1.fig)
    plt.close(fig_tank_fill_2.fig)
    plt.close(fig_tank_temperatures_1.fig)
    plt.close(fig_tank_temperatures_2.fig)
    plt.close(fig_tank_pressures_1.fig)
    plt.close(fig_tank_pressures_2.fig)

    plt.tight_layout()
    plt.show()


def main():
    perform_analysis()

if __name__ == "__main__":
    main()


