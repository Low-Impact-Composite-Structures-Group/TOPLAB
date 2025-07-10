import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux, plot_single_tank_loads
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, InOutTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
import numpy as np
import matplotlib.pyplot as plt

def perform_analysis():
    # Define Tank 1 parameters (reservoir)
    radius_1 = 2.0  # m
    p_init_1 = 4000000  # Pa (40 bar)
    t_init_1 = 70  # K
    fill_1 = 0.0
    p_max_1 = 4500000  # Pa (45 bar)
    p_min_1 = 1500000  # Pa (15 bar)
    ambient_heat_load_1 = 2.0  # W/m²

    # Define Tank 2 parameters (consumer)
    radius_2 = 1.5  # m
    p_init_2 = 2000000  # Pa (20 bar)
    t_init_2 = 70  # K
    fill_2 = 0.0
    p_max_2 = 4000000  # Pa (40 bar)
    p_min_2 = 1000000  # Pa (10 bar)
    ambient_heat_load_2 = 2.0  # W/m²

    # Instantiate tank objects
    tank_material = Composite.carbon(np.radians(55))
    tank_dimensions_1 = TankDimensions(radius_1, 0.0)  # Spherical tank
    tank_dimensions_2 = TankDimensions(radius_2, 0.0)  # Spherical tank

    # Get mission details
    mission_2 = Mission.triathlon()
    total_fuel_mass = mission_2.required_fuel
    total_duration = sum(section.duration for section in mission_2.sections)

    # Calculate required flow rates
    avg_fuel_flow = total_fuel_mass / total_duration
    print(f"Average fuel flow: {avg_fuel_flow:.2f} kg/s")
    print(f"Total fuel mass required for mission: {total_fuel_mass:.2f} kg")

    # Create hydrogen properties for inflow
    hydrogen_requester = SinglePhaseRequester()
    hydrogen_props_1 = hydrogen_requester.get_hydrogen_properties(p_init_1, t_init_1)

    # Define the outflow from Tank 1 (equal to mission requirements)
    outflow_rate_1 = avg_fuel_flow
    outflow_1 = OutFlow(outflow_rate_1, "gas")

    # Define the inflow to Tank 2 (from Tank 1)
    inflow_rate_2 = avg_fuel_flow
    inflow_2 = InFlow(inflow_rate_2, hydrogen_props_1)

    # Create mission sections
    # For Tank 1 - create BOTH inflow and outflow objects
    dummy_inflow_1 = InFlow(0.0, hydrogen_props_1)  # Zero-rate inflow
    outflow_1 = OutFlow(outflow_rate_1, "gas")      # Actual outflow to Tank 2

    # For Tank 1 mission - include both flows
    mission_section_1 = MissionSection(
        duration=total_duration,
        fuel_flows=[dummy_inflow_1, outflow_1],  # Include both inflow and outflow
        altitude=0,
        mach_number=0.0
    )
    mission_1 = Mission([mission_section_1])

    # For Tank 2 - match mission sections but add inflow
    mission_sections_2 = []
    for section in mission_2.sections:
        # Copy each mission section but add inflow
        mission_sections_2.append(
            MissionSection(
                duration=section.duration,
                fuel_flows=[inflow_2] + section.fuel_flows,  # Add inflow to existing flows
                altitude=section.altitude,
                mach_number=section.mach_number
            )
        )
    mission_2_with_inflow = Mission(mission_sections_2)

    # Insulation for both tanks
    insulation_thickness = 0.05  # m
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    # Operating envelopes
    operating_window_1 = OperatingEnvelope(p_max_1, p_min_1, None)
    operating_window_2 = OperatingEnvelope(p_max_2, p_min_2, None)

    # Initial conditions
    initial_conditions_1 = InitialConditions(p_init_1, t_init_1, fill_1, multi_flow=True)
    initial_conditions_2 = InitialConditions(p_init_2, t_init_2, fill_2, multi_flow=True)

    # Run Tank 1 analysis
    print("Analyzing Tank 1 (Reservoir)...")
    tank_performance_1 = InOutTankAnalysisFacade.analyse(
        tank_dimensions_1,
        tank_material,
        insulation,
        mission_1,
        ambient_heat_load_1,
        initial_conditions_1,
        operating_window_1,
        TargetConditions(0.10 * total_fuel_mass, 0.0)  # Stop when 10% fuel left
    )

    # Run Tank 2 analysis
    print("Analyzing Tank 2 (Consumer)...")
    tank_performance_2 = InOutTankAnalysisFacade.analyse(
        tank_dimensions_2,
        tank_material,
        insulation,
        mission_2_with_inflow,
        ambient_heat_load_2,
        initial_conditions_2,
        operating_window_2,
        TargetConditions(5.0, 0.0)  # Drain to 5 kg
    )

    # Extract results
    tank_states_1 = tank_performance_1.tank_states
    tank_states_2 = tank_performance_2.tank_states

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


