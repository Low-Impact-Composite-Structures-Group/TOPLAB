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
    fill_1 = 0.0 # no liquid
    p_max_1 = 4500000  # Pa (45 bar)
    p_min_1 = 1500000  # Pa (15 bar)
    ambient_heat_load_1 = 2.0  # W/m²

    # Define Tank 2 parameters (consumer)
    radius_2 = 1.5  # m
    p_init_2 = 2000000  # Pa (20 bar)
    t_init_2 = 70  # K
    fill_2 = 0.0 # no liquid
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
    print(f"Total fuel mass required for mission: {total_fuel_mass:.2f} kg")

    # Create hydrogen properties for inflow
    hydrogen_requester = SinglePhaseRequester()
    hydrogen_props_1 = hydrogen_requester.get_hydrogen_properties(p_init_1, t_init_1)

    # Create mission sections for Tank 1 that match the outflow needs of Tank 2
    mission_sections_1 = []
    for section in mission_2.sections:
        # Get the outflows from this mission section
        section_outflows = section.get_outflows()

        # Create matching outflows for Tank 1 (with same flow rates but positive)
        tank1_outflows = []
        for outflow in section_outflows:
            # The flow rate might be a single value or a list (for varying flows)
            if isinstance(outflow.mass_flow, list):
                # Maintain the same flow pattern but with positive values (as outflow from Tank 1)
                # We use negative values because outflow is conventionally negative
                tank1_outflow = OutFlow([-flow for flow in outflow.mass_flow], "gas")
            else:
                # Single value flow
                tank1_outflow = OutFlow(-outflow.mass_flow, "gas")
            tank1_outflows.append(tank1_outflow)

        # Create a dummy inflow (needed for the model)
        dummy_inflow_1 = InFlow(0.0, hydrogen_props_1)

        # Create the mission section for Tank 1
        mission_sections_1.append(
            MissionSection(
                duration=section.duration,
                fuel_flows=[dummy_inflow_1] + tank1_outflows,  # Include dummy inflow and real outflows
                altitude=section.altitude,
                mach_number=section.mach_number,
                fuel_flow_key=section.fuel_flow_key  # Keep the same label
            )
        )

    mission_1 = Mission(mission_sections_1)

    # For Tank 2 - add inflow from Tank 1 to each section
    mission_sections_2 = []
    for i, section in enumerate(mission_2.sections):
        # Get the outflows from Tank 1 for this section
        section_tank1_outflows = mission_sections_1[i].get_outflows()

        # Create matching inflows for Tank 2
        tank2_inflows = []
        for outflow in section_tank1_outflows:
            # Convert outflows from Tank 1 to inflows for Tank 2
            if isinstance(outflow.mass_flow, list):
                # For varying flows, create corresponding inflow
                # We flip the sign because outflow (negative) becomes inflow (positive)
                inflow_values = [-flow for flow in outflow.mass_flow]
                tank2_inflow = InFlow(inflow_values, hydrogen_props_1)
            else:
                # Single value flow
                tank2_inflow = InFlow(-outflow.mass_flow, hydrogen_props_1)
            tank2_inflows.append(tank2_inflow)

        # Create mission section with both inflows and original outflows
        mission_sections_2.append(
            MissionSection(
                duration=section.duration,
                fuel_flows=tank2_inflows + section.fuel_flows,  # Add inflows from Tank 1 to original flows
                altitude=section.altitude,
                mach_number=section.mach_number,
                fuel_flow_key=section.fuel_flow_key  # Keep the same label
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
    # Add mass flow plot
    mass_flow_fig = plot_tank_mass_flows(mission_1, mission_2_with_inflow)

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

def plot_tank_mass_flows(mission_1, mission_2_with_inflow):
    """Plot mass flows for both tanks over time"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate cumulative durations for x-axis
    durations = [section.duration for section in mission_1.sections]
    cumulative_durations = [0] + list(np.cumsum(durations))
    total_duration = sum(durations)

    # Create time points array (more points for smoother curves)
    time_points = np.linspace(0, total_duration, 1000)

    # Create arrays to hold the flow values at each time point
    tank1_inflow = np.zeros_like(time_points)  # Always zero (dummy flow)
    tank1_outflow = np.zeros_like(time_points)
    tank2_inflow = np.zeros_like(time_points)
    tank2_outflow = np.zeros_like(time_points)

    # Fill the flow arrays with values from each mission section
    for i, section in enumerate(mission_1.sections):
        start_time = cumulative_durations[i]
        end_time = cumulative_durations[i+1]
        section_mask = (time_points >= start_time) & (time_points <= end_time)

        # Process section flows
        # For each flow in the section, identify its type and add it to the appropriate array
        for flow in section.fuel_flows:
            if hasattr(flow, 'hydrogen'):  # It's an InFlow
                # This is the dummy flow for Tank 1, always zero
                continue  # Already initialized to zero
            else:  # It's an OutFlow
                # Calculate flow value at each time point in this section
                if isinstance(flow.mass_flow, list):
                    # Linear interpolation for varying flows
                    start_flow, end_flow = flow.mass_flow
                    section_times = np.linspace(start_time, end_time, sum(section_mask))
                    section_flows = np.interp(section_times, [start_time, end_time], [start_flow, end_flow])
                    tank1_outflow[section_mask] += section_flows
                else:
                    # Constant flow
                    tank1_outflow[section_mask] += flow.mass_flow

        # Now process Tank 2 flows (both inflow and outflow)
        section2 = mission_2_with_inflow.sections[i]
        for flow in section2.fuel_flows:
            if hasattr(flow, 'hydrogen'):  # It's an InFlow (from Tank 1)
                # Mirror of Tank 1 outflow, positive
                if isinstance(flow.mass_flow, list):
                    start_flow, end_flow = flow.mass_flow
                    section_times = np.linspace(start_time, end_time, sum(section_mask))
                    section_flows = np.interp(section_times, [start_time, end_time], [start_flow, end_flow])
                    tank2_inflow[section_mask] += section_flows
                else:
                    tank2_inflow[section_mask] += flow.mass_flow
            else:  # It's an OutFlow (to the mission)
                if isinstance(flow.mass_flow, list):
                    start_flow, end_flow = flow.mass_flow
                    section_times = np.linspace(start_time, end_time, sum(section_mask))
                    section_flows = np.interp(section_times, [start_time, end_time], [start_flow, end_flow])
                    tank2_outflow[section_mask] += section_flows
                else:
                    tank2_outflow[section_mask] += flow.mass_flow

    # Plot all four flows
    ax.plot(time_points, tank1_inflow, label="Tank 1 Inflow (Dummy)", color='lightgray', linestyle='-')
    ax.plot(time_points, tank1_outflow, label="Tank 1 Outflow", color='red', linestyle='-')
    ax.plot(time_points, tank2_inflow, label="Tank 2 Inflow (from Tank 1)", color='green', linestyle='-')
    ax.plot(time_points, tank2_outflow, label="Tank 2 Outflow (Mission)", color='blue', linestyle='-')

    # Add a zero line for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Add section boundary markers
    for duration in cumulative_durations[1:-1]:  # Skip first and last
        ax.axvline(x=duration, color='gray', linestyle=':', alpha=0.5)

    # Customize plot
    ax.set_title("Mass Flows Between Tanks")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass Flow [kg/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

def main():
    perform_analysis()

if __name__ == "__main__":
    main()


