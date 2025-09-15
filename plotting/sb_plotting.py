from typing import Protocol, Union, List, Optional, Tuple
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from CoolProp.CoolProp import PropsSI, PhaseSI
from plotting.plot_style_sb import DELFT_PALETTE

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# Import from our custom seaborn style module
from plotting.plot_style_sb import (
    set_seaborn_style,
    configure_plot_style,
    create_figure_with_ax,
    apply_custom_ticks,
    format_axis_labels,
    add_legend
)

# Constants (matching those in plot_tank_states.py)
SECONDS_TO_HOURS = 1 / 60 ** 2
PASCAL_TO_BAR = 1e-5
TO_MEGA = 1e-6


class Performances(Protocol):
    volumetric_efficiency: float
    gravimetric_efficiency: float


class TankStates(Protocol):
    pressures: list[float]
    temperatures: list[float]
    timesteps_in_hours: list[float]
    pressures_in_bar: list[float]
    required_fluxes: list[float]
    fills: list[float]
    liquid_masses: list[float]
    gas_masses: list[float]


class SeabornPlotter:
    """A class to handle all Seaborn-based plotting for hydrogen fuel tank analysis."""

    def __init__(self, style: str = "whitegrid", font: str = "Cambria",
                 palette: str = "delft", context: str = "paper"):  # Changed default from "deep" to "delft"
        """Initialize the plotter with styling options.

        Args:
            style: Seaborn style ("whitegrid", "darkgrid", "white", "dark", "ticks")
            font: Font family name
            palette: Color palette name
            context: Scaling parameters ("paper", "notebook", "talk", "poster")
        """
        configure_plot_style(font=font, palette=palette, style=style, context=context)

        # Store the actual color palette for consistent use
        if palette == "delft":
            self.palette = DELFT_PALETTE
        else:
            self.palette = sns.color_palette(palette)

        self.figsize = (8, 6)  # Default figure size

    def plot_tank_loads(self, tank_states: Union[TankStates, List[TankStates]],
                        labels: Optional[List[str]] = None,
                        x_ticks: Optional[List[float]] = None,
                        y_ticks: Optional[List[float]] = None,
                        figsize: Tuple[float, float] = (8, 6)):
        """Plot tank pressure loads over time with Seaborn styling."""
        fig, ax = create_figure_with_ax(figsize)

        # Handle both single TankStates and lists
        if not isinstance(tank_states, list):
            tank_states = [tank_states]

        # Create default labels if not provided
        if labels is None:
            labels = [f"Tank {i+1}" for i in range(len(tank_states))]
        elif len(labels) < len(tank_states):
            # Extend labels if needed
            labels.extend([f"Tank {i+1}" for i in range(len(labels), len(tank_states))])

        # Plot each tank state
        for i, (state, label) in enumerate(zip(tank_states, labels)):
            color = self.palette[i % len(self.palette)]

            # Convert to numpy arrays to avoid pandas indexing issues
            x_data = np.array(state.timesteps_in_hours)
            y_data = np.array(state.pressures_in_bar)

            # Plot using direct matplotlib instead of seaborn for more compatibility
            ax.plot(x_data, y_data, label=label, color=color)

        format_axis_labels(ax, xlabel="Time [hour]", ylabel="Pressure [bar]")
        apply_custom_ticks(ax, xticks=x_ticks, yticks=y_ticks)
        add_legend(ax)
        fig.tight_layout()
        return fig

    def plot_tank_temperatures(self, tank_states: Union[TankStates, List[TankStates]],
                              labels: Optional[List[str]] = None,
                              x_ticks: Optional[List[float]] = None,
                              y_ticks: Optional[List[float]] = None,
                              figsize: Tuple[float, float] = (8, 6)):
        """Plot tank temperatures over time with Seaborn styling."""
        # Implementation similar to plot_tank_loads but for temperatures
        fig, ax = create_figure_with_ax(figsize)

        # Handle both single TankStates and lists
        if not isinstance(tank_states, list):
            tank_states = [tank_states]

        # Create default labels if not provided
        if labels is None:
            labels = [f"Tank {i+1}" for i in range(len(tank_states))]
        elif len(labels) < len(tank_states):
            # Extend labels if needed
            labels.extend([f"Tank {i+1}" for i in range(len(labels), len(tank_states))])

        # Plot each tank state
        for i, (state, label) in enumerate(zip(tank_states, labels)):
            color = self.palette[i % len(self.palette)]
            sns.lineplot(
                x=state.timesteps_in_hours,
                y=state.temperatures,
                label=label,
                color=color,
                ax=ax
            )

        format_axis_labels(ax, xlabel="Time [hour]", ylabel="Temperature [K]")
        apply_custom_ticks(ax, xticks=x_ticks, yticks=y_ticks)
        add_legend(ax)
        fig.tight_layout()
        return fig

    def plot_comparative_tank_states(self, tank_states_1, tank_states_2, figsize=(15, 5), titles=None):
        """
        Create a single figure with three comparative plots for two tanks:
        1. Fuel Mass comparison
        2. Temperature comparison
        3. Pressure comparison

        Args:
            tank_states_1: Tank states for Tank 1 (Reservoir)
            tank_states_2: Tank states for Tank 2 (Consumer)
            figsize: Figure size (width, height) in inches
            titles: Optional custom titles for the three plots

        Returns:
            Figure with three subplots
        """
        from plotting.plot_style_sb import configure_plot_style, KONINGSBLAUW, BORDEAUX

        # Apply styling with Cambria font and Delft palette
        configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

        # Default titles if none provided
        if titles is None:
            titles = [
                "Comparative Fuel Mass",
                "Comparative Tank Temperatures",
                "Comparative Tank Pressures"
            ]

        # Create a single figure with 3 subplots (side by side)
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # Unpack the axes for easier access
        mass_ax, temp_ax, pressure_ax = axs

        # Get time data for both tanks
        times1 = tank_states_1.timesteps_in_hours
        times2 = tank_states_2.timesteps_in_hours

        # SUBPLOT 1: Fuel Mass Comparison
        # Extract masses safely
        try:
            masses1 = []
            for state in tank_states_1.states:
                masses1.append(state.fuel_mass)

            masses2 = []
            for state in tank_states_2.states:
                masses2.append(state.fuel_mass)
        except (AttributeError, IndexError):
            # Fall back to total_masses if available
            try:
                masses1 = tank_states_1.total_masses
                masses2 = tank_states_2.total_masses
            except (AttributeError, IndexError):
                # Last resort - try to compute from liquid+gas masses
                try:
                    masses1 = [l + g for l, g in zip(tank_states_1.liquid_masses, tank_states_1.gas_masses)]
                    masses2 = [l + g for l, g in zip(tank_states_2.liquid_masses, tank_states_2.gas_masses)]
                except (AttributeError, IndexError):
                    print("Warning: Could not access fuel masses")
                    masses1 = [0] * len(times1)
                    masses2 = [0] * len(times2)

        # Plot mass data
        mass_ax.plot(times1, masses1, '-', color=BORDEAUX, label="Tank 1 (Reservoir)", linewidth=2)
        mass_ax.plot(times2, masses2, '-', color=KONINGSBLAUW, label="Tank 2 (Consumer)", linewidth=2)
        mass_ax.set_xlabel("Time [hour]")
        mass_ax.set_ylabel("Fuel Mass [kg]")
        mass_ax.set_title(titles[0])
        mass_ax.legend()
        mass_ax.grid(True, alpha=0.3)

        # SUBPLOT 2: Temperature Comparison
        temp_ax.plot(times1, tank_states_1.temperatures, '-', color=BORDEAUX, label="Tank 1 (Reservoir)", linewidth=2)
        temp_ax.plot(times2, tank_states_2.temperatures, '-', color=KONINGSBLAUW, label="Tank 2 (Consumer)", linewidth=2)
        temp_ax.set_xlabel("Time [hour]")
        temp_ax.set_ylabel("Temperature [K]")
        temp_ax.set_title(titles[1])
        temp_ax.legend()
        temp_ax.grid(True, alpha=0.3)

        # SUBPLOT 3: Pressure Comparison
        pressure_ax.plot(times1, tank_states_1.pressures_in_bar, '-', color=BORDEAUX, label="Tank 1 (Reservoir)", linewidth=2)
        pressure_ax.plot(times2, tank_states_2.pressures_in_bar, '-', color=KONINGSBLAUW, label="Tank 2 (Consumer)", linewidth=2)
        pressure_ax.set_xlabel("Time [hour]")
        pressure_ax.set_ylabel("Pressure [bar]")
        pressure_ax.set_title(titles[2])
        pressure_ax.legend()
        pressure_ax.grid(True, alpha=0.3)

        # Apply tight layout to the entire figure
        fig.tight_layout()

        return fig

    def plot_tank_mass_flows(self, time_points, tank1_inflow, tank1_outflow, tank2_inflow, tank2_outflow,
                         figsize=(10, 6), title="Tank Mass Flow Rates"):
        """
        Plot mass flows between tanks and to mission with Seaborn styling.

        Args:
            time_points: Time points in seconds
            tank1_inflow: Inflow rates for tank 1
            tank1_outflow: Outflow rates from tank 1
            tank2_inflow: Inflow rates for tank 2
            tank2_outflow: Outflow rates from tank 2 (mission flow)
            figsize: Figure size tuple (width, height)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        from plotting.plot_style_sb import configure_plot_style, KONINGSBLAUW, BORDEAUX, ROOD, BOSGROEN

        # Apply consistent styling
        configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure all arrays have the same length by truncating to the time array length
        time_len = min(len(time_points), len(tank1_inflow), len(tank1_outflow),
                        len(tank2_inflow), len(tank2_outflow))

        # Convert time from seconds to hours for consistent plotting
        time_hours = [t * (1 / 60**2) for t in time_points[:time_len]]

        # Plot the flow rates with consistent Delft colors
        ax.plot(time_hours, tank2_outflow[:time_len], '-', color=KONINGSBLAUW,
                label="Tank 2 Outflow (Mission)", linewidth=2)
        ax.plot(time_hours, tank1_inflow[:time_len], '-', color=BORDEAUX,
                label="Tank 1 Inflow", linewidth=2)
        ax.plot(time_hours, tank1_outflow[:time_len], '-', color=ROOD,
                label="Tank 1 Outflow", linewidth=2)
        ax.plot(time_hours, tank2_inflow[:time_len], '-', color=BOSGROEN,
                label="Tank 2 Inflow", linewidth=2)

        # Add a zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Add annotation with better styling
        ax.text(0.02, 0.02, "Tank 1 supplies Tank 2, which supplies the mission",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel("Time [hour]")
        ax.set_ylabel("Mass Flow Rate [kg/s]")

        # Add legend with better positioning
        ax.legend(loc='best', framealpha=0.9)

        # Apply tight layout
        fig.tight_layout()

        return fig

    def plot_single_tank_states(self, tank_states, figsize=(15, 5)):
        """
        Create a single figure with three plots for a single tank:
        1. Fuel Mass
        2. Temperature
        3. Pressure

        Similar to plot_comparative_tank_states but for a single dataset
        """
        # Colors from the Delft palette
        from plotting.plot_style_sb import KONINGSBLAUW

        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # Unpack the axes for easier access
        mass_ax, temp_ax, pressure_ax = axs

        # Get time data
        times = tank_states.timesteps_in_hours

        # SUBPLOT 1: Fuel Mass
        try:
            # Try to get masses directly from states
            masses = []
            for state in tank_states.states:
                masses.append(state.fuel_mass)
        except (AttributeError, IndexError):
            # Fall back to total_masses if available
            try:
                masses = tank_states.total_masses
            except (AttributeError, IndexError):
                # Last resort - compute from liquid+gas masses
                try:
                    masses = [l + g for l, g in zip(tank_states.liquid_masses, tank_states.gas_masses)]
                except (AttributeError, IndexError):
                    print("Warning: Could not access fuel masses")
                    masses = [0] * len(times)

        # Plot mass data
        mass_ax.plot(times, masses, '-', color=KONINGSBLAUW, linewidth=2)
        mass_ax.set_xlabel("Time [hour]")
        mass_ax.set_ylabel("Fuel Mass [kg]")
        mass_ax.set_title("Tank Fuel Mass")
        mass_ax.grid(True, alpha=0.3)

        # SUBPLOT 2: Temperature
        temp_ax.plot(times, tank_states.temperatures, '-', color=KONINGSBLAUW, linewidth=2)
        temp_ax.set_xlabel("Time [hour]")
        temp_ax.set_ylabel("Temperature [K]")
        temp_ax.set_title("Tank Temperature")
        temp_ax.grid(True, alpha=0.3)

        # SUBPLOT 3: Pressure
        pressure_ax.plot(times, tank_states.pressures_in_bar, '-', color=KONINGSBLAUW, linewidth=2)
        pressure_ax.set_xlabel("Time [hour]")
        pressure_ax.set_ylabel("Pressure [bar]")
        pressure_ax.set_title("Tank Pressure")
        pressure_ax.grid(True, alpha=0.3)

        # Apply tight layout
        fig.tight_layout()

        return fig

    def plot_single_mission_flows(self, mass_flows, fuel_flow_keys, durations, total_duration,
                 interpolated_mass_flows=None, figsize=(10, 6)):
        """
        Plot mission mass flows with Seaborn styling for a single tank.

        Standardized convention:
        - Positive values: Flow INTO the tank (Refuelling)
        - Negative values: Flow OUT OF the tank (consumption/draining)

        Args:
            mass_flows: List of flow rates for each mission section
            fuel_flow_keys: Keys/names for each flow section
            durations: Duration of each section in seconds
            total_duration: Total mission duration in hours
            interpolated_mass_flows: Optional interpolated flow data
            figsize: Figure size (width, height) tuple
        """
        # Colors from the Delft palette
        from plotting.plot_style_sb import KONINGSBLAUW, BORDEAUX

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Create time array
        cumulative_time = 0
        time_points = []
        section_mass_flows = []

        # Process each mission section
        for i, (flows, duration) in enumerate(zip(mass_flows, durations)):
            # Add section start and end times
            section_start_time = cumulative_time
            section_end_time = section_start_time + duration

            # For the sign convention, we need to handle InFlow objects differently
            # since their mass_flow values are already positive
            section_flows = []
            for f in flows:
                # Check if this is a value from an InFlow (already in correct convention)
                section_flows.append(f)

            # Add mass flow points
            if len(section_flows) == 2:  # If we have start and end values
                time_points.extend([section_start_time, section_end_time])
                section_mass_flows.extend(section_flows)
            elif len(section_flows) == 1:  # Handle single constant flow value case
                # Create two time points (start and end) with the same flow value
                time_points.extend([section_start_time, section_end_time])
                section_mass_flows.extend([section_flows[0], section_flows[0]])  # Duplicate the flow value
            else:  # If we have more detailed points
                # Create evenly spaced time points for this section
                section_times = np.linspace(section_start_time, section_end_time, len(section_flows))
                time_points.extend(section_times)
                section_mass_flows.extend(section_flows)

            cumulative_time = section_end_time

        # Convert time to hours for plotting
        time_hours = [t * (1 / 3600) for t in time_points]

        # Plot the mission flow rate
        ax.plot(time_hours, section_mass_flows, '-', color=KONINGSBLAUW,
                label="Mission Flow Rate", linewidth=2)

        # If we have interpolated mass flows, plot those too for comparison
        if interpolated_mass_flows is not None:
            # For interpolated flows, use the values directly
            interp_flows = interpolated_mass_flows

            # Create time points matching the length of interpolated_mass_flows
            # interp_times = np.linspace(0, total_duration, len(interp_flows))
            # ax.plot(interp_times, interp_flows, '--', color=BORDEAUX,
            #         label="Interpolated Flow Rate", linewidth=1.5, alpha=0.7)

        # Add a zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Set labels and title with standardized convention
        ax.set_title("Mission Mass Flow Rate")
        ax.set_xlabel("Time [hour]")
        ax.set_ylabel("Mass Flow Rate [kg/s]")

        # Add legend if we have multiple datasets
        # if interpolated_mass_flows is not None:
        #     ax.legend(loc='best', framealpha=0.9)

        # Apply tight layout
        fig.tight_layout()

        return fig

    def plot_density_temperature_combined(self, scenario_data, include_saturation_line=True,
                                 include_isobars=True, include_ref_data=False, figsize=(10, 8),
                                 temperature_range=(15, 80), density_range=(0, 80)):
        """
        Create a combined density-temperature plot for discharge, refuel and dormancy phases.

        Args:
            scenario_data: Dictionary containing temperature, density and pressure data for each scenario
            include_saturation_line: Whether to include hydrogen saturation line
            include_isobars: Whether to include isobar lines
            include_ref_data: Whether to include reference data from literature
            figsize: Figure size tuple
            temperature_range: Min/max temperature range for the plot
            density_range: Min/max density range for the plot

        Returns:
            Figure with the combined density-temperature plot
        """
        from plotting.plot_style_sb import BORDEAUX, KONINGSBLAUW, BOSGROEN, DONKERGRIJS, ORANJE
        import os
        import pandas as pd

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each scenario - using solid lines for consistency
        # Discharge (gray solid line)
        discharge_line, = ax.plot(scenario_data['discharge']['temperatures'],
                scenario_data['discharge']['densities'],
                '-', color=DONKERGRIJS, linewidth=2, label="Discharge")

        # Refuel (red solid line)
        refuel_line, = ax.plot(scenario_data['refuel']['temperatures'],
                scenario_data['refuel']['densities'],
                '-', color=BORDEAUX, linewidth=2, label="Refuelling")

        # Dormancy (blue solid line)
        dormancy_line, = ax.plot(scenario_data['dormancy']['temperatures'],
                scenario_data['dormancy']['densities'],
                '-', color=KONINGSBLAUW, linewidth=2, label="Dormancy")

        # Add reference data from literature if requested
        ref_discharge_line = None
        ref_refuel_line = None
        ref_dormancy_line = None

        if include_ref_data:
            # Base path for reference data files
            ref_data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "analysis", "verification", "reference_data"
            )

            # Use the sorted CSV files
            discharge_ref_file = os.path.join(ref_data_path, "discharge_data_15bar.csv")
            if os.path.exists(discharge_ref_file):
                try:
                    # Read CSV data
                    discharge_ref_data = pd.read_csv(discharge_ref_file, header=None)
                    x_values = discharge_ref_data.iloc[:, 0].to_numpy()  # First column as numpy array
                    y_values = discharge_ref_data.iloc[:, 1].to_numpy()  # Second column as numpy array

                    # Use dashed line style for reference data
                    ref_discharge_line, = ax.plot(x_values, y_values, '--', color=DONKERGRIJS,
                                               linewidth=2, label="Discharge (Ref)")
                except Exception as e:
                    print(f"Warning: Could not load reference discharge data: {e}")

            # Load and plot reference refuel data
            refuel_ref_file = os.path.join(ref_data_path, "refuel_data_15bar.csv")
            if os.path.exists(refuel_ref_file):
                try:
                    # Read CSV data
                    refuel_ref_data = pd.read_csv(refuel_ref_file, header=None)
                    x_values = refuel_ref_data.iloc[:, 0].to_numpy()  # First column as numpy array
                    y_values = refuel_ref_data.iloc[:, 1].to_numpy()  # Second column as numpy array

                    ref_refuel_line, = ax.plot(x_values, y_values, '--', color=BORDEAUX,
                                             linewidth=2, label="Refuelling (Ref)")

                except Exception as e:
                    print(f"Warning: Could not load reference refuel data: {e}")

            # Load and plot reference dormancy data
            dormancy_ref_file = os.path.join(ref_data_path, "dormancy_data_15bar.csv")
            if os.path.exists(dormancy_ref_file):
                try:
                    # Read CSV data
                    dormancy_ref_data = pd.read_csv(dormancy_ref_file, header=None)
                    x_values = dormancy_ref_data.iloc[:, 0].to_numpy()  # First column as numpy array
                    y_values = dormancy_ref_data.iloc[:, 1].to_numpy()  # Second column as numpy array

                    ref_dormancy_line, = ax.plot(x_values, y_values, '--', color=KONINGSBLAUW,
                                               linewidth=2, label="Dormancy (Ref)")

                except Exception as e:
                    print(f"Warning: Could not load reference dormancy data: {e}")        # Add direction arrows to each line
        # For discharge (about 1/3 along the path)
        if len(scenario_data['discharge']['temperatures']) > 10:
            idx = len(scenario_data['discharge']['temperatures']) // 3
            ax.annotate('', xy=(scenario_data['discharge']['temperatures'][idx],
                               scenario_data['discharge']['densities'][idx]),
                       xytext=(scenario_data['discharge']['temperatures'][idx-5],
                               scenario_data['discharge']['densities'][idx-5]),
                       arrowprops=dict(arrowstyle='->', color=DONKERGRIJS, lw=1.5))

        # For refuel (about 1/3 along the path)
        if len(scenario_data['refuel']['temperatures']) > 10:
            idx = len(scenario_data['refuel']['temperatures']) // 3
            ax.annotate('', xy=(scenario_data['refuel']['temperatures'][idx],
                               scenario_data['refuel']['densities'][idx]),
                       xytext=(scenario_data['refuel']['temperatures'][idx-5],
                               scenario_data['refuel']['densities'][idx-5]),
                       arrowprops=dict(arrowstyle='->', color=BORDEAUX, lw=1.5))

        # For dormancy (about 1/3 along the path)
        if len(scenario_data['dormancy']['temperatures']) > 10:
            idx = len(scenario_data['dormancy']['temperatures']) // 3
            ax.annotate('', xy=(scenario_data['dormancy']['temperatures'][idx],
                               scenario_data['dormancy']['densities'][idx]),
                       xytext=(scenario_data['dormancy']['temperatures'][idx-5],
                               scenario_data['dormancy']['densities'][idx-5]),
                       arrowprops=dict(arrowstyle='->', color=KONINGSBLAUW, lw=1.5))

        # Optional: Add saturation line
        saturation_line = None
        if include_saturation_line:
            # Get critical point data
            try:
                from CoolProp.CoolProp import PropsSI
                fluid = 'hydrogen'  # Use the fluid defined in hydrogen_retrievers.py
                T_triple = PropsSI('Ttriple', fluid)
                T_crit = PropsSI('Tcrit', fluid)

                # Create temperature range from triple point to critical point
                # but limited to our plot range
                temps = np.linspace(
                    max(temperature_range[0], T_triple),
                    min(temperature_range[1], T_crit),
                    100
                )

                # Create complete saturation curve
                sat_temps = []
                sat_densities = []

                # First add the saturated liquid branch (bottom to top)
                for temp in temps:
                    try:
                        # Get saturated liquid density (Q=0)
                        density = PropsSI('D', 'T', temp, 'Q', 0, fluid)
                        sat_temps.append(temp)
                        sat_densities.append(density)
                    except Exception:
                        pass

                # Then add the saturated vapor branch (top to bottom)
                for temp in reversed(temps):
                    try:
                        # Get saturated vapor density (Q=1)
                        density = PropsSI('D', 'T', temp, 'Q', 1, fluid)
                        sat_temps.append(temp)
                        sat_densities.append(density)
                    except Exception:
                        pass

                # Plot the complete saturation dome
                if len(sat_temps) > 0:
                    saturation_line, = ax.plot(sat_temps, sat_densities, '--', color=ORANJE,
                            linewidth=1.5, label="Saturation line")
            except Exception as e:
                print(f"Warning: Could not create saturation line: {e}")

        # Optional: Add isobar lines
        isobar_line = None
        if include_isobars:
            # Define key pressure levels in bar
            pressure_levels = [10, 15, 23, 100, 400, 450]

            # Create temperature points
            temps = np.linspace(temperature_range[0], temperature_range[1], 100)

            for pressure in pressure_levels:
                pressure_pa = pressure * 1e5  # Convert bar to Pa
                densities = []

                for temp in temps:
                    try:
                        # Calculate density at this temperature and pressure
                        density = PropsSI('D', 'T', temp, 'P', pressure_pa, 'Hydrogen')
                        densities.append(density)
                    except:
                        densities.append(np.nan)

                # Remove NaN values
                valid_indices = ~np.isnan(densities)
                valid_temps = temps[valid_indices]
                valid_densities = np.array(densities)[valid_indices]

                if len(valid_temps) > 0:
                    # Plot isobar line with dotted gray
                    isobar_line, = ax.plot(valid_temps, valid_densities, ':', color='gray', alpha=0.7, linewidth=1)

                    # Add pressure labels
                    if pressure in [400, 450]:
                        # Place label at 75% of the way through the line
                        idx = int(len(valid_temps) * 0.75)
                    else:
                        # For other pressures, use the midpoint as before
                        idx = len(valid_temps) // 2

                    if idx < len(valid_temps):
                        ax.text(valid_temps[idx], valid_densities[idx],
                                f"{pressure} bar", fontsize=12, alpha=0.8,
                                horizontalalignment='center', verticalalignment='center',
                                color='gray', bbox=dict(facecolor='white', alpha=0.7,
                                                       edgecolor=None, pad=1))

        # Mark starting points of each scenario with X
        # Get the first point from each dataset
        starting_point = None
        if len(scenario_data['discharge']['temperatures']) > 0:
            starting_point, = ax.plot(scenario_data['discharge']['temperatures'][0],
                    scenario_data['discharge']['densities'][0],
                    'x', color='black', markersize=8, markeredgewidth=2, label='Starting point')

        refuel_start = None
        if len(scenario_data['refuel']['temperatures']) > 0:
            refuel_start, = ax.plot(scenario_data['refuel']['temperatures'][0],
                    scenario_data['refuel']['densities'][0],
                    'o', color=BORDEAUX, markersize=8, markerfacecolor='none', markeredgewidth=2)

        if len(scenario_data['dormancy']['temperatures']) > 0:
            ax.plot(scenario_data['dormancy']['temperatures'][0],
                    scenario_data['dormancy']['densities'][0],
                    'x', color='black', markersize=8, markeredgewidth=2)

        # Create a proper legend with actual markers
        legend_elements = [
            discharge_line,
            refuel_line,
            dormancy_line
        ]

        legend_labels = ['Discharge', 'Refuelling', 'Dormancy']

        # Add reference data lines to legend if they exist
        if include_ref_data:
            if ref_discharge_line is not None:
                # Use the actual line objects for correct dashed line style in legend
                legend_elements.append(ref_discharge_line)
                legend_labels.append('Discharge (Ref)')

            if ref_refuel_line is not None:
                legend_elements.append(ref_refuel_line)
                legend_labels.append('Refuelling (Ref)')

            if ref_dormancy_line is not None:
                legend_elements.append(ref_dormancy_line)
                legend_labels.append('Dormancy (Ref)')

        if saturation_line is not None:
            legend_elements.append(saturation_line)
            legend_labels.append('Saturation line')

        if isobar_line is not None:
            legend_elements.append(isobar_line)
            legend_labels.append('Isobars')

        if starting_point is not None:
            legend_elements.append(starting_point)
            legend_labels.append('Starting point')

        legend = ax.legend(legend_elements, legend_labels,
                           loc='upper right', title='Legend')

        # Set labels and limits
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [g/L]')
        ax.set_xlim(temperature_range)
        ax.set_ylim(density_range)
        ax.grid(True, alpha=0.3)

        # Apply tight layout
        fig.tight_layout()

        return fig

    def plot_heat_exchanger_requirements(self, heat_flow_data, scenario_name=None,
                                         ihex_data=None, ohex_data=None, plot_total=True, figsize=(10, 6)):
        """
        Plot heat exchanger requirements over time.

        Args:
            heat_flow_data: Dictionary containing heat flow data from simulation results
                           Expected keys: 't', 'qdot_disch', 'qdot_ohex'
            scenario_name: Optional scenario name for title
            ihex_data: Optional separate iHEX data (time, heat_flow) tuple
            ohex_data: Optional separate oHEX data (time, heat_flow) tuple for future use
            plot_total: Boolean flag to plot total heat flow requirement (iHEX + oHEX) as dashed line
            figsize: Figure size tuple (width, height)

        Returns:
            Matplotlib figure
        """
        from plotting.plot_style_sb import configure_plot_style, KONINGSBLAUW, BORDEAUX, BOSGROEN

        # Apply consistent styling
        configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Convert time from seconds to hours for consistent plotting
        SECONDS_TO_HOURS = 1 / 3600

        # Plot iHEX heat flow requirement (Qdot_disch)
        if ihex_data is not None:
            # Use provided separate iHEX data
            time_hours, ihex_heat_flow = ihex_data
            ax.plot(time_hours, ihex_heat_flow, '-', color=KONINGSBLAUW,
                    label="iHEX Heat Flow Requirement", linewidth=2)
        elif 'qdot_disch' in heat_flow_data and len(heat_flow_data['qdot_disch']) > 0:
            # Use data from simulation results
            time_hours = [t * SECONDS_TO_HOURS for t in heat_flow_data['t']]
            ax.plot(time_hours, heat_flow_data['qdot_disch'], '-', color=KONINGSBLAUW,
                    label="iHEX Heat Flow Requirement", linewidth=2)

        # Plot oHEX heat flow requirement
        ohex_plotted = False
        ohex_values = None
        if ohex_data is not None:
            # Use provided separate oHEX data
            time_hours_ohex, ohex_heat_flow = ohex_data
            ax.plot(time_hours_ohex, ohex_heat_flow, '-', color=BORDEAUX,
                    label="oHEX Heat Flow Requirement", linewidth=2)
            ohex_plotted = True
            ohex_values = ohex_heat_flow
        elif 'qdot_ohex' in heat_flow_data and any(q != 0.0 for q in heat_flow_data['qdot_ohex']):
            # Use data from simulation results (only if non-zero values exist)
            time_hours = [t * SECONDS_TO_HOURS for t in heat_flow_data['t']]
            ax.plot(time_hours, heat_flow_data['qdot_ohex'], '-', color=BORDEAUX,
                    label="oHEX Heat Flow Requirement", linewidth=2)
            ohex_plotted = True
            ohex_values = heat_flow_data['qdot_ohex']

        # Plot total heat flow requirement (iHEX + oHEX) if requested and both curves exist
        if plot_total:
            ihex_values = None
            total_time_hours = None

            # Get iHEX values
            if ihex_data is not None:
                total_time_hours, ihex_values = ihex_data
            elif 'qdot_disch' in heat_flow_data and len(heat_flow_data['qdot_disch']) > 0:
                total_time_hours = [t * SECONDS_TO_HOURS for t in heat_flow_data['t']]
                ihex_values = heat_flow_data['qdot_disch']

            # Get oHEX values for total calculation
            if ohex_values is None and 'qdot_ohex' in heat_flow_data:
                ohex_values = heat_flow_data['qdot_ohex']

            # Calculate and plot total if we have both datasets
            if ihex_values is not None and ohex_values is not None and len(ihex_values) == len(ohex_values):
                total_heat_flow = [ihex + ohex for ihex, ohex in zip(ihex_values, ohex_values)]
                ax.plot(total_time_hours, total_heat_flow, '--', color=BOSGROEN,
                        label="Total Heat Flow Requirement", linewidth=2, alpha=0.8)

        # Add a zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Set labels and title
        title = "Heat Exchanger Requirements"
        if scenario_name:
            title += f" - {scenario_name.capitalize()}"
        ax.set_title(title)
        ax.set_xlabel("Time [hour]")
        ax.set_ylabel("Heat Flow Requirement [W]")

        # Add legend if we have data to plot
        if len(ax.get_lines()) > 1:  # More than just the reference line
            ax.legend(loc='best', framealpha=0.9)

        # Apply grid
        ax.grid(True, alpha=0.3)

        # Apply tight layout
        fig.tight_layout()

        return fig

    def plot_chained_scenarios(self, results, postprocessed_data=None, figsize=(16, 12)):
        """
        Plot results from multiple scenarios with different colors using SeabornPlotter styling.

        Parameters
        ----------
        results : list
            List of result dictionaries from run_hydrogen_tank_simulation()
        postprocessed_data : list, optional
            List of postprocessed data dictionaries. If None, will postprocess automatically.
        figsize : tuple, optional
            Figure size (width, height) in inches
        """
        from plotting.plot_style_sb import BORDEAUX, KONINGSBLAUW, BOSGROEN, DONKERGRIJS, ORANJE

        if postprocessed_data is None:
            raise ValueError("postprocessed_data must be provided for plotting")

        # Color mapping for scenarios using Delft palette colors
        scenario_colors = {
            'DISCHARGE': DONKERGRIJS,
            'REFUEL': BORDEAUX,
            'DORMANCY': KONINGSBLAUW
        }

        # Apply consistent styling
        from plotting.plot_style_sb import configure_plot_style
        configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()  # Make indexing easier

        # 1. Mass vs time
        ax = axes[0]
        for data in postprocessed_data:
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            ax.plot(data['t'], data['m'], color=color, label=data['scenario'], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (kg)")
        ax.set_title("Mass vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Gas Temperature vs time
        ax = axes[1]
        for data in postprocessed_data:
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            ax.plot(data['t'], data['T'], color=color, label=data['scenario'], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Gas Temperature vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Liner/Wall Temperature vs time
        ax = axes[2]
        for data in postprocessed_data:
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            ax.plot(data['t'], data['Ts'], color=color, label=data['scenario'], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Solid Temperature (K)")
        ax.set_title("Liner/Wall Temperature vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Pressure vs time
        ax = axes[3]
        for data in postprocessed_data:
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            ax.plot(data['t'], data['p']/1e5, color=color, label=data['scenario'], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title("Pressure vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Density vs time
        ax = axes[4]
        for i, data in enumerate(postprocessed_data):
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            ax.plot(data['t'], data['rho'], color=color, label=data['scenario'], linewidth=2)
            # Add density stopping threshold for reference
            result = results[i]
            rho_stop = result['metadata']['rho_stop']
            ax.axhline(y=rho_stop, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Density (kg/m³)")
        ax.set_title("Density vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Model usage vs time
        ax = axes[5]
        for data in postprocessed_data:
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            model_numeric = np.where(data['model_used'] == 'single_phase', 0, 1)
            ax.plot(data['t'], model_numeric, color=color, label=data['scenario'], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Model Type")
        ax.set_title("Model Usage vs Time")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Single Phase', 'Two Phase'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 7. Pressure vs Temperature
        ax = axes[6]
        for data in postprocessed_data:
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            single_phase_mask = data['model_used'] == 'single_phase'
            two_phase_mask = data['model_used'] == 'two_phase'

            if np.any(single_phase_mask):
                ax.scatter(data['T'][single_phase_mask], data['p'][single_phase_mask]/1e5,
                          c=color, alpha=0.6, s=10, marker='o',
                          label=f'{data["scenario"]} (Single)')
            if np.any(two_phase_mask):
                ax.scatter(data['T'][two_phase_mask], data['p'][two_phase_mask]/1e5,
                          c=color, alpha=0.6, s=10, marker='s',
                          label=f'{data["scenario"]} (Two)')
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title("Pressure vs Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Density vs Temperature
        ax = axes[7]
        for i, data in enumerate(postprocessed_data):
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            ax.plot(data['T'], data['rho'], color=color, label=data['scenario'], linewidth=2)
            # Add density stopping threshold for reference
            result = results[i]
            rho_stop = result['metadata']['rho_stop']
            ax.axhline(y=rho_stop, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Density (kg/m³)")
        ax.set_title("Density vs Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 9. Summary statistics
        ax = axes[8]
        scenarios = [data['scenario'] for data in postprocessed_data]
        single_percentages = [data['stats']['single_phase_percentage'] for data in postprocessed_data]
        two_percentages = [data['stats']['two_phase_percentage'] for data in postprocessed_data]

        x = np.arange(len(scenarios))
        width = 0.35

        ax.bar(x - width/2, single_percentages, width, label='Single Phase', alpha=0.7, color=KONINGSBLAUW)
        ax.bar(x + width/2, two_percentages, width, label='Two Phase', alpha=0.7, color=BORDEAUX)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Model Usage Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 10. Timeline overview
        ax = axes[9]
        for i, data in enumerate(postprocessed_data):
            color = scenario_colors.get(data['scenario'], KONINGSBLAUW)
            scenario_duration = data['t'][-1] - data['t'][0]
            ax.barh(i, scenario_duration, left=data['t'][0], color=color, alpha=0.7, label=data['scenario'])
            ax.text(data['t'][0] + scenario_duration/2, i, f'{scenario_duration:.0f}s',
                    ha='center', va='center', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scenario')
        ax.set_title('Scenario Timeline')
        ax.set_yticks(range(len(postprocessed_data)))
        ax.set_yticklabels([data['scenario'] for data in postprocessed_data])
        ax.grid(True, alpha=0.3)

        # 11. Final density comparison
        ax = axes[10]
        final_densities = [data['rho'][-1] for data in postprocessed_data]
        target_densities = [results[i]['metadata']['rho_stop'] for i in range(len(results))]

        x = np.arange(len(scenarios))
        colors = [scenario_colors.get(scenario, KONINGSBLAUW) for scenario in scenarios]
        ax.bar(x, final_densities, alpha=0.7, label='Final Density', color=colors)
        ax.scatter(x, target_densities, color='red', s=50, label='Target Density', zorder=5)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Density (kg/m³)')
        ax.set_title('Final vs Target Density')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 12. Summary text
        ax = axes[11]
        ax.axis('off')
        summary_text = "Simulation Summary:\n\n"
        for i, data in enumerate(postprocessed_data):
            result = results[i]
            stats = data['stats']
            summary_text += f"{data['scenario']}:\n"
            summary_text += f"  Duration: {data['t'][-1] - data['t'][0]:.1f}s\n"
            summary_text += f"  Final ρ: {stats['final_density']:.1f} kg/m³\n"
            summary_text += f"  Two-phase: {stats['two_phase_percentage']:.1f}%\n"
            if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
                summary_text += f"  ✓ Stopped at threshold\n"
            else:
                summary_text += f"  ○ Completed time span\n"
            summary_text += "\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=9)

        plt.tight_layout()
        return fig

    def print_detailed_simulation_statistics(self, results, postprocessed_data):
        """
        Print detailed statistics for simulation results.

        Parameters
        ----------
        results : list
            List of result dictionaries from run_hydrogen_tank_simulation()
        postprocessed_data : list
            List of postprocessed data dictionaries
        """
        print(f"\n{'='*80}")
        print("DETAILED SIMULATION STATISTICS")
        print(f"{'='*80}")

        for i, (result, data) in enumerate(zip(results, postprocessed_data)):
            print(f"\n{i+1}. {data['scenario']} SCENARIO:")
            print(f"   Time range: {data['t'][0]:.1f} - {data['t'][-1]:.1f} seconds ({data['t'][-1] - data['t'][0]:.1f}s duration)")
            print(f"   Final density: {data['stats']['final_density']:.2f} kg/m³ (target: {result['metadata']['rho_stop']:.1f} kg/m³)")
            print(f"   Density range: {data['stats']['density_range'][0]:.2f} - {data['stats']['density_range'][1]:.2f} kg/m³")
            print(f"   Model usage: {data['stats']['single_phase_percentage']:.1f}% single-phase, {data['stats']['two_phase_percentage']:.1f}% two-phase")

            if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
                print(f"   ✓ Stopped by density threshold at t={result['stop_info']['stop_time']:.2f}s")
            else:
                print(f"   ○ Completed full time span")

            if 'two_phase_pressure_range' in data['stats']:
                p_range = data['stats']['two_phase_pressure_range']
                T_range = data['stats']['two_phase_temperature_range']
                print(f"   Two-phase region: P={p_range[0]/1e5:.2f}-{p_range[1]/1e5:.2f} bar, T={T_range[0]:.2f}-{T_range[1]:.2f} K")

    def plot_refuel_analysis(self, times, masses, temperatures, densities, pressures,
                           initial_conditions, figsize=(14, 10)):
        """
        Create comprehensive plots for refuel scenario analysis.

        Parameters
        ----------
        times : array_like
            Time points [s]
        masses : array_like
            Mass values [kg]
        temperatures : array_like
            Temperature values [K]
        densities : array_like
            Density values [kg/m³]
        pressures : array_like
            Pressure values [bar]
        initial_conditions : dict
            Dictionary with 'pressure', 'temperature', 'density' keys
        figsize : tuple, optional
            Figure size (width, height) in inches
        """
        from plotting.plot_style_sb import BORDEAUX, KONINGSBLAUW, BOSGROEN, DONKERGRIJS, ORANJE

        # Apply consistent styling
        from plotting.plot_style_sb import configure_plot_style
        configure_plot_style(font="Cambria", palette="delft", style="whitegrid", context="paper")

        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("CCH2 Refuel Scenario Analysis", fontsize=16, fontweight='bold')

        # 1. Mass vs Time
        ax = axes[0, 0]
        ax.plot(times, masses, color=BORDEAUX, linewidth=2.5, label='Tank Mass')
        ax.axhline(y=initial_conditions['density'] * 0.5, color=DONKERGRIJS,
                   linestyle='--', alpha=0.7, label='Target Mass (78 kg/m³)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (kg)")
        ax.set_title("Mass Evolution During Refuel")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation for mass gain
        mass_gain = masses[-1] - masses[0]
        ax.annotate(f'Mass gain: {mass_gain:.2f} kg', xy=(times[-1] * 0.7, masses[-1] * 0.9),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=BORDEAUX, alpha=0.2))

        # 2. Temperature vs Time
        ax = axes[0, 1]
        ax.plot(times, temperatures, color=ORANJE, linewidth=2.5, label='Tank Temperature')
        ax.axhline(y=initial_conditions['temperature'], color=DONKERGRIJS,
                   linestyle='--', alpha=0.7, label='Initial Temperature')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Temperature Evolution During Refuel")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation for temperature change
        temp_change = temperatures[-1] - temperatures[0]
        ax.annotate(f'ΔT: {temp_change:+.2f} K', xy=(times[-1] * 0.7, temperatures[-1] * 1.02),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=ORANJE, alpha=0.2))

        # 3. Pressure vs Time
        ax = axes[1, 0]
        ax.plot(times, pressures, color=KONINGSBLAUW, linewidth=2.5, label='Tank Pressure')
        ax.axhline(y=initial_conditions['pressure'], color=DONKERGRIJS,
                   linestyle='--', alpha=0.7, label='Initial Pressure')
        ax.axhline(y=15.0, color='red', linestyle=':', alpha=0.7, label='Min Pressure (15 bar)')
        ax.axhline(y=450.0, color='red', linestyle=':', alpha=0.7, label='Vent Pressure (450 bar)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title("Pressure Evolution During Refuel")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation for pressure change
        pressure_change = pressures[-1] - pressures[0]
        ax.annotate(f'ΔP: {pressure_change:+.1f} bar', xy=(times[-1] * 0.7, pressures[-1] * 1.05),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=KONINGSBLAUW, alpha=0.2))

        # 4. Density vs Time
        ax = axes[1, 1]
        ax.plot(times, densities, color=BOSGROEN, linewidth=2.5, label='Tank Density')
        ax.axhline(y=initial_conditions['density'], color='red',
                   linestyle='--', alpha=0.7, label='Target Density (78 kg/m³)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Density (kg/m³)")
        ax.set_title("Density Evolution During Refuel")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation for density change
        density_change = densities[-1] - densities[0]
        ax.annotate(f'Δρ: {density_change:+.2f} kg/m³', xy=(times[-1] * 0.7, densities[-1] * 0.95),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=BOSGROEN, alpha=0.2))

        # Add summary text box
        summary_text = f"""Refuel Scenario Summary:
Duration: {times[-1]:.1f} s
Mass added: {mass_gain:.2f} kg
Final density: {densities[-1]:.1f} kg/m³
Config B disabled (REFUEL mode)"""

        fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        return fig