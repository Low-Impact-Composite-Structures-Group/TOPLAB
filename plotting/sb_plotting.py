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
            interp_times = np.linspace(0, total_duration, len(interp_flows))
            ax.plot(interp_times, interp_flows, '--', color=BORDEAUX,
                    label="Interpolated Flow Rate", linewidth=1.5, alpha=0.7)

        # Add a zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Set labels and title with standardized convention
        ax.set_title("Mission Mass Flow Rate (Refuel (+) / Drain (-))")
        ax.set_xlabel("Time [hour]")
        ax.set_ylabel("Mass Flow Rate [kg/s]")

        # Add legend if we have multiple datasets
        if interpolated_mass_flows is not None:
            ax.legend(loc='best', framealpha=0.9)

        # Apply tight layout
        fig.tight_layout()

        return fig

    def plot_density_temperature_combined(self, scenario_data, include_saturation_line=True,
                                 include_isobars=True, include_ref_data=False, figsize=(10, 8),
                                 temperature_range=(30, 80), density_range=(0, 80)):
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