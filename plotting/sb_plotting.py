from typing import Protocol, Union, List, Optional, Tuple
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
        - Positive values: Flow INTO the tank (refueling)
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