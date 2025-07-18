from typing import Protocol, Union, List, Optional, Tuple

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
                 palette: str = "deep", context: str = "paper"):
        """Initialize the plotter with styling options.

        Args:
            style: Seaborn style ("whitegrid", "darkgrid", "white", "dark", "ticks")
            font: Font family name
            palette: Color palette name
            context: Scaling parameters ("paper", "notebook", "talk", "poster")
        """
        configure_plot_style(font=font, palette=palette, style=style, context=context)
        self.palette = sns.color_palette(palette)

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

    # Add more methods as needed, following the same pattern as in the previous message
    # For example:

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

    # Implement other methods as needed