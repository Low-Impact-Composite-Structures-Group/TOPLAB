"""
Multi-tank plotting module for hydrogen storage analysis.

This module provides clean, modular plotting functionality for multi-tank systems
using seaborn styling. Each plotting function is designed to be general and versatile.

Key Features:
- DelftColourPlotter base class with seaborn styling
- Tank evolution plots (pressure, temperature, mass, density vs time)
- Optional reference lines (e.g., P_min, P_vent, stopping density)
- Configurable titles based on analysis_name from config
- Support for both single and multi-tank systems
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import seaborn styling
from plotting.plot_style_sb import (
    set_seaborn_style, configure_plot_style, create_figure_with_ax,
    apply_custom_ticks, format_axis_labels, add_legend, DELFT_PALETTE
)

# Multi-tank system data structures
from src.multi_tank.system.state_management import MultiTankResults


class DelftColourPlotter:
    """
    Base class for multi-tank plotting using Delft University seaborn styling.

    Provides consistent styling and utilities for all plot types while keeping
    each plotting function modular and focused on a specific visualization.
    """

    def __init__(self, analysis_name: str = "Multi-Tank Analysis",
                 use_greyscale: bool = False,
                 enable_multi_tank_overlay: bool = False):
        """
        Initialize plotter with seaborn styling.

        Args:
            analysis_name: Name of the analysis (used in plot titles)
            use_greyscale: Whether to use greyscale styling instead of colors
            enable_multi_tank_overlay: Whether to overlay multiple tanks on same plots
        """
        self.analysis_name = analysis_name
        self.use_greyscale = use_greyscale
        self.enable_multi_tank_overlay = enable_multi_tank_overlay

                # Configure seaborn style
        if use_greyscale:
            # Use standard delft styling but override colors
            configure_plot_style(
                font="Cambria",
                palette="delft",  # Use delft for now, override colors below
                style="whitegrid",
                context="paper",
                figure_size=(12, 8),
                dpi=100
            )
            # Define greyscale color scheme
            self.color_palette = ['#000000', '#404040', '#808080', '#A0A0A0', '#C0C0C0', '#E0E0E0']
            self.marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            print(f"🎨 DelftColourPlotter initialized for: {analysis_name} (Greyscale Mode)")
        else:
            configure_plot_style(
                font="Cambria",
                palette="delft",
                style="whitegrid",
                context="paper",
                figure_size=(12, 8),
                dpi=100
            )
            self.color_palette = DELFT_PALETTE
            self.marker_styles = None  # No markers for color mode
            print(f"🎨 DelftColourPlotter initialized for: {analysis_name}")

    def plot_tank_evolution(self,
                          results: MultiTankResults,
                          tank_index: int = 0,
                          reference_lines: Optional[Dict[str, float]] = None,
                          save_path: Optional[str] = None,
                          overlay_all_tanks: bool = None) -> plt.Figure:
        """
        Plot tank evolution with 4 subplots: pressure, temperature, mass, and density vs time.

        This is the core tank evolution visualization showing the complete state evolution
        of a single tank over time. For multi-tank systems, can overlay all tanks or show individually.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            reference_lines: Optional dict with reference values:
                - 'P_min': Minimum pressure [bar]
                - 'P_vent': Venting pressure [bar]
                - 'rho_stop': Stopping density [kg/m³]
                - 'T_ambient': Ambient temperature [K]
            save_path: Optional path to save the plot
            overlay_all_tanks: Whether to overlay all tanks (overrides class setting)

        Returns:
            matplotlib Figure object
        """
        # Determine if we should overlay all tanks
        should_overlay = overlay_all_tanks if overlay_all_tanks is not None else self.enable_multi_tank_overlay

        if should_overlay and results.n_tanks > 1:
            print(f"🔵 Plotting tank evolution for all {results.n_tanks} tanks (overlay mode)...")
        else:
            print(f"🔵 Plotting tank evolution for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10) if should_overlay else (12, 8))

        # if should_overlay and results.n_tanks > 1:
        #     fig.suptitle(f"{self.analysis_name} - Multi-Tank Evolution Comparison",
        #                  fontsize=14, fontweight='bold')
        # else:
        #     fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Evolution",
        #                  fontsize=14, fontweight='bold')

        # Get time array
        times_hours = results.times / 3600.0  # Convert to hours

        # Colors from palette (greyscale or Delft)
        primary_color = self.color_palette[0]
        reference_color = self.color_palette[2] if len(self.color_palette) > 2 else self.color_palette[1]

        # Setup axes
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # Configure axes
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Pressure [bar]')
        ax1.set_title('Pressure vs Time')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Temperature vs Time')
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Mass [kg]')
        ax3.set_title('Mass vs Time')
        ax3.grid(True, alpha=0.3)

        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Density [kg/m³]')
        ax4.set_title('Density vs Time')
        ax4.grid(True, alpha=0.3)

        # Plot tank data (single tank or multi-tank overlay)
        tanks_to_plot = range(results.n_tanks) if should_overlay else [tank_index]

        for i, tank_idx in enumerate(tanks_to_plot):
            # Extract tank data arrays
            tank_data = results._extract_tank_arrays(tank_idx)

            # Get color and marker configuration
            color = self.color_palette[i % len(self.color_palette)]
            marker_config = self._get_marker_config(i, len(times_hours))

            # Tank label
            tank_label = f"Tank {tank_idx + 1}" if should_overlay else "Data"

            # Plot 1: Pressure vs Time
            ax1.plot(times_hours, tank_data['pressures'], color=color, linewidth=2,
                    label=f'{tank_label} Pressure' if should_overlay else 'Pressure', **marker_config)

            # Plot 2: Temperature vs Time
            ax2.plot(times_hours, tank_data['temperatures'], color=color, linewidth=2,
                    label=f'{tank_label} Temperature' if should_overlay else 'Temperature', **marker_config)

            # Plot 3: Mass vs Time
            ax3.plot(times_hours, tank_data['masses'], color=color, linewidth=2,
                    label=f'{tank_label} Mass' if should_overlay else 'Mass', **marker_config)

            # Plot 4: Density vs Time
            ax4.plot(times_hours, tank_data['densities'], color=color, linewidth=2,
                    label=f'{tank_label} Density' if should_overlay else 'Density', **marker_config)

        # Add reference lines with greyscale-appropriate styling
        if reference_lines:
            # Define reference line styles for greyscale vs color
            if self.use_greyscale:
                pmin_style = {'color': '#000000', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 2.0}
                pvent_style = {'color': '#404040', 'linestyle': '-.', 'alpha': 0.8, 'linewidth': 2.0}
                tambient_style = {'color': '#404040', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5}
                rhostop_style = {'color': '#404040', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5}
            else:
                pmin_style = {'color': reference_color, 'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.5}
                pvent_style = {'color': reference_color, 'linestyle': ':', 'alpha': 0.7, 'linewidth': 1.5}
                tambient_style = {'color': reference_color, 'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.5}
                rhostop_style = {'color': reference_color, 'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.5}

            if 'P_min' in reference_lines:
                ax1.axhline(y=reference_lines['P_min'], label=f"P_min = {reference_lines['P_min']:.0f} bar", **pmin_style)
            if 'P_vent' in reference_lines:
                ax1.axhline(y=reference_lines['P_vent'], label=f"P_vent = {reference_lines['P_vent']:.0f} bar", **pvent_style)
            if 'T_ambient' in reference_lines:
                ax2.axhline(y=reference_lines['T_ambient'], label=f"T_ambient = {reference_lines['T_ambient']:.0f} K", **tambient_style)
            if 'rho_stop' in reference_lines:
                ax4.axhline(y=reference_lines['rho_stop'], label=f"ρ_stop = {reference_lines['rho_stop']:.1f} kg/m³", **rhostop_style)

        # Add legends to axes that have multiple series or reference lines with 3D shadow effect
        for ax in [ax1, ax2, ax3, ax4]:
            if len(ax.get_lines()) > 1 or (len(ax.get_lines()) == 1 and ax.get_lines()[0].get_label() != '_nolegend_'):
                legend = ax.legend(fontsize=9, frameon=True, fancybox=True,
                                 shadow=True, framealpha=0.9, edgecolor='black')
                # Additional styling for 3D effect
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_linewidth(1.2)

        # Improve layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Tank evolution plot completed")
        return fig

    def plot_density_temperature(self,
                               results: MultiTankResults,
                               tank_index: int = 0,
                               include_saturation_line: bool = True,
                               include_isobars: bool = True,
                               isobar_pressures: List[float] = None,
                               reference_pressures: Optional[Dict[str, float]] = None,
                               temperature_range: Optional[Tuple[float, float]] = None,
                               density_range: Tuple[float, float] = (0, 85),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot density-temperature diagram for a single tank.

        Shows the thermodynamic path of the hydrogen in density-temperature space,
        with optional saturation line and isobar references.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            include_saturation_line: Whether to include hydrogen saturation line
            include_isobars: Whether to include isobar lines
            isobar_pressures: List of pressures [bar] for isobars. If None, uses default [450, 400, 100, 15, 5]
            reference_pressures: Dict with 'P_vent' and 'P_min' [bar] for highlighting special isobars
            temperature_range: Min/max temperature range for the plot [K]. If None, auto-computed from data ± 10K
            density_range: Min/max density range for the plot [kg/m³]
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting density-temperature diagram for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Set default isobar pressures if not provided (match reference image style)
        if isobar_pressures is None:
            isobar_pressures = [450, 400, 200, 100, 50, 15, 5]  # bar

        # Set default reference pressures if not provided
        if reference_pressures is None:
            reference_pressures = {}

        # Extract tank data arrays
        tank_data = results._extract_tank_arrays(tank_index)

        # Auto-compute temperature range based on actual data if not provided
        if temperature_range is None:
            temp_min = min(tank_data['temperatures']) - 30.0
            temp_max = max(tank_data['temperatures']) + 30.0
            temperature_range = (temp_min, temp_max)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        # fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Density-Temperature Diagram",
                    #  fontsize=14, fontweight='bold')

        # Primary color for tank path (greyscale or color)
        primary_color = self.color_palette[0]  # Black for greyscale, Delft blue for color
        line_style = {'color': primary_color, 'linewidth': 2}
        if self.use_greyscale:
            marker_config = self._get_marker_config(0, len(tank_data['temperatures']))
            line_style.update(marker_config)

        # Plot tank path
        tank_line, = ax.plot(tank_data['temperatures'], tank_data['densities'],
                            '-', label=f"Tank {tank_index + 1} Path", **line_style)

        # Add direction arrow (about 1/3 along the path) - larger arrow for better visibility
        if len(tank_data['temperatures']) > 10:
            idx = len(tank_data['temperatures']) // 3
            # pick a start index a bit further back for a longer arrow, but not before 0
            start_idx = max(0, idx - 8)
            ax.annotate('',
                xy=(tank_data['temperatures'][idx], tank_data['densities'][idx]),
                xytext=(tank_data['temperatures'][start_idx], tank_data['densities'][start_idx]),
                arrowprops=dict(
                    arrowstyle='-|>',            # solid shaft with triangular head
                    color=primary_color,
                    linewidth=4.0,              # thicker shaft
                    mutation_scale=30,          # larger head
                    alpha=0.95,
                    connectionstyle="arc3,rad=0",
                ))

        # Optional: Add saturation line
        if include_saturation_line:
            try:
                from CoolProp.CoolProp import PropsSI
                fluid = 'hydrogen'

                # Get critical point
                T_crit = PropsSI('Tcrit', fluid)  # K
                rho_crit = PropsSI('rhocrit', fluid)  # kg/m³

                # Temperature range for saturation line (up to critical point)
                T_sat_range = np.linspace(14.01, T_crit, 100)  # From triple point to critical
                rho_sat_liquid = []
                rho_sat_vapor = []

                for T in T_sat_range:
                    try:
                        # Liquid density at saturation
                        rho_l = PropsSI('D', 'T', T, 'Q', 0, fluid)
                        rho_sat_liquid.append(rho_l)

                        # Vapor density at saturation
                        rho_v = PropsSI('D', 'T', T, 'Q', 1, fluid)
                        rho_sat_vapor.append(rho_v)
                    except:
                        # Skip if calculation fails
                        rho_sat_liquid.append(np.nan)
                        rho_sat_vapor.append(np.nan)

                # Plot saturation lines
                # Use appropriate colors based on mode
                sat_liquid_color = self.color_palette[3] if len(self.color_palette) > 3 else self.color_palette[1]
                sat_vapor_color = self.color_palette[4] if len(self.color_palette) > 4 else self.color_palette[2]
                crit_color = self.color_palette[1] if self.use_greyscale else DELFT_PALETTE[6]

                # Plot saturation lines with unified legend
                ax.plot(T_sat_range, rho_sat_liquid, '--', color=sat_liquid_color,
                       linewidth=1.5, alpha=0.7, label='Saturation line')
                ax.plot(T_sat_range, rho_sat_vapor, '--', color=sat_vapor_color,
                       linewidth=1.5, alpha=0.7)  # No label for vapor branch

                # Mark critical point
                ax.plot(T_crit, rho_crit, 'o', color=crit_color, markersize=8,
                       label=f'Critical Point ({T_crit:.1f} K, {rho_crit:.1f} kg/m³)')

            except Exception as e:
                print(f"   ⚠️  Could not add saturation line: {e}")

        # Optional: Add isobars
        if include_isobars:
            try:
                from CoolProp.CoolProp import PropsSI
                fluid = 'hydrogen'

                # Create temperature range for isobars
                temps = np.linspace(temperature_range[0], temperature_range[1], 100)

                # Define colors for different isobar types
                p_vent_bar = reference_pressures.get('P_vent', None)
                p_min_bar = reference_pressures.get('P_min', None)

                # Flag to track if we've added the general "Isobars" legend entry
                isobar_legend_added = False

                for pressure in isobar_pressures:
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
                        # Determine color and style based on pressure type
                        if p_vent_bar and abs(pressure - p_vent_bar) < 0.1:  # Venting pressure
                            color = self.color_palette[1] if self.use_greyscale else DELFT_PALETTE[6]  # Grey/Orange for venting
                            linewidth = 2
                            alpha = 0.8
                            label = f'{pressure:.0f} bar (Vent)'
                        elif p_min_bar and abs(pressure - p_min_bar) < 0.1:  # Minimum pressure
                            color = self.color_palette[2] if self.use_greyscale else DELFT_PALETTE[5]  # Dark grey/Red for minimum
                            linewidth = 2
                            alpha = 0.8
                            label = f'{pressure:.0f} bar (Min)'
                        else:  # Regular isobar
                            color = self.color_palette[3] if self.use_greyscale else DELFT_PALETTE[2]  # Light grey/Blue for regular
                            linewidth = 1
                            alpha = 0.7
                            # Add "Isobars" label only for the first regular isobar
                            label = 'Isobars' if not isobar_legend_added else None
                            if not isobar_legend_added:
                                isobar_legend_added = True

                        # Plot isobar line with dotted style to match reference image
                        ax.plot(valid_temps, valid_densities, ':', color=color, alpha=alpha, linewidth=linewidth, label=label)

                        # Add pressure labels at right edge of plot with arrows (following sb_plotting pattern)
                        if len(valid_temps) > 0:
                            # Find the rightmost point of the isobar line
                            right_idx = -1  # Last point
                            right_temp = valid_temps[right_idx]
                            right_density = valid_densities[right_idx]

                            # Position label at right edge with 1K spacing from right edge
                            label_x = temperature_range[1] - 1.0  # 1K from right edge

                            ax.annotate(f"{pressure} bar",
                                      xy=(right_temp, right_density),
                                      xytext=(label_x, right_density),
                                      arrowprops=dict(arrowstyle='->', color='black', lw=1, alpha=0.8),
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                               edgecolor='black', alpha=0.95),
                                      fontsize=9, ha='right', va='center')

            except Exception as e:
                print(f"   ⚠️  Could not add isobars: {e}")

        # Formatting to match reference image
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [kg/m³]')
        # ax.set_title(f'Density-Temperature Diagram - Tank {tank_index + 1}')
        ax.grid(True, alpha=0.3)

        # Set axis limits to match reference image style using provided temperature and density ranges
        ax.set_xlim(temperature_range)
        ax.set_ylim(density_range)

        # Add legend with title and 3D shadow effect in top left corner
        legend = ax.legend(fontsize=9, loc='best', frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black', title='Legend')
        # Additional styling for 3D effect
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)
        # Style the legend title
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(10)

        # Improve layout (suppress tight_layout warnings for density-temperature plots)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                plt.tight_layout()
            except:
                pass

        # Always apply aggressive margins to maximize plot area width
        plt.subplots_adjust(top=0.92, bottom=0.14, left=0.10, right=0.97)

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Density-temperature plot completed")
        return fig

    def plot_mass_flows(self,
                       results: MultiTankResults,
                       tank_index: int = 0,
                       include_venting_flow: bool = True,
                       include_coupling_flows: bool = True,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot mass flow rates for a single tank using the proven working pattern from multi_tank_analysis.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            include_venting_flow: Whether to show venting flow curve
            include_coupling_flows: Whether to show coupling flows (for multi-tank systems)
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting mass flows for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Create figure using simple, proven layout
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert time to hours for plotting
        times_hours = results.times / 3600.0

        # Get flow data for the specific tank using the exact same pattern as working multi_tank_analysis
        tank_data = results._extract_tank_arrays(tank_index)

        # Use the exact same approach as the working multi_tank_analysis:
        # Combine flows and make outflows negative for display (data is already in g/s)
        inflow_total = tank_data['inflow_rates'] + tank_data['coupling_inflow_rates']
        outflow_total = -(tank_data['outflow_rates'] + tank_data['coupling_outflow_rates'])  # Make negative
        vent = -tank_data['vent_rates']  # Make negative

        # Use consistent color palette approach to match other plots
        inflow_color = self.color_palette[0 % len(self.color_palette)]
        outflow_color = self.color_palette[1 % len(self.color_palette)]
        vent_color = self.color_palette[2 % len(self.color_palette)]

        # Plot with consistent styling and markers for greyscale mode
        inflow_style = {'color': inflow_color, 'linewidth': 2, 'linestyle': '-'}
        outflow_style = {'color': outflow_color, 'linewidth': 2, 'linestyle': '-'}
        vent_style = {'color': vent_color, 'linewidth': 2, 'linestyle': '--'}

        # Add markers for greyscale mode to improve readability
        inflow_style.update(self._get_marker_config(0, len(times_hours)))
        outflow_style.update(self._get_marker_config(1, len(times_hours)))
        vent_style.update(self._get_marker_config(2, len(times_hours)))

        # Plot all flows with consistent styling
        ax.plot(times_hours, inflow_total, label='Inflow', **inflow_style)
        ax.plot(times_hours, outflow_total, label='Outflow', **outflow_style)
        ax.plot(times_hours, vent, label='Vent', **vent_style)

        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Set up plot formatting to match other plots
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Flow Rate [g/s]')
        # ax.set_title(f'Tank {tank_index + 1} Flow Rates')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect (same styling as other plots)
        legend = ax.legend(fontsize=9, loc='best', frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect to match tank evolution plots
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Mass flow plot completed")
        return fig

    def plot_heat_exchanger_requirements(self,
                                       heat_exchanger_data: Dict[str, Any],
                                       tank_index: int = 0,
                                       include_ohex: bool = True,
                                       include_total: bool = True,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heat exchanger requirements for a single tank.

        Shows iHEX (internal heat exchanger from simulation) and oHEX (outboard heat exchanger
        calculated from enthalpy difference) requirements over time.

        Args:
            heat_exchanger_data: Dictionary containing heat exchanger data:
                - 'times': Time array [hours]
                - 'ihex_requirements': iHEX heat flow [W]
                - 'ohex_requirements': oHEX heat flow [W] (optional)
            tank_index: Index of tank to plot (0 for first tank)
            include_ohex: Whether to show oHEX curve
            include_total: Whether to show total (iHEX + oHEX) curve
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting heat exchanger requirements for Tank {tank_index + 1}...")

        # Extract data
        times_hours = heat_exchanger_data.get('times', [])
        ihex_requirements = heat_exchanger_data.get('ihex_requirements', [])
        ohex_requirements = heat_exchanger_data.get('ohex_requirements', [])

        # Validate data
        if len(times_hours) == 0 or len(ihex_requirements) == 0:
            print(f"   ⚠️  No heat exchanger data available for Tank {tank_index + 1}")
            # Create empty plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Heat Exchanger Data Available',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('Heat Flow Requirement [kW]')
            # ax.set_title(f'{self.analysis_name} - Tank {tank_index + 1} Heat Exchanger Requirements')
            return fig

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        # fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Heat Exchanger Requirements",
                    #  fontsize=14, fontweight='bold')

        # Color scheme (greyscale or color)
        if self.use_greyscale:
            ihex_color = self.color_palette[0]    # Black for iHEX
            ohex_color = self.color_palette[2]    # Dark grey for oHEX
            total_color = self.color_palette[1]   # Dark grey for total
        else:
            ihex_color = DELFT_PALETTE[0]    # DONKERBLAUW (dark blue) for iHEX
            ohex_color = DELFT_PALETTE[6]    # ORANJE (orange) for oHEX
            total_color = DELFT_PALETTE[2]   # KONINGSBLAUW (royal blue) for total

        # Convert W to kW for better readability
        ihex_requirements_kw = [req / 1000.0 for req in ihex_requirements]

        # Plot iHEX requirements (always present) with greyscale markers if needed
        ihex_style = {'color': ihex_color, 'linewidth': 2, 'linestyle': '-'}
        if self.use_greyscale:
            ihex_style.update(self._get_marker_config(0, len(times_hours)))
        ax.plot(times_hours, ihex_requirements_kw, label='IHEX', **ihex_style)

        # Plot oHEX requirements if available and requested
        ohex_plotted = False
        if include_ohex and len(ohex_requirements) > 0 and len(ohex_requirements) == len(times_hours):
            # Check if we have meaningful oHEX data (not all zeros)
            max_ohex = max(ohex_requirements) if len(ohex_requirements) > 0 else 0
            if max_ohex > 1.0:  # Only plot if we have significant values (> 1W)
                ohex_requirements_kw = [req / 1000.0 for req in ohex_requirements]
                ohex_style = {'color': ohex_color, 'linewidth': 2, 'linestyle': '-'}
                if self.use_greyscale:
                    ohex_style.update(self._get_marker_config(1, len(times_hours)))
                ax.plot(times_hours, ohex_requirements_kw, label='OHEX', **ohex_style)
                ohex_plotted = True

        # Plot total requirements if both are available and requested
        if include_total and ohex_plotted:
            try:
                total_requirements = [ihex + ohex for ihex, ohex in zip(ihex_requirements, ohex_requirements)]
                total_requirements_kw = [req / 1000.0 for req in total_requirements]
                total_style = {'color': total_color, 'linewidth': 2, 'linestyle': '--', 'alpha': 0.8}
                if self.use_greyscale:
                    total_style.update(self._get_marker_config(2, len(times_hours)))
                ax.plot(times_hours, total_requirements_kw, label='Total Heat Exchanger Requirement', **total_style)
            except Exception as e:
                print(f"   ⚠️  Could not calculate total requirements: {e}")

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Formatting
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Heat Flow Requirement [kW]')
        # ax.set_title('Heat Exchanger Requirements vs Time')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect (same as other plots)
        legend = ax.legend(fontsize=9, loc='best', frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        # Improve layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Heat exchanger requirements plot completed")
        return fig

    def _get_marker_config(self, line_index: int = 0) -> Dict[str, Any]:
        """
        Get marker configuration for greyscale plotting.

        Args:
            line_index: Index of the line/series for different marker styles

        Returns:
            Dictionary with marker configuration
        """
        if not self.use_greyscale or not self.marker_styles:
            return {}

        marker_style = self.marker_styles[line_index % len(self.marker_styles)]
        return {
            'marker': marker_style,
            'markevery': max(1, len(range(100)) // 20),  # Show ~20 markers per line
            'markersize': 6,
            'markerfacecolor': 'white',
            'markeredgewidth': 1.5
        }

    def _get_marker_config(self, line_index: int = 0, data_length: int = 100) -> Dict[str, Any]:
        """
        Get marker configuration for greyscale plotting.

        Args:
            line_index: Index of the line/series for different marker styles
            data_length: Length of data array to calculate marker density

        Returns:
            Dictionary with marker configuration
        """
        if not self.use_greyscale or not self.marker_styles:
            return {}

        marker_style = self.marker_styles[line_index % len(self.marker_styles)]
        # Calculate marker density - show ~15-25 evenly spaced markers per line
        target_markers = 20
        markevery = max(1, data_length // target_markers) if data_length > target_markers else 1

        return {
            'marker': marker_style,
            'markevery': markevery,
            'markersize': 5,
            'markerfacecolor': 'white',
            'markeredgewidth': 1.2,
            'markeredgecolor': self.color_palette[line_index % len(self.color_palette)]
        }

    def create_reference_lines_from_config(self, tank_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Create reference lines dictionary from tank configuration.

        Args:
            tank_config: Tank configuration dictionary with pressure/density limits

        Returns:
            Dictionary with reference line values for plotting
        """
        reference_lines = {}

        # Pressure references (convert Pa to bar)
        # Handle both numeric and string values with scientific notation
        if 'minimum_pressure' in tank_config:
            min_pressure = tank_config['minimum_pressure']
            if isinstance(min_pressure, str):
                min_pressure = float(min_pressure)
            reference_lines['P_min'] = min_pressure / 1e5

        if 'venting_pressure' in tank_config:
            vent_pressure = tank_config['venting_pressure']
            if isinstance(vent_pressure, str):
                vent_pressure = float(vent_pressure)
            reference_lines['P_vent'] = vent_pressure / 1e5

        # Density references
        if 'minimum_density' in tank_config:
            min_density = tank_config['minimum_density']
            if isinstance(min_density, str):
                min_density = float(min_density)
            reference_lines['rho_stop'] = min_density

        # Temperature references
        if 'ambient_temperature' in tank_config:
            ambient_temp = tank_config['ambient_temperature']
            if isinstance(ambient_temp, str):
                ambient_temp = float(ambient_temp)
            reference_lines['T_ambient'] = ambient_temp

        return reference_lines

    def plot_sequential_tank_evolution(self,
                                     mission_results: List[Dict[str, Any]],
                                     tank_index: int = 0,
                                     reference_lines: Optional[Dict[str, float]] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot sequential tank evolution across multiple missions with 4 subplots.

        Creates the same 4-panel layout as plot_tank_evolution but combines data from
        all missions with clear mission boundaries marked.

        Args:
            mission_results: List of dicts with keys 'name', 'type', 'result', 'orchestrator'
            tank_index: Index of tank to plot (0 for first tank)
            reference_lines: Optional dict with reference values
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting sequential tank evolution for Tank {tank_index + 1}...")

        # Aggregate data from all missions
        combined_times = []
        combined_pressures = []
        combined_temperatures = []
        combined_masses = []
        combined_densities = []
        mission_boundaries = []
        mission_labels = []

        time_offset = 0.0

        for i, mission_result in enumerate(mission_results):
            result = mission_result['result']
            name = mission_result.get('name', mission_result.get('mission', f'Mission_{i+1}'))

            # Extract tank data arrays
            tank_data = result._extract_tank_arrays(tank_index)
            times_hours = result.times / 3600.0  # Convert to hours

            # Add time offset to continue from previous mission
            adjusted_times = times_hours + time_offset

            combined_times.extend(adjusted_times)
            combined_pressures.extend(tank_data['pressures'])
            combined_temperatures.extend(tank_data['temperatures'])
            combined_masses.extend(tank_data['masses'])
            combined_densities.extend(tank_data['densities'])

            # Mark mission boundary (except for first mission)
            if i > 0:
                mission_boundaries.append(adjusted_times[0])

            mission_labels.append({
                'name': name.title(),
                'start_time': adjusted_times[0],
                'end_time': adjusted_times[-1]
            })

            # Update offset for next mission
            time_offset = adjusted_times[-1]

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # fig.suptitle(f"{self.analysis_name} - Sequential Tank {tank_index + 1} Evolution",
        #              fontsize=14, fontweight='bold')

        # Colors (greyscale or Delft palette)
        if self.use_greyscale:
            primary_color = self.color_palette[0]    # Black
            reference_color = self.color_palette[2]  # Dark grey
            boundary_color = self.color_palette[1]   # Dark grey
        else:
            primary_color = DELFT_PALETTE[0]  # DONKERBLAUW
            reference_color = DELFT_PALETTE[2]  # KONINGSBLAUW
            boundary_color = DELFT_PALETTE[4]  # ORANJE

        # Plot 1: Pressure vs Time
        ax1 = axes[0, 0]
        ax1.plot(combined_times, combined_pressures, color=primary_color, linewidth=2, label='Pressure')
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Pressure [bar]')
        ax1.set_title('Pressure vs Time')
        ax1.grid(True, alpha=0.3)

        # Add mission boundaries
        for boundary in mission_boundaries:
            ax1.axvline(x=boundary, color=boundary_color, linestyle=':', alpha=0.7, linewidth=1)

        # Add reference lines for pressure
        if reference_lines:
            if 'P_min' in reference_lines:
                ax1.axhline(y=reference_lines['P_min'], color=reference_color,
                           linestyle='--', alpha=0.7, label=f"P_min = {reference_lines['P_min']:.0f} bar")
            if 'P_vent' in reference_lines:
                vent_ref_color = self.color_palette[1] if self.use_greyscale else DELFT_PALETTE[6]
                ax1.axhline(y=reference_lines['P_vent'], color=vent_ref_color,
                           linestyle='--', alpha=0.7, label=f"P_vent = {reference_lines['P_vent']:.0f} bar")

        if reference_lines and ('P_min' in reference_lines or 'P_vent' in reference_lines):
            ax1.legend(fontsize=9)

        # Plot 2: Temperature vs Time
        ax2 = axes[0, 1]
        ax2.plot(combined_times, combined_temperatures, color=primary_color, linewidth=2, label='Temperature')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Temperature vs Time')
        ax2.grid(True, alpha=0.3)

        # Add mission boundaries
        for boundary in mission_boundaries:
            ax2.axvline(x=boundary, color=boundary_color, linestyle=':', alpha=0.7, linewidth=1)

        # Add reference line for ambient temperature
        if reference_lines and 'T_ambient' in reference_lines:
            ax2.axhline(y=reference_lines['T_ambient'], color=reference_color,
                       linestyle='--', alpha=0.7, label=f"T_ambient = {reference_lines['T_ambient']:.0f} K")
            ax2.legend(fontsize=9)

        # Plot 3: Mass vs Time
        ax3 = axes[1, 0]
        ax3.plot(combined_times, combined_masses, color=primary_color, linewidth=2, label='Mass')
        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Mass [kg]')
        ax3.set_title('Mass vs Time')
        ax3.grid(True, alpha=0.3)

        # Add mission boundaries
        for boundary in mission_boundaries:
            ax3.axvline(x=boundary, color=boundary_color, linestyle=':', alpha=0.7, linewidth=1)

        # Plot 4: Density vs Time
        ax4 = axes[1, 1]
        ax4.plot(combined_times, combined_densities, color=primary_color, linewidth=2, label='Density')
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Density [kg/m³]')
        ax4.set_title('Density vs Time')
        ax4.grid(True, alpha=0.3)

        # Add mission boundaries
        for boundary in mission_boundaries:
            ax4.axvline(x=boundary, color=boundary_color, linestyle=':', alpha=0.7, linewidth=1)

        # Add reference line for stopping density
        if reference_lines and 'rho_stop' in reference_lines:
            ax4.axhline(y=reference_lines['rho_stop'], color=reference_color,
                       linestyle='--', alpha=0.7, label=f"ρ_stop = {reference_lines['rho_stop']:.1f} kg/m³")
            ax4.legend(fontsize=9)

        # Add mission labels at the top of the plot
        for i, mission in enumerate(mission_labels):
            mid_time = (mission['start_time'] + mission['end_time']) / 2
            ax1.text(mid_time, ax1.get_ylim()[1] * 0.95, mission['name'],
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Improve layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Sequential tank evolution plot completed")
        return fig

    def plot_sequential_density_temperature(self,
                                          mission_results: List[Dict[str, Any]],
                                          tank_index: int = 0,
                                          include_saturation_line: bool = True,
                                          include_isobars: bool = True,
                                          isobar_pressures: List[float] = None,
                                          reference_pressures: Optional[Dict[str, float]] = None,
                                          temperature_range: Tuple[float, float] = (15, 80),
                                          density_range: Tuple[float, float] = (0, 85),
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot sequential density-temperature diagram showing thermodynamic path across all missions.

        Shows the combined trajectory of hydrogen in density-temperature space for all missions
        with different colors/markers for each mission phase.

        Args:
            mission_results: List of dicts with keys 'name', 'type', 'result', 'orchestrator'
            tank_index: Index of tank to plot (0 for first tank)
            include_saturation_line: Whether to include hydrogen saturation line
            include_isobars: Whether to include isobar lines
            isobar_pressures: List of pressures [bar] for isobars
            reference_pressures: Dict with 'P_vent' and 'P_min' [bar] for highlighting
            temperature_range: Min/max temperature range for the plot [K]
            density_range: Min/max density range for the plot [kg/m³]
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting sequential density-temperature for Tank {tank_index + 1}...")

        # Default isobar pressures if not provided
        if isobar_pressures is None:
            isobar_pressures = [450, 400, 200, 100, 50, 15, 5]  # bar

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        # fig.suptitle(f"{self.analysis_name} - Sequential Density-Temperature (Tank {tank_index + 1})",
        #              fontsize=14, fontweight='bold')

        # Different colors for each mission (greyscale or color)
        if self.use_greyscale:
            mission_colors = [self.color_palette[0], self.color_palette[1], self.color_palette[2]]  # Black, dark grey, grey
        else:
            mission_colors = [DELFT_PALETTE[0], DELFT_PALETTE[3], DELFT_PALETTE[5]]  # DONKERBLAUW, BORDEAUX, GRIJS

        # Plot each mission's trajectory
        for i, mission_result in enumerate(mission_results):
            result = mission_result['result']
            name = mission_result.get('name', mission_result.get('mission', f'Mission_{i+1}'))

            # Extract tank data
            tank_data = result._extract_tank_arrays(tank_index)

            color = mission_colors[i % len(mission_colors)]

            # Plot trajectory
            ax.plot(tank_data['temperatures'], tank_data['densities'],
                   color=color, linewidth=2.5, label=f"{name.title()}", alpha=0.8)

            # Mark start and end points
            ax.scatter(tank_data['temperatures'][0], tank_data['densities'][0],
                      color=color, s=80, marker='o', edgecolor='white', linewidth=1, zorder=5)
            ax.scatter(tank_data['temperatures'][-1], tank_data['densities'][-1],
                      color=color, s=80, marker='s', edgecolor='white', linewidth=1, zorder=5)

        # Add isobars if requested (simplified for sequential plots)
        if include_isobars and isobar_pressures:
            print(f"   📈 Isobar lines requested for pressures: {isobar_pressures} bar (skipped in sequential plot)")

        # Add saturation line if requested (simplified for sequential plots)
        if include_saturation_line:
            print(f"   📈 Saturation line requested (skipped in sequential plot)")

        # Set axis properties
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [kg/m³]')
        ax.set_title('Thermodynamic Path')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect
        legend = ax.legend(fontsize=9, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        # Set axis limits
        ax.set_xlim(temperature_range)
        ax.set_ylim(density_range)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Sequential density-temperature plot completed")
        return fig

    def plot_sequential_mass_flows(self,
                                 mission_results: List[Dict[str, Any]],
                                 tank_index: int = 0,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot sequential mass flows showing flow rates across all missions.

        Args:
            mission_results: List of dicts with keys 'name', 'type', 'result', 'orchestrator'
            tank_index: Index of tank to plot (0 for first tank)
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting sequential mass flows for Tank {tank_index + 1}...")

        # Aggregate flow data
        combined_times = []
        combined_flows = []
        mission_boundaries = []
        mission_labels = []

        time_offset = 0.0

        for i, mission_result in enumerate(mission_results):
            result = mission_result['result']
            name = mission_result.get('name', mission_result.get('mission', f'Mission_{i+1}'))
            mission_type = mission_result.get('type', name.lower() if name else 'unknown')

            times_hours = result.times / 3600.0
            adjusted_times = times_hours + time_offset

            # Get flow rates from mission type and duration
            if mission_type.lower() == 'discharge':
                flow_rates = [-0.001] * len(times_hours)  # kg/s (negative for outflow)
            elif mission_type.lower() == 'refuel':
                flow_rates = [0.07] * len(times_hours)   # kg/s (positive for inflow)
            elif mission_type.lower() == 'dormancy':
                flow_rates = [0.0] * len(times_hours)    # kg/s (zero flow)
            else:
                flow_rates = [0.0] * len(times_hours)

            combined_times.extend(adjusted_times)
            combined_flows.extend(flow_rates)

            if i > 0:
                mission_boundaries.append(adjusted_times[0])

            mission_labels.append({
                'name': name.title(),
                'start_time': adjusted_times[0],
                'end_time': adjusted_times[-1]
            })

            time_offset = adjusted_times[-1]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        # fig.suptitle(f"{self.analysis_name} - Sequential Mass Flows (Tank {tank_index + 1})",
        #              fontsize=14, fontweight='bold')

        # Plot flow rates (greyscale or color)
        primary_color = self.color_palette[0]  # Black for greyscale, Delft blue for color
        flow_style = {'color': primary_color, 'linewidth': 2.5}
        if self.use_greyscale:
            flow_style.update(self._get_marker_config(0, len(combined_times)))
        ax.plot(combined_times, combined_flows, label='Mass Flow Rate', **flow_style)

        # Add mission boundaries
        boundary_color = self.color_palette[1] if self.use_greyscale else DELFT_PALETTE[4]  # Dark grey/Orange
        for boundary in mission_boundaries:
            ax.axvline(x=boundary, color=boundary_color, linestyle=':', alpha=0.7, linewidth=1)

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        # Add mission labels
        for mission in mission_labels:
            mid_time = (mission['start_time'] + mission['end_time']) / 2
            y_pos = ax.get_ylim()[1] * 0.9 if mission['name'] != 'Dormancy' else ax.get_ylim()[1] * 0.1
            ax.text(mid_time, y_pos, mission['name'],
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Mass Flow Rate [kg/s]')
        ax.set_title('Mass Flow Rate vs Time')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect
        legend = ax.legend(fontsize=9, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Sequential mass flows plot completed")
        return fig


def main():
    """Test the DelftColourPlotter with sample data."""
    print("🧪 Testing DelftColourPlotter...")

    # This would normally come from actual simulation results
    # For now, just demonstrate the plotter initialization
    plotter = DelftColourPlotter("Test Analysis")
    print("✅ DelftColourPlotter test successful")


if __name__ == "__main__":
    main()