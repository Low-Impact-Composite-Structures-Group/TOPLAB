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

    def __init__(self, analysis_name: str = "Multi-Tank Analysis"):
        """
        Initialize plotter with seaborn styling.

        Args:
            analysis_name: Name of the analysis (used in plot titles)
        """
        self.analysis_name = analysis_name

        # Configure seaborn style with Delft colors
        configure_plot_style(
            font="Cambria",
            palette="delft",
            style="whitegrid",
            context="paper",
            figure_size=(12, 8),
            dpi=100
        )

        print(f"🎨 DelftColourPlotter initialized for: {analysis_name}")

    def plot_tank_evolution(self,
                          results: MultiTankResults,
                          tank_index: int = 0,
                          reference_lines: Optional[Dict[str, float]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot tank evolution with 4 subplots: pressure, temperature, mass, and density vs time.

        This is the core tank evolution visualization showing the complete state evolution
        of a single tank over time. For multi-tank systems, call this function for each tank.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            reference_lines: Optional dict with reference values:
                - 'P_min': Minimum pressure [bar]
                - 'P_vent': Venting pressure [bar]
                - 'rho_stop': Stopping density [kg/m³]
                - 'T_ambient': Ambient temperature [K]
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting tank evolution for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Extract tank data arrays
        tank_data = results._extract_tank_arrays(tank_index)
        times_hours = results.times / 3600.0  # Convert to hours

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Evolution",
                     fontsize=14, fontweight='bold')

        # Colors from Delft palette
        primary_color = DELFT_PALETTE[0]  # DONKERBLAUW
        reference_color = DELFT_PALETTE[2]  # KONINGSBLAUW

        # Plot 1: Pressure vs Time
        ax1 = axes[0, 0]
        ax1.plot(times_hours, tank_data['pressures'], color=primary_color, linewidth=2, label='Pressure')
        ax1.set_xlabel('Time [hours]')
        ax1.set_ylabel('Pressure [bar]')
        ax1.set_title('Pressure vs Time')
        ax1.grid(True, alpha=0.3)

        # Add reference lines for pressure
        if reference_lines:
            if 'P_min' in reference_lines:
                ax1.axhline(y=reference_lines['P_min'], color=reference_color,
                           linestyle='--', alpha=0.7, label=f"P_min = {reference_lines['P_min']:.0f} bar")
            if 'P_vent' in reference_lines:
                ax1.axhline(y=reference_lines['P_vent'], color=DELFT_PALETTE[6],
                           linestyle='--', alpha=0.7, label=f"P_vent = {reference_lines['P_vent']:.0f} bar")

        if reference_lines and ('P_min' in reference_lines or 'P_vent' in reference_lines):
            ax1.legend(fontsize=9)

        # Plot 2: Temperature vs Time
        ax2 = axes[0, 1]
        ax2.plot(times_hours, tank_data['temperatures'], color=primary_color, linewidth=2, label='Temperature')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Temperature [K]')
        ax2.set_title('Temperature vs Time')
        ax2.grid(True, alpha=0.3)

        # Add reference line for ambient temperature
        if reference_lines and 'T_ambient' in reference_lines:
            ax2.axhline(y=reference_lines['T_ambient'], color=reference_color,
                       linestyle='--', alpha=0.7, label=f"T_ambient = {reference_lines['T_ambient']:.0f} K")
            ax2.legend(fontsize=9)

        # Plot 3: Mass vs Time
        ax3 = axes[1, 0]
        ax3.plot(times_hours, tank_data['masses'], color=primary_color, linewidth=2, label='Mass')
        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Mass [kg]')
        ax3.set_title('Mass vs Time')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Density vs Time
        ax4 = axes[1, 1]
        ax4.plot(times_hours, tank_data['densities'], color=primary_color, linewidth=2, label='Density')
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Density [kg/m³]')
        ax4.set_title('Density vs Time')
        ax4.grid(True, alpha=0.3)

        # Add reference line for stopping density
        if reference_lines and 'rho_stop' in reference_lines:
            ax4.axhline(y=reference_lines['rho_stop'], color=reference_color,
                       linestyle='--', alpha=0.7, label=f"ρ_stop = {reference_lines['rho_stop']:.1f} kg/m³")
            ax4.legend(fontsize=9)

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
                               temperature_range: Tuple[float, float] = (15, 80),
                               density_range: Tuple[float, float] = (0, 80),
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
            temperature_range: Min/max temperature range for the plot [K]
            density_range: Min/max density range for the plot [kg/m³]
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting density-temperature diagram for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Set default isobar pressures if not provided
        if isobar_pressures is None:
            isobar_pressures = [450, 400, 100, 15, 5]  # bar

        # Set default reference pressures if not provided
        if reference_pressures is None:
            reference_pressures = {}

        # Extract tank data arrays
        tank_data = results._extract_tank_arrays(tank_index)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Density-Temperature Diagram",
                     fontsize=14, fontweight='bold')

        # Primary color for tank path
        primary_color = DELFT_PALETTE[0]  # DONKERBLAUW

        # Plot tank path
        tank_line, = ax.plot(tank_data['temperatures'], tank_data['densities'],
                            '-', color=primary_color, linewidth=2,
                            label=f"Tank {tank_index + 1} Path")

        # Add direction arrow (about 1/3 along the path)
        if len(tank_data['temperatures']) > 10:
            idx = len(tank_data['temperatures']) // 3
            ax.annotate('', xy=(tank_data['temperatures'][idx],
                               tank_data['densities'][idx]),
                       xytext=(tank_data['temperatures'][idx-5],
                               tank_data['densities'][idx-5]),
                       arrowprops=dict(arrowstyle='->', color=primary_color, lw=1.5))

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
                ax.plot(T_sat_range, rho_sat_liquid, '--', color=DELFT_PALETTE[3],
                       linewidth=1.5, alpha=0.7, label='Liquid Saturation')
                ax.plot(T_sat_range, rho_sat_vapor, '--', color=DELFT_PALETTE[4],
                       linewidth=1.5, alpha=0.7, label='Vapor Saturation')

                # Mark critical point
                ax.plot(T_crit, rho_crit, 'o', color=DELFT_PALETTE[6], markersize=8,
                       label=f'Critical Point ({T_crit:.1f} K, {rho_crit:.1f} kg/m³)')

            except Exception as e:
                print(f"   ⚠️  Could not add saturation line: {e}")

        # Optional: Add isobars
        if include_isobars:
            try:
                from CoolProp.CoolProp import PropsSI
                fluid = 'hydrogen'

                # Convert pressure levels to Pa
                pressure_levels_pa = [P * 1e5 for P in isobar_pressures]  # Convert bar to Pa

                # Temperature range for isobars (use data range)
                temp_min, temp_max = np.min(tank_data['temperatures']), np.max(tank_data['temperatures'])
                temp_padding = (temp_max - temp_min) * 0.2  # 20% padding for isobars
                T_isobar_range = np.linspace(temp_min - temp_padding, temp_max + temp_padding, 100)

                # Define colors for different isobar types
                p_vent_bar = reference_pressures.get('P_vent', None)
                p_min_bar = reference_pressures.get('P_min', None)

                for i, (P_pa, P_bar) in enumerate(zip(pressure_levels_pa, isobar_pressures)):
                    rho_isobar = []
                    T_valid = []

                    for T in T_isobar_range:
                        try:
                            # Calculate density at this T, P
                            rho = PropsSI('D', 'T', T, 'P', P_pa, fluid)
                            # Use data-based density range for filtering
                            density_min, density_max = np.min(tank_data['densities']), np.max(tank_data['densities'])
                            if 0 <= rho <= density_max * 1.5:  # Allow some range above max data density
                                rho_isobar.append(rho)
                                T_valid.append(T)
                        except:
                            continue

                    if len(T_valid) > 5:  # Only plot if we have enough points
                        # Determine color and style based on pressure type
                        if p_vent_bar and abs(P_bar - p_vent_bar) < 0.1:  # Venting pressure
                            color = DELFT_PALETTE[6]  # ORANJE for venting
                            linewidth = 2
                            alpha = 0.8
                            label = f'{P_bar:.0f} bar (Vent)'
                        elif p_min_bar and abs(P_bar - p_min_bar) < 0.1:  # Minimum pressure
                            color = DELFT_PALETTE[5]  # ROOD for minimum
                            linewidth = 2
                            alpha = 0.8
                            label = f'{P_bar:.0f} bar (Min)'
                        else:  # Regular isobar
                            color = DELFT_PALETTE[2]  # KONINGSBLAUW for regular
                            linewidth = 1
                            alpha = 0.5
                            label = None

                        ax.plot(T_valid, rho_isobar, ':', color=color,
                               linewidth=linewidth, alpha=alpha, label=label)

                        # Add pressure label to the right margin outside the plot area
                        if T_valid and rho_isobar:
                            # Position label at the rightmost end of the isobar line, but offset to the right margin
                            label_x = T_valid[-1]  # Rightmost temperature point on the isobar
                            label_y = rho_isobar[-1]  # Corresponding density

                            # Offset the label to the right of the plot area
                            temp_range = T_valid[-1] - T_valid[0]

                            ax.text(label_x, label_y, f'{P_bar:.0f} bar',
                                   fontsize=10, alpha=0.8, color=color,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.7, edgecolor=color, linewidth=0.5),
                                   verticalalignment='center')

            except Exception as e:
                print(f"   ⚠️  Could not add isobars: {e}")

        # Formatting
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [kg/m³]')
        ax.set_title(f'Density-Temperature Diagram - Tank {tank_index + 1}')
        ax.grid(True, alpha=0.3)

        # Set axis limits based on data range with some padding
        temp_min, temp_max = np.min(tank_data['temperatures']), np.max(tank_data['temperatures'])
        density_min, density_max = np.min(tank_data['densities']), np.max(tank_data['densities'])

        # Add 10% padding to ranges
        temp_padding = (temp_max - temp_min) * 0.1
        density_padding = (density_max - density_min) * 0.1

        ax.set_xlim(temp_min - temp_padding, temp_max + temp_padding)
        ax.set_ylim(max(0, density_min - density_padding), density_max + density_padding)

        # Add legend with 3D shadow effect in top left corner
        legend = ax.legend(fontsize=9, loc='upper left', frameon=True, fancybox=True,
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

        print(f"   ✅ Density-temperature plot completed")
        return fig

    def plot_mass_flows(self,
                       results: MultiTankResults,
                       tank_index: int = 0,
                       include_venting_flow: bool = True,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot mass flow rates for a single tank.

        Shows inflow (positive), outflow (negative), and optionally venting flow (negative)
        over time. For mission sequences, creates subplots for each mission phase.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            include_venting_flow: Whether to show venting flow curve
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting mass flows for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Extract tank flow data
        tank_data = results._extract_tank_arrays(tank_index)
        times_hours = results.times / 3600.0  # Convert to hours

        # Check if we have mission sequence data (for future multi-mission support)
        # For now, assume single mission - can be extended later
        n_missions = 1  # TODO: Extract from results when multi-mission support is added

        # Create figure - single plot for single mission, subplots for multiple missions
        if n_missions == 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            axes = [ax]  # Make it iterable for consistent handling
            fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Mass Flow Rates",
                         fontsize=14, fontweight='bold')
        else:
            # Multi-mission subplot layout (for future implementation)
            fig, axes = plt.subplots(n_missions, 1, figsize=(12, 4*n_missions), sharex=True)
            if n_missions == 1:
                axes = [axes]  # Ensure it's always a list
            fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Mass Flow Rates (Mission Sequence)",
                         fontsize=14, fontweight='bold')

        # Color scheme
        inflow_color = DELFT_PALETTE[1]   # LICHTBLAUW (light blue) for inflows
        outflow_color = DELFT_PALETTE[6]  # ORANJE (orange) for outflows
        vent_color = DELFT_PALETTE[5]     # ROOD (red) for venting

        # Plot data for each mission (currently just one)
        for mission_idx in range(n_missions):
            ax = axes[mission_idx]

            # Extract flow rates (convert from g/s to kg/s)
            inflow_rates = tank_data['inflow_rates'] / 1000.0    # Convert to kg/s, positive values
            outflow_rates = -tank_data['outflow_rates'] / 1000.0 # Convert to kg/s, make negative for plotting
            vent_rates = -tank_data['vent_rates'] / 1000.0       # Convert to kg/s, make negative for plotting

            # Plot inflow (positive)
            ax.plot(times_hours, inflow_rates, color=inflow_color, linewidth=2,
                   label='Inflow', linestyle='-')

            # Plot outflow (negative)
            ax.plot(times_hours, outflow_rates, color=outflow_color, linewidth=2,
                   label='Outflow', linestyle='-')

            # Optionally plot venting flow (negative)
            if include_venting_flow:
                ax.plot(times_hours, vent_rates, color=vent_color, linewidth=2,
                       label='Venting Flow', linestyle='--')

            # Add zero line for reference
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

            # Formatting
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('Mass Flow Rate [kg/s]')
            if n_missions == 1:
                ax.set_title('Mass Flow Rates vs Time')
            else:
                ax.set_title(f'Mission {mission_idx + 1}')
            ax.grid(True, alpha=0.3)

            # Add legend with 3D shadow effect (same as tank evolution)
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
            ax.set_title(f'{self.analysis_name} - Tank {tank_index + 1} Heat Exchanger Requirements')
            return fig

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Heat Exchanger Requirements",
                     fontsize=14, fontweight='bold')

        # Color scheme
        ihex_color = DELFT_PALETTE[0]    # DONKERBLAUW (dark blue) for iHEX
        ohex_color = DELFT_PALETTE[6]    # ORANJE (orange) for oHEX
        total_color = DELFT_PALETTE[2]   # KONINGSBLAUW (royal blue) for total

        # Convert W to kW for better readability
        ihex_requirements_kw = [req / 1000.0 for req in ihex_requirements]

        # Plot iHEX requirements (always present)
        ax.plot(times_hours, ihex_requirements_kw, color=ihex_color, linewidth=2,
               label='iHEX (Internal Heat Exchanger)', linestyle='-')

        # Plot oHEX requirements if available and requested
        ohex_plotted = False
        if include_ohex and len(ohex_requirements) > 0 and len(ohex_requirements) == len(times_hours):
            # Check if we have meaningful oHEX data (not all zeros)
            max_ohex = max(ohex_requirements) if len(ohex_requirements) > 0 else 0
            if max_ohex > 1.0:  # Only plot if we have significant values (> 1W)
                ohex_requirements_kw = [req / 1000.0 for req in ohex_requirements]
                ax.plot(times_hours, ohex_requirements_kw, color=ohex_color, linewidth=2,
                       label='oHEX (Outboard Heat Exchanger)', linestyle='-')
                ohex_plotted = True

        # Plot total requirements if both are available and requested
        if include_total and ohex_plotted:
            try:
                total_requirements = [ihex + ohex for ihex, ohex in zip(ihex_requirements, ohex_requirements)]
                total_requirements_kw = [req / 1000.0 for req in total_requirements]
                ax.plot(times_hours, total_requirements_kw, color=total_color, linewidth=2,
                       label='Total Heat Exchanger Requirement', linestyle='--', alpha=0.8)
            except Exception as e:
                print(f"   ⚠️  Could not calculate total requirements: {e}")

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Formatting
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Heat Flow Requirement [kW]')
        ax.set_title('Heat Exchanger Requirements vs Time')
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


def main():
    """Test the DelftColourPlotter with sample data."""
    print("🧪 Testing DelftColourPlotter...")

    # This would normally come from actual simulation results
    # For now, just demonstrate the plotter initialization
    plotter = DelftColourPlotter("Test Analysis")
    print("✅ DelftColourPlotter test successful")


if __name__ == "__main__":
    main()