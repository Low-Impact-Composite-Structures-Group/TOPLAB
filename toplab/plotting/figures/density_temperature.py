"""DensityTemperatureFigures: figure methods of the Delft plotter (one concern per module)."""


from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import seaborn styling
from toplab.plotting.plot_style_sb import (
    DELFT_PALETTE, FONT_SIZE
)
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.system.state_management import MultiTankResults



class DensityTemperatureFigures:
    def plot_density_temperature(self,
                               results: MultiTankResults,
                               tank_index: int = 0,
                               include_saturation_line: bool = True,
                               include_isobars: bool = True,
                               isobar_pressures: List[float] = None,
                               reference_pressures: Optional[Dict[str, float]] = None,
                               xlim: Optional[Tuple[float, float]] = None,
                               ylim: Optional[Tuple[float, float]] = None,
                               arrow_position: float = 0.25,
                               arrow_size: float = 1.0,
                               valve_events: Optional[List[Dict[str, Any]]] = None,
                               valve_marker_size: float = 8.0,
                               save_path: Optional[str] = None,
                               legend_location: str = 'upper right') -> plt.Figure:
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
            xlim: Optional tuple (xmin, xmax) for x-axis (temperature) range [K]. If None, auto-computed from data
            ylim: Optional tuple (ymin, ymax) for y-axis (density) range [kg/m³]. If None, uses default (0, 85)
            arrow_position: Position along path for direction arrow (0.0-1.0, where 0=start, 1=end)
            arrow_size: Size multiplier for the direction arrow
            valve_events: List of valve events with 'time' and 'type' ('open'/'close') for multi-tank systems
            valve_marker_size: Size of valve opening/closing markers
            save_path: Optional path to save the plot
            legend_location: Location string for legend (e.g., 'best', 'upper left', 'center left')

        Returns:
            matplotlib Figure object
        """
        print(f"Plotting density-temperature diagram for Tank {tank_index + 1}...")

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

        # Auto-compute temperature range (xlim) based on actual data if not provided
        if xlim is None:
            temp_min = min(tank_data['h2_temperatures']) - 4
            temp_max = max(tank_data['h2_temperatures']) + 15
            xlim = (temp_min, temp_max)

        # Set default density range (ylim) if not provided
        if ylim is None:
            ylim = (0, 85)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        # fig.suptitle(f"{self.analysis_name} - Tank {tank_index + 1} Density-Temperature Diagram",
                    #  fontsize=14, fontweight='bold')

        # Primary color for tank path (greyscale or color)
        primary_color = self.color_palette[0]  # Black for greyscale, Delft blue for color

        # Plot tank path (no markers, just lines)
        tank_line, = ax.plot(tank_data['h2_temperatures'], tank_data['densities'],
                            '-', color=primary_color, linewidth=2,
                            label=f"Thermodynamic path")

        # Add configurable direction arrow
        if len(tank_data['h2_temperatures']) > 10:
            # Calculate arrow position along path
            idx = max(1, int(len(tank_data['h2_temperatures']) * arrow_position))
            start_idx = max(0, idx - max(1, int(8 * arrow_size)))

            ax.annotate('',
                xy=(tank_data['h2_temperatures'][idx], tank_data['densities'][idx]),
                xytext=(tank_data['h2_temperatures'][start_idx], tank_data['densities'][start_idx]),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=primary_color,
                    linewidth=2.0 * arrow_size,
                    mutation_scale=20 * arrow_size,
                    alpha=0.95,
                    connectionstyle="arc3,rad=0",
                ))

        # Add valve opening/closing markers if provided
        if valve_events and len(valve_events) > 0:
            for event in valve_events:
                event_time = event.get('time', 0)
                event_type = event.get('type', 'unknown')  # 'open' or 'close'

                # Find closest data point to this time
                times_seconds = results.times
                closest_idx = min(range(len(times_seconds)),
                                key=lambda i: abs(times_seconds[i] - event_time))

                if closest_idx < len(tank_data['h2_temperatures']):
                    temp = tank_data['h2_temperatures'][closest_idx]
                    density = tank_data['densities'][closest_idx]

                    if event_type == 'open':
                        # Circle marker for valve opening - white fill with thin black outline
                        ax.plot(temp, density, 'o', markerfacecolor='white', markeredgecolor='black',
                               markersize=valve_marker_size, markeredgewidth=1.0,
                               label='Valve opening' if not any('Valve opening' in line.get_label()
                                                               for line in ax.get_lines()) else "")
                    elif event_type == 'close':
                        # Cross marker for valve closing - white fill with thin black outline
                        ax.plot(temp, density, 'x', markerfacecolor='white', markeredgecolor='black',
                               markersize=valve_marker_size * 1.2, markeredgewidth=1.0,
                               label='Valve closing' if not any('Valve closing' in line.get_label()
                                                               for line in ax.get_lines()) else "")

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
                       label=f'Critical point ({T_crit:.1f} K, {rho_crit:.1f} kg/m³)')

            except Exception as e:
                print(f"   WARNING: Could not add saturation line: {e}")

        # Optional: Add isobars
        if include_isobars:
            try:
                from CoolProp.CoolProp import PropsSI
                fluid = 'hydrogen'

                # Create temperature range for isobars (use xlim if available)
                temps = np.linspace(xlim[0], xlim[1], 100)

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
                            label = f'{pressure:.0f} bar (vent)'
                        elif p_min_bar and abs(pressure - p_min_bar) < 0.1:  # Minimum pressure
                            color = self.color_palette[2] if self.use_greyscale else DELFT_PALETTE[5]  # Dark grey/Red for minimum
                            linewidth = 2
                            alpha = 0.8
                            label = f'{pressure:.0f} bar (min)'
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
                            label_x = xlim[1] - 1.0  # 1K from right edge

                            ax.annotate(f"{pressure} bar",
                                      xy=(right_temp, right_density),
                                      xytext=(label_x, right_density),
                                      arrowprops=dict(arrowstyle='->', color='black', lw=1, alpha=0.8),
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                               edgecolor='black', alpha=0.95),
                                      fontsize=9, ha='right', va='center')

            except Exception as e:
                print(f"   WARNING: Could not add isobars: {e}")

        # Formatting to match reference image
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [kg/m³]')
        # ax.set_title(f'Density-Temperature Diagram - Tank {tank_index + 1}')
        ax.grid(True, alpha=0.3)

        # Apply custom axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Only show legend if there are multiple curves (tank data + reference lines)
        lines = ax.get_lines()
        labeled_lines = [line for line in lines
                        if line.get_label() and not line.get_label().startswith('_')]

        # Show legend only if there are multiple labeled lines (tank data + reference lines)
        if len(labeled_lines) > 1:
            # Use provided legend location or default
            legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location, frameon=True, fancybox=True,
                              shadow=True, framealpha=0.9, edgecolor='black')
            # Additional styling for 3D effect
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_linewidth(1.2)
            # Style the legend title
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_fontsize(FONT_SIZE)

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
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Density-temperature plot completed")
        return fig

    def plot_sequential_density_temperature(self,
                                          mission_results: List[Dict[str, Any]],
                                          tank_index: int = 0,
                                          include_saturation_line: bool = True,
                                          include_isobars: bool = True,
                                          isobar_pressures: List[float] = None,
                                          reference_pressures: Optional[Dict[str, float]] = None,
                                          temperature_range: Tuple[float, float] = (15, 85),
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
        print(f"Plotting sequential density-temperature for Tank {tank_index + 1}...")

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
            ax.plot(tank_data['h2_temperatures'], tank_data['densities'],
                   color=color, linewidth=2.5, label=f"{name.title()}", alpha=0.8)

            # Mark start and end points
            ax.scatter(tank_data['h2_temperatures'][0], tank_data['densities'][0],
                      color=color, s=80, marker='o', edgecolor='white', linewidth=1, zorder=5)
            ax.scatter(tank_data['h2_temperatures'][-1], tank_data['densities'][-1],
                      color=color, s=80, marker='s', edgecolor='white', linewidth=1, zorder=5)

        # Add isobars if requested (simplified for sequential plots)
        if include_isobars and isobar_pressures:
            print(f"   INFO: Isobar lines requested for pressures: {isobar_pressures} bar (skipped in sequential plot)")

        # Add saturation line if requested (simplified for sequential plots)
        if include_saturation_line:
            print(f"   INFO: Saturation line requested (skipped in sequential plot)")

        # Set axis properties
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [kg/m³]')
        ax.set_title('Thermodynamic Path')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect
        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, frameon=True, fancybox=True,
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
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Sequential density-temperature plot completed")
        return fig
