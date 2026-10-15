"""ThermalFigures: figure methods of the Delft plotter (one concern per module)."""


from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

# Import seaborn styling
from toplab.plotting.plot_style_sb import (
    DELFT_PALETTE
)
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.system.state_management import MultiTankResults



class ThermalFigures:
    def plot_thermal_heat_flows(
            self,
            results: MultiTankResults,
            thermal_model: Any,
            tank_index: int = 0,
            save_path: Optional[str] = None,
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            legend_location: str = 'best') -> plt.Figure:
        """Plot the heat flow through each layer of a tank thermal network."""
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        print(f"Plotting thermal heat flows for Tank {tank_index + 1}...")
        times_hours = results.times / 3600.0
        heat_flows = {
            'Ambient to shell': [],
            'Shell to insulation': [],
            'Insulation to structure': [],
            # 'Structure to H2': [],
        }

        for time, multi_state in zip(results.times, results.multi_tank_states):
            state = multi_state.get_tank_state(tank_index)
            heat_flows['Ambient to shell'].append(
                thermal_model.compute_ambient_to_shell_heat_flux(state.shell_temperature)
            )
            heat_flows['Shell to insulation'].append(
                thermal_model.compute_shell_to_insulation_heat_flux(
                    state.shell_temperature, state.insulation_temperature
                )
            )
            heat_flows['Insulation to structure'].append(
                thermal_model.compute_insulation_to_structure_heat_flux(
                    state.insulation_temperature, state.structure_temperature
                )
            )
            # heat_flows['Structure to H2'].append(
            #     thermal_model.compute_structure_to_h2_heat_flux(time, state)
            # )

        fig, ax = plt.subplots(figsize=(12, 6))
        line_styles = ['-', '--', '-.', ':']
        for series_index, (label, values) in enumerate(heat_flows.items()):
            ax.plot(
                times_hours,
                values,
                color=self.color_palette[series_index % len(self.color_palette)],
                linewidth=2,
                linestyle=line_styles[series_index],
                label=label,
            )

        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Heat flow [W]')
        ax.grid(True, alpha=0.3)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location,
                           frameon=True, fancybox=True, shadow=True,
                           framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Thermal heat-flow plot completed")
        return fig

    def plot_heat_exchanger_requirements(self,
                                       heat_exchanger_data: Dict[str, Any],
                                       tank_index: int = 0,
                                       include_ohex: bool = True,
                                       include_total: bool = True,
                                       save_path: Optional[str] = None,
                                       xlim: Optional[Tuple[float, float]] = None,
                                       ylim: Optional[Tuple[float, float]] = None,
                                       legend_location: str = 'best') -> plt.Figure:
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
            xlim: Optional tuple (xmin, xmax) for x-axis range [hours]
            ylim: Optional tuple (ymin, ymax) for y-axis range [kW]
            legend_location: Location string for legend (e.g., 'best', 'upper left')

        Returns:
            matplotlib Figure object
        """
        print(f"Plotting heat exchanger requirements for Tank {tank_index + 1}...")

        # Extract data
        times_hours = heat_exchanger_data.get('times', [])
        ihex_requirements = heat_exchanger_data.get('ihex_requirements', [])
        ohex_requirements = heat_exchanger_data.get('ohex_requirements', [])

        # Validate data
        if len(times_hours) == 0 or len(ihex_requirements) == 0:
            print(f"   WARNING: No heat exchanger data available for Tank {tank_index + 1}")
            # Create empty plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Heat Exchanger Data Available',
                   transform=ax.transAxes, ha='center', va='center', fontsize=plot_style.LEGEND_FONT_SIZE)
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

        # Plot iHEX requirements (always present) - no markers, use line styles for distinction
        ihex_linestyle = '-'
        ax.plot(times_hours, ihex_requirements_kw, color=ihex_color, linewidth=2,
                linestyle=ihex_linestyle, label='IHEX')

        # Plot oHEX requirements if available and requested
        ohex_plotted = False
        if include_ohex and len(ohex_requirements) > 0 and len(ohex_requirements) == len(times_hours):
            # Check if we have meaningful oHEX data (not all zeros)
            max_ohex = max(ohex_requirements) if len(ohex_requirements) > 0 else 0
            if max_ohex > 1.0:  # Only plot if we have significant values (> 1W)
                ohex_requirements_kw = [req / 1000.0 for req in ohex_requirements]
                ohex_linestyle = '--' if self.use_greyscale else '-'
                ax.plot(times_hours, ohex_requirements_kw, color=ohex_color, linewidth=2,
                        linestyle=ohex_linestyle, label='OHEX')
                ohex_plotted = True

        # Plot total requirements if both are available and requested
        if include_total and ohex_plotted:
            try:
                total_requirements = [ihex + ohex for ihex, ohex in zip(ihex_requirements, ohex_requirements)]
                total_requirements_kw = [req / 1000.0 for req in total_requirements]
                total_linestyle = '-.' if self.use_greyscale else '--'
                ax.plot(times_hours, total_requirements_kw, color=total_color, linewidth=2,
                        linestyle=total_linestyle, alpha=0.8, label='Total heat exchanger requirement')
            except Exception as e:
                print(f"   WARNING: Could not calculate total requirements: {e}")

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Formatting
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Heat Flow Requirement [kW]')
        # ax.set_title('Heat Exchanger Requirements vs Time')
        ax.grid(True, alpha=0.3)

        # Apply custom axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Add legend with 3D shadow effect (same as other plots)
        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        # Improve layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Heat exchanger requirements plot completed")
        return fig
