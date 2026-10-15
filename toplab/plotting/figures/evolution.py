"""EvolutionFigures: figure methods of the Delft plotter (one concern per module)."""


from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt

# Import seaborn styling
from toplab.plotting.plot_style_sb import (
    DELFT_PALETTE
)
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.system.state_management import MultiTankResults



class EvolutionFigures:
    def plot_tank_evolution(self,
                          results: MultiTankResults,
                          tank_index: int = 0,
                          reference_lines: Optional[Dict[str, float]] = None,
                          reference_lines_config: Optional[Dict[str, Dict[str, bool]]] = None,
                          event_lines: Optional[List[Dict[str, any]]] = None,
                          save_path: Optional[str] = None,
                          overlay_all_tanks: bool = None,
                          xlim: Optional[Tuple[float, float]] = None,
                          ylim_pressure: Optional[Tuple[float, float]] = None,
                          ylim_temperature: Optional[Tuple[float, float]] = None,
                          ylim_density: Optional[Tuple[float, float]] = None,
                          legend_locations: Optional[Dict[str, str]] = None,
                          separate_figures: bool = False,
                          save_dir: Optional[str] = None,
                          filename_prefix: Optional[str] = None,
                          dpi: int = 900) -> Any:
        """
        Plot tank evolution with 3 vertical subplots: pressure, temperature, and density vs time.

        This method now creates two separate plots for multi-tank systems:
        1. Main evolution plot (3 vertical subplots without mass)
        2. Mass evolution plot (separate figure)

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            reference_lines: Optional dict with reference values:
                - 'P_min': Minimum pressure [bar]
                - 'P_vent': Venting pressure [bar]
                - 'rho_stop': Stopping density [kg/m³]
                - 'T_ambient': Ambient temperature [K]
            reference_lines_config: Config for which reference lines to show:
                - 'pressure': {'show_p_min': bool, 'show_p_vent': bool}
                - 'temperature': {'show_t_ambient': bool}
                - 'density': {'show_stopping_density': bool}
            event_lines: Optional list of vertical event lines:
                - Each item: {'time': float (hours), 'label': str}
            save_path: Optional path to save the plot
            overlay_all_tanks: Whether to overlay all tanks (overrides class setting)
            xlim: Optional tuple (xmin, xmax) for x-axis range [hours]
            ylim_pressure: Optional tuple (ymin, ymax) for pressure subplot [bar]
            ylim_temperature: Optional tuple (ymin, ymax) for temperature subplot [K]
            ylim_density: Optional tuple (ymin, ymax) for density subplot [kg/m³]
            legend_locations: Optional dict with legend locations per subplot:
                - 'pressure': location string (e.g., 'best', 'upper left')
                - 'temperature': location string
                - 'density': location string
                Each value may also be a dict {'loc': str, 'offset': [dx, dy]} to nudge
                the legend by a fraction of the axes size (e.g. offset [0.02, 0.0]).

        Returns:
            matplotlib Figure object (main evolution plot)
        """
        # Determine if we should overlay all tanks
        should_overlay = overlay_all_tanks if overlay_all_tanks is not None else self.enable_multi_tank_overlay

        if should_overlay and results.n_tanks > 1:
            print(f"Plotting tank evolution for all {results.n_tanks} tanks (overlay mode)...")
        else:
            print(f"Plotting tank evolution for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # If separate_figures is requested, generate three standalone figures instead of a 3x1 grid
        if separate_figures:
            figs: Dict[str, plt.Figure] = {}

            # Helper to decide save location and name
            def _compute_save_file(metric: str) -> Optional[Path]:
                base_dir: Optional[Path] = None
                if save_dir:
                    base_dir = Path(save_dir)
                elif save_path:
                    base_dir = Path(save_path).parent
                # Default to current working directory if no path given
                if base_dir is None:
                    return None
                base_dir.mkdir(parents=True, exist_ok=True)

                # Build filename
                if save_path:
                    ext = Path(save_path).suffix or '.png'
                    base_stem = Path(save_path).stem
                else:
                    ext = '.png'
                    base_stem = filename_prefix or f"tank{tank_index + 1}"

                filename = f"{base_stem}_{metric}{ext}"
                return base_dir / filename

            # Extract tank data once
            tank_data = results._extract_tank_arrays(tank_index)
            times_hours = results.times / 3600.0

            # Colors
            primary_color = self.color_palette[0]
            line_style = self.line_styles[0] if self.use_greyscale else '-'

            # Per-plot common function
            def _format_axes(ax, xlabel: Optional[str], ylabel: str, ylim: Optional[Tuple[float, float]]):
                if xlabel:
                    ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if xlim is not None:
                    ax.set_xlim(xlim)

            # Pressure plot (match combined plot dimensions)
            fig_p, ax_p = plt.subplots(1, 1, figsize=(12, 10))
            ax_p.plot(times_hours, tank_data['pressures'], color=primary_color, linewidth=2, linestyle=line_style,
                      label='Pressure')
            _format_axes(ax_p, None, 'Pressure [bar]', ylim_pressure)

            # Add pressure references if configured
            if reference_lines:
                pmin = reference_lines.get('P_min')
                pvent = reference_lines.get('P_vent')
                if pmin is not None:
                    ax_p.axhline(y=pmin, color=('#000000' if self.use_greyscale else self.color_palette[2]),
                                 linestyle='--', alpha=0.7, linewidth=1.5, label='Minimum allowable pressure')
                if pvent is not None:
                    ax_p.axhline(y=pvent, color=('#404040' if self.use_greyscale else self.color_palette[6]),
                                 linestyle=':', alpha=0.7, linewidth=1.5, label='Maximum allowable / venting pressure')
            # Add event lines and labels only on pressure plot when separate_figures is True
            if event_lines:
                event_line_color = '#606060'  # Medium gray for events
                # Draw vertical lines
                for i, event in enumerate(event_lines):
                    event_time = event.get('time', 0.0)
                    ax_p.axvline(x=event_time, color=event_line_color, linestyle='-',
                                 linewidth=1.5, alpha=0.8, zorder=0.5)

                # Add annotations on pressure axis
                y_min, y_max = ax_p.get_ylim()
                y_range = y_max - y_min
                for i, event in enumerate(event_lines):
                    event_time = event.get('time', 0.0)
                    event_label = event.get('label', 'Event')
                    y_position_norm = max(0.0, min(1.0, event.get('y_position', 0.75)))
                    y_pos = y_min + y_range * y_position_norm

                    x_offset = 30 if i % 2 == 0 else -30
                    y_offset = 15 if i % 2 == 0 else -15
                    ha = 'left' if x_offset > 0 else 'right'

                    ax_p.annotate(
                        event_label,
                        xy=(event_time, y_pos),
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=event_line_color,
                                  alpha=1.0, linewidth=1.2),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                        color=event_line_color, linewidth=1.2),
                        fontsize=plot_style.LEGEND_FONT_SIZE,
                        fontweight='bold',
                        color=event_line_color,
                        ha=ha
                    )
            # Legend if needed
            lines = ax_p.get_lines()
            labeled_lines = [l for l in lines if l.get_label() and not l.get_label().startswith('_')]
            if len(labeled_lines) > 1:
                leg = ax_p.legend(fontsize=plot_style.LEGEND_FONT_SIZE, frameon=True, fancybox=True,
                                  shadow=True, framealpha=0.9, edgecolor='black')
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_linewidth(1.2)
            figs['pressure'] = fig_p

            # Temperature plot (match combined plot dimensions)
            fig_t, ax_t = plt.subplots(1, 1, figsize=(12, 10))
            temperature_series = [
                ('Shell', 'shell_temperatures'),
                ('Insulation', 'insulation_temperatures'),
                ('Structure', 'structure_temperatures'),
                ('H2', 'h2_temperatures'),
            ]
            for series_index, (label, key) in enumerate(temperature_series):
                ax_t.plot(
                    times_hours,
                    tank_data[key],
                    color=self.color_palette[series_index % len(self.color_palette)],
                    linewidth=2,
                    linestyle=self.line_styles[series_index % len(self.line_styles)] if self.use_greyscale else '-',
                    label=label,
                )
            _format_axes(ax_t, None, 'Temperature [K]', ylim_temperature)
            if reference_lines and 'T_ambient' in reference_lines:
                ax_t.axhline(y=reference_lines['T_ambient'], color=('#404040' if self.use_greyscale else self.color_palette[2]),
                             linestyle='--', alpha=0.7, linewidth=1.5, label='Ambient temperature')
            # Add event lines (no labels) to temperature plot
            if event_lines:
                event_line_color = '#606060'
                for event in event_lines:
                    event_time = event.get('time', 0.0)
                    ax_t.axvline(x=event_time, color=event_line_color, linestyle='-',
                                 linewidth=1.5, alpha=0.8, zorder=0.5)
            lines = ax_t.get_lines()
            labeled_lines = [l for l in lines if l.get_label() and not l.get_label().startswith('_')]
            if len(labeled_lines) > 1:
                leg = ax_t.legend(fontsize=plot_style.LEGEND_FONT_SIZE, frameon=True, fancybox=True,
                                  shadow=True, framealpha=0.9, edgecolor='black')
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_linewidth(1.2)
            figs['temperature'] = fig_t

            # Density plot (match combined plot dimensions)
            fig_d, ax_d = plt.subplots(1, 1, figsize=(12, 10))
            ax_d.plot(times_hours, tank_data['densities'], color=primary_color, linewidth=2, linestyle=line_style,
                      label='Density')
            _format_axes(ax_d, 'Time [hours]', 'Density [kg/m³]', ylim_density)
            if reference_lines and 'rho_stop' in reference_lines:
                ax_d.axhline(y=reference_lines['rho_stop'], color=('#404040' if self.use_greyscale else self.color_palette[2]),
                             linestyle='--', alpha=0.7, linewidth=1.5, label='Stopping density')
            # Add event lines (no labels) to density plot
            if event_lines:
                event_line_color = '#606060'
                for event in event_lines:
                    event_time = event.get('time', 0.0)
                    ax_d.axvline(x=event_time, color=event_line_color, linestyle='-',
                                 linewidth=1.5, alpha=0.8, zorder=0.5)
            lines = ax_d.get_lines()
            labeled_lines = [l for l in lines if l.get_label() and not l.get_label().startswith('_')]
            if len(labeled_lines) > 1:
                leg = ax_d.legend(fontsize=plot_style.LEGEND_FONT_SIZE, frameon=True, fancybox=True,
                                  shadow=True, framealpha=0.9, edgecolor='black')
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_linewidth(1.2)
            figs['density'] = fig_d

            # Save if requested
            for metric, fig in figs.items():
                out_file = _compute_save_file(metric)
                if out_file is not None:
                    fig.savefig(str(out_file), dpi=dpi, bbox_inches='tight', facecolor='white')
                    print(f"   Saved {metric} evolution to: {out_file}")

            print(f"   Separate tank evolution plots completed for Tank {tank_index + 1}")
            return figs

        # Create 3x1 vertical subplot grid with shared x-axis
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Get time array
        times_hours = results.times / 3600.0  # Convert to hours

        # Colors from palette (greyscale or Delft)
        primary_color = self.color_palette[0]
        reference_color = self.color_palette[2] if len(self.color_palette) > 2 else self.color_palette[1]

        # Setup axes
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]

        # Configure axes - only bottom subplot gets x-label since they share x-axis
        ax1.set_ylabel('Pressure [bar]')
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel('Temperature [K]')
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel('Time [hours]')
        ax3.set_ylabel('Density [kg/m³]')
        ax3.grid(True, alpha=0.3)

        # Plot tank data (single tank or multi-tank overlay)
        tanks_to_plot = range(results.n_tanks) if should_overlay else [tank_index]

        for i, tank_idx in enumerate(tanks_to_plot):
            # Extract tank data arrays
            tank_data = results._extract_tank_arrays(tank_idx)

            # Get color and line style
            color = self.color_palette[i % len(self.color_palette)]
            line_style = self.line_styles[i % len(self.line_styles)] if self.use_greyscale else '-'

            # Tank label
            tank_label = f"Tank {tank_idx + 1}" if should_overlay else "Data"

            # Plot 1: Pressure vs Time
            ax1.plot(times_hours, tank_data['pressures'], color=color, linewidth=2, linestyle=line_style,
                    label=f'{tank_label} Pressure' if should_overlay else 'Pressure')

            # Plot 2: all thermal-network node temperatures vs time
            temperature_series = [
                ('Shell', 'shell_temperatures'),
                ('Insulation', 'insulation_temperatures'),
                ('Structure', 'structure_temperatures'),
                ('H2', 'h2_temperatures'),
            ]
            for series_index, (node_label, key) in enumerate(temperature_series):
                ax2.plot(
                    times_hours,
                    tank_data[key],
                    color=self.color_palette[series_index % len(self.color_palette)] if not should_overlay else color,
                    linewidth=2,
                    linestyle=(
                        self.line_styles[series_index % len(self.line_styles)]
                        if not should_overlay and self.use_greyscale
                        else ['-', '--', '-.', ':'][series_index]
                    ),
                    label=f'{tank_label} {node_label}' if should_overlay else node_label,
                )

            # Plot 3: Density vs Time
            ax3.plot(times_hours, tank_data['densities'], color=color, linewidth=2, linestyle=line_style,
                    label=f'{tank_label} Density' if should_overlay else None)

        # Add reference lines with configurable visibility and improved labels
        if reference_lines and reference_lines_config:
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

            # Pressure reference lines
            pressure_config = reference_lines_config.get('pressure', {})
            if 'P_min' in reference_lines and pressure_config.get('show_p_min', False):
                ax1.axhline(y=reference_lines['P_min'], label="Minimum allowable pressure", **pmin_style)
            if 'P_vent' in reference_lines and pressure_config.get('show_p_vent', False):
                ax1.axhline(y=reference_lines['P_vent'], label="Maximum allowable / venting pressure", **pvent_style)

            # Temperature reference lines
            temperature_config = reference_lines_config.get('temperature', {})
            if 'T_ambient' in reference_lines and temperature_config.get('show_t_ambient', False):
                ax2.axhline(y=reference_lines['T_ambient'], label="Ambient temperature", **tambient_style)

            # Density reference lines
            density_config = reference_lines_config.get('density', {})
            if 'rho_stop' in reference_lines and density_config.get('show_stopping_density', False):
                ax3.axhline(y=reference_lines['rho_stop'], label="Stopping density", **rhostop_style)
        elif reference_lines:
            # Fallback for backward compatibility - show all reference lines if config not provided
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
                ax1.axhline(y=reference_lines['P_min'], label="Minimum allowable pressure", **pmin_style)
            if 'P_vent' in reference_lines:
                ax1.axhline(y=reference_lines['P_vent'], label="Maximum allowable / venting pressure", **pvent_style)
            if 'T_ambient' in reference_lines:
                ax2.axhline(y=reference_lines['T_ambient'], label="Ambient temperature", **tambient_style)
            if 'rho_stop' in reference_lines:
                ax3.axhline(y=reference_lines['rho_stop'], label="Stopping density", **rhostop_style)

        # Add vertical event lines if provided
        if event_lines:
            event_line_color = '#606060'  # Medium gray (different from reference line gray)

            for i, event in enumerate(event_lines):
                event_time = event.get('time', 0.0)  # Time in hours
                event_label = event.get('label', 'Event')
                # Get normalized y position (0-1 range, where 0=bottom, 1=top)
                # Default to 0.75 (25% from top) if not specified
                y_position_norm = event.get('y_position', 0.75)
                # Clamp to valid range
                y_position_norm = max(0.0, min(1.0, y_position_norm))

                # Add vertical line across all subplots with proper z-order and full extent
                for ax in [ax1, ax2, ax3]:
                    # Draw line with low z-order to appear behind legend and other elements
                    ax.axvline(x=event_time, color=event_line_color, linestyle='--',
                             linewidth=1.5, alpha=0.8, zorder=0.5)

                # Remove gaps between subplots to make lines appear continuous
                # This is handled by tight subplot spacing

                # Add callout annotation only on the top subplot (pressure) to avoid clutter
                # Use normalized position from config (0=bottom, 1=top)
                y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
                y_bottom = ax1.get_ylim()[0]

                # Calculate y position: bottom + (range * normalized_position)
                y_pos = y_bottom + y_range * y_position_norm

                # Use larger offsets and position further from event line to avoid data
                x_offset = 30 if i % 2 == 0 else -30  # Larger horizontal offset
                y_offset = 15 if i % 2 == 0 else -15  # Slightly larger vertical offset

                # Determine horizontal alignment based on position
                ha = 'left' if x_offset > 0 else 'right'

                ax1.annotate(event_label,
                           xy=(event_time, y_pos),
                           xytext=(x_offset, y_offset),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=event_line_color,
                                   alpha=1.0, linewidth=1.2),  # Opaque background
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                         color=event_line_color, linewidth=1.2),
                           fontsize=plot_style.LEGEND_FONT_SIZE,
                           fontweight='bold',
                           color=event_line_color,
                           ha=ha)

        # Apply custom axis limits if provided
        if xlim is not None:
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(xlim)

        if ylim_pressure is not None:
            ax1.set_ylim(ylim_pressure)

        if ylim_temperature is not None:
            ax2.set_ylim(ylim_temperature)

        if ylim_density is not None:
            ax3.set_ylim(ylim_density)

        # Add legends when there are labeled items to display
        for ax in [ax1, ax2, ax3]:
            # Get all lines with actual labels (not '_nolegend_' or None)
            lines = ax.get_lines()
            labeled_lines = [line for line in lines
                           if line.get_label() and not line.get_label().startswith('_')]

            # Count data curves vs reference lines
            data_curves = 0
            reference_lines_count = 0
            for line in labeled_lines:
                label = line.get_label()
                # Reference lines typically contain specific keywords
                if any(keyword in label.lower() for keyword in ['minimum', 'maximum', 'venting', 'ambient', 'stopping']):
                    reference_lines_count += 1
                else:
                    data_curves += 1

            # Show legend only if:
            # 1. Multiple data curves (overlay mode), OR
            # 2. At least one reference line is present
            should_show_legend = (data_curves > 1) or (reference_lines_count > 0)

            if should_show_legend:
                # Determine legend location - use custom if provided, otherwise default to 'best'
                if legend_locations:
                    if ax == ax1:  # Pressure subplot
                        loc = legend_locations.get('pressure', 'best')
                    elif ax == ax2:  # Temperature subplot
                        loc = legend_locations.get('temperature', 'best')
                    else:  # Density subplot (ax3)
                        loc = legend_locations.get('density', 'best')
                else:
                    loc = 'best'

                # A legend entry may be a location string or {loc, offset: [dx, dy]},
                # where the offset shifts the legend in axes-fraction units.
                offset = None
                if isinstance(loc, dict):
                    offset = loc.get('offset')
                    loc = loc.get('loc', 'best')
                legend_kwargs = {}
                if offset is not None and loc != 'best':
                    legend_kwargs = {'bbox_to_anchor': (float(offset[0]), float(offset[1]), 1.0, 1.0),
                                     'bbox_transform': ax.transAxes}
                legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=loc, frameon=True, fancybox=True,
                                 shadow=True, framealpha=0.9, edgecolor='black', **legend_kwargs)
                if not legend_kwargs:
                    # Ensure legend stays within plot boundaries
                    legend.set_bbox_to_anchor(None)
                # Additional styling for 3D effect
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_linewidth(1.2)

        # Improve layout with minimal vertical spacing for continuous event lines
        plt.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10, hspace=0.05)

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"   Saved to: {save_path}")

        print(f"   Tank evolution plot completed")
        return fig

    def plot_mass_evolution(self,
                          results: MultiTankResults,
                          tank_index: int = 0,
                          save_path: Optional[str] = None,
                          overlay_all_tanks: bool = None,
                          dpi: int = 900) -> plt.Figure:
        """
        Plot mass evolution as a separate figure for the new 2-plot system.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            save_path: Optional path to save the plot
            overlay_all_tanks: Whether to overlay all tanks (overrides class setting)

        Returns:
            matplotlib Figure object
        """
        # Determine if we should overlay all tanks
        should_overlay = overlay_all_tanks if overlay_all_tanks is not None else self.enable_multi_tank_overlay

        if should_overlay and results.n_tanks > 1:
            print(f"Plotting mass evolution for all {results.n_tanks} tanks (overlay mode)...")
        else:
            print(f"Plotting mass evolution for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Create single subplot for mass
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Get time array
        times_hours = results.times / 3600.0  # Convert to hours

        # Configure axis
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Mass [kg]')
        ax.set_title('Mass vs Time')
        ax.grid(True, alpha=0.3)

        # Plot tank data (single tank or multi-tank overlay)
        tanks_to_plot = range(results.n_tanks) if should_overlay else [tank_index]

        for i, tank_idx in enumerate(tanks_to_plot):
            # Extract tank data arrays
            tank_data = results._extract_tank_arrays(tank_idx)

            # Get color and line style
            color = self.color_palette[i % len(self.color_palette)]
            line_style = self.line_styles[i % len(self.line_styles)] if self.use_greyscale else '-'

            # Tank label
            tank_label = f"Tank {tank_idx + 1}" if should_overlay else "Mass"

            # Plot mass vs time
            ax.plot(times_hours, tank_data['masses'], color=color, linewidth=2, linestyle=line_style,
                   label=tank_label)

        # Only show legend if multiple tanks are overlaid
        if should_overlay and results.n_tanks > 1:
            legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, frameon=True, fancybox=True,
                              shadow=True, framealpha=0.9, edgecolor='black')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_linewidth(1.2)        # Improve layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"   Saved to: {save_path}")

        print(f"   Mass evolution plot completed")
        return fig

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
        print(f"Plotting sequential tank evolution for Tank {tank_index + 1}...")

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
            combined_temperatures.extend(tank_data['h2_temperatures'])
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
            ax1.legend(fontsize=plot_style.LEGEND_FONT_SIZE)

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
            ax2.legend(fontsize=plot_style.LEGEND_FONT_SIZE)

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
            ax4.legend(fontsize=plot_style.LEGEND_FONT_SIZE)

        # Add mission labels at the top of the plot
        for i, mission in enumerate(mission_labels):
            mid_time = (mission['start_time'] + mission['end_time']) / 2
            ax1.text(mid_time, ax1.get_ylim()[1] * 0.95, mission['name'],
                    ha='center', va='top', fontsize=plot_style.LEGEND_FONT_SIZE, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Improve layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Sequential tank evolution plot completed")
        return fig
