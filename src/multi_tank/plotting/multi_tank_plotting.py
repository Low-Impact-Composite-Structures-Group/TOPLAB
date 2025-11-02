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
    apply_custom_ticks, format_axis_labels, add_legend, DELFT_PALETTE,
    FONT_SIZE, FONT_NAME
)
import plotting.plot_style_sb as plot_style

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
                font=FONT_NAME,
                palette="delft",  # Use delft for now, override colors below
                style="whitegrid",
                context="paper",
                figure_size=(12, 8),
                dpi=100
            )
            # Define greyscale color scheme with better contrast
            self.color_palette = ['#000000', '#404040', '#808080', '#A0A0A0', '#C0C0C0']
            self.line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for distinction
            print(f"🎨 DelftColourPlotter initialized for: {analysis_name} (Greyscale Mode)")
        else:
            configure_plot_style(
                font=FONT_NAME,
                palette="delft",
                style="whitegrid",
                context="paper",
                figure_size=(12, 8),
                dpi=100
            )
            self.color_palette = DELFT_PALETTE
            print(f"🎨 DelftColourPlotter initialized for: {analysis_name}")

    def plot_phase_colour_map(self,
                              temperature_range: Tuple[float, float] = (15.0, 60.0),
                              density_range: Tuple[float, float] = (0.0, 90.0),
                              resolution: Tuple[int, int] = (400, 400),
                              save_path: Optional[str] = None,
                              legend_location: str = 'best',
                              legend_ncols: int = 2,
                              dpi: int = 900,
                              marker_points: Optional[List[Dict[str, Any]]] = None,
                              isobar_pressures_bar: Optional[List[float]] = None,
                              critical_marker_size: float = 50.0,
                              default_marker_size: float = 80.0) -> plt.Figure:
        """
        Plot a hydrogen density–temperature phase map (no trajectory), using a discrete
        colour map for regions and overlaying saturation lines.

        Regions shown:
        - Two-phase (within saturation dome)
        - Superheated vapor (gas) below critical temperature
        - Compressed/subcooled liquid below critical temperature
        - Supercritical (T >= T_crit)
        Plus overlays:
        - Saturated vapor line (Q=1)
        - Saturated liquid line (Q=0)
        - Critical point marker
        - Triple point marker (if available)

        Args:
            temperature_range: (T_min, T_max) in K
            density_range: (rho_min, rho_max) in kg/m³
            resolution: (nx, ny) grid resolution in T and rho
            save_path: optional output path for saving
            legend_location: legend location string

        Returns:
            matplotlib Figure object
        """
        print("🔵 Plotting hydrogen phase colour map…")

        # Late import to avoid hard dependency at module import-time
        try:
            from CoolProp.CoolProp import PropsSI
        except Exception as e:
            # Provide a graceful empty plot if CoolProp isn't available
            return self._create_empty_plot(f"CoolProp unavailable: {e}")

        T_min, T_max = temperature_range
        rho_min, rho_max = density_range
        nx, ny = resolution

        # Critical and triple points (with safe fallbacks)
        try:
            Tcrit = float(PropsSI("Tcrit", "hydrogen"))
            Pcrit = float(PropsSI("pcrit", "hydrogen"))
        except Exception:
            Tcrit, Pcrit = 33.0, 1.3e6

        try:
            Ttriple = float(PropsSI("Ttriple", "hydrogen"))
            Ptriple = float(PropsSI("ptriple", "hydrogen"))
        except Exception:
            Ttriple, Ptriple = 13.8, 7e3  # approximate values

        # Prepare grid
        T_vals = np.linspace(T_min, T_max, nx)
        rho_vals = np.linspace(rho_min, rho_max, ny)

        # Precompute saturation densities as functions of T (for T < Tcrit)
        rho_l_sat = np.full_like(T_vals, np.nan, dtype=float)
        rho_g_sat = np.full_like(T_vals, np.nan, dtype=float)

        for i, T in enumerate(T_vals):
            if T < Tcrit:
                try:
                    rho_l_sat[i] = float(PropsSI("Dmass", "T", T, "Q", 0, "hydrogen"))
                    rho_g_sat[i] = float(PropsSI("Dmass", "T", T, "Q", 1, "hydrogen"))
                except Exception:
                    # leave as NaN if CoolProp can't provide at extreme edges
                    pass

        # Classify grid into discrete phase regions
        # 0: superheated vapor, 1: two-phase, 2: compressed liquid, 3: supercritical
        phase_grid = np.zeros((ny, nx), dtype=int)

        for ix, T in enumerate(T_vals):
            if T >= Tcrit:
                # Supercritical region for all densities displayed
                phase_grid[:, ix] = 3
                continue

            # Below Tcrit, use saturation boundaries when available
            rl = rho_l_sat[ix]
            rg = rho_g_sat[ix]

            if np.isfinite(rl) and np.isfinite(rg) and rg < rl:
                # vapor for rho <= rg, two-phase between, compressed liquid for rho >= rl
                # Note: draw strict interior for dome to avoid visual aliasing on borders
                for iy, rho in enumerate(rho_vals):
                    if rho < rg:
                        phase_grid[iy, ix] = 0  # vapor
                    elif rg <= rho <= rl:
                        phase_grid[iy, ix] = 1  # two-phase
                    else:  # rho > rl
                        phase_grid[iy, ix] = 2  # compressed liquid
            else:
                # If saturation is not available (e.g., near bounds), fall back to single-phase guess
                phase_grid[:, ix] = 0  # assume vapor-like

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Discrete colours (match Delft or greyscale)
        if self.use_greyscale:
            region_colors = [
                '#1a1a1a',  # vapor
                '#7f7f7f',  # two-phase
                '#4c4c4c',  # compressed liquid
                '#bfbfbf',  # supercritical
            ]
        else:
            # Use Delft palette: choose distinct, readable colours
            region_colors = [
                DELFT_PALETTE[6],  # vapor - ROOD (red)
                DELFT_PALETTE[3],  # two-phase - PAARS (purple)
                DELFT_PALETTE[2],  # compressed liquid - KONINGSBLAUW (royal blue)
                DELFT_PALETTE[9],  # supercritical - BOSGROEN (green)
            ]

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(region_colors)

        # Region rendering: color for color mode, hatched patterns for greyscale
        if self.use_greyscale:
            # Build hatched fills per category using contourf on the categorical grid
            Tm, Rhom = np.meshgrid(T_vals, rho_vals)
            levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
            # Order: 0=vapor, 1=two-phase, 2=compressed liquid, 3=supercritical
            # Requested: vapour vertical '|', two-phase square grid '+', subcooled liquid horizontal '-', supercritical lattice dots '.'
            hatches = ['|||', '++', '---', '..']
            # Transparent faces, grey hatch lines
            cf = ax.contourf(Tm, Rhom, phase_grid, levels=levels,
                             colors=['none'] * 4, hatches=hatches, extend='neither')
            for coll in cf.collections:
                try:
                    coll.set_edgecolor('#808080')
                    coll.set_linewidth(0.7)
                    coll.set_facecolor((1, 1, 1, 0))
                except Exception:
                    pass
        else:
            # Show the categorical grid with Delft colors
            im = ax.imshow(
                phase_grid,
                origin='lower',
                aspect='auto',
                cmap=cmap,
                extent=[T_min, T_max, rho_min, rho_max],
                interpolation='nearest',
                alpha=0.6,
            )

        # Overlay saturation lines (only for T < Tcrit)
        valid_mask = np.isfinite(rho_l_sat) & np.isfinite(rho_g_sat) & (T_vals < Tcrit)
        if valid_mask.any():
            T_sat = T_vals[valid_mask].tolist()
            rho_l = rho_l_sat[valid_mask].tolist()
            rho_g = rho_g_sat[valid_mask].tolist()

            # Styling depends on mode
            if self.use_greyscale:
                liq_color = '#000000'
                vap_color = '#000000'
                liq_style = '--'
                vap_style = '-.'
            else:
                # Requested colours/styles: saturated liquid dashed blue, saturated vapor dashed red
                liq_color = DELFT_PALETTE[2]  # KONINGSBLAUW (royal blue)
                vap_color = DELFT_PALETTE[6]  # ROOD (red)
                liq_style = '--'
                vap_style = '--'

            # Extend saturation curves to critical point explicitly
            try:
                rho_crit = float(PropsSI("rhocrit", "hydrogen"))
            except Exception:
                rho_crit = None
            if rho_crit is not None and np.isfinite(rho_crit):
                if (not T_sat) or (T_sat[-1] < Tcrit):
                    T_sat.append(Tcrit)
                    rho_l.append(rho_crit)
                    rho_g.append(rho_crit)

            ax.plot(T_sat, rho_l, color=liq_color, linestyle=liq_style, linewidth=2.0, label='Saturated liquid line')
            ax.plot(T_sat, rho_g, color=vap_color, linestyle=vap_style, linewidth=2.0, label='Saturated vapour line')

        # Add isobar lines at selected pressures (bar)
        # Use provided list if given, otherwise a sensible default
        isobars_list = isobar_pressures_bar if isobar_pressures_bar is not None else [15, 100, 200, 400, 600, 700]
        if isobars_list:
            T_for_isobars = np.linspace(T_min, T_max, max(200, nx))
            for pbar in isobars_list:
                ppa = pbar * 1e5
                Ts_valid = []
                rhos_valid = []
                for T in T_for_isobars:
                    try:
                        rho = float(PropsSI("Dmass", "T", T, "P", ppa, "hydrogen"))
                        if np.isfinite(rho) and (rho_min <= rho <= rho_max):
                            Ts_valid.append(T)
                            rhos_valid.append(rho)
                    except Exception:
                        continue
                if len(Ts_valid) > 1:
                    # Requested: isobars in black, slightly thicker (solid in greyscale, dotted in color)
                    iso_color = 'black'
                    iso_style = '-' if self.use_greyscale else ':'
                    ax.plot(Ts_valid, rhos_valid, color=iso_color, linestyle=iso_style, linewidth=1.8,
                            alpha=0.9)

                    # Add inline label with a small textbox, rotated along local slope
                    mid_idx = int(0.65 * (len(Ts_valid) - 1))
                    mid_idx = max(1, min(mid_idx, len(Ts_valid) - 2))
                    x0, y0 = Ts_valid[mid_idx], rhos_valid[mid_idx]
                    dx = Ts_valid[mid_idx + 1] - Ts_valid[mid_idx - 1]
                    dy = rhos_valid[mid_idx + 1] - rhos_valid[mid_idx - 1]
                    angle = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 else 90.0
                    txt = ax.text(
                        x0, y0,
                        f"{pbar:g} bar",
                        color='black',
                        fontsize=max(plot_style.LEGEND_FONT_SIZE - 2, 10),
                        rotation=angle,
                        rotation_mode='anchor',
                        ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='black',
                                  linewidth=0.6, alpha=1.0),  # opaque white background
                        zorder=6,
                    )
                    try:
                        # Ensure the textbox rotates with the text (aligns with isobar)
                        txt.set_transform_rotates_text(True)
                    except Exception:
                        pass

        # Mark critical point
        if T_min <= Tcrit <= T_max:
            try:
                rho_crit = float(PropsSI("rhocrit", "hydrogen"))
            except Exception:
                rho_crit = 31.0  # approx kg/m³
            if rho_min <= rho_crit <= rho_max:
                ax.scatter([Tcrit], [rho_crit], color='black', s=critical_marker_size, zorder=5, label='Critical point')

        # Mark triple point (optional)
        if T_min <= Ttriple <= T_max:
            try:
                rho_l_tp = float(PropsSI("Dmass", "T", Ttriple, "Q", 0, "hydrogen"))
                rho_g_tp = float(PropsSI("Dmass", "T", Ttriple, "Q", 1, "hydrogen"))
                # Show as two markers at the same T
                markers = []
                if rho_min <= rho_l_tp <= rho_max:
                    markers.append((Ttriple, rho_l_tp))
                if rho_min <= rho_g_tp <= rho_max:
                    markers.append((Ttriple, rho_g_tp))
                if markers:
                    ax.scatter([m[0] for m in markers], [m[1] for m in markers],
                               color='black', marker='x', s=40, zorder=5, label='Triple point (sat)')
            except Exception:
                pass

        # Custom marker points (e.g., CcH2, sLH2, CH2, LH2) if provided
        if marker_points:
            for mp in marker_points:
                try:
                    Tm = float(mp.get('T'))
                    rhom = float(mp.get('rho'))
                except Exception:
                    continue
                # Only plot if within current axis limits
                if not (T_min <= Tm <= T_max and rho_min <= rhom <= rho_max):
                    # Still plot; axis limits may be expanded later
                    pass
                label = mp.get('label', None)
                marker = mp.get('marker', 'o')
                size = float(mp.get('size', default_marker_size))
                # Use black markers by request
                if marker in ['x', '+']:
                    ax.scatter([Tm], [rhom], marker=marker, color='black', s=size, zorder=7, label=label)
                else:
                    ax.scatter([Tm], [rhom], marker=marker, facecolors='black', edgecolors='black',
                               s=size, zorder=7, label=label)

        # Labels and grid
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Density [kg/m³]')
        # ax.set_title('Hydrogen Phase Map (Density–Temperature)')
        # In greyscale, disable the grid to keep hatch patterns crisp
        if self.use_greyscale:
            ax.grid(False)
        else:
            ax.grid(True, alpha=0.3)

        # Build a discrete legend for regions using proxies
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        region_labels = ['Superheated vapour', 'Two-phase', 'Subcooled liquid', 'Supercritical']
        if self.use_greyscale:
            # Legend proxies that display the same hatch pattern per region
            hatch_patterns = ['|||', '++', '---', '..']
            proxies = [
                mpatches.Patch(
                    facecolor='white', edgecolor='#808080', hatch=hatch_patterns[i],
                    linewidth=0.8, label=region_labels[i]
                ) for i in range(4)
            ]
        else:
            proxies = [mpatches.Patch(color=region_colors[i], label=region_labels[i]) for i in range(len(region_colors))]

        # Add a single proxy line for isobars to the legend (solid in greyscale, dotted in color)
        isobar_proxy = Line2D(
            [0], [0], color='black', linestyle='-' if self.use_greyscale else ':',
            linewidth=1.8, label='Isobars'
        )

        # Collect existing line handles (sat lines, points)
        handles, labels = ax.get_legend_handles_labels()
        handles = proxies + [isobar_proxy] + handles
        labels = region_labels + ['Isobars'] + labels

        legend = ax.legend(
            handles,
            labels,
            fontsize=plot_style.LEGEND_FONT_SIZE,
            loc=legend_location,
            ncol=max(1, int(legend_ncols)),
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            edgecolor='black',
            columnspacing=0.8,
            handletextpad=0.6,
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        # Limits
        ax.set_xlim(temperature_range)
        ax.set_ylim(density_range)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print("   ✅ Phase colour map completed")
        return fig

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

        Returns:
            matplotlib Figure object (main evolution plot)
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
            ax_t.plot(times_hours, tank_data['temperatures'], color=primary_color, linewidth=2, linestyle=line_style,
                      label='Temperature')
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
                    print(f"   💾 Saved {metric} evolution to: {out_file}")

            print(f"   ✅ Separate tank evolution plots completed for Tank {tank_index + 1}")
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

            # Plot 2: Temperature vs Time
            ax2.plot(times_hours, tank_data['temperatures'], color=color, linewidth=2, linestyle=line_style,
                    label=f'{tank_label} Temperature' if should_overlay else 'Temperature')

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

                legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=loc, frameon=True, fancybox=True,
                                 shadow=True, framealpha=0.9, edgecolor='black')
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
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Tank evolution plot completed")
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
            print(f"🔵 Plotting mass evolution for all {results.n_tanks} tanks (overlay mode)...")
        else:
            print(f"🔵 Plotting mass evolution for Tank {tank_index + 1}...")

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
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Mass evolution plot completed")
        return fig

    def _extract_valve_events(self, results: MultiTankResults) -> List[Dict[str, Any]]:
        """
        Extract valve opening/closing events from simulation results.

        This method analyzes coupling flow data to identify when valves open and close
        for multi-tank systems. Returns empty list for single-tank systems.

        Args:
            results: MultiTankResults containing simulation data

        Returns:
            List of valve event dictionaries with 'time' and 'type' keys
        """
        valve_events = []

        # Only extract valve events for multi-tank systems
        if results.n_tanks < 2:
            return valve_events

        try:
            # Look for coupling flow changes to identify valve events
            for tank_idx in range(results.n_tanks):
                tank_data = results._extract_tank_arrays(tank_idx)
                coupling_inflows = tank_data.get('coupling_inflow_rates', [])
                coupling_outflows = tank_data.get('coupling_outflow_rates', [])

                if len(coupling_inflows) == 0 and len(coupling_outflows) == 0:
                    continue

                # Combine flows to detect valve activity
                total_coupling_flow = []
                for i in range(len(coupling_inflows)):
                    inflow = coupling_inflows[i] if i < len(coupling_inflows) else 0
                    outflow = coupling_outflows[i] if i < len(coupling_outflows) else 0
                    total_coupling_flow.append(abs(inflow) + abs(outflow))

                # Detect valve opening/closing events
                valve_active = False
                # Note: flows in tank_data are in g/s (see MultiTankResults._extract_tank_arrays)
                # Set a sensible threshold in g/s to avoid false positives from tiny numerical noise
                flow_threshold = 1.0  # g/s threshold for valve activity

                for i, flow in enumerate(total_coupling_flow):
                    current_active = flow > flow_threshold

                    if current_active != valve_active and i > 0:
                        event_time = results.times[i]
                        event_type = 'open' if current_active else 'close'

                        valve_events.append({
                            'time': event_time,
                            'type': event_type,
                            'tank': tank_idx
                        })

                    valve_active = current_active

        except Exception as e:
            print(f"   ⚠️  Could not extract valve events: {e}")

        return valve_events

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

        # Auto-compute temperature range (xlim) based on actual data if not provided
        if xlim is None:
            temp_min = min(tank_data['temperatures']) - 4
            temp_max = max(tank_data['temperatures']) + 15
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
        tank_line, = ax.plot(tank_data['temperatures'], tank_data['densities'],
                            '-', color=primary_color, linewidth=2,
                            label=f"Thermodynamic path")

        # Add configurable direction arrow
        if len(tank_data['temperatures']) > 10:
            # Calculate arrow position along path
            idx = max(1, int(len(tank_data['temperatures']) * arrow_position))
            start_idx = max(0, idx - max(1, int(8 * arrow_size)))

            ax.annotate('',
                xy=(tank_data['temperatures'][idx], tank_data['densities'][idx]),
                xytext=(tank_data['temperatures'][start_idx], tank_data['densities'][start_idx]),
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

                if closest_idx < len(tank_data['temperatures']):
                    temp = tank_data['temperatures'][closest_idx]
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
                print(f"   ⚠️  Could not add saturation line: {e}")

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
                print(f"   ⚠️  Could not add isobars: {e}")

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
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Density-temperature plot completed")
        return fig

    def plot_mass_flows(self,
                       results: MultiTankResults,
                       tank_index: int = 0,
                       include_venting_flow: bool = True,
                       include_coupling_flows: bool = True,
                       save_path: Optional[str] = None,
                       xlim: Optional[Tuple[float, float]] = None,
                       ylim: Optional[Tuple[float, float]] = None,
                       legend_location: str = 'best') -> plt.Figure:
        """
        Plot mass flow rates for a single tank using the proven working pattern from multi_tank_analysis.

        Args:
            results: MultiTankResults containing simulation data
            tank_index: Index of tank to plot (0 for first tank)
            include_venting_flow: Whether to show venting flow curve
            include_coupling_flows: Whether to show coupling flows (for multi-tank systems)
            save_path: Optional path to save the plot
            xlim: Optional tuple (xmin, xmax) for x-axis range [hours]
            ylim: Optional tuple (ymin, ymax) for y-axis range [g/s]
            legend_location: Location string for legend (e.g., 'best', 'upper left')

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

        # Get flow data - coupling flows are incorporated into inflow/outflow
        # For multi-tank systems:
        # - Tank 1 (source): coupling appears as additional outflow
        # - Tank 2 (target): coupling appears as additional inflow
        # This correctly represents the mass transfer between tanks

        coupling_inflow_rates = tank_data['coupling_inflow_rates']
        coupling_outflow_rates = tank_data['coupling_outflow_rates']

        # Total inflow = mission inflow + coupling inflow
        total_inflow = []
        for i in range(len(tank_data['inflow_rates'])):
            mission_inflow = tank_data['inflow_rates'][i] if i < len(tank_data['inflow_rates']) else 0
            coupling_inflow = coupling_inflow_rates[i] if i < len(coupling_inflow_rates) else 0
            total_inflow.append(mission_inflow + coupling_inflow)

        # Total outflow = mission outflow + coupling outflow (make negative for display)
        total_outflow = []
        for i in range(len(tank_data['outflow_rates'])):
            mission_outflow = tank_data['outflow_rates'][i] if i < len(tank_data['outflow_rates']) else 0
            coupling_outflow = coupling_outflow_rates[i] if i < len(coupling_outflow_rates) else 0
            total_outflow.append(-(mission_outflow + coupling_outflow))  # Negative for display

        vent = -tank_data['vent_rates']  # Make negative

        # Use consistent color palette approach to match other plots
        inflow_color = self.color_palette[0 % len(self.color_palette)]
        outflow_color = self.color_palette[1 % len(self.color_palette)]
        vent_color = self.color_palette[2 % len(self.color_palette)]

        # Plot with different line styles for greyscale distinction
        inflow_linestyle = '-'
        outflow_linestyle = '--' if self.use_greyscale else '-'
        vent_linestyle = '-.' if self.use_greyscale else '--'  # dash-dot pattern

        # Plot only inflow, outflow, and vent (coupling flows are incorporated into inflow/outflow)
        ax.plot(times_hours, total_inflow, color=inflow_color, linewidth=2,
                linestyle=inflow_linestyle, label='Inflow')
        ax.plot(times_hours, total_outflow, color=outflow_color, linewidth=2,
                linestyle=outflow_linestyle, label='Outflow')
        ax.plot(times_hours, vent, color=vent_color, linewidth=2,
                linestyle=vent_linestyle, label='Vent')

        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Set up plot formatting to match other plots
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Flow Rate [g/s]')
        # ax.set_title(f'Tank {tank_index + 1} Flow Rates')
        ax.grid(True, alpha=0.3)

        # Apply custom axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Add legend with 3D shadow effect (same styling as other plots)
        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect to match tank evolution plots
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Mass flow plot completed")
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
                print(f"   ⚠️  Could not calculate total requirements: {e}")

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
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Heat exchanger requirements plot completed")
        return fig



    def create_reference_lines_from_config(self, tank_config: Dict[str, Any], ref_line_config: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Create reference lines dictionary from tank configuration.

        Args:
            tank_config: Tank configuration dictionary with pressure/density limits
            ref_line_config: Reference line configuration controlling which lines to show

        Returns:
            Dictionary with reference line values for plotting
        """
        reference_lines = {}

        # Use empty dict as default if no config provided
        if ref_line_config is None:
            ref_line_config = {}

        # Pressure references (convert Pa to bar)
        # Handle both numeric and string values with scientific notation
        pressure_config = ref_line_config.get('pressure', {})

        if 'minimum_pressure' in tank_config and pressure_config.get('show_p_min', True):
            min_pressure = tank_config['minimum_pressure']
            if isinstance(min_pressure, str):
                min_pressure = float(min_pressure)
            reference_lines['P_min'] = min_pressure / 1e5

        if 'venting_pressure' in tank_config and pressure_config.get('show_p_vent', True):
            vent_pressure = tank_config['venting_pressure']
            if isinstance(vent_pressure, str):
                vent_pressure = float(vent_pressure)
            reference_lines['P_vent'] = vent_pressure / 1e5

        # Density references
        density_config = ref_line_config.get('density', {})

        if 'minimum_density' in tank_config and density_config.get('show_stopping_density', True):
            min_density = tank_config['minimum_density']
            if isinstance(min_density, str):
                min_density = float(min_density)
            reference_lines['rho_stop'] = min_density

        # Temperature references
        temperature_config = ref_line_config.get('temperature', {})

        if 'ambient_temperature' in tank_config and temperature_config.get('show_t_ambient', True):
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
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
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
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
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

        # Plot flow rates (greyscale or color) - no markers
        primary_color = self.color_palette[0]  # Black for greyscale, Delft blue for color
        ax.plot(combined_times, combined_flows, color=primary_color, linewidth=2.5,
                linestyle='-', label='Mass Flow Rate')

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
                   ha='center', va='center', fontsize=plot_style.LEGEND_FONT_SIZE, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Mass Flow Rate [kg/s]')
        ax.set_title('Mass Flow Rate vs Time')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect
        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        # Additional styling for 3D effect
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Sequential mass flows plot completed")
        return fig

    def plot_pressure_requirements(self,
                                 orchestrator,
                                 tank_index: int = 1,  # LH2 tank is typically tank 2 (index 1)
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pressure requirements evolution for mission-adaptive pressure control.

        Shows the dynamic pressure requirements, activation thresholds, and actual tank pressure
        for mission-adaptive coupling systems.

        Args:
            orchestrator: System orchestrator containing coupling valve diagnostic data
            tank_index: Index of tank to analyze (default 1 for LH2 tank)
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"🔵 Plotting pressure requirements for Tank {tank_index + 1}...")

        # Check if we have a mission-adaptive coupling valve
        if not hasattr(orchestrator.tank_system, 'coupling_valves'):
            print("   ⚠️ No coupling valves found")
            return self._create_empty_plot("No coupling valves found")

        # Find the mission-adaptive valve
        adaptive_valve = None
        for valve in orchestrator.tank_system.coupling_valves:
            if hasattr(valve, 'get_diagnostic_data'):
                adaptive_valve = valve
                break

        if adaptive_valve is None:
            print("   ⚠️ No mission-adaptive valve found")
            return self._create_empty_plot("No mission-adaptive valve found")

        # Get diagnostic data from the valve
        try:
            diag_data = adaptive_valve.get_diagnostic_data()
        except AttributeError as e:
            print(f"   ⚠️ Cannot get diagnostic data: {e}")
            # Try to access data directly
            if hasattr(adaptive_valve, 'time_history') and hasattr(adaptive_valve, 'required_pressure_history'):
                diag_data = {
                    'time_history': adaptive_valve.time_history,
                    'required_pressure_history': adaptive_valve.required_pressure_history,
                    'activation_threshold_history': getattr(adaptive_valve, 'activation_threshold_history', [])
                }
            else:
                print("   ⚠️ No pressure history data available")
                return self._create_empty_plot("No pressure history data available")

        if not diag_data['time_history']:
            print("   ⚠️ No pressure history data available")
            return self._create_empty_plot("No pressure history data available")

        # Convert to numpy arrays for plotting
        times = np.array(diag_data['time_history']) / 3600.0  # Convert to hours
        required_pressures = np.array(diag_data['required_pressure_history']) / 1e5  # Convert to bar (no margin)
        activation_thresholds = np.array(diag_data.get('activation_threshold_history', [])) / 1e5  # Convert to bar (with margin)

        # Ensure diagnostic series are time-sorted to avoid visual artifacts and warnings
        if len(times) > 1:
            order = np.argsort(times)
            if not np.all(order == np.arange(len(times))):
                times = times[order]
                required_pressures = required_pressures[order]
                if len(activation_thresholds) == len(order):
                    activation_thresholds = activation_thresholds[order]
                print("   ℹ️ Sorted diagnostic pressure series by time for plotting")

            # Optionally drop exact-duplicate time samples (keep last)
            # This reduces over-plotting at t≈0 and prevents dense clumps from looking like lines
            unique_times, first_idx = np.unique(times, return_index=True)
            if len(unique_times) != len(times):
                # Keep the last occurrence for each time by reversing first
                rev_times = times[::-1]
                rev_req = required_pressures[::-1]
                if len(activation_thresholds) == len(times):
                    rev_act = activation_thresholds[::-1]
                else:
                    rev_act = None
                unique_rev_times, last_rev_idx = np.unique(rev_times, return_index=True)
                # Recover forward order of last occurrences
                keep_mask_rev = np.zeros_like(rev_times, dtype=bool)
                keep_mask_rev[last_rev_idx] = True
                keep_mask = keep_mask_rev[::-1]
                times = times[keep_mask]
                required_pressures = required_pressures[keep_mask]
                if rev_act is not None:
                    activation_thresholds = activation_thresholds[keep_mask]
                print(f"   ℹ️ Deduplicated diagnostic times: {len(unique_times)} unique of {len(order)} samples")

        # Get actual tank pressure from results for comparison
        combined_data = orchestrator.results.get_combined_data()
        tank_pressures = combined_data['pressures'][tank_index]  # Already in bar
        result_times = combined_data['times'] / 3600.0  # Result times in hours

        # Get minimum pressure from tank configuration
        minimum_pressure_bar = orchestrator.tank_system.config.tanks[tank_index].P_MIN / 1e5  # Convert Pa to bar

    # Create the plot with proper styling
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use greyscale colors if specified
        if self.use_greyscale:
            required_color = 'darkgrey'                 # Dark grey for required (no margin)
            activation_color = 'lightgrey'              # Light grey for activation (with margin)
            actual_color = 'black'                      # Black solid for actual
            min_color = 'darkgrey'                      # Dark grey dotted for minimum

            required_style = '--'                       # Dark grey dashed
            activation_style = '-'                      # Light grey solid
            actual_style = '-'                          # Black solid
            min_style = ':'                             # Dark grey dotted
        else:
            required_color = self.color_palette[3]      # Delft orange for required (no margin)
            activation_color = self.color_palette[1]    # Delft red for activation (with margin)
            actual_color = self.color_palette[2]        # Delft green for actual
            min_color = self.color_palette[4]           # Delft purple for minimum

            required_style = '--'
            activation_style = '-.'
            actual_style = '-'
            min_style = ':'

        # Plot pressure curves - avoid connecting discontinuous segments
        # Split data where there are gaps to prevent straight line connections

        def plot_with_gaps(ax, x_data, y_data, gap_threshold_hours=0.002, **plot_kwargs):
            """Plot data with gaps to avoid connecting discontinuous segments."""
            if len(x_data) <= 1:
                return ax.plot(x_data, y_data, **plot_kwargs)

            # Convert to numpy arrays for easier handling
            x_data = np.array(x_data)
            y_data = np.array(y_data)

            # Find gaps in time data AND pressure jumps
            segments = []
            current_segment_x = [x_data[0]]
            current_segment_y = [y_data[0]]

            for i in range(1, len(x_data)):
                time_gap = x_data[i] - x_data[i-1]
                pressure_jump = abs(y_data[i] - y_data[i-1])

                # Calculate adaptive pressure jump threshold based on data range
                pressure_range = np.ptp(y_data) if len(y_data) > 1 else 1.0
                pressure_threshold = max(0.2, pressure_range * 0.05)  # At least 0.2 bar or 5% of range

                # End segment if there's a significant time gap OR pressure jump
                if time_gap > gap_threshold_hours or pressure_jump > pressure_threshold:
                    # Only save segment if it has multiple points
                    if len(current_segment_x) > 1:
                        segments.append((current_segment_x.copy(), current_segment_y.copy()))
                    # Start new segment
                    current_segment_x = [x_data[i]]
                    current_segment_y = [y_data[i]]
                else:
                    # Continue current segment
                    current_segment_x.append(x_data[i])
                    current_segment_y.append(y_data[i])

            # Add final segment if it has multiple points
            if len(current_segment_x) > 1:
                segments.append((current_segment_x, current_segment_y))

            # Plot each segment separately
            lines = []
            for i, (seg_x, seg_y) in enumerate(segments):
                if len(seg_x) > 1:  # Only plot segments with multiple points
                    # Only show label on first segment
                    label = plot_kwargs.get('label') if i == 0 else None
                    plot_kwargs_copy = plot_kwargs.copy()
                    plot_kwargs_copy['label'] = label
                    line = ax.plot(seg_x, seg_y, **plot_kwargs_copy)
                    lines.extend(line)

            return lines

    # Plot pressure requirements - identify continuous segments to avoid connecting discontinuous data
        # The valve switches between different operational states, creating disconnected segments

        def plot_continuous_segments(ax, times, pressures, min_threshold=2.1, **plot_kwargs):
            """Plot only continuous segments where valve is active, avoiding connecting ends."""
            if len(times) <= 1 or len(pressures) <= 1:
                return

            # Find where valve is active (pressure above minimum threshold)
            active_mask = pressures > min_threshold

            if not np.any(active_mask):
                return

            # Find continuous segments by detecting breaks in the active mask
            diff_mask = np.diff(active_mask.astype(int))
            start_indices = np.where(diff_mask == 1)[0] + 1  # Start of active segments
            end_indices = np.where(diff_mask == -1)[0] + 1   # End of active segments

            # Handle edge cases
            if active_mask[0]:  # Starts active
                start_indices = np.concatenate(([0], start_indices))
            if active_mask[-1]:  # Ends active
                end_indices = np.concatenate((end_indices, [len(active_mask)]))

            # Plot each continuous segment separately
            for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
                if end_idx > start_idx + 1:  # Need at least 2 points to draw a line
                    segment_times = times[start_idx:end_idx]
                    segment_pressures = pressures[start_idx:end_idx]

                    # Only show label on first segment
                    label = plot_kwargs.get('label') if i == 0 else None
                    plot_kwargs_copy = plot_kwargs.copy()
                    plot_kwargs_copy['label'] = label
                    ax.plot(segment_times, segment_pressures, **plot_kwargs_copy)

        # Debug: Check for time ordering issues (post-sort this should be zero)
        if len(times) > 1:
            time_diffs = np.diff(times)
            negative_diffs = np.sum(time_diffs < 0)
            if negative_diffs > 0:
                print(f"   ⚠️ Found {negative_diffs} negative time differences in pressure data")

        # Plot required pressure (no margin) - shows minimum pressure needed for mission discharge
        if len(required_pressures) > 0:
            ax.plot(times, required_pressures,
                    color=required_color, linestyle=required_style, linewidth=1.5, alpha=0.8,
                    label='Required pressure (mission discharge)')

        # Plot activation threshold (required + margin) - conservative target with safety buffer
        if len(activation_thresholds) > 0:
            ax.plot(times, activation_thresholds,
                    color=activation_color, linestyle=activation_style, linewidth=1.5, alpha=0.9,
                    label='Target pressure (required + margin)')

        # Plot actual tank pressure (continuous data, no filtering needed)
        ax.plot(result_times, tank_pressures,
               color=actual_color, linestyle=actual_style, linewidth=2.0, alpha=0.8,
               label=f'Actual tank {tank_index + 1} pressure')

        # =========================
        # Quantitative tracking audit
        # =========================
        try:
            # Interpolate required pressures to the results timeline (hours) without extrapolation
            if len(times) > 1 and len(result_times) > 1 and len(required_pressures) == len(times):
                # Only evaluate within the diagnostic time bounds
                within = (result_times >= np.nanmin(times)) & (result_times <= np.nanmax(times))
                required_interp = np.full_like(result_times, np.nan, dtype=float)

                # Use numpy.interp for the overlapping window; avoid extrapolation by slicing
                if np.any(within):
                    required_interp[within] = np.interp(result_times[within], times, required_pressures)

                # Active segments only (use all finite values)
                active_mask = np.isfinite(required_interp)

                if np.any(active_mask):
                    delta = tank_pressures[active_mask] - required_interp[active_mask]

                    # Stats over active segments
                    min_delta = float(np.nanmin(delta))
                    mean_delta = float(np.nanmean(delta))
                    frac_below = float(np.mean(delta < 0.0))

                    # Early-time audit (0–0.1 h)
                    early_mask = active_mask & (result_times <= 0.1)
                    if np.any(early_mask):
                        early_delta = tank_pressures[early_mask] - required_interp[early_mask]
                        early_min = float(np.nanmin(early_delta))
                        early_frac_below = float(np.mean(early_delta < 0.0))
                    else:
                        early_min = float('nan')
                        early_frac_below = float('nan')

                    print("   📈 Tracking audit (active segments):")
                    print(f"      min(P_actual - P_required) = {min_delta:.3f} bar, mean = {mean_delta:.3f} bar, fraction below = {frac_below:.3%}")
                    if np.isfinite(early_min):
                        print(f"      Early [0–0.1 h]: min Δ = {early_min:.3f} bar, fraction below = {early_frac_below:.3%}")

                    # CSV export intentionally disabled to keep output clean
                else:
                    print("   ℹ️ Tracking audit skipped: no active segments detected in required pressure.")
            else:
                print("   ℹ️ Tracking audit skipped: insufficient diagnostic or results data.")
        except Exception as e:
            print(f"   ⚠️ Tracking audit failed: {e}")

        # Add reference line for minimum tank pressure from configuration
        ax.axhline(y=minimum_pressure_bar, color=min_color, linestyle=min_style, alpha=0.7,
                  label='Absolute minimum tank pressure')

        # Formatting with consistent style
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Pressure [bar]')
        ax.grid(True, alpha=0.3)

        # Add legend with 3D shadow effect (same styling as other plots)
        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc='best', frameon=True,
                          fancybox=True, shadow=True, framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        # Set reasonable y-limits
        if len(required_pressures) > 0 and len(tank_pressures) > 0:
            y_min = min(min(required_pressures), min(tank_pressures), 1.5) - 0.5
            if len(activation_thresholds) > 0:
                y_max = max(max(activation_thresholds), max(tank_pressures)) + 0.5
            else:
                y_max = max(max(required_pressures), max(tank_pressures)) + 0.5
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")

        print(f"   ✅ Pressure requirements plot completed")
        return fig

    def _create_empty_plot(self, message: str) -> plt.Figure:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center',
               fontsize=plot_style.LEGEND_FONT_SIZE)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Pressure [bar]')
        return fig

    def plot_atr72_mass_flow(self,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ATR72 mission mass flow vs time in greyscale with mission segment labels.

        Shows positive fuel flow with labeled segments based on the mission profile.
        No legend since there's only one curve. Matches reference image styling.

        Args:
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print("🔵 Plotting ATR72 mission mass flow...")

        # Import mission here to avoid circular imports
        from src.mission.mission import Mission

        # Get ATR72 mission
        atr72_mission = Mission.atr72()

        # Extract time and flow data
        times = []
        flows = []
        segment_boundaries = []
        segment_labels = []

        current_time = 0.0

        # Create meaningful labels from the numerical fuel_flow_keys
        label_mapping = {
            'one': 'Pre-taxi',
            'two': 'Taxi Out',
            'three': 'Take Off',
            'four': 'Cruise',
            'five': 'Initial Descent',
            'six': 'Approach',
            'seven': 'Landing',
            'eight': 'Taxi In',
            'nine': 'Hold',
            'ten': 'Final Approach',
            'eleven': 'Ground'
        }

        for i, section in enumerate(atr72_mission.sections):
            # Get section duration in hours
            duration_hours = section.duration / 3600.0

            # Get fuel flow - make positive since we want to show consumption as positive
            if section.fuel_flows:
                fuel_flow = section.fuel_flows[0]
                if isinstance(fuel_flow.mass_flow, list) and len(fuel_flow.mass_flow) >= 2:
                    # Linear interpolation for variable flow segments
                    start_rate = abs(fuel_flow.mass_flow[0])
                    end_rate = abs(fuel_flow.mass_flow[-1])

                    # Create interpolated points for smooth visualization
                    num_points = max(10, int(duration_hours * 3600 / 60))  # At least 10 points or 1 per minute
                    time_points = np.linspace(0, duration_hours, num_points)

                    for j, dt in enumerate(time_points):
                        # Linear interpolation: start + (end - start) * (t / duration)
                        if duration_hours > 0:
                            interpolation_factor = dt / duration_hours
                        else:
                            interpolation_factor = 0
                        flow_rate = start_rate + (end_rate - start_rate) * interpolation_factor
                        flow_rate_gs = flow_rate * 1000.0  # Convert to g/s

                        times.append(current_time + dt)
                        flows.append(flow_rate_gs)

                elif isinstance(fuel_flow.mass_flow, list) and len(fuel_flow.mass_flow) == 1:
                    # Single value in list - treat as constant
                    flow_rate = abs(fuel_flow.mass_flow[0])
                    flow_rate_gs = flow_rate * 1000.0

                    # Add start and end points for step function
                    times.append(current_time)
                    flows.append(flow_rate_gs)
                    times.append(current_time + duration_hours)
                    flows.append(flow_rate_gs)
                else:
                    # Constant flow rate
                    flow_rate = abs(fuel_flow.mass_flow)
                    flow_rate_gs = flow_rate * 1000.0

                    # Add start and end points for step function
                    times.append(current_time)
                    flows.append(flow_rate_gs)
                    times.append(current_time + duration_hours)
                    flows.append(flow_rate_gs)
            else:
                # No fuel flow - add zero flow points
                times.append(current_time)
                flows.append(0.0)
                times.append(current_time + duration_hours)
                flows.append(0.0)

            # Calculate end time for this section
            end_time = current_time + duration_hours

            # Store segment info with meaningful labels
            segment_key = section.fuel_flow_key or f'segment_{i+1}'
            meaningful_label = label_mapping.get(segment_key, segment_key.title())

            # For variable flow segments, get representative flow for labeling
            if section.fuel_flows:
                fuel_flow = section.fuel_flows[0]
                if isinstance(fuel_flow.mass_flow, list) and len(fuel_flow.mass_flow) >= 2:
                    # Use average flow for labeling
                    start_rate = abs(fuel_flow.mass_flow[0])
                    end_rate = abs(fuel_flow.mass_flow[-1])
                    representative_flow_gs = ((start_rate + end_rate) / 2) * 1000.0
                else:
                    representative_flow_gs = abs(fuel_flow.mass_flow) * 1000.0 if isinstance(fuel_flow.mass_flow, (int, float)) else abs(fuel_flow.mass_flow[0]) * 1000.0
            else:
                representative_flow_gs = 0.0

            # Only add label if segment is significant (> 0.005 hours or has flow > 1 g/s)
            if duration_hours > 0.005 or representative_flow_gs > 1.0:
                segment_boundaries.append(current_time)
                segment_labels.append({
                    'name': meaningful_label,
                    'start_time': current_time,
                    'end_time': end_time,
                    'mid_time': current_time + duration_hours / 2,
                    'duration': duration_hours,
                    'flow': representative_flow_gs
                })

            current_time = end_time

        # Create figure with greyscale styling
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot mass flow - single black line with clean styling
        ax.plot(times, flows, color='black', linewidth=2.5, linestyle='-')        # Set up plot formatting to match reference image
        ax.set_xlabel('Time [h]', fontsize=FONT_SIZE)
        ax.set_ylabel('Mass Flow [g/s]', fontsize=FONT_SIZE)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Set y-limits to show data clearly
        if flows:
            y_max = max(flows) * 1.05  # Minimal headroom for clean appearance
            ax.set_ylim(0, y_max)

        # Set x-limits with small margins
        if times:
            x_max = max(times)
            ax.set_xlim(-0.01, x_max * 1.02)

        # Apply tight layout
        plt.tight_layout()

        # Save figure - either to provided path or default location
        if save_path:
            fig.savefig(save_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {save_path}")
        else:
            # Save in same directory as the calling script
            import inspect
            import os

            # Get the directory of the calling script
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back
                while caller_frame:
                    filename = caller_frame.f_code.co_filename
                    if not filename.endswith('multi_tank_plotting.py'):
                        script_dir = os.path.dirname(os.path.abspath(filename))
                        break
                    caller_frame = caller_frame.f_back
                else:
                    # Fallback to current working directory
                    script_dir = os.getcwd()
            finally:
                del frame

            default_path = os.path.join(script_dir, 'atr72_mass_flow.png')
            fig.savefig(default_path, dpi=900, bbox_inches='tight', facecolor='white')
            print(f"   💾 Saved to: {default_path}")

        print("   ✅ ATR72 mass flow plot completed")
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