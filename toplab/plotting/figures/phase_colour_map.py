"""PhaseColourMapFigures: figure methods of the Delft plotter (one concern per module)."""


from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import seaborn styling
from toplab.plotting.plot_style_sb import (
    DELFT_PALETTE
)
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.fluids.fluid import DEFAULT_COOLPROP_FLUID



class PhaseColourMapFigures:
    def plot_phase_colour_map(self,
                              fluid: str = DEFAULT_COOLPROP_FLUID,
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
                              default_marker_size: float = 80.0,
                              isobar_label_font_size: Optional[float] = None,
                              callout_font_size: Optional[float] = None) -> plt.Figure:
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
        print("Plotting hydrogen phase colour map...")

        resolved_isobar_label_font_size = (
            float(isobar_label_font_size)
            if isobar_label_font_size is not None
            else max(plot_style.LEGEND_FONT_SIZE - 2, 10)
        )
        resolved_callout_font_size = (
            float(callout_font_size)
            if callout_font_size is not None
            else max(plot_style.LEGEND_FONT_SIZE - 1, 10)
        )

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
            Tcrit = float(PropsSI("Tcrit", fluid))
            Pcrit = float(PropsSI("pcrit", fluid))
        except Exception:
            Tcrit, Pcrit = 33.0, 1.3e6

        try:
            Ttriple = float(PropsSI("Ttriple", fluid))
            Ptriple = float(PropsSI("ptriple", fluid))
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
                    rho_l_sat[i] = float(PropsSI("Dmass", "T", T, "Q", 0, fluid))
                    rho_g_sat[i] = float(PropsSI("Dmass", "T", T, "Q", 1, fluid))
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
            # Show the categorical grid with Delft colors using pcolormesh
            # Rasterize the background to keep PGF file size manageable
            Tm, Rhom = np.meshgrid(T_vals, rho_vals)
            im = ax.pcolormesh(
                Tm,
                Rhom,
                phase_grid,
                cmap=cmap,
                shading='nearest',
                alpha=0.6,
                rasterized=True,  # Rasterize background, keep lines/text as vectors
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
                rho_crit = float(PropsSI("rhocrit", fluid))
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
                        rho = float(PropsSI("Dmass", "T", T, "P", ppa, fluid))
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
                        fontsize=resolved_isobar_label_font_size,
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
                rho_crit = float(PropsSI("rhocrit", fluid))
            except Exception:
                rho_crit = 31.0  # approx kg/m³
            if rho_min <= rho_crit <= rho_max:
                ax.scatter([Tcrit], [rho_crit], color='black', s=critical_marker_size, zorder=5, label='Critical point')

        # Mark triple point (optional)
        if T_min <= Ttriple <= T_max:
            try:
                rho_l_tp = float(PropsSI("Dmass", "T", Ttriple, "Q", 0, fluid))
                rho_g_tp = float(PropsSI("Dmass", "T", Ttriple, "Q", 1, fluid))
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
                include_in_legend = bool(mp.get('include_in_legend', False))
                label = mp.get('label', None) if include_in_legend else None
                marker = mp.get('marker', 'o')
                size = float(mp.get('size', default_marker_size))
                # Use black markers by request
                if marker in ['x', '+']:
                    ax.scatter([Tm], [rhom], marker=marker, color='black', s=size, zorder=7, label=label)
                else:
                    ax.scatter([Tm], [rhom], marker=marker, facecolors='black', edgecolors='black',
                               s=size, zorder=7, label=label)

                callout_text = mp.get('callout')
                if callout_text:
                    callout_offset = mp.get('callout_offset', (10, 10))
                    if not isinstance(callout_offset, (list, tuple)) or len(callout_offset) != 2:
                        callout_offset = (10, 10)
                    callout_ha = mp.get('callout_ha', 'left')
                    callout_va = mp.get('callout_va', 'center')
                    ax.annotate(
                        callout_text,
                        xy=(Tm, rhom),
                        xytext=(float(callout_offset[0]), float(callout_offset[1])),
                        textcoords='offset points',
                        ha=callout_ha,
                        va=callout_va,
                        fontsize=resolved_callout_font_size,
                        color='black',
                        bbox=dict(
                            boxstyle='round,pad=0.25',
                            facecolor='white',
                            edgecolor='black',
                            linewidth=0.8,
                            alpha=0.98,
                        ),
                        arrowprops=dict(
                            arrowstyle='-',
                            color='black',
                            linewidth=0.8,
                            shrinkA=4,
                            shrinkB=4,
                        ),
                        zorder=8,
                    )

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

        region_labels = ['Superheated vapour', 'Two-phase', 'Compressed liquid', 'Supercritical']
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
            self._save_figure(fig, save_path, dpi=dpi)

        print("   Phase colour map completed")
        return fig
