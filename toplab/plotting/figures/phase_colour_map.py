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

# use normal hydrogen for phase map plotting, as it is the most common and well-characterized form of hydrogen
DEFAULT_COOLPROP_FLUID = "Hydrogen"  # default fluid for phase map plotting

class PhaseColourMapFigures:
    def plot_phase_colour_map(self,
                              fluid: str = DEFAULT_COOLPROP_FLUID,
                              temperature_range: Tuple[float, float] = (15.0, 60.0),
                              density_range: Tuple[float, float] = (0.0, 90.0),
                              resolution: Tuple[int, int] = (400, 400),
                              save_path: Optional[str] = None,
                              legend_location: str = 'best',
                              legend_ncols: int = 2,
                              dpi: int = 1000,
                              marker_points: Optional[List[Dict[str, Any]]] = None,
                              isobar_pressures_bar: Optional[List[float]] = None,
                              critical_marker_size: float = 50.0,
                              default_marker_size: float = 80.0,
                              isobar_label_font_size: Optional[float] = None,
                              callout_font_size: Optional[float] = None,
                              cryocompressed_envelope: Optional[Tuple[float, float, float, float]] = None,
                              max_display_pressure_bar: float = 800.0,
                              cch2_temperature_ceiling_K: float = 233.15,
                              show_cch2_region: bool = True,
                              show_fluid_model_warning: bool = False) -> plt.Figure:
        """
        Plot a hydrogen density–temperature phase map (no trajectory), using a discrete
        colour map for regions and overlaying saturation lines.

        Regions shown:
        - Two-phase (within saturation dome)
        - Superheated vapor (gas) below critical temperature
        - Compressed/subcooled liquid below critical temperature
        - Supercritical fluid (T > T_crit and P > P_crit), up to pressure cap
        - Gas (T > T_crit and P < P_crit)
        Plus overlays:
        - Saturated vapor line (Q=1)
        - Saturated liquid line (Q=0)
        - Critical point marker
        - Triple point marker (if available)
        - White/unclassified area beyond max_display_pressure_bar
        - CcH2 engineering overlay from Tc to -40 C (configurable)

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
        if not (T_min < T_max and density_range[0] < density_range[1]):
            raise ValueError("Temperature and density ranges must be increasing")
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

        if T_min < Ttriple:
            raise ValueError(
                f"CoolProp fluid EOS does not model solid hydrogen: "
                f"T_min={T_min:g} K is below the triple temperature {Ttriple:g} K."
            )
        # Published normal-hydrogen melting pressures (bar), tabulated at
        # temperatures (K).  These are independent of CoolProp's melting_line(),
        # whose hydrogen results should not be used here without validation.
        # Source: NBS Monograph 168, Selected Properties of Hydrogen (1981),
        # melting-pressure table; selected points.  The table is for normal H2.
        # Only a short extension of the last tabulated segment (30 K, 733.5 bar)
        # to the 800 bar plotting cap is used; it is explicitly an extrapolation.
        melt_T_data = np.array([14., 15., 16., 18., 20., 22., 24., 26., 30.])
        melt_P_bar_data = np.array([1.70, 33.24, 66.47, 140.1, 221.8,
                                    310.9, 406.8, 509.4, 733.5])
        if fluid.lower() not in ('hydrogen', 'normalhydrogen', 'n-hydrogen'):
            raise ValueError('The melting-pressure data are for normal hydrogen only')
        # Linear interpolation is deliberately used: no overshoot between
        # published tabulated points.  A limited linear extrapolation above
        # 30 K is needed to meet the 800 bar pressure cap.
        last_slope = ((melt_P_bar_data[-1] - melt_P_bar_data[-2]) /
                      (melt_T_data[-1] - melt_T_data[-2]))
        melt_T_cap = melt_T_data[-1] + (max_display_pressure_bar - melt_P_bar_data[-1]) / last_slope
        if max_display_pressure_bar > 800.0:
            raise ValueError('Melting mask is only supported up to 800 bar')
        if max_display_pressure_bar < melt_P_bar_data[-1]:
            melt_T_cap = np.interp(max_display_pressure_bar, melt_P_bar_data, melt_T_data)

        def melting_pressure_bar(T):
            if T < melt_T_data[0] or T > melt_T_cap:
                return np.nan
            if T <= melt_T_data[-1]:
                return float(np.interp(T, melt_T_data, melt_P_bar_data))
            return float(melt_P_bar_data[-1] + last_slope * (T - melt_T_data[-1]))

        # Prepare grid
        T_vals = np.linspace(T_min, T_max, nx)
        rho_vals = np.linspace(rho_min, rho_max, ny)

        # Precompute saturation densities as functions of T (for T < Tcrit)
        rho_l_sat = np.full_like(T_vals, np.nan, dtype=float)
        rho_g_sat = np.full_like(T_vals, np.nan, dtype=float)

        for i, T in enumerate(T_vals):
            if Ttriple <= T < Tcrit:
                try:
                    rho_l_sat[i] = float(PropsSI("Dmass", "T", T, "Q", 0, fluid))
                    rho_g_sat[i] = float(PropsSI("Dmass", "T", T, "Q", 1, fluid))
                except Exception:
                    # leave as NaN if CoolProp can't provide at extreme edges
                    pass

        # At supercritical temperatures the critical isobar separates gas from
        # supercritical fluid. Compute its density once per temperature column;
        # no million-point pressure evaluation is necessary.
        rho_pcrit = np.full_like(T_vals, np.nan, dtype=float)
        for i, T in enumerate(T_vals):
            if T > Tcrit:
                try:
                    rho_pcrit[i] = float(PropsSI("Dmass", "T", float(T),
                                                "P", Pcrit, fluid))
                except (ValueError, RuntimeError):
                    pass

        # Upper plotting boundary: states beyond this isobar are not displayed.
        # This is an engineering pressure cut-off, NOT a melting curve.
        if max_display_pressure_bar <= Pcrit / 1e5:
            raise ValueError("max_display_pressure_bar must exceed the critical pressure")
        rho_pmax = np.full_like(T_vals, np.nan, dtype=float)
        for i, T in enumerate(T_vals):
            try:
                rho_pmax[i] = float(PropsSI("Dmass", "T", float(T),
                                             "P", max_display_pressure_bar * 1e5, fluid))
            except (ValueError, RuntimeError):
                # EOS cannot establish this boundary at this temperature.
                # Do not interpolate or extrapolate it silently.
                pass

        # Classify grid into discrete phase regions
        # 0: superheated vapor, 1: two-phase, 2: compressed liquid, 3: supercritical
        phase_grid = np.zeros((ny, nx), dtype=int)

        for ix, T in enumerate(T_vals):
            if T > Tcrit:
                boundary = rho_pcrit[ix]
                if not np.isfinite(boundary):
                    raise RuntimeError(
                        f"Cannot calculate critical-isobar density at T={T:g} K"
                    )
                # 0: gas below critical pressure; 3: supercritical fluid.
                phase_grid[:, ix] = np.where(rho_vals > boundary, 3, 0)
                continue
            if T == Tcrit:
                # The exact critical-temperature column has zero width in the
                # continuum; classify by critical density for plotting.
                rho_c = float(PropsSI("rhocrit", fluid))
                phase_grid[:, ix] = np.where(rho_vals > rho_c, 3, 0)
                continue

            # Below Tcrit, use saturation boundaries when available
            rl = rho_l_sat[ix]
            rg = rho_g_sat[ix]

            if np.isfinite(rl) and np.isfinite(rg) and rg < rl:
                # vapor for rho <= rg, two-phase between, compressed liquid for rho >= rl
                # Note: draw strict interior for dome to avoid visual aliasing on borders
                phase_grid[:, ix] = np.where(
                    rho_vals < rg, 0, np.where(rho_vals <= rl, 1, 2)
                )
            else:
                raise RuntimeError(
                    f"No valid saturation densities at T={T:g} K; "
                    "refusing to guess a phase region."
                )

        # Mask high-pressure states above the chosen engineering cut-off.
        # Do not confuse this white wedge with a verified solid phase.
        above_pmax = np.zeros_like(phase_grid, dtype=bool)
        for ix, boundary in enumerate(rho_pmax):
            if np.isfinite(boundary):
                above_pmax[:, ix] = rho_vals > boundary
        # Liquid-side solid--liquid exclusion.  At each temperature, the
        # highest stable liquid density is rho(T, P_melt(T)).  Mask higher
        # densities rather than extrapolating the fluid EOS into solid-stable
        # states.  This does not attempt to draw solid or coexistence densities.
        solid_excluded = np.zeros_like(phase_grid, dtype=bool)
        rho_melt_liquid = np.full_like(T_vals, np.nan, dtype=float)
        for ix, T in enumerate(T_vals):
            p_melt_bar = melting_pressure_bar(float(T))
            if not np.isfinite(p_melt_bar):
                continue
            try:
                rho_boundary = float(PropsSI('Dmass', 'T', float(T),
                                              'P', p_melt_bar * 1e5, fluid))
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f'Cannot calculate liquid-side melting density at {T:g} K'
                ) from exc
            if not np.isfinite(rho_boundary):
                raise RuntimeError(f'Nonfinite melting density at {T:g} K')
            rho_melt_liquid[ix] = rho_boundary
            # Only compressed-liquid states can be excluded by this mask.
            solid_excluded[:, ix] = ((phase_grid[:, ix] == 2) &
                                     (rho_vals > rho_boundary))
        phase_display = np.ma.array(phase_grid, mask=(above_pmax | solid_excluded))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Discrete colours (match Delft or greyscale)
        if self.use_greyscale:
            region_colors = [
                '#1a1a1a',  # gas / vapor
                '#7f7f7f',  # two-phase
                '#4c4c4c',  # compressed liquid
                '#bfbfbf',  # supercritical
            ]
        else:
            # Use Delft palette: choose distinct, readable colours
            region_colors = [
                DELFT_PALETTE[6],  # gas / vapor - ROOD (red)
                DELFT_PALETTE[3],  # two-phase - PAARS (purple)
                DELFT_PALETTE[2],  # compressed liquid - KONINGSBLAUW (royal blue)
                DELFT_PALETTE[9],  # supercritical - BOSGROEN (green)
            ]

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(region_colors)
        cmap.set_bad('white')

        # Region rendering: color for color mode, hatched patterns for greyscale
        if self.use_greyscale:
            # Build hatched fills per category using contourf on the categorical grid
            Tm, Rhom = np.meshgrid(T_vals, rho_vals)
            levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
            # Order: 0=vapor, 1=two-phase, 2=compressed liquid, 3=supercritical
            # Requested: vapour vertical '|', two-phase square grid '+', subcooled liquid horizontal '-', supercritical lattice dots '.'
            hatches = ['|||', '++', '---', '..']
            # Transparent faces, grey hatch lines
            cf = ax.contourf(Tm, Rhom, phase_display.filled(np.nan), levels=levels,
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
                phase_display,
                cmap=cmap,
                vmin=-0.5, vmax=3.5,
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

        # Critical-pressure isobar: actual gas/supercritical boundary above Tc.
        valid_pc = np.isfinite(rho_pcrit) & (T_vals > Tcrit)
        if np.any(valid_pc):
            T_pc = np.concatenate(([Tcrit], T_vals[valid_pc]))
            rho_pc = np.concatenate(([float(PropsSI("rhocrit", fluid))],
                                     rho_pcrit[valid_pc]))
            ax.plot(T_pc, rho_pc, color='black', linestyle='--',
                    linewidth=2.2, zorder=4, label='Critical isobar (12.96 bar)')

        # Add isobar lines at selected pressures (bar)
        # Use provided list if given, otherwise a sensible default
        isobars_list = isobar_pressures_bar if isobar_pressures_bar is not None else [15, 100, 200, 400, 600, 700]
        if isobars_list:
            T_for_isobars = np.linspace(max(T_min, Ttriple + 1e-4), T_max, max(200, nx))
            for pbar in isobars_list:
                ppa = pbar * 1e5
                # Already plotted explicitly as the critical boundary.
                if abs(ppa - Pcrit) / Pcrit < 1e-3:
                    continue
                Ts_valid = []
                rhos_valid = []
                for T in T_for_isobars:
                    try:
                        rho = float(PropsSI("Dmass", "T", T, "P", ppa, fluid))
                        p_melt_bar = melting_pressure_bar(float(T))
                        # Suppress the solid-stable segment of each isobar.
                        solid_stable = (np.isfinite(p_melt_bar) and
                                        pbar > p_melt_bar and T < Tcrit)
                        if (np.isfinite(rho) and rho_min <= rho <= rho_max
                                and not solid_stable and pbar <= max_display_pressure_bar):
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

        # Engineering CcH2 classification used by the reference figure:
        # Tcrit < T <= -40 C, Pcrit < P <= 800 bar. This is NOT a
        # thermodynamic phase. It is a highlighted subset of supercritical H2.
        if show_cch2_region:
            ceiling = min(cch2_temperature_ceiling_K, T_max)
            if ceiling > max(Tcrit, T_min):
                tt = np.linspace(max(Tcrit + 1e-4, T_min), ceiling, 450)
                bottom = np.array([PropsSI('Dmass', 'T', float(t), 'P', Pcrit, fluid)
                                   for t in tt], dtype=float)
                top = np.array([PropsSI('Dmass', 'T', float(t), 'P',
                                              max_display_pressure_bar * 1e5, fluid)
                                for t in tt], dtype=float)
                valid = np.isfinite(bottom) & np.isfinite(top) & (top > bottom)
                if np.any(valid):
                    # Use a gentle hatch to distinguish the *engineering* CcH2
                    # envelope from the thermodynamic supercritical region.
                    ax.fill_between(tt, bottom, top, where=valid,
                                    facecolor='none', edgecolor='#216b51' if not self.use_greyscale else '#555555',
                                    hatch='//', linewidth=0, zorder=2.5,
                                    label='_nolegend_')
                    # Vertical division at -40 C, only within the pressure band.
                    if T_min < cch2_temperature_ceiling_K < T_max:
                        ax.plot([ceiling, ceiling], [bottom[-1], top[-1]],
                                color='#216b51' if not self.use_greyscale else '#555555',
                                lw=1.7, zorder=4)

        # Optional narrower design envelope, if explicitly supplied.
        if cryocompressed_envelope is not None:
            t_lo, t_hi, p_lo_bar, p_hi_bar = map(float, cryocompressed_envelope)
            if not (Tcrit < t_lo < t_hi and Pcrit / 1e5 < p_lo_bar < p_hi_bar <= max_display_pressure_bar):
                raise ValueError("Narrow CcH2 envelope must be supercritical and within the pressure cap")
            lo, hi = max(t_lo, T_min), min(t_hi, T_max)
            if lo < hi:
                tt = np.linspace(lo, hi, 240)
                low_rho = np.array([PropsSI('Dmass', 'T', float(t), 'P', p_lo_bar * 1e5, fluid)
                                    for t in tt], dtype=float)
                high_rho = np.array([PropsSI('Dmass', 'T', float(t), 'P', p_hi_bar * 1e5, fluid)
                                     for t in tt], dtype=float)
                ax.fill_between(tt, low_rho, high_rho,
                                facecolor='none', edgecolor='black', hatch='xx',
                                linewidth=0.0, zorder=3, label='Narrow CcH$_2$ design envelope')
                ax.plot(tt, low_rho, color='black', linewidth=1.2, zorder=4)
                ax.plot(tt, high_rho, color='black', linewidth=1.2, zorder=4)

        # Draw the pressure cap as a real CoolProp isobar, not an arbitrary
        # straight diagonal or a solid--liquid phase boundary.
        cap_valid = (np.isfinite(rho_pmax) & (rho_pmax >= rho_min) &
                     (rho_pmax <= rho_max) & (T_vals >= melt_T_cap))
        if np.any(cap_valid):
            ax.plot(T_vals[cap_valid], rho_pmax[cap_valid],
                    color='black', linestyle='-', linewidth=2.1, zorder=5,
                    label='_nolegend_')

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

        # Storage-concept names are indicative engineering regimes, not phases.
        # Keep them separate from the thermodynamic phase legend.
        def storage_label(x, y, label, size=19, color='black'):
            if T_min <= x <= T_max and rho_min <= y <= rho_max:
                ax.text(x, y, label, ha='center', va='center',
                        fontsize=size, fontweight='bold', color=color,
                        alpha=0.85, zorder=6)

        if show_cch2_region:
            storage_label(130, 13, r'CcH$_2$', color='k')
        if cch2_temperature_ceiling_K < T_max:
            storage_label(266, 19, r'CH$_2$', color='k')
        storage_label(20.7, 51, r'LH$_2$', size=15, color='k')
        storage_label(25.5, 85, r'sLH$_2$', size=14, color='k')

        # Label the 800 bar isobar directly; do not describe it as a display limit.
        cap_label_T = min(max(250.0, T_min), T_max)
        try:
            cap_label_rho = float(PropsSI('Dmass', 'T', cap_label_T,
                                          'P', max_display_pressure_bar * 1e5, fluid))
            if rho_min <= cap_label_rho <= rho_max:
                ax.annotate(f'{max_display_pressure_bar:g} bar',
                            (cap_label_T, cap_label_rho), xytext=(0, 7),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=resolved_isobar_label_font_size,
                            bbox=dict(facecolor='white', edgecolor='black',
                                      linewidth=0.5, boxstyle='round,pad=0.2'), zorder=7)
        except Exception:
            pass

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

        region_labels = ['Gas / superheated vapour', 'Two-phase', 'Compressed liquid', 'Supercritical fluid']
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
