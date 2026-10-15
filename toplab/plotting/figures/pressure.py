"""PressureFigures: figure methods of the Delft plotter (one concern per module)."""


from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# Import seaborn styling
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.system.state_management import MultiTankResults



class PressureFigures:
    def plot_pressure_requirements(self,
                                 coupling_edges,
                                 results: MultiTankResults,
                                 minimum_pressure_pa: float,
                                 tank_index: int = 1,  # LH2 tank is typically tank 2 (index 1)
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pressure requirements evolution for mission-adaptive pressure control.

        Shows the dynamic pressure requirements, activation thresholds, and actual tank pressure
        for mission-adaptive coupling systems.

        Args:
            coupling_edges: Coupling edges; the one exposing ``get_diagnostic_data`` is plotted
            results: Simulation results
            minimum_pressure_pa: Minimum allowed pressure of the analysed tank [Pa]
            tank_index: Index of tank to analyze (default 1 for LH2 tank)
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"Plotting pressure requirements for Tank {tank_index + 1}...")

        # Check if we have a mission-adaptive coupling valve
        if not coupling_edges:
            print("   WARNING: No coupling valves found")
            return self._create_empty_plot("No coupling valves found")

        # Find the mission-adaptive valve
        adaptive_valve = None
        for valve in coupling_edges:
            if hasattr(valve, 'get_diagnostic_data'):
                adaptive_valve = valve
                break

        if adaptive_valve is None:
            print("   WARNING: No mission-adaptive valve found")
            return self._create_empty_plot("No mission-adaptive valve found")

        # Get diagnostic data from the valve
        try:
            diag_data = adaptive_valve.get_diagnostic_data()
        except AttributeError as e:
            print(f"   WARNING: Cannot get diagnostic data: {e}")
            # Try to access data directly
            if hasattr(adaptive_valve, 'time_history') and hasattr(adaptive_valve, 'required_pressure_history'):
                diag_data = {
                    'time_history': adaptive_valve.time_history,
                    'required_pressure_history': adaptive_valve.required_pressure_history,
                    'activation_threshold_history': getattr(adaptive_valve, 'activation_threshold_history', [])
                }
            else:
                print("   WARNING: No pressure history data available")
                return self._create_empty_plot("No pressure history data available")

        if not diag_data['time_history']:
            print("   WARNING: No pressure history data available")
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
                print("   INFO: Sorted diagnostic pressure series by time for plotting")

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
                print(f"   INFO: Deduplicated diagnostic times: {len(unique_times)} unique of {len(order)} samples")

        # Get actual tank pressure from results for comparison
        combined_data = results.get_combined_data()
        tank_pressures = combined_data['pressures'][tank_index]  # Already in bar
        result_times = combined_data['times'] / 3600.0  # Result times in hours

        # Get minimum pressure from tank configuration
        minimum_pressure_bar = minimum_pressure_pa / 1e5  # Convert Pa to bar

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
                print(f"   WARNING: Found {negative_diffs} negative time differences in pressure data")

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

                    print("   INFO: Tracking audit (active segments):")
                    print(f"      min(P_actual - P_required) = {min_delta:.3f} bar, mean = {mean_delta:.3f} bar, fraction below = {frac_below:.3%}")
                    if np.isfinite(early_min):
                        print(f"      Early [0–0.1 h]: min Δ = {early_min:.3f} bar, fraction below = {early_frac_below:.3%}")

                    # CSV export intentionally disabled to keep output clean
                else:
                    print("   INFO: Tracking audit skipped: no active segments detected in required pressure.")
            else:
                print("   INFO: Tracking audit skipped: insufficient diagnostic or results data.")
        except Exception as e:
            print(f"   WARNING: Tracking audit failed: {e}")

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
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Pressure requirements plot completed")
        return fig
