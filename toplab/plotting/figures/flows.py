"""FlowFigures: figure methods of the Delft plotter (one concern per module)."""


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



class FlowFigures:
    def plot_mass_flows(self,
                       results: MultiTankResults,
                       tank_index: int = 0,
                       include_venting_flow: bool = True,
                       include_coupling_flows: bool = True,
                       save_path: Optional[str] = None,
                       xlim: Optional[Tuple[float, float]] = None,
                       ylim: Optional[Tuple[float, float]] = None,
                       legend_location: str = 'best',
                       per_stream_outflows: Optional[Dict[str, Any]] = None) -> plt.Figure:
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
            per_stream_outflows: Optional dict mapping stream label → outflow array [g/s, negative].
                                 When provided, individual stream curves are plotted instead of a
                                 single combined outflow, e.g. {'FC': array, 'GT': array}.

        Returns:
            matplotlib Figure object
        """
        print(f"Plotting mass flows for Tank {tank_index + 1}...")

        # Validate inputs
        if tank_index >= results.n_tanks:
            raise ValueError(f"Tank index {tank_index} exceeds available tanks ({results.n_tanks})")

        # Create figure using simple, proven layout
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert time to hours for plotting
        times_hours = results.times / 3600.0

        # Get flow data for the specific tank using the exact same pattern as working multi_tank_analysis
        tank_data = results._extract_tank_arrays(tank_index)

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

        inflow_linestyle = '-'
        outflow_linestyle = '--' if self.use_greyscale else '-'
        vent_linestyle = '-.' if self.use_greyscale else '--'

        ax.plot(times_hours, total_inflow, color=inflow_color, linewidth=2,
                linestyle=inflow_linestyle, label='Inflow')

        if per_stream_outflows:
            # Plot each discharge stream separately, then the total as a dashed sum
            stream_items = list(per_stream_outflows.items())
            for stream_idx, (stream_label, stream_flows) in enumerate(stream_items):
                color = self.color_palette[(stream_idx + 1) % len(self.color_palette)]
                linestyle = self.line_styles[stream_idx % len(self.line_styles)] if self.use_greyscale else '-'
                ax.plot(times_hours, stream_flows, color=color, linewidth=2,
                        linestyle=linestyle, label=stream_label)
            # Total outflow as dashed reference
            ax.plot(times_hours, total_outflow, color='black', linewidth=1.5,
                    linestyle='--', alpha=0.6, label='Total outflow')
        else:
            ax.plot(times_hours, total_outflow, color=outflow_color, linewidth=2,
                    linestyle=outflow_linestyle, label='Outflow')

        ax.plot(times_hours, vent, color=vent_color, linewidth=2,
                linestyle=vent_linestyle, label='Vent')

        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Set up plot formatting to match other plots
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Flow Rate [g/s]')
        ax.grid(True, alpha=0.3)

        # Apply custom axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Add legend with 3D shadow effect (same styling as other plots)
        legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.2)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Mass flow plot completed")
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
        print(f"Plotting sequential mass flows for Tank {tank_index + 1}...")

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
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Sequential mass flows plot completed")
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
        print("Plotting ATR72 mission mass flow...")

        # Import mission here to avoid circular imports
        from toplab.missions.mission import Mission

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
            self._save_figure(fig, save_path, dpi=900)
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
            print(f"   Saved to: {default_path}")

        print("   ATR72 mass flow plot completed")
        return fig

    # ------------------------------------------------------------------
    # Human-readable component name helper
    # ------------------------------------------------------------------
