"""PlotterBase: figure methods of the Delft plotter (one concern per module)."""


from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

# Import seaborn styling
from toplab.plotting.plot_style_sb import (
    configure_plot_style, DELFT_PALETTE, FONT_NAME
)
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.system.state_management import MultiTankResults



class PlotterBase:
    def __init__(self, analysis_name: str = "Multi-Tank Analysis",
                 use_greyscale: bool = False,
                 enable_multi_tank_overlay: bool = False,
                 output_formats: Optional[List[str]] = None,
                 enable_pgf: bool = False):
        """
        Initialize plotter with seaborn styling.

        Args:
            analysis_name: Name of the analysis (used in plot titles)
            use_greyscale: Whether to use greyscale styling instead of colors
            enable_multi_tank_overlay: Whether to overlay multiple tanks on same plots
            output_formats: List of output formats (e.g., ['png', 'pdf', 'pgf']). If None, defaults to ['png']
            enable_pgf: Whether to enable PGF backend for LaTeX compatibility
        """
        self.analysis_name = analysis_name
        self.use_greyscale = use_greyscale
        self.enable_multi_tank_overlay = enable_multi_tank_overlay
        self.output_formats = output_formats if output_formats else ['png']
        self.enable_pgf = enable_pgf or ('pgf' in self.output_formats)

        # Configure matplotlib backend for PGF if needed
        if self.enable_pgf:
            self._configure_pgf_backend()

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
            print(f"DelftColourPlotter initialised for: {analysis_name} (Greyscale Mode)")
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
            print(f"DelftColourPlotter initialised for: {analysis_name}")

        # Report output formats
        if len(self.output_formats) > 1:
            print(f"   Output formats: {', '.join(self.output_formats)}")

    def _configure_pgf_backend(self):
        """Configure matplotlib for PGF output compatible with LaTeX."""
        # Note: Backend should be set before importing pyplot, but we work with current backend
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{graphicx}",
        })
        print(f"   PGF backend configured for LaTeX compatibility")

    def _save_figure(self, fig, save_path: str, dpi: int = 900, formats: Optional[List[str]] = None):
        """
        Save figure in multiple formats.

        Args:
            fig: Matplotlib figure object
            save_path: Base path for saving (without extension)
            dpi: Resolution for raster formats
            formats: List of formats to save. If None, uses self.output_formats
        """
        from pathlib import Path

        if formats is None:
            formats = self.output_formats

        # Parse the save_path to extract directory and base name
        save_path_obj = Path(save_path)
        base_dir = save_path_obj.parent
        base_name = save_path_obj.stem  # Name without extension

        # Ensure directory exists
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save in each requested format
        saved_files = []
        for fmt in formats:
            output_file = base_dir / f"{base_name}.{fmt}"

            try:
                if fmt == 'pgf':
                    # For PGF, use lower DPI and ensure rasterization is set correctly
                    fig.savefig(str(output_file), format='pgf', bbox_inches='tight')
                elif fmt == 'pdf':
                    fig.savefig(str(output_file), format='pdf', dpi=dpi, bbox_inches='tight')
                elif fmt == 'svg':
                    fig.savefig(str(output_file), format='svg', bbox_inches='tight')
                else:  # png or other raster formats
                    fig.savefig(str(output_file), dpi=dpi, bbox_inches='tight', facecolor='white')

                saved_files.append(str(output_file))
            except Exception as e:
                print(f"   WARNING: Failed to save {fmt} format: {e}")

        if saved_files:
            if len(saved_files) == 1:
                print(f"   Saved to: {saved_files[0]}")
            else:
                print(f"   Saved {len(saved_files)} formats: {', '.join([Path(f).suffix for f in saved_files])}")

        return saved_files

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

    def _create_empty_plot(self, message: str) -> plt.Figure:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center',
               fontsize=plot_style.LEGEND_FONT_SIZE)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Pressure [bar]')
        return fig

    @staticmethod
    def _component_label(class_name: str) -> str:
        """Convert a Python class name to a readable label."""
        _map = {
            'IdealHeatExchanger': 'HEX',
            'Compressor': 'Compressor',
            'Cryopump': 'Pump',
        }
        return _map.get(class_name, class_name)

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
            print(f"   WARNING: Could not extract valve events: {e}")

        return valve_events
