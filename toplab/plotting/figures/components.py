"""ComponentFigures: figure methods of the Delft plotter (one concern per module)."""


from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import seaborn styling
from toplab.plotting.plot_style_sb import (
    FONT_SIZE
)
import toplab.plotting.plot_style_sb as plot_style

# Multi-tank system data structures
from toplab.fluids.fluid import DEFAULT_COOLPROP_FLUID
from toplab.system.state_management import MultiTankResults



class ComponentFigures:
    def plot_peripheral_components(
        self,
        stream_history: dict,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Plot inlet/outlet conditions (mdot, T, P) for every component in one
        coupling valve's peripheral-component chain.

        Parameters
        ----------
        stream_history : dict
            One entry from the list returned by
            ``toplab.post.streams.coupling_stream_history()``.
        save_path : str, optional
            If given, the figure is saved here.
        xlim : tuple, optional
            (t_min, t_max) in hours.

        Returns
        -------
        matplotlib Figure
        """
        components = stream_history['components']
        times_h = stream_history['times_h']
        coupling_id = stream_history['coupling_id']

        if not components:
            raise ValueError("stream_history has no components to plot.")

        print(f"Plotting peripheral components for coupling '{coupling_id}'...")

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax_mdot, ax_temp, ax_pres = axes

        # Build an ordered list of "station" labels and stream data:
        # station 0 = inlet of first component (= source tank outlet)
        # station i+1 = outlet of component i
        stations = []
        n = len(components)
        for ci, comp in enumerate(components):
            lbl = self._component_label(comp['name'])
            if ci == 0:
                stations.append((f"Source → {lbl} inlet", comp['inlet']))
            stations.append((f"{lbl} outlet", comp['outlet']))

        # Assign colours from palette (cycle if many components)
        palette = self.color_palette
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

        for si, (label, data) in enumerate(stations):
            col = palette[si % len(palette)]
            ls  = line_styles[si % len(line_styles)]
            mdot = data['mdot_gs']
            temp = data['temperature_K']
            pres = data['pressure_bar']

            # Only mdot for first station (inlet) once; skip repeated inlet mdot lines
            ax_mdot.plot(times_h, mdot, color=col, linewidth=2, linestyle=ls,
                         label=label)
            ax_temp.plot(times_h, temp, color=col, linewidth=2, linestyle=ls,
                         label=label)
            ax_pres.plot(times_h, pres, color=col, linewidth=2, linestyle=ls,
                         label=label)

        # Formatting
        ax_mdot.set_ylabel('Mass Flow [g/s]')
        ax_temp.set_ylabel('Temperature [K]')
        ax_pres.set_ylabel('Pressure [bar]')
        ax_pres.set_xlabel('Time [hours]')

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.6)
            legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc='best',
                               frameon=True, fancybox=True, shadow=True,
                               framealpha=0.9, edgecolor='black')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_linewidth(1.2)
            if xlim is not None:
                ax.set_xlim(xlim)

        plt.suptitle(f"Component Chain: {coupling_id}", fontsize=FONT_SIZE + 1,
                     fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print(f"   Peripheral components plot completed")
        return fig

    def plot_hex_compressor_power(
        self,
        stream_history: dict,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim_power: Optional[Tuple[float, float]] = None,
        ylim_energy: Optional[Tuple[float, float]] = None,
        legend_location: str = 'best',
        fluid: str = DEFAULT_COOLPROP_FLUID,
    ) -> plt.Figure:
        """
        Plot instantaneous power [kW] and cumulative energy [MJ] for the
        ideal_heat_exchanger and compressor in a peripheral-component chain.

        Power is computed from the per-component inlet/outlet states already
        stored by ``toplab.post.streams.coupling_stream_history()`` using:

            Qdot / Wdot = mdot * (h_out - h_in)

        where enthalpies are recovered from (T, P) via CoolProp.

        Parameters
        ----------
        stream_history : dict
            One entry from ``compute_coupling_stream_history()`` whose
            ``stream_type`` is ``'split_chain'``.
        save_path : str, optional
        xlim : tuple, optional   (t_min, t_max) in hours.
        """
        from CoolProp.CoolProp import PropsSI

        times_h = stream_history['times_h']
        components = stream_history['components']
        coupling_id = stream_history['coupling_id']

        # Identify HEX and compressor entries by class name
        hex_comp = None
        comp_comp = None
        for comp in components:
            name_lc = comp['name'].lower()
            if 'heatexchanger' in name_lc or 'hex' in name_lc:
                hex_comp = comp
            elif 'compressor' in name_lc:
                comp_comp = comp

        if hex_comp is None and comp_comp is None:
            raise ValueError(
                f"No IdealHeatExchanger or Compressor found in stream_history "
                f"for coupling '{coupling_id}'. Available: "
                f"{[c['name'] for c in components]}"
            )

        def _power_kw(comp_data: dict) -> np.ndarray:
            """Return instantaneous power [kW] at each time step."""
            n = len(times_h)
            power = np.full(n, np.nan)
            mdot_gs = comp_data['inlet']['mdot_gs']
            T_in    = comp_data['inlet']['temperature_K']
            P_in    = comp_data['inlet']['pressure_bar']
            T_out   = comp_data['outlet']['temperature_K']
            P_out   = comp_data['outlet']['pressure_bar']
            for i in range(n):
                mdot = mdot_gs[i] / 1000.0          # g/s → kg/s
                if np.isnan(mdot) or mdot <= 0.0:
                    continue
                if np.isnan(T_in[i]) or np.isnan(T_out[i]):
                    continue
                try:
                    h_in  = PropsSI("Hmass", "P", P_in[i]  * 1e5, "T", T_in[i],  fluid)
                    h_out = PropsSI("Hmass", "P", P_out[i] * 1e5, "T", T_out[i], fluid)
                    power[i] = mdot * (h_out - h_in) / 1000.0  # W → kW
                except Exception:
                    pass
            return power

        def _cumulative_energy_mj(power_kw: np.ndarray) -> np.ndarray:
            """Cumulative energy [MJ] via trapezoidal integration over simulation time."""
            times_s = times_h * 3600.0
            n = len(times_s)
            energy = np.full(n, np.nan)
            # treat NaN (zero-flow intervals) as zero power for the integral
            p_w = np.where(np.isnan(power_kw), 0.0, power_kw * 1000.0)  # kW → W
            dt = np.diff(times_s)
            increments = 0.5 * (p_w[:-1] + p_w[1:]) * dt
            energy[0] = 0.0
            energy[1:] = np.cumsum(increments) / 1e6               # J → MJ
            return energy

        hex_pwr  = _power_kw(hex_comp)  if hex_comp  is not None else None
        comp_pwr = _power_kw(comp_comp) if comp_comp is not None else None
        hex_eng  = _cumulative_energy_mj(hex_pwr)  if hex_pwr  is not None else None
        comp_eng = _cumulative_energy_mj(comp_pwr) if comp_pwr is not None else None

        fig, (ax_pwr, ax_eng) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        palette   = self.color_palette
        col_hex   = palette[0]
        col_comp  = palette[1]

        if hex_pwr is not None:
            ax_pwr.plot(times_h, hex_pwr,  color=col_hex,  linewidth=2,
                        label='HEX — heat duty')
            ax_eng.plot(times_h, hex_eng,  color=col_hex,  linewidth=2,
                        label='HEX — cumulative heat')
        if comp_pwr is not None:
            ax_pwr.plot(times_h, comp_pwr, color=col_comp, linewidth=2,
                        linestyle='--', label='Compressor — shaft power')
            ax_eng.plot(times_h, comp_eng, color=col_comp, linewidth=2,
                        linestyle='--', label='Compressor — cumulative work')

        for ax in (ax_pwr, ax_eng):
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.6)
            ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(xlim)

        for ax, ylim, loc in (
            (ax_pwr, ylim_power,  legend_location),
            (ax_eng, ylim_energy, legend_location),
        ):
            legend = ax.legend(
                fontsize=plot_style.LEGEND_FONT_SIZE, loc=loc,
                frameon=True, fancybox=True, shadow=True,
                framealpha=0.9, edgecolor='black',
            )
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_linewidth(1.2)
            if ylim is not None:
                ax.set_ylim(ylim)

        ax_pwr.set_ylabel('Power [kW]')
        ax_eng.set_ylabel('Cumulative Energy [MJ]')
        ax_eng.set_xlabel('Time [hours]')

        plt.suptitle(
            f"HEX & Compressor Power: {coupling_id}",
            fontsize=FONT_SIZE + 1, fontweight='bold', y=1.01,
        )
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print("   HEX/compressor power plot completed")
        return fig

    def plot_fuel_cell_inlet(
        self,
        results: MultiTankResults,
        source_tank_idx: int = 1,
        stream_histories: Optional[list] = None,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim_mdot: Optional[Tuple[float, float]] = None,
        ylim_temp: Optional[Tuple[float, float]] = None,
        ylim_pres: Optional[Tuple[float, float]] = None,
        legend_location: str = 'best',
    ) -> plt.Figure:
        """
        Plot conditions delivered to the fuel cell at the mixer junction.

        When *stream_histories* contains the pre-computed stream data (from
        ``toplab.post.streams.coupling_stream_history``), the plot shows:
          • the conditioned main stream (95% of Tank 1 outflow after HEX + PressureReg)
          • the buffer-tank discharge stream (Tank 2 outflow after PressureReg)
          • the enthalpy-weighted mixed stream that the fuel cell actually receives

        Without *stream_histories* the method falls back to showing the raw
        Tank 2 coupling outflow / temperature / pressure (legacy behaviour).

        Parameters
        ----------
        results : MultiTankResults
        source_tank_idx : int
            0-based index of the buffer tank (used for fallback mode only).
        stream_histories : list, optional
            Return value of ``toplab.post.streams.coupling_stream_history``.
        save_path : str, optional
        xlim : tuple, optional   (t_min, t_max) in hours.
        ylim_mdot : tuple, optional   (y_min, y_max) for the mass-flow panel [g/s].

        Returns
        -------
        matplotlib Figure
        """
        print("Plotting fuel cell inlet conditions...")

        # ----------------------------------------------------------------
        # Try to use pre-computed mixed stream data
        # ----------------------------------------------------------------
        mixed_entry       = None
        main_entry        = None
        discharge_entry   = None

        if stream_histories:
            for sh in stream_histories:
                st = sh.get('stream_type', '')
                if st == 'mixed_fuel_cell':
                    mixed_entry = sh
                elif st == 'main_discharge':
                    main_entry = sh
                elif st == 'fuel_cell_discharge':
                    discharge_entry = sh

        # ----------------------------------------------------------------
        # Plot
        # ----------------------------------------------------------------
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax_mdot, ax_temp, ax_pres = axes

        colors = self.color_palette

        if mixed_entry is not None:
            times_h     = mixed_entry['times_h']
            mdot_mix    = mixed_entry['mdot_gs']
            T_mix       = mixed_entry['temperature_K']
            P_mix       = mixed_entry['pressure_bar']
            mdot_main   = mixed_entry.get('mdot_main_gs')
            mdot_disc   = mixed_entry.get('mdot_discharge_gs')

            # mass-flow panel
            if mdot_main is not None:
                ax_mdot.plot(times_h, mdot_main,  color=colors[0 % len(colors)], linewidth=1.5,
                             linestyle='--', label='Tank 1 main (95%)')
            if mdot_disc is not None:
                ax_mdot.plot(times_h, mdot_disc,  color=colors[1 % len(colors)], linewidth=1.5,
                             linestyle='--', label='Tank 2 discharge')
            ax_mdot.plot(times_h, mdot_mix,       color=colors[2 % len(colors)], linewidth=2.2,
                         label='Fuel cell (mixed)')

            # temperature panel — show outlet T of each stream + mix
            if main_entry is not None and main_entry['components']:
                last_comp = main_entry['components'][-1]
                ax_temp.plot(times_h, last_comp['outlet']['temperature_K'],
                             color=colors[0 % len(colors)], linewidth=1.5, linestyle='--',
                             label='Tank 1 main (95%)')
            if discharge_entry is not None and discharge_entry['components']:
                last_comp = discharge_entry['components'][-1]
                ax_temp.plot(times_h, last_comp['outlet']['temperature_K'],
                             color=colors[1 % len(colors)], linewidth=1.5, linestyle='--',
                             label='Tank 2 discharge')
            ax_temp.plot(times_h, T_mix, color=colors[2 % len(colors)], linewidth=2.2,
                         label='Fuel cell (mixed)')

            # pressure panel — all streams should be at ~3 bar at the mixer
            if main_entry is not None and main_entry['components']:
                last_comp = main_entry['components'][-1]
                ax_pres.plot(times_h, last_comp['outlet']['pressure_bar'],
                             color=colors[0 % len(colors)], linewidth=1.5, linestyle='--',
                             label='Tank 1 main (95%)')
            if discharge_entry is not None and discharge_entry['components']:
                last_comp = discharge_entry['components'][-1]
                ax_pres.plot(times_h, last_comp['outlet']['pressure_bar'],
                             color=colors[1 % len(colors)], linewidth=1.5, linestyle='--',
                             label='Tank 2 discharge')
            ax_pres.plot(times_h, P_mix, color=colors[2 % len(colors)], linewidth=2.2,
                         label='Fuel cell (mixed)')

            title = "Fuel Cell Inlet — Mixed Stream (Tank 1 + Tank 2)"

        else:
            # ---- Fallback: raw Tank 2 discharge ----
            print(f"   (No stream_histories provided; showing Tank {source_tank_idx + 1} raw discharge)")
            if source_tank_idx >= results.n_tanks:
                raise ValueError(
                    f"source_tank_idx={source_tank_idx} out of range (n_tanks={results.n_tanks})"
                )
            times_h = results.times / 3600.0
            tank_data = results._extract_tank_arrays(source_tank_idx)
            mdot_gs       = tank_data['coupling_outflow_rates']
            temperature_K = tank_data['h2_temperatures']
            pressure_bar  = tank_data['pressures']
            col = colors[0 % len(colors)]
            ax_mdot.plot(times_h, mdot_gs,       color=col, linewidth=2, label='Fuel Cell Inlet')
            ax_temp.plot(times_h, temperature_K, color=col, linewidth=2, label='Fuel Cell Inlet')
            ax_pres.plot(times_h, pressure_bar,  color=col, linewidth=2, label='Fuel Cell Inlet')
            title = f"Fuel Cell Inlet — Tank {source_tank_idx + 1} Discharge"

        ax_mdot.set_ylabel('Mass Flow [g/s]')
        ax_temp.set_ylabel('Temperature [K]')
        ax_pres.set_ylabel('Pressure [bar]')
        ax_pres.set_xlabel('Time [hours]')

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.6)
            legend = ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location,
                               frameon=True, fancybox=True, shadow=True,
                               framealpha=0.9, edgecolor='black')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_linewidth(1.2)
            if xlim is not None:
                ax.set_xlim(xlim)

        if ylim_mdot is not None:
            ax_mdot.set_ylim(ylim_mdot)
        if ylim_temp is not None:
            ax_temp.set_ylim(ylim_temp)
        if ylim_pres is not None:
            ax_pres.set_ylim(ylim_pres)

        plt.suptitle(title, fontsize=FONT_SIZE + 1, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print("   Fuel cell inlet plot completed")
        return fig

    def plot_mixer_flows(
        self,
        history,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        legend_location: str = 'best',
    ) -> plt.Figure:
        """
        Flow into and out of one mixer junction (a ``toplab.post.mixer.MixerStreamHistory``).

        Panels: mass flow (each inlet, merged outlet, requested demand), temperature and pressure
        (each inlet stream and the merged stream), mixer mass balance (inlets - demand) and
        enthalpy balance (outlet - inlets) as a residual, and cumulative mass per stream.
        """
        print(f"Plotting mixer {history.mixer_id} flows...")
        colors = self.color_palette
        t = history.times_h

        fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
        ax_mdot, ax_temp, ax_pres, ax_res, ax_cum = axes

        for k, inlet in enumerate(history.inlets):
            color, label = colors[k % len(colors)], f"Tank {inlet.tank_id}"
            ax_mdot.plot(t, inlet.mdot_gs, color=color, linewidth=1.5, linestyle='--', label=label)
            ax_temp.plot(t, np.where(inlet.mdot_gs > 0, inlet.temperature_K, np.nan),
                         color=color, linewidth=1.5, linestyle='--', label=label)
            ax_pres.plot(t, np.where(inlet.mdot_gs > 0, inlet.pressure_bar, np.nan),
                         color=color, linewidth=1.5, linestyle='--', label=label)

        merged_color = colors[len(history.inlets) % len(colors)]
        ax_mdot.plot(t, history.mdot_gs, color=merged_color, linewidth=2.2, label='Mixer outlet')
        ax_mdot.plot(t, history.demand_gs, color='black', linewidth=1.0, linestyle=':', label='Requested demand')
        ax_temp.plot(t, history.temperature_K, color=merged_color, linewidth=2.2, label='Mixer outlet')
        ax_pres.plot(t, history.pressure_bar, color=merged_color, linewidth=2.2, label='Mixer outlet')

        ax_res.plot(t, history.mass_residual_gs, color=colors[0 % len(colors)], linewidth=1.5,
                    label='Mass: inlets - demand [g/s]')
        ax_res.set_ylabel('Mass residual [g/s]')
        ax_res_h = ax_res.twinx()
        ax_res_h.plot(t, history.enthalpy_residual_W, color=colors[1 % len(colors)], linewidth=1.5,
                      linestyle='--', label='Enthalpy: outlet - inlets [W]')
        ax_res_h.set_ylabel('Enthalpy-flux residual [W]')
        lines = ax_res.get_lines() + ax_res_h.get_lines()
        ax_res.legend(lines, [line.get_label() for line in lines], fontsize=plot_style.LEGEND_FONT_SIZE,
                      loc=legend_location, frameon=True, framealpha=0.9, edgecolor='black')

        for name, mass in history.cumulative_mass_kg().items():
            style = {'outlet': dict(color=merged_color, linewidth=2.2),
                     'demand': dict(color='black', linewidth=1.0, linestyle=':')}.get(name)
            if style is None:
                k = [i.tank_id for i in history.inlets].index(int(name.split('_')[1]))
                style = dict(color=colors[k % len(colors)], linewidth=1.5, linestyle='--')
            ax_cum.plot(t, mass, label=name.replace('_', ' ').capitalize(), **style)

        ax_mdot.set_ylabel('Mass Flow [g/s]')
        ax_temp.set_ylabel('Temperature [K]')
        ax_pres.set_ylabel('Pressure [bar]')
        ax_cum.set_ylabel('Cumulative mass [kg]')
        ax_cum.set_xlabel('Time [hours]')

        for ax in axes:
            ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(xlim)
        for ax in (ax_mdot, ax_temp, ax_pres, ax_cum):
            ax.legend(fontsize=plot_style.LEGEND_FONT_SIZE, loc=legend_location, frameon=True,
                      framealpha=0.9, edgecolor='black')

        plt.suptitle(f"Mixer {history.mixer_id} — Flow In and Out", fontsize=FONT_SIZE + 1,
                     fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, dpi=900)

        print("   Mixer flow plot completed")
        return fig
