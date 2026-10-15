"""Peripheral-component stream states along each coupling edge, derived from results."""

import numpy as np

from toplab.peripheral_components.base import PeripheralFlowState
from toplab.system.state_management import MultiTankResults


def coupling_stream_history(system, results) -> list:
    """
    Post-processing helper: re-run each coupling valve's peripheral component chain
    at every output time step and return the stream state (T, P, mdot) before and
    after each component.

    Only valves that have a non-empty component_chain (for split_chain entries) or
    non-empty main_conditioning_chain / discharge_conditioning_chain are included.

    Returns
    -------
    list of dicts, one per qualifying entry::

        {
            'stream_type': 'split_chain' | 'main_discharge' | 'fuel_cell_discharge' | 'mixed_fuel_cell',
            'coupling_id': str,
            'source_tank': int,   # 0-based
            'target_tank': int,   # 0-based, or -1 for sink
            'times_h':    ndarray,   # output times [h]
            'components': [          # for split_chain / main_discharge / fuel_cell_discharge
                {
                    'name':   str,
                    'inlet':  {'mdot_gs': ndarray, 'temperature_K': ndarray, 'pressure_bar': ndarray},
                    'outlet': {'mdot_gs': ndarray, 'temperature_K': ndarray, 'pressure_bar': ndarray},
                },
                ...
            ],
            # Additional keys for 'mixed_fuel_cell' only:
            'mdot_gs':        ndarray,   # total mixed mass flow [g/s]
            'temperature_K':  ndarray,   # mixed temperature [K]
            'pressure_bar':   ndarray,   # mixing pressure [bar]
            'mdot_main_gs':   ndarray,   # 95% conditioned stream [g/s]
            'mdot_discharge_gs': ndarray, # Tank 2 discharge stream [g/s]
        }
    """
    import numpy as np
    from CoolProp.CoolProp import PropsSI
    from toplab.peripheral_components.base import PeripheralFlowState
    from toplab.coupling import (
        ProportionalSplitCoupling,
        PressureTriggeredDischarge,
    )

    histories = []
    n_timesteps = len(results.times)
    times_h = results.times / 3600.0

    def _process_chain(chain, source_state, mdot, target_pressure):
        """Walk *chain* starting from *source_state* at *mdot* kg/s.
        Returns list of dicts {name, inlet_mdot, inlet_T, inlet_P, outlet_mdot, outlet_T, outlet_P}.
        """
        comp_data = [
            {
                'name':        type(comp).__name__,
                'inlet_mdot':  np.full(n_timesteps, np.nan),
                'inlet_T':     np.full(n_timesteps, np.nan),
                'inlet_P':     np.full(n_timesteps, np.nan),
                'outlet_mdot': np.full(n_timesteps, np.nan),
                'outlet_T':    np.full(n_timesteps, np.nan),
                'outlet_P':    np.full(n_timesteps, np.nan),
            }
            for comp in chain
        ]
        return comp_data  # pre-allocated; caller fills per timestep

    def _fill_chain_at_ts(comp_data, chain, source_state, mdot, target_pressure, ts_idx):
        """Fill one time-step's worth of inlet/outlet data for a component chain."""
        stream = PeripheralFlowState.from_tank_state(source_state, mdot)
        for ci, component in enumerate(chain):
            cd = comp_data[ci]
            cd['inlet_mdot'][ts_idx] = stream.mass_flow_rate * 1000.0
            cd['inlet_T'][ts_idx]    = stream.temperature
            cd['inlet_P'][ts_idx]    = stream.pressure / 1e5

            stream = component.process_stream(stream, target_pressure=target_pressure)

            cd['outlet_mdot'][ts_idx] = stream.mass_flow_rate * 1000.0
            cd['outlet_T'][ts_idx]    = stream.temperature
            cd['outlet_P'][ts_idx]    = stream.pressure / 1e5
        return stream  # final outlet stream after full chain

    def _comp_data_to_list(comp_data):
        return [
            {
                'name': cd['name'],
                'inlet':  {'mdot_gs': cd['inlet_mdot'], 'temperature_K': cd['inlet_T'],  'pressure_bar': cd['inlet_P']},
                'outlet': {'mdot_gs': cd['outlet_mdot'], 'temperature_K': cd['outlet_T'], 'pressure_bar': cd['outlet_P']},
            }
            for cd in comp_data
        ]

    # ----------------------------------------------------------------
    # Pass 1: split_chain and main_discharge entries
    # ----------------------------------------------------------------
    main_discharge_entry = None   # will be used to compute mixed stream

    for valve in system.coupling_valves:
        source_tank_idx = valve.source_tank

        # --- split_chain (full ODE chain: main conditioning + compressor) ---
        if valve.component_chain:
            n_components = len(valve.component_chain)
            comp_data = _process_chain(valve.component_chain, None, 0, None)

            for ts_idx in range(n_timesteps):
                t = results.times[ts_idx]
                source_state = results.multi_tank_states[ts_idx].tank_states[source_tank_idx]
                if source_state.pressure is None and hasattr(source_state, 'compute_pressure'):
                    source_state.compute_pressure()
                if source_state.pressure is None:
                    continue

                target_pressure = None
                if valve.target_tank >= 0:
                    tgt = results.multi_tank_states[ts_idx].tank_states[valve.target_tank]
                    if tgt.pressure is None and hasattr(tgt, 'compute_pressure'):
                        tgt.compute_pressure()
                    target_pressure = tgt.pressure

                try:
                    mdot = float(system.edge_flow_history[system.coupling_valves.index(valve), ts_idx])
                except Exception:
                    continue
                if mdot <= 0.0:
                    continue

                try:
                    _fill_chain_at_ts(comp_data, valve.component_chain, source_state, mdot, target_pressure, ts_idx)
                except Exception as e:
                    print(f"   WARNING: split_chain re-run failed at t={t:.1f}s: {e}")

            histories.append({
                'stream_type': 'split_chain',
                'coupling_id': valve.coupling_id,
                'source_tank': valve.source_tank,
                'target_tank': valve.target_tank,
                'times_h':    times_h,
                'components': _comp_data_to_list(comp_data),
            })

        # --- main_discharge (95% of Tank 1 outflow, through HEX + PressureReg only) ---
        if (
            isinstance(valve, ProportionalSplitCoupling)
            and valve.main_conditioning_chain
        ):
            chain = valve.main_conditioning_chain
            comp_data_main = _process_chain(chain, None, 0, None)

            # Arrays for the outlet state (used later for mixing)
            main_outlet_T   = np.full(n_timesteps, np.nan)
            main_outlet_P   = np.full(n_timesteps, np.nan)
            main_outlet_h   = np.full(n_timesteps, np.nan)
            main_mdot_gs    = np.full(n_timesteps, np.nan)

            for ts_idx in range(n_timesteps):
                t = results.times[ts_idx]
                source_state = results.multi_tank_states[ts_idx].tank_states[source_tank_idx]
                if source_state.pressure is None and hasattr(source_state, 'compute_pressure'):
                    source_state.compute_pressure()
                if source_state.pressure is None:
                    continue

                # Main discharge: (1 - split_fraction) * full mission outflow
                try:
                    mdot_split = float(system.edge_flow_history[system.coupling_valves.index(valve), ts_idx])
                except Exception:
                    continue
                if mdot_split <= 0.0:
                    continue
                mdot_main = mdot_split * (1.0 - valve.split_fraction) / valve.split_fraction

                try:
                    outlet_stream = _fill_chain_at_ts(
                        comp_data_main, chain, source_state, mdot_main, None, ts_idx
                    )
                    main_outlet_T[ts_idx]  = outlet_stream.temperature
                    main_outlet_P[ts_idx]  = outlet_stream.pressure
                    main_outlet_h[ts_idx]  = outlet_stream.enthalpy
                    main_mdot_gs[ts_idx]   = mdot_main * 1000.0
                except Exception as e:
                    print(f"   WARNING: main_discharge re-run failed at t={t:.1f}s: {e}")

            main_discharge_entry = {
                'stream_type': 'main_discharge',
                'coupling_id': valve.coupling_id + '_main',
                'source_tank': valve.source_tank,
                'target_tank': -1,
                'times_h':    times_h,
                'components': _comp_data_to_list(comp_data_main),
                # Extra arrays for mixing
                '_outlet_T_K':  main_outlet_T,
                '_outlet_P_Pa': main_outlet_P,
                '_outlet_h':    main_outlet_h,
                '_mdot_gs':     main_mdot_gs,
            }
            histories.append(main_discharge_entry)

    # ----------------------------------------------------------------
    # Pass 2: fuel_cell_discharge (Tank 2 discharge through PressureReg)
    # ----------------------------------------------------------------
    fuel_cell_entry = None

    for valve in system.coupling_valves:
        if not (isinstance(valve, PressureTriggeredDischarge) and valve.discharge_conditioning_chain):
            continue

        source_tank_idx = valve.source_tank
        chain = valve.discharge_conditioning_chain
        comp_data_fc = _process_chain(chain, None, 0, None)

        # Get Tank 2 gross coupling outflow from post-processed results [g/s]
        tank_data = results._extract_tank_arrays(source_tank_idx)
        tank2_outflow_gs = tank_data.get('coupling_outflow_rates', np.zeros(n_timesteps))

        # Arrays for the outlet state (used for mixing)
        fc_outlet_T   = np.full(n_timesteps, np.nan)
        fc_outlet_P   = np.full(n_timesteps, np.nan)
        fc_outlet_h   = np.full(n_timesteps, np.nan)
        fc_mdot_gs    = np.full(n_timesteps, np.nan)

        for ts_idx in range(n_timesteps):
            t = results.times[ts_idx]
            mdot = tank2_outflow_gs[ts_idx] / 1000.0  # g/s → kg/s
            if mdot <= 0.0 or np.isnan(mdot):
                continue

            source_state = results.multi_tank_states[ts_idx].tank_states[source_tank_idx]
            if source_state.pressure is None and hasattr(source_state, 'compute_pressure'):
                source_state.compute_pressure()
            if source_state.pressure is None:
                continue

            try:
                outlet_stream = _fill_chain_at_ts(
                    comp_data_fc, chain, source_state, mdot, None, ts_idx
                )
                fc_outlet_T[ts_idx]  = outlet_stream.temperature
                fc_outlet_P[ts_idx]  = outlet_stream.pressure
                fc_outlet_h[ts_idx]  = outlet_stream.enthalpy
                fc_mdot_gs[ts_idx]   = mdot * 1000.0
            except Exception as e:
                print(f"   WARNING: fuel_cell_discharge re-run failed at t={t:.1f}s: {e}")

        fuel_cell_entry = {
            'stream_type': 'fuel_cell_discharge',
            'coupling_id': valve.coupling_id + '_discharge',
            'source_tank': source_tank_idx,
            'target_tank': -1,
            'times_h':    times_h,
            'components': _comp_data_to_list(comp_data_fc),
            # Extra arrays for mixing
            '_outlet_T_K':  fc_outlet_T,
            '_outlet_P_Pa': fc_outlet_P,
            '_outlet_h':    fc_outlet_h,
            '_mdot_gs':     fc_mdot_gs,
        }
        histories.append(fuel_cell_entry)

    # ----------------------------------------------------------------
    # Pass 3: mixed_fuel_cell  (enthalpy-weighted mix at mixer junction)
    # ----------------------------------------------------------------
    if main_discharge_entry is not None and fuel_cell_entry is not None:
        A_T   = main_discharge_entry['_outlet_T_K']
        A_P   = main_discharge_entry['_outlet_P_Pa']
        A_h   = main_discharge_entry['_outlet_h']
        A_mdot = main_discharge_entry['_mdot_gs']

        B_T   = fuel_cell_entry['_outlet_T_K']
        B_P   = fuel_cell_entry['_outlet_P_Pa']
        B_h   = fuel_cell_entry['_outlet_h']
        B_mdot = fuel_cell_entry['_mdot_gs']

        mix_fluid = system._coolprop_fluids[main_discharge_entry['source_tank']]
        mix_mdot_gs = np.full(n_timesteps, np.nan)
        mix_T_K     = np.full(n_timesteps, np.nan)
        mix_P_bar   = np.full(n_timesteps, np.nan)

        for ts_idx in range(n_timesteps):
            mA = A_mdot[ts_idx]
            mB = B_mdot[ts_idx]
            has_A = (not np.isnan(mA)) and mA > 0.0
            has_B = (not np.isnan(mB)) and mB > 0.0

            if not has_A and not has_B:
                continue

            if has_A and has_B:
                # Enthalpy-weighted mix
                m_total = mA + mB
                h_mix = (mA * A_h[ts_idx] + mB * B_h[ts_idx]) / m_total
                # Use average outlet pressure (both should be ~3 bar)
                P_mix = (A_P[ts_idx] + B_P[ts_idx]) / 2.0
                try:
                    T_mix = PropsSI("T", "P", P_mix, "Hmass", h_mix, mix_fluid)
                except Exception:
                    T_mix = np.nan
                mix_mdot_gs[ts_idx] = m_total
                mix_T_K[ts_idx]     = T_mix
                mix_P_bar[ts_idx]   = P_mix / 1e5
            elif has_A:
                mix_mdot_gs[ts_idx] = mA
                mix_T_K[ts_idx]     = A_T[ts_idx]
                mix_P_bar[ts_idx]   = A_P[ts_idx] / 1e5
            else:
                mix_mdot_gs[ts_idx] = mB
                mix_T_K[ts_idx]     = B_T[ts_idx]
                mix_P_bar[ts_idx]   = B_P[ts_idx] / 1e5

        histories.append({
            'stream_type':      'mixed_fuel_cell',
            'coupling_id':      'fuel_cell_mixed',
            'source_tank':      -1,
            'target_tank':      -1,
            'times_h':          times_h,
            'components':       [],
            'mdot_gs':          mix_mdot_gs,
            'temperature_K':    mix_T_K,
            'pressure_bar':     mix_P_bar,
            'mdot_main_gs':     A_mdot,
            'mdot_discharge_gs': B_mdot,
        })

    return histories
