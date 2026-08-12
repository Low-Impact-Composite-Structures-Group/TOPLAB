"""
State management for multi-tank systems.

This module provides classes for managing the state of multiple tanks
and storing analysis results.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from toplab.thermodynamics.tank_states import IsochoricTankState, IsochoricTankStates


@dataclass
class MultiTankState:
    """
    State container for multiple tanks.

    Manages the N-tank state vector [m1, T1, Ts1, m2, T2, Ts2, ..., mN, TN, TsN]
    and provides convenient access to individual tank states.
    """
    tank_states: List[IsochoricTankState]
    time: float = 0.0

    @property
    def n_tanks(self) -> int:
        """Number of tanks in the system"""
        return len(self.tank_states)

    @property
    def state_vector(self) -> np.ndarray:
        """Combined state vector [m1, T1, Ts1, m2, T2, Ts2, ...]"""
        vector = []
        for tank_state in self.tank_states:
            vector.extend([
                tank_state.fuel_mass,
                tank_state.temperature,
                tank_state.solid_temperature
            ])
        return np.array(vector)

    @classmethod
    def from_state_vector(cls,
                         state_vector: np.ndarray,
                         tank_objects: List[Any],
                         time: float = 0.0,
                         flow_data: List[Dict] = None) -> 'MultiTankState':
        """Create MultiTankState from combined state vector"""
        n_tanks = len(tank_objects)
        if len(state_vector) != 3 * n_tanks:
            raise ValueError(f"State vector length {len(state_vector)} != 3 * {n_tanks} tanks")

        tank_states = []
        for i in range(n_tanks):
            idx = 3 * i
            tank_state = IsochoricTankState(
                tank=tank_objects[i],
                fuel_mass=state_vector[idx],
                temperature=state_vector[idx + 1],
                solid_temperature=state_vector[idx + 2]
            )

            # Add flow rate data if provided
            if flow_data and i < len(flow_data):
                flow_info = flow_data[i]
                tank_state.inflow_rate = flow_info.get('inflow_rate', 0.0)
                tank_state.outflow_rate = flow_info.get('outflow_rate', 0.0)
                tank_state.vent_rate = flow_info.get('vent_rate', 0.0)
                tank_state.coupling_inflow_rate = flow_info.get('coupling_inflow_rate', 0.0)
                tank_state.coupling_outflow_rate = flow_info.get('coupling_outflow_rate', 0.0)

            tank_states.append(tank_state)

        return cls(tank_states=tank_states, time=time)

    def get_tank_state(self, tank_index: int) -> IsochoricTankState:
        """Get state for specific tank"""
        return self.tank_states[tank_index]

    def update_from_state_vector(self, state_vector: np.ndarray):
        """Update all tank states from combined state vector"""
        for i, tank_state in enumerate(self.tank_states):
            idx = 3 * i
            tank_state.fuel_mass = state_vector[idx]
            tank_state.temperature = state_vector[idx + 1]
            tank_state.solid_temperature = state_vector[idx + 2]

            # Recompute derived properties
            tank_state.compute_pressure()
            tank_state.get_hydrogen_properties()


@dataclass
class MultiTankResults:
    """
    Results container for multi-tank analysis.

    Provides both individual tank access and unified time series data
    for convenient post-processing and plotting.
    """
    times: np.ndarray
    multi_tank_states: List[MultiTankState]
    tank_metadata: List[Dict[str, Any]]

    @property
    def n_tanks(self) -> int:
        """Number of tanks"""
        return len(self.tank_metadata)

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self.times)

    def get_tank_series(self, tank_index: int) -> IsochoricTankStates:
        """Get time series for specific tank (compatible with single-tank plotting)"""
        tank_states = []
        for multi_state in self.multi_tank_states:
            tank_states.append(multi_state.get_tank_state(tank_index))

        # Use the first tank's time step for compatibility
        timestep = self.times[1] - self.times[0] if len(self.times) > 1 else 1.0

        return IsochoricTankStates(states=tank_states, timestep=timestep)

    def get_combined_data(self) -> Dict[str, np.ndarray]:
        """Get combined data arrays for all tanks"""
        data = {
            'times': self.times,
            'masses': [],
            'temperatures': [],
            'solid_temperatures': [],
            'pressures': [],
            'densities': [],
            'inflow_rates': [],
            'outflow_rates': [],
            'vent_rates': [],
            'coupling_inflow_rates': [],
            'coupling_outflow_rates': []
        }

        for tank_idx in range(self.n_tanks):
            tank_data = self._extract_tank_arrays(tank_idx)
            for key in ['masses', 'temperatures', 'solid_temperatures', 'pressures', 'densities',
                       'inflow_rates', 'outflow_rates', 'vent_rates', 'coupling_inflow_rates', 'coupling_outflow_rates']:
                data[key].append(tank_data[key])

        # Convert lists to numpy arrays
        for key in ['masses', 'temperatures', 'solid_temperatures', 'pressures', 'densities',
                   'inflow_rates', 'outflow_rates', 'vent_rates', 'coupling_inflow_rates', 'coupling_outflow_rates']:
            data[key] = np.array(data[key])

        return data

    def _extract_tank_arrays(self, tank_index: int) -> Dict[str, np.ndarray]:
        """Extract time series arrays for specific tank"""
        masses = []
        temperatures = []
        solid_temperatures = []
        pressures = []
        densities = []
        inflow_rates = []
        outflow_rates = []
        vent_rates = []
        coupling_inflow_rates = []
        coupling_outflow_rates = []

        for multi_state in self.multi_tank_states:
            tank_state = multi_state.get_tank_state(tank_index)
            masses.append(tank_state.fuel_mass)
            temperatures.append(tank_state.temperature)
            solid_temperatures.append(tank_state.solid_temperature)

            # Calculate pressure if not available
            if tank_state.pressure is None:
                tank_state.compute_pressure()
            pressures.append(tank_state.pressure / 1e5)  # Convert to bar

            densities.append(tank_state.density)

            # Extract flow rates (convert to g/s for plotting)
            inflow_rates.append(getattr(tank_state, 'inflow_rate', 0.0) * 1000)
            outflow_rates.append(getattr(tank_state, 'outflow_rate', 0.0) * 1000)
            vent_rates.append(getattr(tank_state, 'vent_rate', 0.0) * 1000)
            coupling_inflow_rates.append(getattr(tank_state, 'coupling_inflow_rate', 0.0) * 1000)
            coupling_outflow_rates.append(getattr(tank_state, 'coupling_outflow_rate', 0.0) * 1000)

        return {
            'masses': np.array(masses),
            'temperatures': np.array(temperatures),
            'solid_temperatures': np.array(solid_temperatures),
            'pressures': np.array(pressures),
            'densities': np.array(densities),
            'inflow_rates': np.array(inflow_rates),
            'outflow_rates': np.array(outflow_rates),
            'vent_rates': np.array(vent_rates),
            'coupling_inflow_rates': np.array(coupling_inflow_rates),
            'coupling_outflow_rates': np.array(coupling_outflow_rates)
        }