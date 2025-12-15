"""
Isochoric Dynamic Models for stops_model integration within multi_tank.

This module mirrors the functionality of `src/dynamics/isochoric_dynamic_models.py`
but is colocated under `src/multi_tank/` to keep multi-state code self-contained.
"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from CoolProp.CoolProp import PropsSI

from src.thermodynamics.tank_states import (
    IsochoricTankState,
    IsochoricStateDerivatives,
    IsochoricTankStates,
)
from src.fluids.convective_mediums import IsochoricHydrogen
from src.fluids.hydrogen_retrievers import IsochoricHydrogenRequester

_heat_flow_data = None


def set_heat_flow_data_collector(data_dict):
    global _heat_flow_data
    _heat_flow_data = data_dict


class IsochoricDynamicModel(ABC):
    @abstractmethod
    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs,
    ) -> IsochoricStateDerivatives:
        pass

    @abstractmethod
    def is_applicable(self, state: IsochoricTankState) -> bool:
        pass


class SinglePhaseIsochoricModel(IsochoricDynamicModel):
    def __init__(self, scenario: str = "DISCHARGE", p_min: float = 15e5, p_vent: float = 450e5, tank_volume: float = 0.5):
        self.scenario = scenario
        self.p_min = p_min
        self.p_vent = p_vent
        self.tank_volume = tank_volume

    def is_applicable(self, state: IsochoricTankState) -> bool:
        if state.hydrogen is None:
            return True
        return not state.hydrogen.is_near_saturation

    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs,
    ) -> IsochoricStateDerivatives:
        m = max(state.fuel_mass, 1.0)
        T = max(state.temperature, 1.0)
        Ts = state.solid_temperature
        rho = state.density
        p = state.pressure

        if state.hydrogen is None:
            requester = IsochoricHydrogenRequester()
            hydrogen = requester.get_hydrogen_properties(p, T, rho)
        else:
            hydrogen = state.hydrogen

        try:
            h = PropsSI("Hmass", "T", T, "Dmass", rho, "hydrogen")
            c_v = PropsSI("Cvmass", "T", T, "Dmass", rho, "hydrogen")
            dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, "hydrogen")
        except:
            h = 0.0
            c_v = 14000.0
            dp_dT_rho = p / T

        config = self._determine_configuration(p)
        state.configuration = config
        self._last_config = config

        mdot_fuel = fuel_flow_func(time)
        mdot_discharge = discharge_flow_func(time)
        mdot_vent = self._get_vent_flow_rate(config, T, rho, p, time, discharge_flow_func, Ts)

        h_fuel = self._compute_fuel_enthalpy(p, T)
        coupling_enthalpy = kwargs.get('coupling_enthalpy', None)
        if coupling_enthalpy is not None and coupling_enthalpy != 0.0:
            h_fuel = coupling_enthalpy

        h_discharge = h
        h_vent = h

        h_term = mdot_fuel * (h_fuel - h) - mdot_discharge * (h_discharge - h) - mdot_vent * (h_vent - h)
        net_mass_flow = mdot_fuel - mdot_discharge - mdot_vent
        work_term = (T / rho) * dp_dT_rho * net_mass_flow

        Q_solid = kwargs.get('Q_solid', 0.0)
        Q_discharge = kwargs.get('Q_discharge', 0.0)
        dm_dt = net_mass_flow
        actual_qdot_disch = 0.0

        if config == "B" and self.scenario != "REFUEL":
            try:
                p_constrained = self.p_min
                dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, 'hydrogen')
                term1 = (T / rho) * dp_dT_rho
                dT_drho_p = PropsSI('d(T)/d(D)|P', 'P', p_constrained, 'T', T, 'hydrogen')
                term2 = rho * c_v * dT_drho_p
                qdot_disch_B = mdot_discharge * (term1 - term2) - Q_solid
                actual_qdot_disch = qdot_disch_B
                dT_dt = (h_term + work_term + Q_solid + qdot_disch_B) / (m * c_v)
            except Exception:
                dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v)
                actual_qdot_disch = 0.0
        else:
            dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v)
            actual_qdot_disch = 0.0

        if _heat_flow_data is not None:
            _heat_flow_data['t'].append(time)
            _heat_flow_data['qdot_disch'].append(actual_qdot_disch)
            _heat_flow_data['qdot_ohex'].append(0.0)
            _heat_flow_data['mdot_disch'].append(mdot_discharge)
            _heat_flow_data['T'].append(T)
            _heat_flow_data['rho'].append(rho)

        dTs_dt = kwargs.get('dTs_dt', 0.0)

        return IsochoricStateDerivatives(
            fuel_mass_derivative=dm_dt,
            temperature_derivative=dT_dt,
            solid_temperature_derivative=dTs_dt,
            heat_flux=Q_solid,
            discharge_heat_flux=Q_discharge,
        )

    def _determine_configuration(self, pressure: float) -> str:
        p_min_hysteresis = self.p_min * 0.01
        current_config = getattr(self, '_last_config', 'A')
        if pressure >= self.p_vent:
            return "C"
        elif current_config == "B" and pressure <= (self.p_min + p_min_hysteresis):
            return "B"
        elif current_config != "B" and pressure <= (self.p_min - p_min_hysteresis):
            return "B"
        else:
            return "A"

    def _get_vent_flow_rate(self, config: str, T: float, rho: float, p: float, time: float, discharge_func: Callable, Ts: float) -> float:
        if config == "C":
            return max(0.0, (p - self.p_vent) * 1e-8)
        else:
            return 0.0

    def _compute_fuel_enthalpy(self, pressure: float, temperature: float) -> float:
        if self.scenario == "REFUEL":
            return self._compute_cryopump_enthalpy(pressure, temperature)
        else:
            try:
                return PropsSI("Hmass", "P", pressure, "T", temperature, "hydrogen")
            except:
                return 0.0

    def _compute_cryopump_enthalpy(self, tank_pressure: float, tank_temperature: float) -> float:
        P1 = 3e5
        P2 = tank_pressure
        eta_p = 0.78
        try:
            h1 = PropsSI("H", "P", P1, "Q", 0, "hydrogen")
            s1 = PropsSI("S", "P", P1, "Q", 0, "hydrogen")
            h2s = PropsSI("H", "P", P2, "S", s1, "hydrogen")
            h2 = h1 + (h2s - h1) / eta_p
            return h2
        except:
            return 0.0


class TwoPhaseIsochoricModel(IsochoricDynamicModel):
    def __init__(self, scenario: str = "DISCHARGE", p_min: float = 15e5, p_vent: float = 450e5, tank_volume: float = 0.5):
        self.scenario = scenario
        self.p_min = p_min
        self.p_vent = p_vent
        self.tank_volume = tank_volume

    def is_applicable(self, state: IsochoricTankState) -> bool:
        if state.hydrogen is None:
            return False
        return state.hydrogen.is_near_saturation

    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs,
    ) -> IsochoricStateDerivatives:
        m = max(state.fuel_mass, 1e-12)
        T = max(state.temperature, 1.0)
        Ts = state.solid_temperature
        rho = state.density

        if state.hydrogen is None:
            requester = IsochoricHydrogenRequester()
            hydrogen = requester.get_hydrogen_properties(state.pressure, T, rho)
        else:
            hydrogen = state.hydrogen

        try:
            p_sat = PropsSI("P", "T", T, "Q", 0, "hydrogen")
            h = PropsSI("Hmass", "T", T, "Dmass", rho, "hydrogen")
            x = hydrogen.vapor_fraction if hydrogen.vapor_fraction is not None else 0.0
            c_v_liquid = PropsSI("Cvmass", "T", T, "Q", 0, "hydrogen")
            c_v_vapor = PropsSI("Cvmass", "T", T, "Q", 1, "hydrogen")
            c_v2P = x * c_v_vapor + (1.0 - x) * c_v_liquid
            h_vapor = PropsSI("Hmass", "T", T, "Q", 1, "hydrogen")
            h_liquid = PropsSI("Hmass", "T", T, "Q", 0, "hydrogen")
            rho_vapor = PropsSI("Dmass", "T", T, "Q", 1, "hydrogen")
            rho_liquid = PropsSI("Dmass", "T", T, "Q", 0, "hydrogen")
            L_v = h_vapor - h_liquid
            delta_v = (1.0 / rho_vapor) - (1.0 / rho_liquid)
            dp_sat_dT = L_v / (T * delta_v)
        except:
            p_sat = state.pressure
            h = 0.0
            c_v2P = 14000.0
            dp_sat_dT = p_sat / T

        config = self._determine_configuration(p_sat)
        state.configuration = config
        state.pressure = p_sat

        mdot_fuel = fuel_flow_func(time)
        mdot_discharge = discharge_flow_func(time)
        mdot_vent = self._get_vent_flow_rate(config, T, rho, p_sat, time, discharge_flow_func, Ts)

        h_fuel = self._compute_fuel_enthalpy(p_sat, T)
        h_discharge = h
        h_vent = h
        h_term = mdot_fuel * (h_fuel - h) - mdot_discharge * (h_discharge - h) - mdot_vent * (h_vent - h)
        net_mass_flow = mdot_fuel - mdot_discharge - mdot_vent
        work_term = (T / rho) * dp_sat_dT * net_mass_flow

        Q_solid = kwargs.get('Q_solid', 0.0)
        Q_discharge = kwargs.get('Q_discharge', 0.0)
        dm_dt = net_mass_flow
        actual_qdot_disch = 0.0

        if config == "B" and self.scenario != "REFUEL":
            try:
                term1 = (T / rho) * dp_sat_dT
                h_disch = h
                term2 = h_disch - h
                qdot_disch_B = mdot_discharge * (term1 + term2) - Q_solid
                actual_qdot_disch = qdot_disch_B
                dT_dt = (h_term + work_term + Q_solid + qdot_disch_B) / (m * c_v2P)
            except Exception:
                dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v2P)
                actual_qdot_disch = 0.0
        else:
            dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v2P)
            actual_qdot_disch = 0.0

        if _heat_flow_data is not None:
            _heat_flow_data['t'].append(time)
            _heat_flow_data['qdot_disch'].append(actual_qdot_disch)
            _heat_flow_data['qdot_ohex'].append(0.0)
            _heat_flow_data['mdot_disch'].append(mdot_discharge)
            _heat_flow_data['T'].append(T)
            _heat_flow_data['rho'].append(rho)

        dTs_dt = kwargs.get('dTs_dt', 0.0)

        return IsochoricStateDerivatives(
            fuel_mass_derivative=dm_dt,
            temperature_derivative=dT_dt,
            solid_temperature_derivative=dTs_dt,
            heat_flux=Q_solid,
            discharge_heat_flux=Q_discharge,
        )

    def _determine_configuration(self, pressure: float) -> str:
        if pressure >= self.p_vent:
            return "C"
        elif pressure <= self.p_min:
            return "B"
        else:
            return "A"

    def _get_vent_flow_rate(self, config: str, T: float, rho: float, p: float, time: float, discharge_func: Callable, Ts: float) -> float:
        if config == "C":
            return max(0.0, (p - self.p_vent) * 1e-7)
        else:
            return 0.0

    def _compute_fuel_enthalpy(self, pressure: float, temperature: float) -> float:
        if self.scenario == "REFUEL":
            return self._compute_cryopump_enthalpy(pressure, temperature)
        else:
            try:
                return PropsSI("Hmass", "P", pressure, "T", temperature, "hydrogen")
            except Exception:
                R_specific = 4124.0
                cp = 14300.0
                return cp * temperature

    def _compute_cryopump_enthalpy(self, tank_pressure: float, tank_temperature: float) -> float:
        P1 = 3e5
        P2 = tank_pressure
        eta_p = 0.78
        try:
            h1 = PropsSI("H", "P", P1, "Q", 0, "hydrogen")
            s1 = PropsSI("S", "P", P1, "Q", 0, "hydrogen")
            h2s = PropsSI("H", "P", P2, "S", s1, "hydrogen")
            h2 = h1 + (h2s - h1) / eta_p
            return h2
        except:
            return 0.0


class IsochoricModelSwitcher:
    def __init__(self, scenario: str = "DISCHARGE", p_min: float = 15e5, p_vent: float = 450e5, tank_volume: float = 0.5):
        self.single_phase_model = SinglePhaseIsochoricModel(scenario, p_min, p_vent, tank_volume)
        self.two_phase_model = TwoPhaseIsochoricModel(scenario, p_min, p_vent, tank_volume)
        self.model_history = []

    def select_model(self, state: IsochoricTankState) -> IsochoricDynamicModel:
        if self.two_phase_model.is_applicable(state):
            selected_model = self.two_phase_model
            model_name = "two_phase"
        else:
            selected_model = self.single_phase_model
            model_name = "single_phase"
        self.model_history.append(model_name)
        return selected_model

    def compute_state_derivatives(
        self,
        time: float,
        state: IsochoricTankState,
        fuel_flow_func: Callable[[float], float],
        discharge_flow_func: Callable[[float], float],
        **kwargs,
    ) -> IsochoricStateDerivatives:
        model = self.select_model(state)
        return model.compute_state_derivatives(time, state, fuel_flow_func, discharge_flow_func, **kwargs)
