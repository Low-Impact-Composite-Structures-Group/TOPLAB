"""
Isochoric Dynamic Models for stops_model integration within multi_tank.

This module mirrors the functionality of `src/dynamics/isochoric_dynamic_models.py`
but is colocated under `src/multi_tank/` to keep multi-state code self-contained.

Author: Dante Raso
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np

from .edge_flow import EdgeFlow

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
        edge_flows: List[EdgeFlow],
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
        edge_flows: List[EdgeFlow],
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

        tank_idx = kwargs.get('tank_index', 0)
        mdot_discharge = sum(e.mdot for e in edge_flows if e.is_outflow_for(tank_idx))
        mdot_vent = self._get_vent_flow_rate(config, p)

        h_term = 0.0
        for e in edge_flows:
            if e.is_inflow_for(tank_idx):
                h_in = e.h if (e.edge_type == 'coupling' and e.h != 0.0) else self._compute_fuel_enthalpy(p, T)
                h_term += e.mdot * (h_in - h)
            # outflows carry h_tank  →  (h_edge − h_tank) = 0, no contribution

        net_mass_flow = sum(e.mass_contribution(tank_idx) for e in edge_flows) - mdot_vent
        work_term = (T / rho) * dp_dT_rho * net_mass_flow

        Q_solid = kwargs.get('Q_solid', 0.0)
        Q_discharge = kwargs.get('Q_discharge', 0.0)
        dm_dt = net_mass_flow
        actual_qdot_disch = 0.0

        if config == "B" and self.scenario != "REFUEL":
            try:
                term1 = (T / rho) * dp_dT_rho
                dT_drho_p = PropsSI('d(T)/d(D)|P', 'P', p, 'T', T, 'hydrogen')
                term2 = rho * c_v * dT_drho_p
                qdot_disch_B = mdot_discharge * (term1 - term2) - Q_solid
                actual_qdot_disch = qdot_disch_B
                dT_dt = (h_term + work_term + Q_solid + qdot_disch_B) / (m * c_v)
            except Exception:
                dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v)
                actual_qdot_disch = 0.0
        else:
            dT_dt = (h_term + work_term + Q_solid + Q_discharge) / (m * c_v)

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
            config = "C"
        elif current_config == "B" and pressure <= (self.p_min + p_min_hysteresis):
            config = "B"
        elif current_config != "B" and pressure <= (self.p_min - p_min_hysteresis):
            config = "B"
        else:
            config = "A"
        # Track whether this model has ever operated in Config A so that the
        # ode_system throttling only fires for tanks that naturally entered
        # Config B from above p_min (not via a two-phase → single-phase transition).
        return config

    def _get_vent_flow_rate(self, config: str, p: float) -> float:
        if config == "C":
            return max(0.0, (p - self.p_vent) * 1e-8)
        else:
            return 0.0

    def _compute_fuel_enthalpy(self, pressure: float, temperature: float) -> float:
        if self.scenario == "REFUEL":
            return self._compute_cryopump_enthalpy(pressure, temperature)
        try:
            return PropsSI("Hmass", "P", pressure, "T", temperature, "hydrogen")
        except ValueError:
            # (P, T) on the saturation curve — return saturated-vapour enthalpy
            # (single-phase model represents gas-phase tanks)
            try:
                return PropsSI("Hmass", "T", temperature, "Q", 1, "hydrogen")
            except Exception:
                return 14300.0 * temperature
        except Exception:
            return 14300.0 * temperature

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
        edge_flows: List[EdgeFlow],
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

        tank_idx = kwargs.get('tank_index', 0)
        mdot_discharge = sum(e.mdot for e in edge_flows if e.is_outflow_for(tank_idx))
        mdot_vent = self._get_vent_flow_rate(config, p_sat)

        # NOTE: Old TwoPhaseIsochoricModel always used _compute_fuel_enthalpy for
        # ALL inflows (did not honour coupling_enthalpy / e.h).  We replicate that
        # behaviour here so that the two-phase energy balance is unchanged from the
        # pre-EdgeFlow baseline.  The physically-correct CH2 enthalpy path belongs
        # in a future change once the full graph refactor is in place.
        h_fuel = self._compute_fuel_enthalpy(p_sat, T)
        h_term = 0.0
        for e in edge_flows:
            if e.is_inflow_for(tank_idx):
                h_term += e.mdot * (h_fuel - h)
            # outflows carry h_tank  →  (h_edge − h_tank) = 0, no contribution

        net_mass_flow = sum(e.mass_contribution(tank_idx) for e in edge_flows) - mdot_vent
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

    def _get_vent_flow_rate(self, config: str, p: float) -> float:
        if config == "C":
            return max(0.0, (p - self.p_vent) * 1e-7)
        else:
            return 0.0

    def _compute_fuel_enthalpy(self, pressure: float, temperature: float) -> float:
        if self.scenario == "REFUEL":
            return self._compute_cryopump_enthalpy(pressure, temperature)
        # In two-phase equilibrium (P, T) is ambiguous — CoolProp cannot resolve
        # the phase from that pair alone.  Use saturated-vapour enthalpy instead,
        # which is the appropriate value for gas leaving a two-phase LH2 tank.
        try:
            return PropsSI("Hmass", "T", temperature, "Q", 1, "hydrogen")
        except Exception:
            return 14300.0 * temperature  # ideal-gas fallback

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
        edge_flows: List[EdgeFlow],
        **kwargs,
    ) -> IsochoricStateDerivatives:
        model = self.select_model(state)
        return model.compute_state_derivatives(time, state, edge_flows, **kwargs)
