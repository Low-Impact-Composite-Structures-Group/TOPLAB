"""
Isochoric mission framework colocated under multi_tank.

Copied from `src/mission/isochoric_missions.py` with imports updated to
use multi_tank dynamics/thermals.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from src.mission.mission import Mission
from src.mission.mission_sections import MissionSection, OutFlow, InFlow
from src.multi_tank.thermodynamics.isochoric_thermal_model import IsochoricThermalModel
from src.multi_tank.dynamics.isochoric_dynamic_models import IsochoricModelSwitcher
from src.multistep_methods.linear_multistep_methods import ScipyMethod
from src.thermodynamics.tank_states import (
    IsochoricTankState,
    IsochoricInitialState,
    IsochoricTankStates,
)


@dataclass
class IsochoricMissionParameters:
    tank_volume: float = 0.5
    p_min: float = 15e5
    p_vent: float = 450e5
    initial_mass: float = 10.0
    initial_temperature: float = 20.0
    initial_solid_temperature: float = 288.15
    ambient_temperature: float = 288.15
    time_step: float = 1.0
    rtol: float = 1e-6
    atol: float = 1e-9
    use_density_stopping_events: bool = False


class IsochoricMission(Mission):
    def __init__(self, sections: list[MissionSection], parameters: IsochoricMissionParameters, scenario: str = "DISCHARGE"):
        super().__init__(sections)
        self.parameters = parameters
        self.scenario = scenario
        self.dynamic_model_switcher = IsochoricModelSwitcher(
            scenario=scenario,
            p_min=parameters.p_min,
            p_vent=parameters.p_vent,
            tank_volume=parameters.tank_volume,
        )
        self.integration_method = ScipyMethod(timestep=parameters.time_step, rtol=parameters.rtol, atol=parameters.atol)

    def set_thermal_model(self, thermal_model: IsochoricThermalModel):
        self.thermal_model = thermal_model

    def create_initial_state(self) -> IsochoricInitialState:
        return IsochoricInitialState(
            fuel_mass=self.parameters.initial_mass,
            temperature=self.parameters.initial_temperature,
            solid_temperature=self.parameters.initial_solid_temperature,
            scenario=self.scenario,
        )

    def get_fuel_flow_function(self, section: MissionSection) -> Callable[[float], float]:
        flow_functions = []
        for flow in section.fuel_flows:
            if isinstance(flow, InFlow):
                if isinstance(flow.mass_flow, list) and len(flow.mass_flow) >= 2:
                    start_rate = flow.mass_flow[0]
                    end_rate = flow.mass_flow[-1]
                    duration = section.duration
                    if abs(end_rate - start_rate) < 1e-12:
                        flow_func = lambda t, rate=start_rate: rate
                    else:
                        flow_func = lambda t, s=start_rate, e=end_rate, d=duration: s + (e - s) * min(max(t, 0.0), d) / d
                elif isinstance(flow.mass_flow, list) and len(flow.mass_flow) == 1:
                    rate = flow.mass_flow[0]
                    flow_func = lambda t, rate=rate: rate
                else:
                    rate = flow.mass_flow
                    flow_func = lambda t, rate=rate: rate
                flow_functions.append(flow_func)
        if len(flow_functions) == 0:
            return lambda t: 0.0
        elif len(flow_functions) == 1:
            return flow_functions[0]
        else:
            return lambda t: sum(func(t) for func in flow_functions)

    def get_discharge_flow_function(self, section: MissionSection) -> Callable[[float], float]:
        flow_functions = []
        for flow in section.fuel_flows:
            if isinstance(flow, OutFlow):
                if isinstance(flow.mass_flow, list) and len(flow.mass_flow) >= 2:
                    start_rate = abs(flow.mass_flow[0])
                    end_rate = abs(flow.mass_flow[-1])
                    duration = section.duration
                    if abs(end_rate - start_rate) < 1e-12:
                        flow_func = lambda t, rate=start_rate: rate
                    else:
                        flow_func = lambda t, s=start_rate, e=end_rate, d=duration: s + (e - s) * min(max(t, 0.0), d) / d
                elif isinstance(flow.mass_flow, list) and len(flow.mass_flow) == 1:
                    rate = abs(flow.mass_flow[0])
                    flow_func = lambda t, rate=rate: rate
                else:
                    rate = abs(flow.mass_flow)
                    flow_func = lambda t, rate=rate: rate
                flow_functions.append(flow_func)
        if len(flow_functions) == 0:
            return lambda t: 0.0
        elif len(flow_functions) == 1:
            return flow_functions[0]
        else:
            return lambda t: sum(func(t) for func in flow_functions)


class DischargeMission(IsochoricMission):
    def __init__(self, discharge_rate: float = None, duration: float = None, parameters: IsochoricMissionParameters = None, altitude: float = 0.0, mach_number: float = 0.0, sections: list = None):
        if parameters is None:
            parameters = IsochoricMissionParameters()
        if sections is not None:
            super().__init__(sections, parameters, "DISCHARGE")
            self.discharge_rate = self._calculate_average_discharge_rate()
        else:
            if discharge_rate is None or duration is None:
                raise ValueError("For single-section missions, discharge_rate and duration are required")
            discharge_section = MissionSection(duration=duration, fuel_flows=[OutFlow(-discharge_rate, "gas")], altitude=altitude, mach_number=mach_number)
            super().__init__([discharge_section], parameters, "DISCHARGE")
            self.discharge_rate = discharge_rate

    def _calculate_average_discharge_rate(self) -> float:
        total_fuel = 0.0
        total_duration = 0.0
        for section in self.sections:
            for flow in section.fuel_flows:
                if isinstance(flow, OutFlow):
                    if isinstance(flow.mass_flow, list):
                        avg_rate = sum(abs(rate) for rate in flow.mass_flow) / len(flow.mass_flow)
                    else:
                        avg_rate = abs(flow.mass_flow)
                    total_fuel += avg_rate * section.duration
            total_duration += section.duration
        return total_fuel / total_duration if total_duration > 0 else 0.0

    @classmethod
    def constant_discharge(cls, discharge_rate: float, duration: float, initial_mass: float = 10.0, initial_temperature: float = 20.0) -> "DischargeMission":
        parameters = IsochoricMissionParameters(initial_mass=initial_mass, initial_temperature=initial_temperature)
        return cls(discharge_rate, duration, parameters)

    @classmethod
    def time_varying_discharge(cls, discharge_profile: list[float], duration: float, initial_mass: float = 10.0, initial_temperature: float = 20.0) -> "DischargeMission":
        parameters = IsochoricMissionParameters(initial_mass=initial_mass, initial_temperature=initial_temperature)
        discharge_section = MissionSection(duration=duration, fuel_flows=[OutFlow([-rate for rate in discharge_profile], "gas")], altitude=0.0, mach_number=0.0)
        mission = cls.__new__(cls)
        IsochoricMission.__init__(mission, [discharge_section], parameters, "DISCHARGE")
        return mission

    @classmethod
    def atr72_mission(cls, initial_mass: float = 25.0, initial_temperature: float = 53.25) -> "DischargeMission":
        parameters = IsochoricMissionParameters(initial_mass=initial_mass, initial_temperature=initial_temperature)
        atr72_mission = Mission.atr72()
        discharge_sections = []
        for i, section in enumerate(atr72_mission.sections):
            original_flow = section.fuel_flows[0]
            if isinstance(original_flow.mass_flow, list):
                discharge_rates = [abs(rate) for rate in original_flow.mass_flow]
                discharge_flow = OutFlow(discharge_rates, "gas")
            else:
                discharge_rate = abs(original_flow.mass_flow)
                discharge_flow = OutFlow(discharge_rate, "gas")
            discharge_section = MissionSection(duration=section.duration, fuel_flows=[discharge_flow], altitude=section.altitude, mach_number=section.mach_number, fuel_flow_key=getattr(section, 'fuel_flow_key', f'section_{i+1}'))
            discharge_sections.append(discharge_section)
        return cls(sections=discharge_sections, parameters=parameters)


class RefuelMission(IsochoricMission):
    def __init__(self, refuel_rate: float, duration: float, parameters: IsochoricMissionParameters = None, dewar_pressure: float = 3e5, pump_efficiency: float = 0.78):
        if parameters is None:
            parameters = IsochoricMissionParameters()
        class DummyHydrogen:
            pass
        refuel_section = MissionSection(duration=duration, fuel_flows=[InFlow(refuel_rate, DummyHydrogen())], altitude=0.0, mach_number=0.0)
        super().__init__([refuel_section], parameters, "REFUEL")
        self.refuel_rate = refuel_rate
        self.dewar_pressure = dewar_pressure
        self.pump_efficiency = pump_efficiency

    @classmethod
    def constant_refuel(cls, refuel_rate: float, duration: float, target_mass: float = 20.0, initial_temperature: float = 20.0) -> "RefuelMission":
        initial_mass = max(1.0, target_mass - refuel_rate * duration)
        parameters = IsochoricMissionParameters(initial_mass=initial_mass, initial_temperature=initial_temperature)
        return cls(refuel_rate, duration, parameters)


class DormancyMission(IsochoricMission):
    def __init__(self, duration: float, parameters: IsochoricMissionParameters = None, ambient_temperature: float = 288.15):
        if parameters is None:
            parameters = IsochoricMissionParameters()
        parameters.ambient_temperature = ambient_temperature
        dormancy_section = MissionSection(duration=duration, fuel_flows=[], altitude=0.0, mach_number=0.0)
        super().__init__([dormancy_section], parameters, "DORMANCY")

    @classmethod
    def long_term_storage(cls, duration: float, initial_mass: float = 10.0, initial_temperature: float = 20.0, ambient_temperature: float = 288.15) -> "DormancyMission":
        parameters = IsochoricMissionParameters(initial_mass=initial_mass, initial_temperature=initial_temperature, ambient_temperature=ambient_temperature)
        return cls(duration, parameters, ambient_temperature)


class IsochoricMissionAnalysis:
    def __init__(self, mission: IsochoricMission, thermal_model: IsochoricThermalModel):
        self.mission = mission
        self.thermal_model = thermal_model
        self.mission.set_thermal_model(thermal_model)
        self.results = None
        self.analysis_complete = False

    def run_analysis(self) -> IsochoricTankStates:
        initial_state = self.mission.create_initial_state()
        all_times = []
        all_states = []
        current_time = 0.0
        class MinimalTank:
            def __init__(self, volume):
                self.volume = volume
        minimal_tank = MinimalTank(self.mission.parameters.tank_volume)
        current_state = IsochoricTankState(tank=minimal_tank, fuel_mass=initial_state.fuel_mass, temperature=initial_state.temperature, solid_temperature=initial_state.solid_temperature, scenario=initial_state.scenario)
        for section in self.mission.sections:
            fuel_flow_func = self.mission.get_fuel_flow_function(section)
            discharge_flow_func = self.mission.get_discharge_flow_function(section)
            def ode_system(t, y):
                min_mass = 0.1
                min_temp = 10.0
                max_temp = 1000.0
                if y[0] <= min_mass:
                    y[0] = max(y[0], min_mass)
                    bounded_mass = True
                else:
                    bounded_mass = False
                y[1] = max(min(y[1], max_temp), min_temp)
                y[2] = max(min(y[2], max_temp), min_temp)
                if y[0] <= 0 or y[1] <= 0 or y[2] <= 0:
                    rho = max(y[0], 0.001) / self.mission.parameters.tank_volume
                    return np.array([-0.001, 0.001, 0.001])
                try:
                    state = IsochoricTankState(tank=minimal_tank, fuel_mass=y[0], temperature=y[1], solid_temperature=y[2])
                except Exception:
                    return np.array([0.0, 0.0, 0.0])
                Q_solid = self.thermal_model.compute_heat_flux(current_time + t, state)
                dTs_dt = self.thermal_model.compute_solid_temperature_derivative(current_time + t, state)
                def section_fuel_flow_func(abs_time):
                    section_time = abs_time - current_time
                    return fuel_flow_func(section_time)
                def section_discharge_flow_func(abs_time):
                    section_time = abs_time - current_time
                    return discharge_flow_func(section_time)
                derivatives = self.mission.dynamic_model_switcher.compute_state_derivatives(current_time + t, state, section_fuel_flow_func, section_discharge_flow_func, Q_solid=Q_solid, dTs_dt=dTs_dt)
                mass_derivative = derivatives.fuel_mass_derivative
                if bounded_mass and mass_derivative < 0:
                    mass_derivative = 0.0
                temp_derivative = max(min(derivatives.temperature_derivative, 100.0), -100.0)
                solid_temp_derivative = max(min(derivatives.solid_temperature_derivative, 10.0), -10.0)
                return [mass_derivative, temp_derivative, solid_temp_derivative]
            y0 = [current_state.fuel_mass, current_state.temperature, current_state.solid_temperature]
            t_span = (0.0, section.duration)
            t_eval = np.arange(0.0, section.duration, self.mission.parameters.time_step)
            if len(t_eval) == 0 or t_eval[-1] < section.duration:
                t_eval = np.append(t_eval, section.duration)
            self.mission.integration_method.set_ode_function(ode_system)
            section_results = self.mission.integration_method.integrate_full(t_span, y0, t_eval, events=None)
            section_times = current_time + section_results.t
            section_states = []
            for i, t in enumerate(section_results.t):
                state = IsochoricTankState(tank=minimal_tank, fuel_mass=section_results.y[0][i], temperature=section_results.y[1][i], solid_temperature=section_results.y[2][i])
                section_states.append(state)
            all_times.extend(section_times)
            all_states.extend(section_states)
            current_time = section_times[-1]
            current_state = section_states[-1]
        self.results = IsochoricTankStates(states=all_states, timestep=self.mission.parameters.time_step)
        self.analysis_complete = True
        return self.results

    def get_results(self) -> Optional[IsochoricTankStates]:
        return self.results if self.analysis_complete else None
