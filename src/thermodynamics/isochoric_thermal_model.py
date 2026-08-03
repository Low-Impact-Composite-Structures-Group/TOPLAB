"""

Thermal models for isochoric tanks adapted from Stops framework (see 10.1016/j.cryogenics.2024.103826)

Author: Dante Raso
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from CoolProp.CoolProp import PropsSI

from src.thermodynamics.tank_states import IsochoricTankState
from src.materials.nist_materials import NISTMetal, NISTComposite


class IsochoricThermalModel(ABC):
    @abstractmethod
    def compute_solid_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        pass

    @abstractmethod
    def compute_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        pass


@dataclass
class AlphaSParameters:
    k_s: float = 237.0
    rho_s: float = 2700.0
    c_s: float = 900.0

    @property
    def alpha_s(self) -> float:
        return self.k_s / (self.rho_s * self.c_s)


@dataclass
class TankGeometry:
    volume: float = 0.5
    radius: float = None
    wall_thickness: float = 0.01
    surface_area: float = None

    def __post_init__(self):
        if self.radius is None:
            self.radius = (3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        if self.surface_area is None:
            self.surface_area = 4.0 * np.pi * self.radius ** 2


class CoupledSolidFluidThermalModel(IsochoricThermalModel):
    def __init__(self, alpha_s_params: AlphaSParameters = None, tank_geometry: TankGeometry = None, ambient_temperature: float = 288.15, heat_transfer_coefficient: float = 100.0):
        self.alpha_s_params = alpha_s_params if alpha_s_params else AlphaSParameters()
        self.tank_geometry = tank_geometry if tank_geometry else TankGeometry()
        self.ambient_temperature = ambient_temperature
        self.h_sf = heat_transfer_coefficient
        self.heat_flux_history = []
        self.temperature_history = []

    def compute_solid_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        Q_sf = self.compute_heat_flux(time, state, **kwargs)
        Q_amb = self._compute_ambient_heat_flux(state.solid_temperature, **kwargs)
        solid_mass = self._compute_solid_mass()
        thermal_capacity = solid_mass * self.alpha_s_params.c_s
        return (Q_amb - Q_sf) / thermal_capacity

    def compute_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        T_fluid = state.temperature
        T_solid = state.solid_temperature
        Q_sf = self.h_sf * self.tank_geometry.surface_area * (T_solid - T_fluid)
        self.heat_flux_history.append({'time': time, 'Q_sf': Q_sf, 'T_solid': T_solid, 'T_fluid': T_fluid})
        return Q_sf

    def _compute_ambient_heat_flux(self, T_solid: float, **kwargs) -> float:
        ambient_htc = kwargs.get('ambient_htc', 1.0)
        return ambient_htc * self.tank_geometry.surface_area * (self.ambient_temperature - T_solid)

    def _compute_solid_mass(self) -> float:
        r_outer = self.tank_geometry.radius
        r_inner = r_outer - self.tank_geometry.wall_thickness
        volume_shell = (4.0 / 3.0) * np.pi * (r_outer ** 3 - r_inner ** 3)
        return self.alpha_s_params.rho_s * volume_shell

    def get_thermal_properties(self) -> dict:
        return {
            'alpha_s': self.alpha_s_params.alpha_s,
            'k_s': self.alpha_s_params.k_s,
            'rho_s': self.alpha_s_params.rho_s,
            'c_s': self.alpha_s_params.c_s,
            'solid_mass': self._compute_solid_mass(),
            'surface_area': self.tank_geometry.surface_area,
            'h_sf': self.h_sf,
        }


class StopsModelThermalModel(IsochoricThermalModel):
    def __init__(self, tank_volume: float = 0.5, inner_surface_area: float = 4.0, outer_surface_area: float = 4.1, inner_diameter: float = 1.0, ambient_temperature: float = 298.15, ambient_htc: float = 0.025, liner_mass: float = 100.0, wall_mass: float = 150.0):
        self.V_t = tank_volume
        self.A_in = inner_surface_area
        self.A_out = outer_surface_area
        self.diameter = inner_diameter
        self.T_amb = ambient_temperature
        self.k_amb = ambient_htc
        self.m_liner = liner_mass
        self.m_wall = wall_mass
        self.liner_material = NISTMetal.aluminum_6061T6_nist()
        self.wall_material = NISTComposite.g10_nist(winding_angle=0.0)

    def get_alpha_s(self, T_fluid: float, T_solid: float, pressure: float = 101325) -> float:
        T_film = max(min(0.5 * (T_fluid + T_solid), 1000.0), 15.0)
        try:
            k = PropsSI('L', 'T', T_film, 'P', pressure, 'hydrogen')
            mu = PropsSI('V', 'T', T_film, 'P', pressure, 'hydrogen')
            rho = PropsSI('D', 'T', T_film, 'P', pressure, 'hydrogen')
            cp = PropsSI('C', 'T', T_film, 'P', pressure, 'hydrogen')
            nu = mu / rho
            alpha = k / (rho * cp)
            Pr = nu / alpha
            beta = 1.0 / T_film
            Ra_D = 9.81 * beta * abs(T_solid - T_fluid) * self.diameter ** 3 / (nu * alpha)
            Nu_D = (0.60 + (0.387 * Ra_D ** (1 / 6)) / ((1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27))) ** 2
            h = Nu_D * k / self.diameter
            return h
        except Exception:
            return 150.0

    def compute_heat_flux(self, time: float, state, **kwargs) -> float:
        T_fluid = state.temperature
        T_solid = state.solid_temperature
        pressure = getattr(getattr(state, 'hydrogen', None), 'pressure', 101325)
        alpha_s = self.get_alpha_s(T_fluid, T_solid, pressure)
        return alpha_s * self.A_in * (T_solid - T_fluid)

    def compute_solid_temperature_derivative(self, time: float, state, **kwargs) -> float:
        T_solid = state.solid_temperature
        Q_amb = self.k_amb * self.A_out * (self.T_amb - T_solid)
        Q_s = self.compute_heat_flux(time, state, **kwargs)
        T_solid_bounded = max(4.0, min(T_solid, 300.0))
        c_liner = float(self.liner_material.determine_specific_heat(T_solid_bounded))
        c_wall = float(self.wall_material.determine_specific_heat(T_solid_bounded))
        thermal_capacity = self.m_liner * c_liner + self.m_wall * c_wall
        return (Q_amb - Q_s) / thermal_capacity

    def calculate_thermal_equilibrium_Ts(self, T_fluid: float) -> float:
        alpha_s_approx = 150.0
        numerator = self.k_amb * self.A_out * self.T_amb + alpha_s_approx * self.A_in * T_fluid
        denominator = self.k_amb * self.A_out + alpha_s_approx * self.A_in
        return numerator / denominator


def create_default_coupled_thermal_model(tank_volume: float = 0.5, ambient_temperature: float = 288.15, wall_material: str = "aluminum") -> CoupledSolidFluidThermalModel:
    if wall_material.lower() == "aluminum":
        alpha_s_params = AlphaSParameters(k_s=237.0, rho_s=2700.0, c_s=900.0)
    elif wall_material.lower() == "steel":
        alpha_s_params = AlphaSParameters(k_s=50.0, rho_s=7850.0, c_s=450.0)
    else:
        alpha_s_params = AlphaSParameters()
    tank_geometry = TankGeometry(volume=tank_volume)
    return CoupledSolidFluidThermalModel(alpha_s_params=alpha_s_params, tank_geometry=tank_geometry, ambient_temperature=ambient_temperature, heat_transfer_coefficient=100.0)
