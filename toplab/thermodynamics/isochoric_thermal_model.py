"""

Thermal models for isochoric tanks adapted from Stops framework (see 10.1016/j.cryogenics.2024.103826)

Author: Dante Raso
"""

from abc import ABC, abstractmethod

from CoolProp.CoolProp import PropsSI

from toplab.thermodynamics.tank_states import IsochoricTankState
from toplab.materials.nist_materials import NISTMetal, NISTComposite


class IsochoricThermalModel(ABC):
    @abstractmethod
    def compute_solid_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        pass

    @abstractmethod
    def compute_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        pass


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
