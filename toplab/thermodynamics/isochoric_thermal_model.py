"""
Thermal models for isochoric tanks.

The InsulatedTankThermalModel implements the five-state thermal network:

    H2  ←(Q_structure_to_h2)←  structure  ←(Q_insulation_to_structure)←
    insulation  ←(Q_shell_to_insulation)←  shell  ←(Q_ambient_to_shell)← ambient

Structure (liner + composite), insulation (Rohacell foam), and shell each carry
an ODE state. The insulation layer is divided at its equal-resistance midpoint;
each half-layer heat flow integrates the temperature-dependent conductivity.

Author: Dante Raso
"""

import math
from abc import ABC, abstractmethod

from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq
from toplab.thermodynamics.tank_states import IsochoricTankState
from toplab.materials.nist_materials import NISTMaterial
from toplab.materials.rohacell_properties import (
    specific_heat as rohacell_cp,
    integrated_thermal_conductivity,
    thermal_conductivity as rohacell_k,
)


class IsochoricThermalModel(ABC):
    @abstractmethod
    def compute_structure_to_h2_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """Heat flow from structure to H2 [W]. Positive when structure is warmer."""

    @abstractmethod
    def compute_structure_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """dT_structure/dt [K/s]."""

    @abstractmethod
    def compute_insulation_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """dT_insulation/dt [K/s]."""

    @abstractmethod
    def compute_shell_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """dT_shell/dt [K/s]."""


class InsulatedTankThermalModel(IsochoricThermalModel):
    """
    Five-state tank thermal model with transient insulation.

    Geometry layers (inside → outside):
      fluid | liner + wall (structure) | Rohacell foam (insulation) | Al shell | ambient

        ODEs:
            dT_structure/dt  = (Q_insulation_to_structure − Q_structure_to_h2) / C_structure
            dT_insulation/dt = (Q_shell_to_insulation − Q_insulation_to_structure) / C_insulation
            dT_shell/dt      = (Q_ambient_to_shell − Q_shell_to_insulation) / C_shell

    Algebraic:
    Q_structure_to_h2 = α_s · A_in · (T_structure − T_H2)        [inner natural convection]
    Q_shell_to_insulation and Q_insulation_to_structure use the
    temperature-dependent Fourier conductivity integral.
    Q_ambient_to_shell = α_amb · A_shell · (T_amb − T_shell)
                 + ε · σ · A_shell · (T_amb⁴ − T_shell⁴)  [conv + radiation]
    """

    STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴

    def __init__(
        self,
        tank_volume: float,
        inner_surface_area: float,
        inner_diameter: float,
        r_structure: float,      # outer radius of liner+wall [m]
        r_shell: float,          # r_structure + t_insulation [m]
        cylinder_length: float,  # cylindrical section length L [m]
        liner_mass: float,
        wall_mass: float,
        foam_mass: float,
        shell_mass: float,
        ambient_temperature: float,
        alpha_amb: float,        # convective HTC ambient→shell [W/m²K]
        emissivity_shell: float, # shell surface emissivity [−]
        liner_material: NISTMaterial,
        wall_material: NISTMaterial,
        shell_material: NISTMaterial,
    ):
        if r_shell <= r_structure:
            raise ValueError(
                f"Shell radius {r_shell:.4f} m must be greater than structure radius "
                f"{r_structure:.4f} m (check insulation thickness)"
            )
        self.V_t = tank_volume
        self.A_in = inner_surface_area
        self.diameter = inner_diameter
        self.r_structure = r_structure
        self.r_shell = r_shell
        self.L = cylinder_length
        self.m_liner = liner_mass
        self.m_wall = wall_mass
        self.m_foam = foam_mass
        self.m_shell = shell_mass
        self.T_amb = ambient_temperature
        self.alpha_amb = alpha_amb
        self.eps_shell = emissivity_shell
        self.liner_material = liner_material
        self.wall_material = wall_material
        self.shell_material = shell_material

        # A_shell = 2π r_shell L + 4π r_shell² (per governing equations)
        self.A_shell = (
            2.0 * math.pi * r_shell * cylinder_length
            + 4.0 * math.pi * r_shell ** 2
        )
        self.r_m_cyl = math.sqrt(r_structure * r_shell)
        self.r_m_sph = 2.0 * r_structure * r_shell / (r_structure + r_shell)

    # ------------------------------------------------------------------
    # Inner convection: structure ↔ fluid
    # ------------------------------------------------------------------

    def get_alpha_s(self, h2_temperature: float, structure_temperature: float, pressure: float) -> float:
        """Churchill-Chu natural convection HTC [W/m²K] for inner surface."""
        T_film = max(min(0.5 * (h2_temperature + structure_temperature), 1000.0), 15.0)
        try:
            k   = PropsSI('L', 'T', T_film, 'P', pressure, 'hydrogen')
            mu  = PropsSI('V', 'T', T_film, 'P', pressure, 'hydrogen')
            rho = PropsSI('D', 'T', T_film, 'P', pressure, 'hydrogen')
            cp  = PropsSI('C', 'T', T_film, 'P', pressure, 'hydrogen')
        except ValueError as e:
            if 'saturation' not in str(e).lower():
                raise
            # (T_film, P) is on the saturation line; evaluate as saturated liquid
            k   = PropsSI('L', 'T', T_film, 'Q', 0, 'hydrogen')
            mu  = PropsSI('V', 'T', T_film, 'Q', 0, 'hydrogen')
            rho = PropsSI('D', 'T', T_film, 'Q', 0, 'hydrogen')
            cp  = PropsSI('C', 'T', T_film, 'Q', 0, 'hydrogen')
        nu     = mu / rho
        alpha  = k / (rho * cp)
        Pr     = nu / alpha
        beta   = 1.0 / T_film
        Ra_D   = 9.81 * beta * abs(structure_temperature - h2_temperature) * self.diameter ** 3 / (nu * alpha)
        Nu_D   = (0.60 + (0.387 * Ra_D ** (1 / 6)) / ((1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27))) ** 2
        return Nu_D * k / self.diameter

    def compute_structure_to_h2_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """Heat flow from structure to H2 [W]. Positive when structure is warmer."""
        pressure = state.hydrogen.pressure if state.hydrogen is not None else state.pressure
        alpha_s = self.get_alpha_s(state.h2_temperature, state.structure_temperature, pressure)
        return alpha_s * self.A_in * (state.structure_temperature - state.h2_temperature)

    # ------------------------------------------------------------------
    # Insulation: Rohacell foam (variable-conductivity Fourier conduction)
    # ------------------------------------------------------------------

    @staticmethod
    def _integrated_conductivity(temperature_low: float, temperature_high: float) -> float:
        return integrated_thermal_conductivity(temperature_low, temperature_high)

    def compute_shell_to_insulation_heat_flux(self, shell_temperature: float, insulation_temperature: float) -> float:
        """Heat flow from shell to insulation [W]. Positive when shell is warmer."""
        geometry_factor = (
            2.0 * math.pi * self.L / math.log(self.r_shell / self.r_m_cyl)
            + 4.0 * math.pi * self.r_shell * self.r_m_sph / (self.r_shell - self.r_m_sph)
        )
        return geometry_factor * self._integrated_conductivity(insulation_temperature, shell_temperature)

    def compute_insulation_to_structure_heat_flux(self, insulation_temperature: float, structure_temperature: float) -> float:
        """Heat flow from insulation to structure [W]. Positive when insulation is warmer."""
        geometry_factor = (
            2.0 * math.pi * self.L / math.log(self.r_m_cyl / self.r_structure)
            + 4.0 * math.pi * self.r_structure * self.r_m_sph / (self.r_m_sph - self.r_structure)
        )
        return geometry_factor * self._integrated_conductivity(structure_temperature, insulation_temperature)

    def determine_initial_insulation_temperature(
        self, structure_temperature: float, shell_temperature: float
    ) -> float:
        """Return the insulation temperature that balances its two initial heat flows."""
        if structure_temperature == shell_temperature:
            return structure_temperature

        lower = min(structure_temperature, shell_temperature)
        upper = max(structure_temperature, shell_temperature)
        return brentq(
            lambda insulation_temperature: (
                self.compute_shell_to_insulation_heat_flux(shell_temperature, insulation_temperature)
                - self.compute_insulation_to_structure_heat_flux(insulation_temperature, structure_temperature)
            ),
            lower,
            upper,
            xtol=1e-8,
        )

    # ------------------------------------------------------------------
    # Ambient: convection + radiation to/from shell
    # ------------------------------------------------------------------

    def compute_ambient_to_shell_heat_flux(self, shell_temperature: float) -> float:
        """Heat flow from ambient to shell [W]. Positive when ambient is warmer."""
        Q_conv = self.alpha_amb * self.A_shell * (self.T_amb - shell_temperature)
        Q_rad  = (
            self.eps_shell * self.STEFAN_BOLTZMANN * self.A_shell
            * (self.T_amb ** 4 - shell_temperature ** 4)
        )
        return Q_conv + Q_rad

    # ------------------------------------------------------------------
    # ODEs
    # ------------------------------------------------------------------

    def compute_structure_temperature_derivative(
        self, time: float, state: IsochoricTankState, **kwargs
    ) -> float:
        """dT_structure/dt [K/s]. Structure gains from insulation and loses to H2."""
        structure_temperature = state.structure_temperature
        Q_insulation_to_structure = self.compute_insulation_to_structure_heat_flux(
            state.insulation_temperature, structure_temperature
        )
        Q_structure_to_h2 = self.compute_structure_to_h2_heat_flux(time, state, **kwargs)
        T_bounded = max(4.0, min(structure_temperature, 400.0))
        c_liner = float(self.liner_material.determine_specific_heat(T_bounded))
        c_wall  = float(self.wall_material.determine_specific_heat(T_bounded))
        thermal_capacity = self.m_liner * c_liner + self.m_wall * c_wall
        return (Q_insulation_to_structure - Q_structure_to_h2) / thermal_capacity

    def compute_insulation_temperature_derivative(
        self, time: float, state: IsochoricTankState, **kwargs
    ) -> float:
        """dT_insulation/dt [K/s]."""
        insulation_temperature = state.insulation_temperature
        Q_shell_to_insulation = self.compute_shell_to_insulation_heat_flux(
            state.shell_temperature, insulation_temperature
        )
        Q_insulation_to_structure = self.compute_insulation_to_structure_heat_flux(
            insulation_temperature, state.structure_temperature
        )
        heat_capacity = self.m_foam * float(rohacell_cp(max(4.0, min(insulation_temperature, 400.0))))
        return (Q_shell_to_insulation - Q_insulation_to_structure) / heat_capacity

    def compute_shell_temperature_derivative(
        self, time: float, state: IsochoricTankState, **kwargs
    ) -> float:
        """dT_shell/dt [K/s]. Shell gains from ambient and loses to insulation."""
        shell_temperature = state.shell_temperature
        Q_ambient_to_shell = self.compute_ambient_to_shell_heat_flux(shell_temperature)
        Q_shell_to_insulation = self.compute_shell_to_insulation_heat_flux(
            shell_temperature, state.insulation_temperature
        )
        T_bounded = max(4.0, min(shell_temperature, 400.0))
        c_shell = float(self.shell_material.determine_specific_heat(T_bounded))
        return (Q_ambient_to_shell - Q_shell_to_insulation) / (self.m_shell * c_shell)
