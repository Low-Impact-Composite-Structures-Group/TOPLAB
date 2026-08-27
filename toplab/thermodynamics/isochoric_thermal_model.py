"""
Thermal models for isochoric tanks.

The InsulatedTankThermalModel implements the four-layer thermal network:

    fluid  ←(Q_structure)←  structure  ←(Q_insulation)←  shell  ←(Q_amb)←  ambient

Structure (liner + composite) and shell each carry an ODE state.
Insulation (Rohacell foam) is a static conductive path with k(T_mean).

Author: Dante Raso
"""

import math
from abc import ABC, abstractmethod

from CoolProp.CoolProp import PropsSI

from toplab.thermodynamics.tank_states import IsochoricTankState
from toplab.materials.nist_materials import NISTMaterial
from toplab.materials.rohacell_properties import thermal_conductivity as rohacell_k


class IsochoricThermalModel(ABC):
    @abstractmethod
    def compute_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """Q_structure: heat from structure to fluid [W]. Positive when T_structure > T_fluid."""

    @abstractmethod
    def compute_structure_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """dT_structure/dt [K/s]."""

    @abstractmethod
    def compute_shell_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """dT_shell/dt [K/s]."""

    def compute_solid_temperature_derivative(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """Legacy alias for compute_structure_temperature_derivative."""
        return self.compute_structure_temperature_derivative(time, state, **kwargs)


class InsulatedTankThermalModel(IsochoricThermalModel):
    """
    Four-layer tank thermal model with physics-based insulation.

    Geometry layers (inside → outside):
      fluid | liner + wall (structure) | Rohacell foam (insulation) | Al shell | ambient

    ODEs:
      dT_structure/dt = (Q_insulation − Q_structure) / (m_liner·c_liner + m_wall·c_wall)
      dT_shell/dt     = (Q_amb − Q_insulation) / (m_shell·c_shell)

    Algebraic:
      Q_structure  = α_s · A_in · (T_structure − T_fluid)          [inner natural convection]
      Q_insulation = Q_cyl + Q_cap                                  [Fourier through Rohacell]
      Q_amb        = α_amb · A_shell · (T_amb − T_shell)
                   + ε · σ · A_shell · (T_amb⁴ − T_shell⁴)        [conv + radiation]
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

    # ------------------------------------------------------------------
    # Inner convection: structure ↔ fluid
    # ------------------------------------------------------------------

    def get_alpha_s(self, T_fluid: float, T_structure: float, pressure: float) -> float:
        """Churchill-Chu natural convection HTC [W/m²K] for inner surface."""
        T_film = max(min(0.5 * (T_fluid + T_structure), 1000.0), 15.0)
        k   = PropsSI('L', 'T', T_film, 'P', pressure, 'hydrogen')
        mu  = PropsSI('V', 'T', T_film, 'P', pressure, 'hydrogen')
        rho = PropsSI('D', 'T', T_film, 'P', pressure, 'hydrogen')
        cp  = PropsSI('C', 'T', T_film, 'P', pressure, 'hydrogen')
        nu     = mu / rho
        alpha  = k / (rho * cp)
        Pr     = nu / alpha
        beta   = 1.0 / T_film
        Ra_D   = 9.81 * beta * abs(T_structure - T_fluid) * self.diameter ** 3 / (nu * alpha)
        Nu_D   = (0.60 + (0.387 * Ra_D ** (1 / 6)) / ((1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27))) ** 2
        return Nu_D * k / self.diameter

    def compute_heat_flux(self, time: float, state: IsochoricTankState, **kwargs) -> float:
        """Q_structure [W]: heat flux from structure to fluid. Positive when T_structure > T_fluid."""
        pressure = state.hydrogen.pressure if state.hydrogen is not None else state.pressure
        alpha_s = self.get_alpha_s(state.temperature, state.solid_temperature, pressure)
        return alpha_s * self.A_in * (state.solid_temperature - state.temperature)

    # ------------------------------------------------------------------
    # Insulation: Rohacell foam (Fourier conduction)
    # ------------------------------------------------------------------

    def compute_insulation_heat_flux(self, T_structure: float, T_shell: float) -> float:
        """Q_insulation [W]: heat through foam from shell to structure. Positive when T_shell > T_structure."""
        T_mid = 0.5 * (T_structure + T_shell)
        k_ins = rohacell_k(T_mid)
        r_s   = self.r_structure
        r_sh  = self.r_shell
        t_ins = r_sh - r_s
        # Cylindrical section
        Q_cyl = 2.0 * math.pi * self.L * k_ins / math.log(r_sh / r_s) * (T_shell - T_structure)
        # Two spherical endcaps (combined formula for concentric spherical shells)
        Q_cap = 4.0 * math.pi * k_ins * r_s * r_sh / t_ins * (T_shell - T_structure)
        return Q_cyl + Q_cap

    # ------------------------------------------------------------------
    # Ambient: convection + radiation to/from shell
    # ------------------------------------------------------------------

    def compute_ambient_heat_flux(self, T_shell: float) -> float:
        """Q_amb [W]: heat from ambient to shell. Positive when T_amb > T_shell."""
        Q_conv = self.alpha_amb * self.A_shell * (self.T_amb - T_shell)
        Q_rad  = (
            self.eps_shell * self.STEFAN_BOLTZMANN * self.A_shell
            * (self.T_amb ** 4 - T_shell ** 4)
        )
        return Q_conv + Q_rad

    # ------------------------------------------------------------------
    # ODEs
    # ------------------------------------------------------------------

    def compute_structure_temperature_derivative(
        self, time: float, state: IsochoricTankState, **kwargs
    ) -> float:
        """dT_structure/dt [K/s]. Structure gains from insulation, loses to fluid."""
        T_s  = state.solid_temperature
        T_sh = state.shell_temperature
        Q_insulation = self.compute_insulation_heat_flux(T_s, T_sh)
        Q_structure  = self.compute_heat_flux(time, state, **kwargs)
        T_bounded = max(4.0, min(T_s, 400.0))
        c_liner = float(self.liner_material.determine_specific_heat(T_bounded))
        c_wall  = float(self.wall_material.determine_specific_heat(T_bounded))
        thermal_capacity = self.m_liner * c_liner + self.m_wall * c_wall
        return (Q_insulation - Q_structure) / thermal_capacity

    def compute_shell_temperature_derivative(
        self, time: float, state: IsochoricTankState, **kwargs
    ) -> float:
        """dT_shell/dt [K/s]. Shell gains from ambient, loses to insulation."""
        T_s  = state.solid_temperature
        T_sh = state.shell_temperature
        Q_amb        = self.compute_ambient_heat_flux(T_sh)
        Q_insulation = self.compute_insulation_heat_flux(T_s, T_sh)
        T_bounded = max(4.0, min(T_sh, 400.0))
        c_shell = float(self.shell_material.determine_specific_heat(T_bounded))
        return (Q_amb - Q_insulation) / (self.m_shell * c_shell)
