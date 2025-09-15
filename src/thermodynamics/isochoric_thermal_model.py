"""
Isochoric Thermal Model for stops_model integration with HFT framework.

This module implements the solid temperature evolution and heat flux calculation
for the stops_model approach. It handles:
- Solid temperature derivatives (dTs/dt)
- Heat flux from solid to fluid (Q_solid)
- Alpha_s thermal diffusivity calculations
- Coupling with isochoric dynamic models

Integration with HFT Framework:
Victor Kees Poorte, 2025
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Optional, Union, Tuple
import numpy as np

from CoolProp.CoolProp import PropsSI

from src.thermodynamics.tank_states import IsochoricTankState
from src.materials.nist_materials import NISTMetal, NISTComposite


class IsochoricThermalModel(ABC):
    """
    Base class for isochoric thermal models.

    This class defines the interface for computing solid temperature
    derivatives and heat flux between solid and fluid in the stops_model
    approach.
    """

    @abstractmethod
    def compute_solid_temperature_derivative(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """
        Compute solid temperature derivative dTs/dt.

        Args:
            time: Current time [s]
            state: Current tank state with [m, T, Ts]
            **kwargs: Additional model parameters

        Returns:
            float: Solid temperature derivative [K/s]
        """
        pass

    @abstractmethod
    def compute_heat_flux(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """
        Compute heat flux from solid to fluid.

        Args:
            time: Current time [s]
            state: Current tank state
            **kwargs: Additional model parameters

        Returns:
            float: Heat flux from solid to fluid [W]
        """
        pass


@dataclass
class AlphaSParameters:
    """Parameters for alpha_s thermal diffusivity calculation"""
    k_s: float = 237.0  # Solid thermal conductivity [W/m/K] (aluminum)
    rho_s: float = 2700.0  # Solid density [kg/m³] (aluminum)
    c_s: float = 900.0  # Solid specific heat capacity [J/kg/K] (aluminum)

    @property
    def alpha_s(self) -> float:
        """Compute thermal diffusivity [m²/s]"""
        return self.k_s / (self.rho_s * self.c_s)


@dataclass
class TankGeometry:
    """Tank geometry parameters for heat transfer calculations"""
    volume: float = 0.5  # Tank volume [m³]
    radius: float = None  # Tank radius [m] (computed if None)
    wall_thickness: float = 0.01  # Wall thickness [m]
    surface_area: float = None  # Surface area [m²] (computed if None)

    def __post_init__(self):
        """Compute derived parameters"""
        if self.radius is None:
            # Assume spherical tank
            self.radius = (3.0 * self.volume / (4.0 * np.pi)) ** (1.0/3.0)

        if self.surface_area is None:
            # Spherical surface area
            self.surface_area = 4.0 * np.pi * self.radius**2


class CoupledSolidFluidThermalModel(IsochoricThermalModel):
    """
    Coupled solid-fluid thermal model for stops_model.

    This model implements the heat transfer between the tank wall (solid)
    and the hydrogen fluid, based on the thermal diffusivity approach
    from the original stops_model analysis.

    Key features:
    - Computes alpha_s thermal diffusivity
    - Handles convective heat transfer at solid-fluid interface
    - Tracks solid temperature evolution
    - Provides heat flux coupling for dynamic model
    """

    def __init__(self,
                 alpha_s_params: AlphaSParameters = None,
                 tank_geometry: TankGeometry = None,
                 ambient_temperature: float = 288.15,
                 heat_transfer_coefficient: float = 100.0):
        """
        Initialize coupled solid-fluid thermal model.

        Args:
            alpha_s_params: Solid thermal properties
            tank_geometry: Tank geometry parameters
            ambient_temperature: External ambient temperature [K]
            heat_transfer_coefficient: Convective HTC between solid and fluid [W/m²/K]
        """
        self.alpha_s_params = alpha_s_params if alpha_s_params else AlphaSParameters()
        self.tank_geometry = tank_geometry if tank_geometry else TankGeometry()
        self.ambient_temperature = ambient_temperature
        self.h_sf = heat_transfer_coefficient  # Solid-fluid HTC

        # Storage for computation history
        self.heat_flux_history = []
        self.temperature_history = []

    def compute_solid_temperature_derivative(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """
        Compute solid temperature derivative using thermal diffusivity approach.

        This implements the heat conduction equation in the solid wall:
        dTs/dt = alpha_s * d²Ts/dr² + (2/r) * dTs/dr - heat_loss_to_fluid/capacity

        Simplified approach assumes uniform solid temperature with
        convective boundary conditions.
        """
        T_fluid = state.temperature
        T_solid = state.solid_temperature

        # Heat flux from solid to fluid [W]
        Q_sf = self.compute_heat_flux(time, state, **kwargs)

        # Heat flux from ambient to solid (external heat input) [W]
        Q_amb = self._compute_ambient_heat_flux(T_solid, **kwargs)

        # Solid thermal mass [J/K]
        solid_mass = self._compute_solid_mass()
        thermal_capacity = solid_mass * self.alpha_s_params.c_s

        # Energy balance on solid: dTs/dt = (Q_in - Q_out) / (m_s * c_s)
        dTs_dt = (Q_amb - Q_sf) / thermal_capacity

        return dTs_dt

    def compute_heat_flux(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """
        Compute heat flux from solid to fluid using convective model.

        Q = h_sf * A * (T_solid - T_fluid)
        """
        T_fluid = state.temperature
        T_solid = state.solid_temperature

        # Convective heat transfer
        Q_sf = self.h_sf * self.tank_geometry.surface_area * (T_solid - T_fluid)

        # Store for history tracking
        self.heat_flux_history.append({
            'time': time,
            'Q_sf': Q_sf,
            'T_solid': T_solid,
            'T_fluid': T_fluid
        })

        return Q_sf

    def _compute_ambient_heat_flux(self, T_solid: float, **kwargs) -> float:
        """
        Compute heat flux from ambient to solid.

        This can be customized based on mission conditions, insulation, etc.
        """
        # Simple model: constant ambient heat input
        ambient_htc = kwargs.get('ambient_htc', 1.0)  # [W/m²/K]
        Q_amb = ambient_htc * self.tank_geometry.surface_area * (self.ambient_temperature - T_solid)

        return Q_amb

    def _compute_solid_mass(self) -> float:
        """Compute solid wall mass assuming spherical shell"""
        r_outer = self.tank_geometry.radius
        r_inner = r_outer - self.tank_geometry.wall_thickness

        # Spherical shell volume
        volume_shell = (4.0/3.0) * np.pi * (r_outer**3 - r_inner**3)

        # Mass
        mass = self.alpha_s_params.rho_s * volume_shell

        return mass

    def get_thermal_properties(self) -> dict:
        """Get current thermal properties for analysis"""
        return {
            'alpha_s': self.alpha_s_params.alpha_s,
            'k_s': self.alpha_s_params.k_s,
            'rho_s': self.alpha_s_params.rho_s,
            'c_s': self.alpha_s_params.c_s,
            'solid_mass': self._compute_solid_mass(),
            'surface_area': self.tank_geometry.surface_area,
            'h_sf': self.h_sf
        }


class StopsModelThermalModel(IsochoricThermalModel):
    """
    Thermal model implementing the exact stops_model physics.

    This implements the proper coupled heat transfer with:
    - Churchill & Chu natural convection correlation for alpha_s
    - NIST material properties for liner (aluminum) and wall (G10 composite)
    - Proper heat transfer coupling: Qdot_amb -> solid -> fluid
    - Mass-weighted thermal capacity calculations
    """

    def __init__(self,
                 tank_volume: float = 0.5,
                 inner_surface_area: float = 4.0,
                 outer_surface_area: float = 4.1,
                 inner_diameter: float = 1.0,
                 ambient_temperature: float = 298.15,
                 ambient_htc: float = 0.025,
                 liner_mass: float = 100.0,
                 wall_mass: float = 150.0):
        """
        Initialize stops_model thermal model.

        Args:
            tank_volume: Tank volume [m³]
            inner_surface_area: Inner surface area A_in [m²]
            outer_surface_area: Outer surface area A_out [m²]
            inner_diameter: Inner diameter for alpha_s calculation [m]
            ambient_temperature: Ambient temperature T_amb [K]
            ambient_htc: Ambient heat transfer coefficient k_amb [W/m²K]
            liner_mass: Aluminum liner mass [kg]
            wall_mass: G10 composite wall mass [kg]
        """
        self.V_t = tank_volume
        self.A_in = inner_surface_area
        self.A_out = outer_surface_area
        self.diameter = inner_diameter
        self.T_amb = ambient_temperature
        self.k_amb = ambient_htc
        self.m_liner = liner_mass
        self.m_wall = wall_mass

        # Initialize NIST materials for temperature-dependent properties
        self.liner_material = NISTMetal.aluminum_6061T6_nist()
        self.wall_material = NISTComposite.g10_nist(winding_angle=0.0)

    def get_alpha_s(self, T_fluid: float, T_solid: float, pressure: float = 101325) -> float:
        """
        Compute convective heat transfer coefficient using Churchill & Chu correlation.

        This is the exact implementation from stops_model get_alpha_s() function.
        """
        from CoolProp.CoolProp import PropsSI

        # Film temperature
        T_film = 0.5 * (T_fluid + T_solid)

        # Ensure reasonable film temperature bounds
        T_film = max(T_film, 15.0)  # Above hydrogen triple point
        T_film = min(T_film, 1000.0)  # Reasonable upper bound

        try:
            # Fluid properties at film temperature using hydrogen
            k = PropsSI('L', 'T', T_film, 'P', pressure, 'hydrogen')    # W/m-K
            mu = PropsSI('V', 'T', T_film, 'P', pressure, 'hydrogen')   # Pa·s
            rho = PropsSI('D', 'T', T_film, 'P', pressure, 'hydrogen')  # kg/m^3
            cp = PropsSI('C', 'T', T_film, 'P', pressure, 'hydrogen')   # J/kg-K

            # Derived properties
            nu = mu / rho                  # kinematic viscosity [m^2/s]
            alpha = k / (rho * cp)         # thermal diffusivity [m^2/s]
            Pr = nu / alpha                # Prandtl number
            beta = 1.0 / T_film            # thermal expansion coeff (ideal gas approx)

            # Rayleigh number (based on diameter for horizontal cylinder)
            Ra_D = 9.81 * beta * abs(T_solid - T_fluid) * self.diameter**3 / (nu * alpha)

            # Churchill & Chu correlation for horizontal cylinder
            Nu_D = (0.60 + (0.387 * Ra_D**(1/6)) /
                   ((1 + (0.559/Pr)**(9/16))**(8/27)))**2

            # Heat transfer coefficient
            h = Nu_D * k / self.diameter
            return h

        except Exception as e:
            # Fallback to constant value if CoolProp fails
            return 150.0  # W/m²K - reasonable fallback from stops_model

    def compute_heat_flux(self, time: float, state, **kwargs) -> float:
        """
        Compute heat flux from solid to fluid: Q_s = alpha_s * A_in * (Ts - T)
        """
        T_fluid = state.temperature
        T_solid = state.solid_temperature

        # Get pressure for alpha_s calculation
        try:
            pressure = state.hydrogen.pressure  # Use actual pressure
        except:
            pressure = 101325  # Fallback to atmospheric

        # Calculate alpha_s using Churchill & Chu correlation
        alpha_s = self.get_alpha_s(T_fluid, T_solid, pressure)

        # Heat flux from solid to fluid
        Q_s = alpha_s * self.A_in * (T_solid - T_fluid)

        return Q_s

    def compute_solid_temperature_derivative(self, time: float, state, **kwargs) -> float:
        """
        Compute solid temperature derivative: dTs/dt = (Q_amb - Q_s) / (m_s * c_s)

        This implements the exact stops_model thermal balance equation.
        """
        T_fluid = state.temperature
        T_solid = state.solid_temperature

        # Heat flux from ambient to solid: Q_amb = k_amb * A_out * (T_amb - Ts)
        Q_amb = self.k_amb * self.A_out * (self.T_amb - T_solid)

        # Heat flux from solid to fluid (already computed)
        Q_s = self.compute_heat_flux(time, state, **kwargs)

        # Temperature-dependent specific heats using NIST data with bounds checking
        T_solid_bounded = max(4.0, min(T_solid, 300.0))  # Keep within NIST range

        c_liner = float(self.liner_material.determine_specific_heat(T_solid_bounded))
        c_wall = float(self.wall_material.determine_specific_heat(T_solid_bounded))

        # Mass-weighted thermal capacity: m_s * c_s = m_liner * c_liner + m_wall * c_wall
        thermal_capacity = self.m_liner * c_liner + self.m_wall * c_wall

        # Energy balance on solid: dTs/dt = (Q_in - Q_out) / (thermal_capacity)
        dTs_dt = (Q_amb - Q_s) / thermal_capacity

        return dTs_dt

    def calculate_thermal_equilibrium_Ts(self, T_fluid: float) -> float:
        """
        Calculate initial solid temperature for thermal equilibrium.

        At equilibrium: dTs/dt = 0, so Q_amb = Q_s
        k_amb * A_out * (T_amb - Ts) = alpha_s * A_in * (Ts - T_fluid)

        This matches stops_model calculate_thermal_equilibrium_Ts() function.
        """
        # Use approximate alpha_s for equilibrium calculation
        alpha_s_approx = 150.0  # W/m²K - typical value from stops_model

        # Solve: k_amb*A_out*T_amb + alpha_s*A_in*T_fluid = Ts*(k_amb*A_out + alpha_s*A_in)
        numerator = self.k_amb * self.A_out * self.T_amb + alpha_s_approx * self.A_in * T_fluid
        denominator = self.k_amb * self.A_out + alpha_s_approx * self.A_in

        Ts_equilibrium = numerator / denominator

        return Ts_equilibrium


class SimplifiedIsochoricThermalModel(IsochoricThermalModel):
    """
    Simplified thermal model for basic isochoric analysis.

    This model provides simplified heat transfer calculations
    without detailed solid temperature evolution, useful for
    scenarios where solid thermal dynamics are not critical.
    """

    def __init__(self,
                 constant_heat_flux: float = 100.0,
                 solid_temperature_rate: float = 0.0):
        """
        Initialize simplified thermal model.

        Args:
            constant_heat_flux: Constant heat flux from solid to fluid [W]
            solid_temperature_rate: Constant solid temperature change rate [K/s]
        """
        self.constant_heat_flux = constant_heat_flux
        self.solid_temperature_rate = solid_temperature_rate

    def compute_solid_temperature_derivative(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """Return constant solid temperature derivative"""
        return self.solid_temperature_rate

    def compute_heat_flux(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """Return constant heat flux"""
        return self.constant_heat_flux


class MissionBasedThermalModel(IsochoricThermalModel):
    """
    Mission-based thermal model that switches behavior based on scenario.

    This model adapts the thermal behavior based on the current mission
    phase (DISCHARGE, REFUEL, DORMANCY) with different heat transfer
    characteristics for each phase.
    """

    def __init__(self,
                 discharge_model: IsochoricThermalModel,
                 refuel_model: IsochoricThermalModel,
                 dormancy_model: IsochoricThermalModel,
                 current_scenario: str = "DISCHARGE"):
        """
        Initialize mission-based thermal model.

        Args:
            discharge_model: Thermal model for discharge phase
            refuel_model: Thermal model for refuel phase
            dormancy_model: Thermal model for dormancy phase
            current_scenario: Current scenario name
        """
        self.models = {
            "DISCHARGE": discharge_model,
            "REFUEL": refuel_model,
            "DORMANCY": dormancy_model
        }
        self.current_scenario = current_scenario

    def set_scenario(self, scenario: str):
        """Change current scenario"""
        if scenario in self.models:
            self.current_scenario = scenario
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def compute_solid_temperature_derivative(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """Delegate to current scenario model"""
        model = self.models[self.current_scenario]
        return model.compute_solid_temperature_derivative(time, state, **kwargs)

    def compute_heat_flux(
        self,
        time: float,
        state: IsochoricTankState,
        **kwargs
    ) -> float:
        """Delegate to current scenario model"""
        model = self.models[self.current_scenario]
        return model.compute_heat_flux(time, state, **kwargs)


def create_default_coupled_thermal_model(
    tank_volume: float = 0.5,
    ambient_temperature: float = 288.15,
    wall_material: str = "aluminum"
) -> CoupledSolidFluidThermalModel:
    """
    Create a default coupled thermal model with standard parameters.

    Args:
        tank_volume: Tank volume [m³]
        ambient_temperature: Ambient temperature [K]
        wall_material: Wall material type

    Returns:
        CoupledSolidFluidThermalModel: Configured thermal model
    """
    # Material properties
    if wall_material.lower() == "aluminum":
        alpha_s_params = AlphaSParameters(
            k_s=237.0,      # W/m/K
            rho_s=2700.0,   # kg/m³
            c_s=900.0       # J/kg/K
        )
    elif wall_material.lower() == "steel":
        alpha_s_params = AlphaSParameters(
            k_s=50.0,       # W/m/K
            rho_s=7850.0,   # kg/m³
            c_s=450.0       # J/kg/K
        )
    else:
        # Default to aluminum
        alpha_s_params = AlphaSParameters()

    # Tank geometry
    tank_geometry = TankGeometry(volume=tank_volume)

    # Create model
    return CoupledSolidFluidThermalModel(
        alpha_s_params=alpha_s_params,
        tank_geometry=tank_geometry,
        ambient_temperature=ambient_temperature,
        heat_transfer_coefficient=100.0  # W/m²/K
    )


def main():
    pass


if __name__ == "__main__":
    main()


# End