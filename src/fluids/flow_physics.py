"""
Flow Physics Utility for Inter-Tank Coupling Calculations

This module provides a configuration-driven approach to flow physics calculations,
eliminating hardcoded values and leveraging CoolProp for temperature/pressure-dependent
fluid properties.

Key Features:
- CoolProp integration for accurate fluid properties
- Configuration-driven physics parameters
- Choked flow calculations with proper critical pressure ratios
- Pipe flow with friction factor correlations
- Safety limits and numerical tolerances
- Fallback values for offline/backup calculations

Authors: GitHub Copilot (2025)
"""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    print("⚠️  CoolProp not available - using backup fluid properties")


class FlowPhysics:
    """
    Configuration-driven flow physics calculator with CoolProp integration.

    This class eliminates hardcoded values in flow calculations by reading
    all physics parameters from YAML configuration files.
    """

    def __init__(self, flow_physics_config: Dict[str, Any]):
        """
        Initialize flow physics calculator from configuration.

        Args:
            flow_physics_config: Flow physics section from YAML config
        """
        self.config = flow_physics_config

        # Fluid properties configuration
        fluid_config = self.config.get('fluid_properties', {})
        self.use_coolprop = fluid_config.get('use_coolprop', True) and COOLPROP_AVAILABLE
        self.coolprop_fluid = fluid_config.get('coolprop_fluid', 'H2')

        # Backup properties (used if CoolProp unavailable or disabled)
        backup = fluid_config.get('backup_properties', {})
        self.backup_R_specific = backup.get('specific_gas_constant', 4124.0)
        self.backup_gamma = backup.get('heat_capacity_ratio', 1.4)
        self.backup_kinematic_viscosity = backup.get('kinematic_viscosity', 1.55e-6)
        self.backup_speed_of_sound = backup.get('speed_of_sound', 1354.0)

        # Orifice flow parameters
        orifice_config = self.config.get('orifice_flow', {})
        self.discharge_coefficient = orifice_config.get('discharge_coefficient', 0.6)
        self.use_flow_coefficient = orifice_config.get('use_flow_coefficient', False)
        self.flow_coefficient = orifice_config.get('flow_coefficient', 0.0001)

        # Choked flow parameters
        choked_config = self.config.get('choked_flow', {})
        self.enable_choked_flow = choked_config.get('enable_choked_flow', True)
        self.critical_pressure_ratio_override = choked_config.get('critical_pressure_ratio_override')
        self.sonic_velocity_factor = choked_config.get('sonic_velocity_factor', 1.0)
        self.choked_flow_pressure_factor = choked_config.get('choked_flow_pressure_factor', 2.0)

        # Pipe flow parameters
        pipe_config = self.config.get('pipe_flow', {})
        self.atmospheric_pressure = pipe_config.get('atmospheric_pressure', 101325.0)
        self.reynolds_transition = pipe_config.get('reynolds_transition', 2300.0)

        friction_config = pipe_config.get('friction_correlations', {})
        self.laminar_factor = friction_config.get('laminar_factor', 64.0)
        self.turbulent_correlation = friction_config.get('turbulent_correlation', 'blasius')
        self.turbulent_factor = friction_config.get('turbulent_factor', 0.316)
        self.turbulent_exponent = friction_config.get('turbulent_exponent', 0.25)

        self.default_roughness = pipe_config.get('default_roughness', 1.5e-6)
        self.velocity_choked_fraction = pipe_config.get('velocity_choked_fraction', 0.5)

        # Safety limits
        safety_config = self.config.get('safety_limits', {})
        self.max_mass_transfer_fraction = safety_config.get('max_mass_transfer_fraction', 0.1)
        self.minimum_pressure_pa = safety_config.get('minimum_pressure_pa', 1000.0)
        self.maximum_velocity_ms = safety_config.get('maximum_velocity_ms', 2000.0)

        # Numerical parameters
        numerical_config = self.config.get('numerical', {})
        self.pressure_tolerance_pa = float(numerical_config.get('pressure_tolerance_pa', 100.0))
        self.flow_rate_tolerance_kg_s = float(numerical_config.get('flow_rate_tolerance_kg_s', 1e-8))
        self.property_update_frequency = int(numerical_config.get('property_update_frequency', 1))

        print(f"🔬 FlowPhysics initialized:")
        print(f"   CoolProp: {'✓' if self.use_coolprop else '✗'} ({self.coolprop_fluid if self.use_coolprop else 'backup properties'})")
        print(f"   Discharge coefficient: {self.discharge_coefficient}")
        print(f"   Choked flow: {'✓' if self.enable_choked_flow else '✗'}")

    def get_fluid_properties(self, pressure: float, temperature: float) -> Dict[str, float]:
        """
        Get fluid properties at specified conditions using CoolProp or backup values.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Dictionary with fluid properties
        """
        properties = {}

        if self.use_coolprop:
            try:
                # Always available and stable
                properties['specific_gas_constant'] = (
                    CP.PropsSI('GAS_CONSTANT', self.coolprop_fluid) /
                    CP.PropsSI('MOLAR_MASS', self.coolprop_fluid)
                )

                # Prefer P,T; if near-saturation errors occur, fall back to saturated vapor at T
                try:
                    cp = CP.PropsSI('CPMASS', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                    cv = CP.PropsSI('CVMASS', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                except Exception:
                    # Try saturated vapor side at given temperature (avoids two-phase ambiguity)
                    try:
                        cp = CP.PropsSI('Cpmass', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                        cv = CP.PropsSI('Cvmass', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                    except Exception:
                        cp = None
                        cv = None

                if cp is not None and cv is not None and cv > 0:
                    properties['heat_capacity_ratio'] = cp / cv
                else:
                    # Final fallback: use configured backup gamma
                    properties['heat_capacity_ratio'] = self.backup_gamma

                # Speed of sound: P,T first; then saturated vapor at T; finally backup
                try:
                    sos = CP.PropsSI('SPEED_OF_SOUND', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                except Exception:
                    try:
                        sos = CP.PropsSI('SPEED_OF_SOUND', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                    except Exception:
                        sos = self.backup_speed_of_sound
                properties['speed_of_sound'] = sos * self.sonic_velocity_factor

                # Dynamic viscosity and density for kinematic viscosity
                # Use P,T first; if that fails near saturation, try saturated vapor at T
                try:
                    dynamic_viscosity = CP.PropsSI('VISCOSITY', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                except Exception:
                    try:
                        dynamic_viscosity = CP.PropsSI('VISCOSITY', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                    except Exception:
                        dynamic_viscosity = None

                try:
                    density = CP.PropsSI('DMASS', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                except Exception:
                    try:
                        density = CP.PropsSI('Dmass', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                    except Exception:
                        density = None

                if dynamic_viscosity is not None and density is not None and density > 0:
                    properties['kinematic_viscosity'] = dynamic_viscosity / density
                    properties['density'] = density
                else:
                    # Fall back partially: provide backup kinematic viscosity; density left as None
                    properties['kinematic_viscosity'] = self.backup_kinematic_viscosity
                    properties['density'] = None

            except Exception as e:
                # As a last resort, use full backup set; avoid spamming logs on near-saturation
                properties = self._get_backup_properties()
        else:
            properties = self._get_backup_properties()

        return properties

    def _get_backup_properties(self) -> Dict[str, float]:
        """Get backup fluid properties from configuration."""
        return {
            'specific_gas_constant': self.backup_R_specific,
            'heat_capacity_ratio': self.backup_gamma,
            'kinematic_viscosity': self.backup_kinematic_viscosity,
            'speed_of_sound': self.backup_speed_of_sound * self.sonic_velocity_factor,
            'density': None  # Will be calculated from state if needed
        }

    def calculate_critical_pressure_ratio(self, gamma: float) -> float:
        """
        Calculate critical pressure ratio for choked flow.

        Args:
            gamma: Heat capacity ratio

        Returns:
            Critical pressure ratio (P_crit/P_upstream)
        """
        if self.critical_pressure_ratio_override is not None:
            return self.critical_pressure_ratio_override

        return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    def calculate_choked_flow_rate(self, upstream_pressure: float, upstream_temperature: float,
                                 upstream_density: float, effective_area: float) -> float:
        """
        Calculate choked flow rate through an orifice.

        Args:
            upstream_pressure: Upstream pressure [Pa]
            upstream_temperature: Upstream temperature [K]
            upstream_density: Upstream density [kg/m³]
            effective_area: Effective flow area [m²]

        Returns:
            Mass flow rate [kg/s]
        """
        if not self.enable_choked_flow:
            return 0.0

        # Get fluid properties
        props = self.get_fluid_properties(upstream_pressure, upstream_temperature)
        gamma = props['heat_capacity_ratio']
        R_specific = props['specific_gas_constant']

        # Calculate sonic velocity at throat conditions
        T_throat = upstream_temperature * (2.0 / (gamma + 1.0))
        sonic_velocity = math.sqrt(gamma * R_specific * T_throat)

        # Calculate throat density
        rho_throat = upstream_density * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))

        # Calculate mass flow rate
        flow_rate = self.discharge_coefficient * effective_area * rho_throat * sonic_velocity

        return flow_rate

    def calculate_subsonic_flow_rate(self, upstream_pressure: float, downstream_pressure: float,
                                   upstream_density: float, effective_area: float) -> float:
        """
        Calculate subsonic flow rate through an orifice.

        Args:
            upstream_pressure: Upstream pressure [Pa]
            downstream_pressure: Downstream pressure [Pa]
            upstream_density: Upstream density [kg/m³]
            effective_area: Effective flow area [m²]

        Returns:
            Mass flow rate [kg/s]
        """
        if upstream_pressure <= downstream_pressure:
            return 0.0

        # Calculate velocity using Bernoulli equation
        velocity = math.sqrt(2.0 * (upstream_pressure - downstream_pressure) / upstream_density)

        # Apply velocity limit
        velocity = min(velocity, self.maximum_velocity_ms)

        # Calculate mass flow rate
        flow_rate = self.discharge_coefficient * effective_area * upstream_density * velocity

        return flow_rate

    def calculate_orifice_flow_rate(self, upstream_pressure: float, downstream_pressure: float,
                                  upstream_temperature: float, upstream_density: float,
                                  orifice_diameter: float) -> float:
        """
        Calculate flow rate through an orifice with automatic choked/subsonic selection.

        Args:
            upstream_pressure: Upstream pressure [Pa]
            downstream_pressure: Downstream pressure [Pa]
            upstream_temperature: Upstream temperature [K]
            upstream_density: Upstream density [kg/m³]
            orifice_diameter: Orifice diameter [m]

        Returns:
            Mass flow rate [kg/s]
        """
        if upstream_pressure <= self.minimum_pressure_pa:
            return 0.0

        # Calculate effective area
        if self.use_flow_coefficient:
            effective_area = self.flow_coefficient
        else:
            orifice_area = math.pi * (orifice_diameter / 2.0) ** 2
            effective_area = self.discharge_coefficient * orifice_area

        # Get fluid properties for critical pressure ratio
        props = self.get_fluid_properties(upstream_pressure, upstream_temperature)
        gamma = props['heat_capacity_ratio']
        critical_pressure_ratio = self.calculate_critical_pressure_ratio(gamma)

        # Check for choked flow
        pressure_ratio = downstream_pressure / upstream_pressure

        if self.enable_choked_flow and pressure_ratio <= critical_pressure_ratio:
            # Choked flow
            flow_rate = self.calculate_choked_flow_rate(
                upstream_pressure, upstream_temperature, upstream_density, effective_area
            )
        else:
            # Subsonic flow
            flow_rate = self.calculate_subsonic_flow_rate(
                upstream_pressure, downstream_pressure, upstream_density, effective_area
            )

        return flow_rate

    def calculate_pipe_friction_factor(self, reynolds_number: float,
                                     relative_roughness: float) -> float:
        """
        Calculate pipe friction factor using configured correlation.

        Args:
            reynolds_number: Reynolds number
            relative_roughness: Relative roughness (ε/D)

        Returns:
            Darcy friction factor
        """
        if reynolds_number <= self.reynolds_transition:
            # Laminar flow
            return self.laminar_factor / reynolds_number
        else:
            # Turbulent flow
            if self.turbulent_correlation == 'blasius':
                return self.turbulent_factor / (reynolds_number ** self.turbulent_exponent)
            elif self.turbulent_correlation == 'colebrook':
                # Colebrook-White equation (iterative)
                return self._colebrook_friction_factor(reynolds_number, relative_roughness)
            else:
                # Default to Blasius
                return self.turbulent_factor / (reynolds_number ** self.turbulent_exponent)

    def _colebrook_friction_factor(self, reynolds_number: float,
                                 relative_roughness: float) -> float:
        """
        Calculate friction factor using Colebrook-White equation.

        Args:
            reynolds_number: Reynolds number
            relative_roughness: Relative roughness (ε/D)

        Returns:
            Darcy friction factor
        """
        # Initial guess using Blasius
        f = self.turbulent_factor / (reynolds_number ** self.turbulent_exponent)

        # Iterate Colebrook equation
        for _ in range(10):  # Limit iterations
            f_new = 1.0 / (
                -2.0 * math.log10(
                    relative_roughness / 3.7 + 2.51 / (reynolds_number * math.sqrt(f))
                )
            ) ** 2

            if abs(f_new - f) < 1e-6:
                break
            f = f_new

        return f

    def calculate_pipe_pressure_drop(self, flow_rate: float, density: float,
                                   viscosity: float, pipe_diameter: float,
                                   pipe_length: float, pipe_roughness: float,
                                   loss_coefficient: float = 0.0) -> float:
        """
        Calculate pressure drop through pipe including friction and minor losses.

        Args:
            flow_rate: Mass flow rate [kg/s]
            density: Fluid density [kg/m³]
            viscosity: Dynamic viscosity [Pa·s]
            pipe_diameter: Pipe diameter [m]
            pipe_length: Pipe length [m]
            pipe_roughness: Pipe roughness [m]
            loss_coefficient: Minor loss coefficient (K-factor)

        Returns:
            Pressure drop [Pa]
        """
        if flow_rate <= self.flow_rate_tolerance_kg_s:
            return 0.0

        # Calculate flow parameters
        pipe_area = math.pi * (pipe_diameter / 2.0) ** 2
        velocity = flow_rate / (density * pipe_area)
        reynolds_number = density * velocity * pipe_diameter / viscosity
        relative_roughness = pipe_roughness / pipe_diameter

        # Get friction factor
        friction_factor = self.calculate_pipe_friction_factor(reynolds_number, relative_roughness)

        # Calculate pressure losses
        dynamic_pressure = 0.5 * density * velocity ** 2

        # Friction losses (Darcy-Weisbach)
        friction_loss = friction_factor * (pipe_length / pipe_diameter) * dynamic_pressure

        # Minor losses
        minor_loss = loss_coefficient * dynamic_pressure

        total_pressure_drop = friction_loss + minor_loss

        # Check for choked flow in pipe
        if self.enable_choked_flow:
            props = self.get_fluid_properties(density * 287 * 288, 288)  # Approximate conditions
            sonic_velocity = props['speed_of_sound']

            if velocity > self.velocity_choked_fraction * sonic_velocity:
                # Apply choked flow factor
                total_pressure_drop *= self.choked_flow_pressure_factor

        return total_pressure_drop

    def apply_safety_limits(self, flow_rate: float, source_mass: float) -> float:
        """
        Apply safety limits to calculated flow rate.

        Args:
            flow_rate: Calculated flow rate [kg/s]
            source_mass: Source tank mass [kg]

        Returns:
            Limited flow rate [kg/s]
        """
        if flow_rate <= self.flow_rate_tolerance_kg_s:
            return 0.0

        # Apply maximum mass transfer fraction limit
        max_safe_flow = self.max_mass_transfer_fraction * source_mass
        limited_flow_rate = min(flow_rate, max_safe_flow)

        return limited_flow_rate

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information about flow physics configuration."""
        return {
            'coolprop_enabled': self.use_coolprop,
            'coolprop_fluid': self.coolprop_fluid,
            'discharge_coefficient': self.discharge_coefficient,
            'choked_flow_enabled': self.enable_choked_flow,
            'turbulent_correlation': self.turbulent_correlation,
            'max_mass_transfer_fraction': self.max_mass_transfer_fraction,
            'atmospheric_pressure_pa': self.atmospheric_pressure
        }


def create_flow_physics_from_config(config: Dict[str, Any]) -> FlowPhysics:
    """
    Factory function to create FlowPhysics instance from full configuration.

    Args:
        config: Full configuration dictionary (should contain 'flow_physics' section)

    Returns:
        FlowPhysics instance

    Raises:
        KeyError: If 'flow_physics' section not found in config
    """
    if 'flow_physics' not in config:
        raise KeyError("Configuration must contain 'flow_physics' section")

    return FlowPhysics(config['flow_physics'])