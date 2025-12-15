"""
Copy of `src/fluids/flow_physics.py` colocated under multi_tank for self-contained imports.
"""

from typing import Dict, Any
import math
import numpy as np

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False


class FlowPhysics:
    def __init__(self, flow_physics_config: Dict[str, Any]):
        self.config = flow_physics_config
        fluid_config = self.config.get('fluid_properties', {})
        self.use_coolprop = fluid_config.get('use_coolprop', True) and COOLPROP_AVAILABLE
        self.coolprop_fluid = fluid_config.get('coolprop_fluid', 'H2')
        backup = fluid_config.get('backup_properties', {})
        self.backup_R_specific = backup.get('specific_gas_constant', 4124.0)
        self.backup_gamma = backup.get('heat_capacity_ratio', 1.4)
        self.backup_kinematic_viscosity = backup.get('kinematic_viscosity', 1.55e-6)
        self.backup_speed_of_sound = backup.get('speed_of_sound', 1354.0)
        orifice_config = self.config.get('orifice_flow', {})
        self.discharge_coefficient = orifice_config.get('discharge_coefficient', 0.6)
        self.use_flow_coefficient = orifice_config.get('use_flow_coefficient', False)
        self.flow_coefficient = orifice_config.get('flow_coefficient', 0.0001)
        choked_config = self.config.get('choked_flow', {})
        self.enable_choked_flow = choked_config.get('enable_choked_flow', True)
        self.critical_pressure_ratio_override = choked_config.get('critical_pressure_ratio_override')
        self.sonic_velocity_factor = choked_config.get('sonic_velocity_factor', 1.0)
        self.choked_flow_pressure_factor = choked_config.get('choked_flow_pressure_factor', 2.0)
        pipe_config = self.config.get('pipe_flow', {})
        self.atmospheric_pressure = pipe_config.get('atmospheric_pressure', 101325.0)
        self.reynolds_transition = pipe_config.get('reynolds_transition', 2300.0)
        self.reynolds_blend_width = pipe_config.get('reynolds_blend_width', 400.0)
        friction_config = pipe_config.get('friction_correlations', {})
        self.laminar_factor = friction_config.get('laminar_factor', 64.0)
        self.turbulent_correlation = friction_config.get('turbulent_correlation', 'blasius')
        self.turbulent_factor = friction_config.get('turbulent_factor', 0.316)
        self.turbulent_exponent = friction_config.get('turbulent_exponent', 0.25)
        self.default_roughness = pipe_config.get('default_roughness', 1.5e-6)
        self.velocity_choked_fraction = pipe_config.get('velocity_choked_fraction', 0.5)
        self.gas_density_threshold = pipe_config.get('gas_density_threshold', 10.0)
        safety_config = self.config.get('safety_limits', {})
        self.max_mass_transfer_fraction = safety_config.get('max_mass_transfer_fraction', 0.1)
        self.minimum_pressure_pa = safety_config.get('minimum_pressure_pa', 1000.0)
        self.maximum_velocity_ms = safety_config.get('maximum_velocity_ms', 2000.0)
        numerical_config = self.config.get('numerical', {})
        self.pressure_tolerance_pa = float(numerical_config.get('pressure_tolerance_pa', 100.0))
        self.flow_rate_tolerance_kg_s = float(numerical_config.get('flow_rate_tolerance_kg_s', 1e-8))
        self.property_update_frequency = int(numerical_config.get('property_update_frequency', 1))

    def get_fluid_properties(self, pressure: float, temperature: float) -> Dict[str, float]:
        properties = {}
        if self.use_coolprop:
            try:
                properties['specific_gas_constant'] = (
                    CP.PropsSI('GAS_CONSTANT', self.coolprop_fluid) /
                    CP.PropsSI('MOLAR_MASS', self.coolprop_fluid)
                )
                try:
                    cp = CP.PropsSI('CPMASS', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                    cv = CP.PropsSI('CVMASS', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                except Exception:
                    try:
                        cp = CP.PropsSI('Cpmass', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                        cv = CP.PropsSI('Cvmass', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                    except Exception:
                        cp = None
                        cv = None
                properties['heat_capacity_ratio'] = (cp / cv) if (cp is not None and cv is not None and cv > 0) else self.backup_gamma
                try:
                    sos = CP.PropsSI('SPEED_OF_SOUND', 'P', pressure, 'T', temperature, self.coolprop_fluid)
                except Exception:
                    try:
                        sos = CP.PropsSI('SPEED_OF_SOUND', 'T', max(temperature, 1.0), 'Q', 1, self.coolprop_fluid)
                    except Exception:
                        sos = self.backup_speed_of_sound
                properties['speed_of_sound'] = sos * self.sonic_velocity_factor
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
                    properties['kinematic_viscosity'] = self.backup_kinematic_viscosity
                    properties['density'] = None
            except Exception:
                properties = self._get_backup_properties()
        else:
            properties = self._get_backup_properties()
        return properties

    def _get_backup_properties(self) -> Dict[str, float]:
        return {
            'specific_gas_constant': self.backup_R_specific,
            'heat_capacity_ratio': self.backup_gamma,
            'kinematic_viscosity': self.backup_kinematic_viscosity,
            'speed_of_sound': self.backup_speed_of_sound * self.sonic_velocity_factor,
            'density': None,
        }

    def calculate_critical_pressure_ratio(self, gamma: float) -> float:
        if self.critical_pressure_ratio_override is not None:
            return self.critical_pressure_ratio_override
        return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    def calculate_choked_flow_rate(self, upstream_pressure: float, upstream_temperature: float, upstream_density: float, effective_area: float) -> float:
        if not self.enable_choked_flow:
            return 0.0
        props = self.get_fluid_properties(upstream_pressure, upstream_temperature)
        gamma = props['heat_capacity_ratio']
        R_specific = props['specific_gas_constant']
        T_throat = upstream_temperature * (2.0 / (gamma + 1.0))
        sonic_velocity = math.sqrt(gamma * R_specific * T_throat)
        rho_throat = upstream_density * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))
        flow_rate = self.discharge_coefficient * effective_area * rho_throat * sonic_velocity
        return flow_rate

    def calculate_subsonic_flow_rate(self, upstream_pressure: float, downstream_pressure: float, upstream_density: float, effective_area: float) -> float:
        if upstream_pressure <= downstream_pressure:
            return 0.0
        velocity = math.sqrt(2.0 * (upstream_pressure - downstream_pressure) / upstream_density)
        velocity = min(velocity, self.maximum_velocity_ms)
        return self.discharge_coefficient * effective_area * upstream_density * velocity

    def calculate_orifice_flow_rate(self, upstream_pressure: float, downstream_pressure: float, upstream_temperature: float, upstream_density: float, orifice_diameter: float) -> float:
        if upstream_pressure <= self.minimum_pressure_pa:
            return 0.0
        if self.use_flow_coefficient:
            effective_area = self.flow_coefficient
        else:
            orifice_area = math.pi * (orifice_diameter / 2.0) ** 2
            effective_area = self.discharge_coefficient * orifice_area
        props = self.get_fluid_properties(upstream_pressure, upstream_temperature)
        gamma = props['heat_capacity_ratio']
        critical_pressure_ratio = self.calculate_critical_pressure_ratio(gamma)
        pressure_ratio = downstream_pressure / upstream_pressure
        if self.enable_choked_flow and pressure_ratio <= critical_pressure_ratio:
            flow_rate = self.calculate_choked_flow_rate(upstream_pressure, upstream_temperature, upstream_density, effective_area)
        else:
            flow_rate = self.calculate_subsonic_flow_rate(upstream_pressure, downstream_pressure, upstream_density, effective_area)
        return flow_rate

    def calculate_pipe_friction_factor(self, reynolds_number: float, relative_roughness: float) -> float:
        re = max(1e-6, reynolds_number)
        f_lam = self.laminar_factor / re
        if self.turbulent_correlation == 'blasius':
            f_turb = self.turbulent_factor / (re ** self.turbulent_exponent)
        elif self.turbulent_correlation == 'colebrook':
            f_turb = self._colebrook_friction_factor(re, relative_roughness)
        else:
            f_turb = self.turbulent_factor / (re ** self.turbulent_exponent)
        w = 0.5 * (1.0 + math.tanh((re - self.reynolds_transition) / max(1e-6, self.reynolds_blend_width)))
        return (1.0 - w) * f_lam + w * f_turb

    def _colebrook_friction_factor(self, reynolds_number: float, relative_roughness: float) -> float:
        f = self.turbulent_factor / (reynolds_number ** self.turbulent_exponent)
        for _ in range(10):
            f_new = 1.0 / (-2.0 * math.log10(relative_roughness / 3.7 + 2.51 / (reynolds_number * math.sqrt(f)))) ** 2
            if abs(f_new - f) < 1e-6:
                break
            f = f_new
        return f

    def calculate_pipe_pressure_drop(self, flow_rate: float, density: float, viscosity: float, pipe_diameter: float, pipe_length: float, pipe_roughness: float, loss_coefficient: float = 0.0) -> float:
        if flow_rate <= self.flow_rate_tolerance_kg_s:
            return 0.0
        pipe_area = math.pi * (pipe_diameter / 2.0) ** 2
        velocity = flow_rate / (max(1e-12, density) * pipe_area)
        reynolds_number = max(1e-12, density) * velocity * pipe_diameter / max(1e-12, viscosity)
        relative_roughness = pipe_roughness / max(1e-12, pipe_diameter)
        friction_factor = self.calculate_pipe_friction_factor(reynolds_number, relative_roughness)
        dynamic_pressure = 0.5 * max(1e-12, density) * velocity ** 2
        friction_loss = friction_factor * (pipe_length / max(1e-12, pipe_diameter)) * dynamic_pressure
        minor_loss = loss_coefficient * dynamic_pressure
        total_pressure_drop = friction_loss + minor_loss
        if self.enable_choked_flow and density <= self.gas_density_threshold:
            try:
                sos = self.get_fluid_properties(self.atmospheric_pressure, 288.0)['speed_of_sound']
            except Exception:
                sos = 300.0
            mach = velocity / max(1e-6, sos)
            softness = 0.05
            s = 0.5 * (1.0 + math.tanh((mach - self.velocity_choked_fraction) / max(1e-6, softness)))
            total_pressure_drop *= (1.0 + (self.choked_flow_pressure_factor - 1.0) * s)
        return total_pressure_drop

    def apply_safety_limits(self, flow_rate: float, source_mass: float) -> float:
        if flow_rate <= self.flow_rate_tolerance_kg_s:
            return 0.0
        max_safe_flow = self.max_mass_transfer_fraction * source_mass
        return min(flow_rate, max_safe_flow)

    def get_diagnostic_info(self) -> Dict[str, Any]:
        return {
            'coolprop_enabled': self.use_coolprop,
            'coolprop_fluid': self.coolprop_fluid,
            'discharge_coefficient': self.discharge_coefficient,
            'choked_flow_enabled': self.enable_choked_flow,
            'turbulent_correlation': self.turbulent_correlation,
            'max_mass_transfer_fraction': self.max_mass_transfer_fraction,
            'atmospheric_pressure_pa': self.atmospheric_pressure,
        }

def create_flow_physics_from_config(config: Dict[str, Any]) -> FlowPhysics:
    if 'flow_physics' not in config:
        raise KeyError("Configuration must contain 'flow_physics' section")
    return FlowPhysics(config['flow_physics'])
