"""
Coupling Rules for Multi-Tank Systems

This module defines coupling rule classes that handle inter-tank interactions
such as pressure compensation, vented mass recovery, and temperature regulation.

Each coupling rule implements time-dependent physics with hysteresis and
activation conditions to provide realistic multi-tank behavior.

Key Features:
- Pressure compensation with bidirectional flow
- Hysteresis loops for activation/deactivation
- Time-dependent coupling activation
- Flow resistance and rate limiting
- Extensible base class for new coupling types

Authors: Dante Raso (2025)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class CouplingState:
    """State tracking for coupling rule activation and history"""
    is_active: bool = False
    activation_time: float = 0.0
    last_flow_rate: float = 0.0
    cumulative_mass_transferred: float = 0.0


class CouplingRule(ABC):
    """
    Abstract base class for all coupling rules.

    Coupling rules define how tanks interact with each other through
    mass, energy, or momentum exchange. Each rule can be time-dependent
    and include activation conditions with hysteresis.
    """

    def __init__(self, coupling_config: Dict[str, Any]):
        """
        Initialize coupling rule from configuration dictionary.

        Args:
            coupling_config: Configuration dict from YAML coupling_rules section
        """
        self.coupling_id = coupling_config['coupling_id']
        self.coupling_type = coupling_config['coupling_type']
        self.description = coupling_config.get('description', '')
        self.participants = coupling_config['participants']

        # Initialize state tracking
        self.state = CouplingState()

        # Store configuration
        self.config = coupling_config

    @abstractmethod
    def evaluate_coupling(self, tank_states: Dict[str, Any], time: float) -> Dict[str, float]:
        """
        Evaluate coupling and return flow rates for each participant tank.

        Args:
            tank_states: Dictionary of tank states {tank_id: state}
            time: Current simulation time [s]

        Returns:
            Dictionary of flow rates {tank_id: flow_rate_kg_s}
            Positive = inflow to tank, Negative = outflow from tank
        """
        pass

    @abstractmethod
    def check_activation_conditions(self, tank_states: Dict[str, Any], time: float) -> bool:
        """
        Check if coupling rule should be activated.

        Args:
            tank_states: Dictionary of tank states {tank_id: state}
            time: Current simulation time [s]

        Returns:
            True if coupling should be active, False otherwise
        """
        pass

    def update_state(self, tank_states: Dict[str, Any], time: float, flow_rates: Dict[str, float]) -> None:
        """
        Update internal state tracking for the coupling rule.

        Args:
            tank_states: Dictionary of tank states
            time: Current simulation time [s]
            flow_rates: Applied flow rates from this coupling
        """
        # Update cumulative mass transfer
        if flow_rates:
            # Sum absolute flow rates (total mass transferred)
            total_flow = sum(abs(rate) for rate in flow_rates.values())
            dt = 1.0  # Assume 1 second timestep for now
            self.state.cumulative_mass_transferred += total_flow * dt
            self.state.last_flow_rate = total_flow

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about this coupling rule."""
        return {
            'coupling_id': self.coupling_id,
            'coupling_type': self.coupling_type,
            'is_active': self.state.is_active,
            'cumulative_mass_kg': self.state.cumulative_mass_transferred,
            'last_flow_rate_kg_s': self.state.last_flow_rate
        }


class PressureCompensationRule(CouplingRule):
    """
    Pressure compensation coupling rule.

    Implements bidirectional pressure-driven flow between two tanks
    with hysteresis, flow resistance, and rate limiting.
    """

    def __init__(self, coupling_config: Dict[str, Any]):
        """Initialize pressure compensation coupling."""
        super().__init__(coupling_config)

        # Extract configuration parameters
        self.source_tank = self.participants['source']
        self.target_tank = self.participants['target']

        # Activation conditions
        activation = self.config['activation_conditions']
        self.pressure_diff_threshold = activation['pressure_differential_bar'] * 1e5  # Convert to Pa
        self.min_source_pressure = activation['minimum_source_pressure_bar'] * 1e5
        self.max_target_pressure = activation['maximum_target_pressure_bar'] * 1e5

        # Flow parameters
        flow_params = self.config['flow_parameters']
        self.flow_resistance = flow_params['flow_resistance']
        self.bidirectional = flow_params['bidirectional']
        self.max_flow_rate = flow_params['max_flow_rate_kg_s']

        # Hysteresis parameters
        hysteresis = self.config.get('hysteresis', {})
        self.activation_threshold = hysteresis.get('activation_threshold_bar', 5.0) * 1e5
        self.deactivation_threshold = hysteresis.get('deactivation_threshold_bar', 2.0) * 1e5

    def check_activation_conditions(self, tank_states: Dict[str, Any], time: float) -> bool:
        """Check if pressure compensation should be activated."""
        source_state = tank_states[self.source_tank]
        target_state = tank_states[self.target_tank]

        source_pressure = source_state.pressure
        target_pressure = target_state.pressure

        # Calculate pressure differential
        pressure_diff = abs(source_pressure - target_pressure)

        # Check basic pressure limits
        if source_pressure < self.min_source_pressure:
            return False
        if target_pressure > self.max_target_pressure:
            return False

        # Apply hysteresis
        if self.state.is_active:
            # Currently active - check deactivation threshold
            return pressure_diff > self.deactivation_threshold
        else:
            # Currently inactive - check activation threshold
            return pressure_diff > self.activation_threshold

    def evaluate_coupling(self, tank_states: Dict[str, Any], time: float) -> Dict[str, float]:
        """Evaluate pressure compensation coupling and return flow rates."""
        # Check activation conditions
        should_be_active = self.check_activation_conditions(tank_states, time)

        # Update activation state
        if should_be_active and not self.state.is_active:
            self.state.is_active = True
            self.state.activation_time = time
        elif not should_be_active and self.state.is_active:
            self.state.is_active = False

        # If not active, return zero flows
        if not self.state.is_active:
            return {self.source_tank: 0.0, self.target_tank: 0.0}

        # Calculate pressure-driven flow
        source_state = tank_states[self.source_tank]
        target_state = tank_states[self.target_tank]

        source_pressure = source_state.pressure
        target_pressure = target_state.pressure
        pressure_diff = source_pressure - target_pressure

        # Determine flow direction
        if pressure_diff > 0:
            # Flow from source to target
            high_pressure_tank = self.source_tank
            low_pressure_tank = self.target_tank
        elif pressure_diff < 0 and self.bidirectional:
            # Flow from target to source
            high_pressure_tank = self.target_tank
            low_pressure_tank = self.source_tank
            pressure_diff = -pressure_diff
        else:
            # No flow
            return {self.source_tank: 0.0, self.target_tank: 0.0}

        # Calculate flow rate using orifice-like equation
        # Q = Cd * A * sqrt(2 * rho * dP)
        # Simplified: flow_rate = pressure_diff / flow_resistance
        flow_rate = pressure_diff / (self.flow_resistance * 1e6)  # Scale for reasonable flow rates

        # Apply maximum flow rate limit
        flow_rate = min(flow_rate, self.max_flow_rate)

        # Create flow rate dictionary
        flow_rates = {self.source_tank: 0.0, self.target_tank: 0.0}

        if high_pressure_tank == self.source_tank:
            # Flow from source to target
            flow_rates[self.source_tank] = -flow_rate  # Outflow from source
            flow_rates[self.target_tank] = flow_rate   # Inflow to target
        else:
            # Flow from target to source
            flow_rates[self.source_tank] = flow_rate   # Inflow to source
            flow_rates[self.target_tank] = -flow_rate  # Outflow from target

        # Update internal state
        self.update_state(tank_states, time, flow_rates)

        return flow_rates


class VentedMassRecuperationRule(CouplingRule):
    """
    Vented mass recuperation coupling rule.

    Captures vented hydrogen from one tank and feeds it to another tank,
    with efficiency losses and rate limiting.
    """

    def __init__(self, coupling_config: Dict[str, Any]):
        """Initialize vented mass recuperation coupling."""
        super().__init__(coupling_config)

        self.source_tank = self.participants['source']  # Tank that vents
        self.target_tank = self.participants['target']  # Tank that receives

        # Recovery parameters
        recovery_params = self.config['recovery_parameters']
        self.recovery_efficiency = recovery_params['efficiency']
        self.max_recovery_rate = recovery_params['max_recovery_rate_kg_s']

        # Storage for vented mass tracking
        self.vented_mass_available = 0.0

    def check_activation_conditions(self, tank_states: Dict[str, Any], time: float) -> bool:
        """Check if vented mass recovery should be activated."""
        # Active when source tank is venting and target tank can accept mass
        source_state = tank_states[self.source_tank]
        target_state = tank_states[self.target_tank]

        # Check if source is venting (simplified - assume venting when above vent pressure)
        source_venting = hasattr(source_state, 'vent_rate') and source_state.vent_rate > 0

        # Check if target can accept more mass (not at maximum pressure)
        target_can_accept = target_state.pressure < (target_state.vent_pressure * 0.95)

        return source_venting and target_can_accept

    def evaluate_coupling(self, tank_states: Dict[str, Any], time: float) -> Dict[str, float]:
        """Evaluate vented mass recovery coupling and return flow rates."""
        if not self.check_activation_conditions(tank_states, time):
            return {self.source_tank: 0.0, self.target_tank: 0.0}

        # Get vented mass from source
        source_state = tank_states[self.source_tank]
        vented_rate = getattr(source_state, 'vent_rate', 0.0)

        # Calculate recoverable mass with efficiency
        recoverable_rate = vented_rate * self.recovery_efficiency

        # Apply maximum recovery rate limit
        recovery_rate = min(recoverable_rate, self.max_recovery_rate)

        # The source tank doesn't lose additional mass (it's already venting)
        # The target tank gains the recovered mass
        flow_rates = {
            self.source_tank: 0.0,        # No additional loss
            self.target_tank: recovery_rate  # Gain recovered mass
        }

        # Update internal state
        self.update_state(tank_states, time, flow_rates)

        return flow_rates


class FlowControlledPressurizationRule(CouplingRule):
    """
    Flow-controlled pressurization coupling rule for CH2 → LH2 systems.

    This coupling provides controlled CH2 gas injection to maintain adequate
    LH2 tank pressure for mission discharge requirements. Uses feedback control
    to adjust CH2 flow based on LH2 discharge rate and target pressure.
    """

    def __init__(self, coupling_config: Dict[str, Any]):
        """Initialize flow-controlled pressurization coupling."""
        super().__init__(coupling_config)

        # Extract configuration parameters
        self.source_tank = self.participants['source']  # CH2 tank
        self.target_tank = self.participants['target']  # LH2 tank

        # Control parameters
        control_params = self.config['control_parameters']
        self.target_pressure_offset_bar = control_params['target_pressure_offset_bar']  # Above required discharge pressure
        self.control_gain = control_params.get('control_gain', 1.0)  # PID-like gain
        self.max_pressurization_rate = control_params['max_pressurization_rate_kg_s']
        self.min_source_pressure_bar = control_params['min_source_pressure_bar']

        # Flow calculation parameters
        flow_params = self.config['flow_parameters']
        self.flow_coefficient = flow_params.get('flow_coefficient', 1.0)
        self.response_time = flow_params.get('response_time_s', 5.0)  # Control response time

        # Mission flow rate tracking (for feedback control)
        self.target_mission_flow_rate = 0.0
        self.last_lh2_pressure = 0.0

        # Control state for feedback
        self.pressure_error_integral = 0.0
        self.last_pressure_error = 0.0

    def check_activation_conditions(self, tank_states: Dict[str, Any], time: float) -> bool:
        """Check if flow-controlled pressurization should be activated."""
        source_state = tank_states[self.source_tank]
        target_state = tank_states[self.target_tank]

        # Must have sufficient CH2 pressure
        source_pressure_bar = source_state.pressure / 1e5
        if source_pressure_bar < self.min_source_pressure_bar:
            return False

        # Must have active mission flow rate (LH2 discharge)
        # Note: This would need to be provided by the mission system
        # For now, assume active if LH2 pressure is dropping or needs support
        target_pressure_bar = target_state.pressure / 1e5

        # Activate if LH2 pressure is below target for discharge support
        required_pressure = self._calculate_required_pressure(target_state, time)

        return target_pressure_bar < (required_pressure + self.target_pressure_offset_bar)

    def _calculate_required_pressure(self, lh2_state, time: float) -> float:
        """
        Calculate required LH2 pressure to maintain mission discharge rate.

        This is simplified - in reality would depend on:
        - Mission discharge rate requirement
        - LH2 properties and flow resistance
        - Tank geometry and outlet configuration
        """
        # Simplified: assume need 3-5 bar minimum for reasonable discharge
        base_pressure = 3.0  # bar

        # Add pressure based on discharge requirements
        # This would be fed from mission controller in full implementation
        if hasattr(lh2_state, 'discharge_rate') and lh2_state.discharge_rate > 0:
            # Higher discharge rates need higher pressure
            flow_pressure = lh2_state.discharge_rate * 50.0  # Simplified relationship
            return base_pressure + flow_pressure

        return base_pressure

    def evaluate_coupling(self, tank_states: Dict[str, Any], time: float) -> Dict[str, float]:
        """Evaluate flow-controlled pressurization and return flow rates."""
        # Check activation conditions
        should_be_active = self.check_activation_conditions(tank_states, time)

        # Update activation state
        if should_be_active and not self.state.is_active:
            self.state.is_active = True
            self.state.activation_time = time
        elif not should_be_active and self.state.is_active:
            self.state.is_active = False

        # If not active, return zero flows
        if not self.state.is_active:
            return {self.source_tank: 0.0, self.target_tank: 0.0}

        # Get tank states
        ch2_state = tank_states[self.source_tank]
        lh2_state = tank_states[self.target_tank]

        # Calculate target pressure for LH2 tank
        required_pressure = self._calculate_required_pressure(lh2_state, time)
        target_pressure = required_pressure + self.target_pressure_offset_bar

        # Calculate pressure error
        current_pressure_bar = lh2_state.pressure / 1e5
        pressure_error = target_pressure - current_pressure_bar

        # Simple proportional control for CH2 flow rate
        # In full implementation, this would be PID control
        control_signal = self.control_gain * pressure_error

        # Convert control signal to mass flow rate
        # Positive error (low pressure) → increase CH2 flow
        ch2_flow_rate = max(0.0, control_signal * self.flow_coefficient)

        # Apply maximum flow rate limit
        ch2_flow_rate = min(ch2_flow_rate, self.max_pressurization_rate)

        # Ensure CH2 tank has sufficient mass and pressure
        ch2_pressure_bar = ch2_state.pressure / 1e5
        if ch2_state.fuel_mass < 0.1 or ch2_state.pressure < (self.min_source_pressure_bar * 1e5):
            # Debug output for pressure constraint
            if time > 0 and (time % 300 < 5 or ch2_pressure_bar < (self.min_source_pressure_bar + 5)):  # Every 5 min or near limit
                if ch2_state.fuel_mass < 0.1:
                    print(f"  CH2 Constraint t={time/3600:.2f}h: Low mass ({ch2_state.fuel_mass:.1f}kg < 0.1kg)")
                if ch2_pressure_bar < self.min_source_pressure_bar:
                    print(f"  CH2 Constraint t={time/3600:.2f}h: Low pressure ({ch2_pressure_bar:.1f}bar < {self.min_source_pressure_bar:.1f}bar) - FLOW DISABLED")
                elif ch2_pressure_bar < (self.min_source_pressure_bar + 5):
                    print(f"  CH2 Constraint t={time/3600:.2f}h: Approaching limit ({ch2_pressure_bar:.1f}bar, limit={self.min_source_pressure_bar:.1f}bar)")
            ch2_flow_rate = 0.0

        # Create flow rate dictionary
        flow_rates = {
            self.source_tank: -ch2_flow_rate,  # Outflow from CH2
            self.target_tank: ch2_flow_rate    # Inflow to LH2 (pressurization gas)
        }

        # Update internal state tracking
        self.update_state(tank_states, time, flow_rates)
        self.last_lh2_pressure = current_pressure_bar

        return flow_rates


class OHEXExtractionRule(CouplingRule):
    """
    OHEX (Outboard Heat Exchanger) extraction coupling rule.

    This coupling extracts liquid hydrogen from the LH2 tank at the rate required
    by the mission profile. The extraction pressure must be sufficient to meet
    the mission mass flow rate demand. After extraction, the LH2 is warmed to
    target conditions (200K, 20 bar) in the OHEX.

    The coupling only extracts at mission-required flow rates and relies on
    other coupling rules (like CH2 pressurization) to maintain adequate pressure.
    """

    def __init__(self, coupling_config: Dict[str, Any]):
        """Initialize OHEX extraction coupling."""
        super().__init__(coupling_config)

        # Extract configuration parameters
        self.source_tank = self.participants['source']  # LH2 tank
        self.target_tank = None  # No target tank for OHEX extraction

        # Mission flow parameters
        mission_params = self.config['mission_parameters']
        self.mission_profile = mission_params.get('mission_profile', {})

        # OHEX target conditions (for reference/reporting)
        ohex_params = self.config.get('ohex_parameters', {})
        self.target_temperature_K = ohex_params.get('target_temperature_K', 200.0)
        self.target_pressure_bar = ohex_params.get('target_pressure_bar', 20.0)

        # Extraction parameters
        extraction_params = self.config.get('extraction_parameters', {})
        self.min_extraction_pressure_bar = extraction_params.get('min_extraction_pressure_bar', 3.0)

        # Mission flow rate tracking
        self.current_mission_flow_rate = 0.0

    def check_activation_conditions(self, tank_states: Dict[str, Any], time: float) -> bool:
        """Check if OHEX extraction should be activated."""
        # Get current mission flow rate requirement
        mission_flow_rate = self._get_mission_flow_rate(time)

        # Only activate if mission requires fuel
        if mission_flow_rate <= 0:
            return False

        # Check if LH2 tank has sufficient pressure for extraction
        source_state = tank_states[self.source_tank]
        source_pressure_bar = source_state.pressure / 1e5

        return source_pressure_bar >= self.min_extraction_pressure_bar

    def evaluate_coupling(self, tank_states: Dict[str, Any], time: float) -> Dict[str, float]:
        """Evaluate OHEX extraction and return flow rates."""
        # Get mission flow rate requirement
        mission_flow_rate = self._get_mission_flow_rate(time)
        self.current_mission_flow_rate = mission_flow_rate

        # Check activation conditions
        if not self.check_activation_conditions(tank_states, time):
            self.state.is_active = False
            return {self.source_tank: 0.0}

        # Extract at mission-required rate
        # Negative flow rate = outflow from LH2 tank
        extraction_flow_rate = -mission_flow_rate

        # Update state
        self.state.is_active = True
        if not hasattr(self.state, 'activation_time') or self.state.activation_time == 0:
            self.state.activation_time = time

        # Create flow rate dictionary
        flow_rates = {self.source_tank: extraction_flow_rate}

        # Update internal state tracking
        self.update_state(tank_states, time, flow_rates)

        return flow_rates

    def _get_mission_flow_rate(self, time: float) -> float:
        """
        Get mission flow rate requirement at given time.

        Args:
            time: Current simulation time [s]

        Returns:
            Required mass flow rate [kg/s]
        """
        # For now, use a simple time-based lookup
        # This could be replaced with interpolation from mission profile data

        if not self.mission_profile:
            return 0.0

        # Extract time points and flow rates from mission profile
        time_points = self.mission_profile.get('time_s', [])
        flow_rates = self.mission_profile.get('flow_rate_kg_s', [])

        if not time_points or not flow_rates:
            return 0.0

        # Simple linear interpolation
        if time <= time_points[0]:
            return flow_rates[0]
        elif time >= time_points[-1]:
            return flow_rates[-1]
        else:
            # Find surrounding points and interpolate
            for i in range(len(time_points) - 1):
                if time_points[i] <= time <= time_points[i + 1]:
                    # Linear interpolation
                    t1, t2 = time_points[i], time_points[i + 1]
                    f1, f2 = flow_rates[i], flow_rates[i + 1]
                    return f1 + (f2 - f1) * (time - t1) / (t2 - t1)

        return 0.0

    def get_ohex_conditions(self, tank_states: Dict[str, Any]) -> Dict[str, float]:
        """
        Get OHEX output conditions for current extraction.

        Returns:
            Dictionary with OHEX output temperature and pressure conditions
        """
        return {
            'ohex_output_temperature_K': self.target_temperature_K,
            'ohex_output_pressure_bar': self.target_pressure_bar,
            'extraction_flow_rate_kg_s': self.current_mission_flow_rate
        }


class MissionAdaptivePressurizationRule(CouplingRule):
    """
    Mission-adaptive pressurization coupling rule for CH2 → LH2 systems.

    This coupling provides dynamic CH2 pressurization based on real-time mission
    flow requirements. At each timestep, it:
    1. Determines required mission flow rate from profile
    2. Calculates minimum LH2 pressure needed for discharge piping
    3. Sets dynamic activation/deactivation thresholds with safety margins
    4. Provides controlled CH2 injection to maintain adequate pressure
    """

    def __init__(self, coupling_config: Dict[str, Any]):
        """Initialize mission-adaptive pressurization coupling."""
        super().__init__(coupling_config)

        # Extract configuration parameters
        self.source_tank = self.participants['source']  # CH2 tank
        self.target_tank = self.participants['target']  # LH2 tank

        # Mission profile parameters
        mission_params = self.config['mission_parameters']['mission_profile']
        self.mission_times = np.array(mission_params['time_s'])
        self.mission_flow_rates = np.array(mission_params['flow_rate_kg_s'])

        # Discharge piping characteristics for pressure calculations
        piping_params = self.config['discharge_piping']
        self.pipe_diameter = piping_params['diameter_m']
        self.pipe_length = piping_params['length_m']
        self.pipe_roughness = piping_params['roughness_m']
        self.loss_coefficient = piping_params['loss_coefficient']
        self.choked_flow_enabled = piping_params.get('choked_flow_enabled', True)

        # Control parameters
        control_params = self.config['control_parameters']
        self.pressure_margin_bar = control_params['pressure_margin_bar']
        self.max_pressurization_rate = control_params['max_pressurization_rate_kg_s']
        self.min_source_pressure_bar = control_params['min_source_pressure_bar']

        # Flow calculation parameters for CH2 pressurization flow
        flow_params = self.config['flow_parameters']
        self.flow_coefficient = flow_params.get('flow_coefficient', 0.01)
        self.response_time = flow_params.get('response_time_s', 5.0)
        self.max_ch2_flow_rate = flow_params.get('max_flow_rate_kg_s', 0.005)  # 5 g/s limit
        self.ch2_orifice_diameter = flow_params.get('orifice_diameter_m', 0.001)  # 1mm

        # State tracking for dynamic thresholds
        self.current_mission_flow_rate = 0.0
        self.current_activation_threshold = 3.0  # Default 3 bar
        self.current_deactivation_threshold = 4.0  # Default 4 bar
        self.last_required_pressure = 3.0

    def get_mission_flow_rate(self, time: float) -> float:
        """Get required mission flow rate at current time from profile."""
        if time <= self.mission_times[0]:
            return self.mission_flow_rates[0]
        elif time >= self.mission_times[-1]:
            return self.mission_flow_rates[-1]
        else:
            # Linear interpolation between mission profile points
            return np.interp(time, self.mission_times, self.mission_flow_rates)

    def calculate_minimum_discharge_pressure(self, flow_rate_kg_s: float, lh2_density: float) -> float:
        """
        Calculate minimum LH2 tank pressure required to achieve target flow rate
        through discharge piping with losses and choked flow considerations.

        Args:
            flow_rate_kg_s: Required mass flow rate [kg/s]
            lh2_density: LH2 density [kg/m³]

        Returns:
            Minimum tank pressure [Pa]
        """
        if flow_rate_kg_s <= 0:
            return 1e5  # 1 bar minimum for no flow

        # Convert mass flow to volumetric flow
        volumetric_flow = flow_rate_kg_s / lh2_density  # m³/s

        # Calculate flow velocity in pipe
        pipe_area = np.pi * (self.pipe_diameter / 2) ** 2
        velocity = volumetric_flow / pipe_area  # m/s

        # Calculate Reynolds number for friction factor
        # Simplified: assume kinematic viscosity ~ 1e-7 m²/s for LH2
        reynolds = velocity * self.pipe_diameter / 1e-7

        # Calculate Darcy friction factor (Colebrook-White approximation)
        if reynolds > 2300:  # Turbulent flow
            # Simplified turbulent friction factor
            friction_factor = 0.316 / (reynolds ** 0.25)
        else:  # Laminar flow
            friction_factor = 64 / reynolds

        # Calculate pressure losses
        # Frictional losses (Darcy-Weisbach equation)
        friction_loss = friction_factor * (self.pipe_length / self.pipe_diameter) * (lh2_density * velocity**2 / 2)

        # Minor losses (K-factors)
        minor_loss = self.loss_coefficient * (lh2_density * velocity**2 / 2)

        # Total pressure drop
        total_pressure_drop = friction_loss + minor_loss

        # Check for choked flow condition
        if self.choked_flow_enabled:
            # Simplified choked flow check: if velocity > 0.5 * sonic velocity
            # Assume sonic velocity ~ 1000 m/s for LH2
            if velocity > 500:  # Approaching choked flow
                # Increase pressure requirement for choked flow
                choked_flow_factor = 2.0
                total_pressure_drop *= choked_flow_factor

        # Minimum tank pressure = atmospheric + pressure drops + safety margin
        atmospheric_pressure = 1.01325e5  # Pa
        min_tank_pressure = atmospheric_pressure + total_pressure_drop

        return min_tank_pressure

    def update_dynamic_thresholds(self, time: float, lh2_density: float):
        """Update activation and deactivation thresholds based on current mission requirements."""
        # Get current mission flow requirement
        self.current_mission_flow_rate = self.get_mission_flow_rate(time)

        # Calculate minimum pressure required for this flow rate
        min_pressure_pa = self.calculate_minimum_discharge_pressure(self.current_mission_flow_rate, lh2_density)
        min_pressure_bar = min_pressure_pa / 1e5

        # Set activation threshold with safety margin
        self.current_activation_threshold = min_pressure_bar + self.pressure_margin_bar

        # Set deactivation threshold with additional margin to prevent oscillation
        self.current_deactivation_threshold = self.current_activation_threshold + self.deactivation_margin_bar

        # Store for logging/debugging
        self.last_required_pressure = min_pressure_bar

    def check_activation_conditions(self, tank_states: Dict[str, Any], time: float) -> bool:
        """Check if mission-adaptive pressurization should be activated."""
        source_state = tank_states[self.source_tank]
        target_state = tank_states[self.target_tank]

        # Update dynamic thresholds based on current mission requirements
        self.update_dynamic_thresholds(time, target_state.density)

        # Must have sufficient CH2 pressure
        source_pressure_bar = source_state.pressure / 1e5
        if source_pressure_bar < self.min_source_pressure_bar:
            return False

        # Must have adequate pressure difference
        target_pressure_bar = target_state.pressure / 1e5
        pressure_difference = source_pressure_bar - target_pressure_bar
        if pressure_difference < 5.0:  # Default 5 bar minimum pressure difference
            return False

        # Check LH2 pressure against dynamic activation threshold
        # Use hysteresis: different thresholds for activation vs deactivation
        if not self.state.is_active:
            # Activation condition: LH2 pressure below activation threshold
            return target_pressure_bar < self.current_activation_threshold
        else:
            # Already active - use deactivation threshold to prevent oscillation
            return target_pressure_bar < self.current_deactivation_threshold

    def evaluate_coupling(self, tank_states: Dict[str, Any], time: float) -> Dict[str, float]:
        """Evaluate mission-adaptive pressurization and return flow rates."""
        # Check activation conditions (also updates dynamic thresholds)
        should_be_active = self.check_activation_conditions(tank_states, time)

        # Update activation state
        if should_be_active and not self.state.is_active:
            self.state.is_active = True
            self.state.activation_time = time
        elif not should_be_active and self.state.is_active:
            self.state.is_active = False

        # If not active, return zero flows
        if not self.state.is_active:
            return {self.source_tank: 0.0, self.target_tank: 0.0}

        # Get tank states
        ch2_state = tank_states[self.source_tank]
        lh2_state = tank_states[self.target_tank]

        # Calculate pressure error using dynamic activation threshold
        current_pressure_bar = lh2_state.pressure / 1e5
        target_pressure_bar = self.current_activation_threshold
        pressure_error = target_pressure_bar - current_pressure_bar

        # Proportional control for CH2 flow rate
        control_signal = self.control_gain * pressure_error

        # Convert control signal to mass flow rate
        # Positive error (low pressure) → increase CH2 flow
        ch2_flow_rate = max(0.0, control_signal * self.flow_coefficient)

        # Apply maximum flow rate limit
        ch2_flow_rate = min(ch2_flow_rate, min(self.max_pressurization_rate, self.max_ch2_flow_rate))

        # Ensure CH2 tank has sufficient mass and pressure
        if ch2_state.fuel_mass < 0.1 or ch2_state.pressure < (self.min_source_pressure_bar * 1e5):
            ch2_flow_rate = 0.0

        # Create flow rate dictionary
        flow_rates = {
            self.source_tank: -ch2_flow_rate,  # Outflow from CH2
            self.target_tank: ch2_flow_rate    # Inflow to LH2 (pressurization gas)
        }

        # Update internal state tracking
        self.update_state(tank_states, time, flow_rates)

        return flow_rates

    def get_coupling_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for coupling rule analysis and debugging."""
        diagnostics = super().get_coupling_diagnostics()

        # Add mission-adaptive specific diagnostics
        diagnostics.update({
            'current_mission_flow_rate_kg_s': self.current_mission_flow_rate,
            'current_activation_threshold_bar': self.current_activation_threshold,
            'current_deactivation_threshold_bar': self.current_deactivation_threshold,
            'last_required_pressure_bar': self.last_required_pressure,
            'pipe_diameter_m': self.pipe_diameter,
            'pipe_length_m': self.pipe_length,
            'pressure_margin_bar': self.pressure_margin_bar
        })

        return diagnostics


def create_coupling_rule(coupling_config: Dict[str, Any]) -> CouplingRule:
    """
    Factory function to create coupling rules from configuration.

    Args:
        coupling_config: Configuration dictionary for coupling rule

    Returns:
        Appropriate CouplingRule instance

    Raises:
        ValueError: If coupling_type is not supported
    """
    coupling_type = coupling_config['coupling_type']

    if coupling_type == 'pressure_compensation':
        return PressureCompensationRule(coupling_config)
    elif coupling_type == 'vented_mass_recuperation':
        return VentedMassRecuperationRule(coupling_config)
    elif coupling_type == 'flow_controlled_pressurization':
        return FlowControlledPressurizationRule(coupling_config)
    elif coupling_type == 'ohex_extraction':
        return OHEXExtractionRule(coupling_config)
    elif coupling_type == 'mission_adaptive_pressurization':
        return MissionAdaptivePressurizationRule(coupling_config)
    else:
        raise ValueError(f"Unsupported coupling type: {coupling_type}")


def parse_coupling_rules(coupling_rules_config: List[Dict[str, Any]]) -> List[CouplingRule]:
    """
    Parse coupling rules from configuration list.

    Args:
        coupling_rules_config: List of coupling rule configuration dictionaries

    Returns:
        List of CouplingRule instances
    """
    coupling_rules = []

    for rule_config in coupling_rules_config:
        try:
            rule = create_coupling_rule(rule_config)
            coupling_rules.append(rule)
        except Exception as e:
            print(f"⚠️ Failed to create coupling rule {rule_config.get('coupling_id', 'unknown')}: {e}")

    return coupling_rules