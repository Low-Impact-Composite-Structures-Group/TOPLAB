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