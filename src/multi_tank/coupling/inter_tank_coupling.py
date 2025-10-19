"""
Inter-tank coupling mechanisms for multi-tank systems.

This module provides base classes and specific implementations for mass transfer
between tanks in a multi-tank hydrogen storage system.
"""

import math
from typing import List, Optional, Dict, Any
from src.fluids.flow_physics import FlowPhysics


class InterTankCoupling:
    """Base class for inter-tank mass transfer mechanisms."""

    def __init__(self, source_idx: int, target_idx: int, coupling_id: str = None):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.coupling_id = coupling_id or f"Coupling_{source_idx}→{target_idx}"
        self.is_active = False

    def evaluate(self, time_s, source_tank, dest_tank):
        """Evaluate the coupling (base implementation - should be overridden)."""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate mass flow rate [kg/s] when coupling is active."""
        raise NotImplementedError("Subclasses must implement calculate_flow_rate()")


class PressureTriggeredValve(InterTankCoupling):
    """Pressure-triggered valve with choked flow physics and hysteresis control."""

    def __init__(self, source_idx: int, target_idx: int,
                 p_open: float, p_close: float,
                 max_flow_rate: float = 0.005,
                 orifice_diameter: float = 0.002,
                 coupling_id: str = None,
                 flow_physics: Optional[FlowPhysics] = None):
        super().__init__(source_idx, target_idx, coupling_id)
        # Correct pressure logic: p_open is activation threshold, p_close is deactivation threshold
        self.p_open = p_open     # Valve opens when target pressure <= p_open
        self.p_close = p_close   # Valve closes when target pressure >= p_close
        self.max_flow_rate = max_flow_rate
        self.orifice_diameter = orifice_diameter

        # Hysteresis thresholds for clear logic
        self.activation_threshold = p_open    # Open valve when P_target ≤ this
        self.deactivation_threshold = p_close # Close valve when P_target ≥ this

        # Flow physics calculator (configuration-driven)
        self.flow_physics = flow_physics

        # Calculate effective area using flow physics or fallback
        if self.flow_physics and not self.flow_physics.use_flow_coefficient:
            orifice_area = math.pi * (orifice_diameter / 2)**2
            self.effective_area = self.flow_physics.discharge_coefficient * orifice_area
        elif self.flow_physics and self.flow_physics.use_flow_coefficient:
            self.effective_area = self.flow_physics.flow_coefficient
        else:
            # Fallback for backward compatibility
            self.effective_area = 0.6 * math.pi * (orifice_diameter / 2)**2

        if p_close <= p_open:
            raise ValueError(f"deactivation_threshold ({p_close/1e5:.1f} bar) must be > activation_threshold ({p_open/1e5:.1f} bar)")

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Evaluate valve state with hysteresis logic.

        Valve opens when target pressure ≤ activation_threshold
        Valve closes when target pressure ≥ deactivation_threshold
        """
        target_state = tank_states[self.target_idx]

        if target_state.pressure is None:
            target_state.compute_pressure()

        target_pressure = target_state.pressure

        # Valve opens when target pressure drops to or below activation threshold
        if not self.is_active and target_pressure <= self.activation_threshold:
            self.is_active = True
            print(f"t={t/3600:.2f}h: Valve {self.source_idx}→{self.target_idx} OPENED (P={target_pressure/1e5:.1f} bar ≤ {self.activation_threshold/1e5:.1f} bar)")

        # Valve closes when target pressure rises to or above deactivation threshold
        elif self.is_active and target_pressure >= self.deactivation_threshold:
            self.is_active = False
            print(f"t={t/3600:.2f}h: Valve {self.source_idx}→{self.target_idx} CLOSED (P={target_pressure/1e5:.1f} bar ≥ {self.deactivation_threshold/1e5:.1f} bar)")

        # Debug: Show valve state periodically when active
        if self.is_active and int(t*10) % 50 == 0:  # Every 5 seconds
            print(f"  Valve ACTIVE at t={t/3600:.3f}h: P_target={target_pressure/1e5:.1f}bar")

        return self.is_active

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate flow rate using configuration-driven flow physics."""
        if not self.is_active:
            return 0.0

        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        if source_state.fuel_mass < 1.0:
            return 0.0

        if source_state.pressure is None:
            source_state.compute_pressure()
        if target_state.pressure is None:
            target_state.compute_pressure()

        P1, P2 = source_state.pressure, target_state.pressure

        if P1 <= P2:
            return 0.0

        T1 = source_state.temperature
        rho1 = source_state.fuel_mass / source_state.tank.volume

        # Use configuration-driven flow physics if available
        if self.flow_physics:
            flow_rate = self.flow_physics.calculate_orifice_flow_rate(
                upstream_pressure=P1,
                downstream_pressure=P2,
                upstream_temperature=T1,
                upstream_density=rho1,
                orifice_diameter=self.orifice_diameter
            )

            # Apply valve capacity limit
            flow_rate = min(flow_rate, self.max_flow_rate)

            # Apply safety limits
            flow_rate = self.flow_physics.apply_safety_limits(flow_rate, source_state.fuel_mass)

        else:
            # Fallback calculation for backward compatibility
            # Gas properties - FALLBACK VALUES ONLY
            gamma = 1.4  # Heat capacity ratio for hydrogen
            R_specific = 4124  # J/(kg⋅K) specific gas constant for hydrogen
            P_crit_ratio = (2/(gamma+1))**(gamma/(gamma-1))  # Critical pressure ratio ≈ 0.528
            discharge_coeff = 0.6  # Discharge coefficient for sharp-edged orifice

            if P2/P1 < P_crit_ratio:
                # Choked flow - sonic velocity condition
                sonic_velocity = math.sqrt(gamma * R_specific * T1)
                flow_rate = discharge_coeff * self.effective_area * rho1 * sonic_velocity
            else:
                # Subsonic flow
                velocity = math.sqrt(2 * (P1 - P2) / rho1)
                flow_rate = discharge_coeff * self.effective_area * rho1 * velocity

            # Apply valve capacity limit
            flow_rate = min(flow_rate, self.max_flow_rate)

            # Safety limit: prevent excessive mass transfer rate
            max_safe_flow = 0.1 * source_state.fuel_mass
            flow_rate = min(flow_rate, max_safe_flow)

        return flow_rate

    def update_valve_state(self, target_pressure, t):
        """Update valve open/close state based on hysteresis logic

        Opens when target_pressure ≤ activation_threshold
        Closes when target_pressure ≥ deactivation_threshold
        """
        if not self.is_active:
            # Valve opens when target pressure drops to or below activation threshold
            if target_pressure <= self.activation_threshold:
                self.is_active = True
                print(f"  Valve {self.source_idx}→{self.target_idx} OPENED: P={target_pressure/1e5:.1f} ≤ {self.activation_threshold/1e5:.1f} bar")
        else:
            # Valve closes when target pressure rises to or above deactivation threshold
            if target_pressure >= self.deactivation_threshold:
                self.is_active = False
                print(f"  Valve {self.source_idx}→{self.target_idx} CLOSED: P={target_pressure/1e5:.1f} ≥ {self.deactivation_threshold/1e5:.1f} bar")

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Update valve state based on current pressures
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # For post-processing scenarios where valve state isn't preserved,
        # evaluate flow possibility directly based on current pressure conditions
        target_pressure = target_state.pressure
        source_pressure = source_state.pressure

        # Check if flow should occur based on activation criteria
        # (target pressure low enough AND source pressure higher than target)
        should_flow = (target_pressure <= self.activation_threshold and
                      source_pressure > target_pressure)

        if not should_flow:
            # Also update valve state for runtime consistency
            self.update_valve_state(target_pressure, t)
            if not self.is_active:
                return 0.0
        else:
            # Force valve active for flow calculation if conditions are met
            # This handles post-processing where valve state isn't persistent
            if not self.is_active:
                self.is_active = True

        # Create mock tank_states list for compatibility with calculate_flow_rate
        tank_states = [None] * max(self.source_idx + 1, self.target_idx + 1)
        tank_states[self.source_idx] = source_state
        tank_states[self.target_idx] = target_state

        # Call existing calculate_flow_rate method
        return self.calculate_flow_rate(t, tank_states)


class OHEXExtractionCoupling(InterTankCoupling):
    """OHEX (Outboard Heat Exchanger) extraction coupling for mission fuel demand."""

    def __init__(self, source_idx: int, mission_profile: dict,
                 min_extraction_pressure: float = 3.0e5,
                 coupling_id: str = None):
        # OHEX extraction is unidirectional (source → OHEX), so target_idx = -1 (no target tank)
        super().__init__(source_idx, -1, coupling_id)
        self.mission_profile = mission_profile
        self.min_extraction_pressure = min_extraction_pressure
        self.current_mission_flow_rate = 0.0

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Check if OHEX extraction should be active."""
        # Get mission flow rate requirement
        mission_flow_rate = self._get_mission_flow_rate(t)

        # Only activate if mission requires fuel
        if mission_flow_rate <= 0:
            return False

        # Check if source tank has sufficient pressure for extraction
        source_state = tank_states[self.source_idx]
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()

        source_pressure = source_state.pressure
        return source_pressure >= self.min_extraction_pressure

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate OHEX extraction flow rate based on mission demand."""
        # Get mission flow rate requirement
        mission_flow_rate = self._get_mission_flow_rate(t)
        self.current_mission_flow_rate = mission_flow_rate

        # Check activation conditions
        if not self.evaluate(t, tank_states):
            return 0.0

        # Return mission-required flow rate (positive = extraction from source)
        return mission_flow_rate

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Update source state pressure
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()

        # Create mock tank_states list for compatibility
        tank_states = [None] * (self.source_idx + 1)
        tank_states[self.source_idx] = source_state

        # Call existing calculate_flow_rate method
        return self.calculate_flow_rate(t, tank_states)

    def _get_mission_flow_rate(self, time: float) -> float:
        """
        Get mission flow rate requirement at given time.

        Args:
            time: Current simulation time [s]

        Returns:
            Required mass flow rate [kg/s]
        """
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


class MissionAdaptivePressureValve(InterTankCoupling):
    """
    Mission-adaptive pressure valve with dynamic threshold calculation.

    This valve updates its activation/deactivation thresholds based on real-time
    mission flow requirements and discharge piping characteristics.
    """

    def __init__(self, source_idx: int, target_idx: int,
                 mission_profile: dict,
                 discharge_piping: dict,
                 control_params: dict,
                 max_flow_rate: float = 0.005,
                 orifice_diameter: float = 0.001,
                 coupling_id: str = None,
                 flow_physics: Optional[FlowPhysics] = None,
                 target_tank_config: dict = None):
        print(f"DEBUG: Creating MissionAdaptivePressureValve {coupling_id} from tank {source_idx} to tank {target_idx}")
        super().__init__(source_idx, target_idx, coupling_id)

        # Mission profile parameters - handle both hardcoded and system-loaded missions
        self.mission_profile = mission_profile
        if mission_profile and 'time_s' in mission_profile:
            # Hardcoded mission profile in coupling rule
            self.mission_times = mission_profile['time_s']
            self.mission_flow_rates = mission_profile['flow_rate_kg_s']
        else:
            # Mission will be loaded from system configuration - set to None for now
            self.mission_times = None
            self.mission_flow_rates = None

        # Discharge piping characteristics
        self.pipe_diameter = discharge_piping['diameter_m']
        self.pipe_length = discharge_piping['length_m']
        self.pipe_roughness = discharge_piping['roughness_m']
        self.loss_coefficient = discharge_piping['loss_coefficient']
        self.choked_flow_enabled = discharge_piping.get('choked_flow_enabled', True)

        # Control parameters
        self.pressure_margin_bar = control_params['pressure_margin_bar']
        self.minimum_safety_pressure_bar = control_params.get('minimum_safety_pressure_bar', 3.0)  # Safety floor
        self.activation_delay_seconds = control_params.get('activation_delay_seconds', 0.0)  # Time delay before controller starts
        # Actuator dynamics (optional, with sensible defaults)
        self.valve_response_time_s = control_params.get('valve_response_time_s', 2.0)  # ~63% time constant
        self.max_valve_rate_per_s = control_params.get('max_valve_rate_per_s', 0.5)     # fraction per second

        # Target tank configuration (for minimum pressure)
        self.target_tank_config = target_tank_config or {}
        self.target_minimum_pressure_pa = self.target_tank_config.get('minimum_pressure', 300000)  # Default 3 bar

        # Flow parameters
        self.max_flow_rate = max_flow_rate
        self.orifice_diameter = orifice_diameter

        # Flow physics calculator (configuration-driven)
        self.flow_physics = flow_physics

        # Continuous control system state tracking
        self.current_flow_coefficient = 0.0  # Variable flow control (0 = closed, >0 = open)
        self.previous_flow_coefficient = 0.0  # For rate limiting valve movements
        self.is_active = False  # Track if valve is currently open
        self.first_timestep = True  # Flag to handle t=0 case
        # Monotonic control time to tolerate solver backtracking
        self._control_time = 0.0

        # Calculate effective area using flow physics or fallback
        if self.flow_physics and not self.flow_physics.use_flow_coefficient:
            orifice_area = math.pi * (orifice_diameter / 2)**2
            self.effective_area = self.flow_physics.discharge_coefficient * orifice_area
        elif self.flow_physics and self.flow_physics.use_flow_coefficient:
            self.effective_area = self.flow_physics.flow_coefficient
        else:
            # Fallback for backward compatibility
            self.effective_area = 0.6 * math.pi * (orifice_diameter / 2)**2

        # Dynamic threshold tracking
        self.current_activation_threshold = 3.0e5  # Default 3 bar
        self.current_deactivation_threshold = 4.0e5  # Default 4 bar
        self.current_mission_flow_rate = 0.0
        self.last_required_pressure = 3.0e5

        # PID Controller parameters (conservative tuning)
        self.kp = control_params.get('pid_kp', 0.1)  # Proportional gain
        self.ki = control_params.get('pid_ki', 0.01)  # Integral gain
        self.kd = control_params.get('pid_kd', 0.05)  # Derivative gain

        # Debug: print activation delay value
        print(f"  🔧 DEBUG: Activation delay set to {self.activation_delay_seconds:.1f} seconds")

        # PID state variables
        self.pid_integral = 0.0
        self.pid_previous_error = None  # Start with None to avoid stale derivative on first call
        self.pid_previous_time = 0.0

        # Anti-windup limits
        self.integral_max = 1000.0  # Prevent integral windup
        self.integral_min = -1000.0

        # Data collection for plotting
        self.time_history = []
        self.required_pressure_history = []
        self.activation_threshold_history = []
        self.mission_flow_history = []

    def set_mission_profile(self, mission_profile: dict):
        """Set mission profile after initialization (for system-loaded missions)."""
        if 'time_s' in mission_profile and 'flow_rate_kg_s' in mission_profile:
            self.mission_times = mission_profile['time_s']
            self.mission_flow_rates = mission_profile['flow_rate_kg_s']
            self.mission_profile = mission_profile
            print(f"   ✓ Mission profile loaded: {len(self.mission_times)} time points, max flow: {max(self.mission_flow_rates):.3f} kg/s")
        else:
            print(f"   ⚠️ Mission profile loading failed: missing keys in {list(mission_profile.keys())}")

    def get_mission_flow_rate(self, time: float) -> float:
        """Get mission flow rate from actual mission profile using safe interpolation."""
        if not hasattr(self, 'mission_times') or not hasattr(self, 'mission_flow_rates'):
            # Return fallback value instead of crashing
            return 0.0

        if len(self.mission_times) == 0 or len(self.mission_flow_rates) == 0:
            # Return fallback value instead of crashing
            return 0.0

        # Boundary conditions with safety checks
        if time <= self.mission_times[0]:
            return self.mission_flow_rates[0]
        elif time >= self.mission_times[-1]:
            return self.mission_flow_rates[-1]
        else:
            # Safe manual linear interpolation to avoid numpy issues
            for i in range(len(self.mission_times) - 1):
                if self.mission_times[i] <= time <= self.mission_times[i + 1]:
                    # Manual linear interpolation
                    t1, t2 = self.mission_times[i], self.mission_times[i + 1]
                    f1, f2 = self.mission_flow_rates[i], self.mission_flow_rates[i + 1]

                    if t2 - t1 == 0:  # Avoid division by zero
                        return f1

                    return f1 + (f2 - f1) * (time - t1) / (t2 - t1)

            # Fallback if no interpolation range found
            return self.mission_flow_rates[-1]

    def calculate_minimum_discharge_pressure(self, flow_rate_kg_s: float, lh2_density: float) -> float:
        """Calculate minimum tank pressure required to achieve target flow rate through discharge piping."""
        if flow_rate_kg_s <= 0:
            return 1e5  # 1 bar minimum for no flow

        # Use configuration-driven flow physics if available
        if self.flow_physics:
            # Get fluid properties for LH2 (approximate conditions)
            props = self.flow_physics.get_fluid_properties(300000, 20.4)  # 3 bar, 20.4K (approx LH2)
            kinematic_viscosity = props['kinematic_viscosity']
            sonic_velocity = props['speed_of_sound']

            # Use flow physics pipe pressure drop calculation
            pressure_drop = self.flow_physics.calculate_pipe_pressure_drop(
                flow_rate=flow_rate_kg_s,
                density=lh2_density,
                viscosity=kinematic_viscosity * lh2_density,  # Convert to dynamic viscosity
                pipe_diameter=self.pipe_diameter,
                pipe_length=self.pipe_length,
                pipe_roughness=self.pipe_roughness,
                loss_coefficient=self.loss_coefficient
            )

            min_tank_pressure = self.flow_physics.atmospheric_pressure + pressure_drop

        else:
            # Fallback calculation with hardcoded values
            # Convert mass flow to volumetric flow
            volumetric_flow = flow_rate_kg_s / lh2_density  # m³/s

            # Calculate flow velocity in pipe
            pipe_area = math.pi * (self.pipe_diameter / 2) ** 2
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

            # Minimum tank pressure = atmospheric + pressure drops
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

        # Store required pressure for diagnostic data
        self.last_required_pressure = min_pressure_pa

        # Set activation threshold with safety margin
        activation_pressure_bar = min_pressure_bar + self.pressure_margin_bar
        self.current_activation_threshold = activation_pressure_bar * 1e5

        # No deactivation threshold needed for PID control

        # Store for logging/debugging
        self.last_required_pressure = min_pressure_pa

        # Store data for plotting
        self.time_history.append(time)
        self.required_pressure_history.append(min_pressure_pa)  # Store in Pa for consistency
        self.activation_threshold_history.append(activation_pressure_bar * 1e5)  # Store in Pa
        self.mission_flow_history.append(self.current_mission_flow_rate)

    def calculate_pid_flow_rate(self, current_pressure_pa: float, target_pressure_pa: float, time: float) -> float:
        """
        Calculate desired flow rate using PID controller to maintain target pressure.

        Args:
            current_pressure_pa: Current tank pressure [Pa]
            target_pressure_pa: Target pressure to maintain [Pa]
            time: Current simulation time [s]

        Returns:
            Desired flow rate [kg/s] (0 to max_flow_rate)
        """
        # Calculate time step with stability checks
        if self.pid_previous_time == 0.0:
            dt = 1.0  # Default for first call
        else:
            dt = time - self.pid_previous_time

        # More robust time step handling to prevent numerical instability
        if dt <= 0.01:  # Prevent very small time steps that cause instability (increased to 0.01s)
            # Return the previous result to avoid derivative spikes and control chatter
            return getattr(self, '_last_pid_output', 0.0)

        # Limit maximum time step to prevent integration errors
        if dt > 5.0:  # Reduced from 10s to 5s for better control responsiveness
            dt = 5.0

        # Minimum pressure maintenance logic with continuous deadband control
        # Use smooth transition functions instead of discrete zones
        pressure_deficit = target_pressure_pa - current_pressure_pa

        # Control deadband: symmetric around target pressure for smooth oscillatory control
        control_deadband = 0.3e5  # 0.3 bar deadband 
        extended_deadband = 2.0 * control_deadband  # 0.6 bar for gradual reduction

        # Continuous gain modulation based on distance from target
        # Uses smooth sigmoid/tanh functions instead of discrete zones
        
        # Calculate distance-based gain factors (continuous, not discrete)
        deficit_normalized = pressure_deficit / control_deadband
        
        # Smooth gain reduction when pressure is above target (deficit < 0)
        # Using tanh for smooth continuous transition instead of discrete zones
        if pressure_deficit < -extended_deadband:
            # Pressure well above target - apply gradual flow reduction
            excess_normalized = abs(pressure_deficit + extended_deadband) / control_deadband
            reduction_factor = 0.1 + 0.9 * math.exp(-excess_normalized)  # Exponential decay, minimum 10%
        else:
            reduction_factor = 1.0
        
        # Continuous gain scaling based on proximity to target
        # Far from target (deficit > 1): use full gains
        # Near target (|deficit| < 1): reduce gains smoothly
        proximity_factor = math.tanh(abs(deficit_normalized))  # 0 to 1, smooth transition
        
        # Integral control factor: reduce integration when close to target
        integral_factor = math.tanh(max(0, deficit_normalized))  # Only integrate when below target
        
        # Use PID gains directly from configuration with continuous modulation
        kp_effective = self.kp * proximity_factor
        ki_effective = self.ki * integral_factor * 0.5  # More conservative integral
        kd_effective = self.kd * proximity_factor

        # Controller is active when pressure deficit warrants action
        if pressure_deficit >= -extended_deadband:
            # Use error as pressure deficit for PID calculation
            error = pressure_deficit
            
            # Proportional term with continuous gain modulation
            proportional = kp_effective * error

            # Integral term with anti-windup and continuous modulation
            if dt < 10.0 and integral_factor > 0.1:  # Only integrate when significantly below target
                self.pid_integral += error * dt * integral_factor
                self.pid_integral = max(self.integral_min, min(self.integral_max, self.pid_integral))
            elif pressure_deficit < -control_deadband:
                # Gradually reduce integral when pressure is above target
                decay_rate = 1.0 - 0.05 * min(1.0, abs(pressure_deficit) / control_deadband)
                self.pid_integral *= decay_rate
            integral = ki_effective * self.pid_integral

            # Derivative term with continuous smoothing
            if self.pid_previous_error is not None and dt > 0:
                raw_derivative = (error - self.pid_previous_error) / dt
                derivative = kd_effective * max(-1e4, min(1e4, raw_derivative))

                # Enhanced derivative smoothing for continuity
                if hasattr(self, '_smoothed_derivative'):
                    alpha = 0.85  # Higher smoothing for more continuity
                    self._smoothed_derivative = alpha * self._smoothed_derivative + (1 - alpha) * derivative
                    derivative = self._smoothed_derivative
                else:
                    self._smoothed_derivative = derivative
            else:
                derivative = 0.0

            # PID output with continuous modulation
            pid_output = proportional + integral + derivative

            # Scale PID output from pressure units to flow rate units
            pressure_to_flow_scaling = self.max_flow_rate / 1e5  # kg/s per Pa
            desired_flow = pid_output * pressure_to_flow_scaling

            # Apply continuous flow reduction (no discrete switching)
            desired_flow *= reduction_factor

            # Constrain to physical limits
            desired_flow = max(0.0, min(self.max_flow_rate, desired_flow))
        else:
            # Controller inactive (pressure well above extended deadband)
            desired_flow = 0.0

        # Cache the result for stability
        self._last_pid_output = desired_flow

        # Debug output every 100 seconds to track PID behavior
        if abs(time % 100) < 1.0 and pressure_deficit > -extended_deadband:
            deficit_bar = pressure_deficit / 1e5
            current_bar = current_pressure_pa / 1e5
            target_bar = target_pressure_pa / 1e5
            deadband_bar = control_deadband / 1e5

            # Show continuous factors instead of discrete zones
            proximity_pct = int(proximity_factor * 100)
            integral_pct = int(integral_factor * 100) 
            reduction_pct = int(reduction_factor * 100)
            
            print(f"  PID Debug t={time:.1f}s: P_current={current_bar:.1f}bar, P_target={target_bar:.1f}bar, deficit={deficit_bar:.2f}bar, proximity={proximity_pct}%, integral={integral_pct}%, reduction={reduction_pct}%, flow={desired_flow*1000:.1f}g/s, dt={dt:.6f}s")

        # Update state for next iteration
        # Only track error when controller is active (within extended deadband range)
        if pressure_deficit > -extended_deadband:
            self.pid_previous_error = pressure_deficit
        else:
            self.pid_previous_error = None  # Reset when outside control range
        self.pid_previous_time = time

        return desired_flow

    def get_future_flow_rate(self, time: float, lookahead_seconds: float = 5.0) -> float:
        """Get maximum flow rate expected in the next lookahead_seconds for predictive control."""
        future_time = time + lookahead_seconds
        current_flow = self.get_mission_flow_rate(time)
        future_flow = self.get_mission_flow_rate(future_time)
        return max(current_flow, future_flow)

    def update_continuous_control(self, time: float, lh2_pressure: float, lh2_density: float, base_flow_capacity: float = None):
        """Update valve control with realistic actuator dynamics and smooth interpolation.

        Args:
            time: Simulation time [s]
            lh2_pressure: Target tank pressure [Pa]
            lh2_density: Target tank density [kg/m³]
            base_flow_capacity: Optional current coupling flow capacity [kg/s] for fully open valve
                                computed from source/target states (P1,P2,T1,ρ1). When provided,
                                we map desired PID flow to valve coefficient against this capacity
                                to avoid systematic under/over-delivery when capacity ≠ max_flow_rate.
        """

        # Initialize control state tracking
        if not hasattr(self, '_last_control_time'):
            self._last_control_time = 0.0
            self._previous_coefficient = 0.0
            self._target_coefficient = 0.0
            self._pid_update_time = 0.0
            self._last_interpolation_time = 0.0
            self._last_coefficient = 0.0

        # Advance internal control clock monotonically (handles solver backtracking)
        if not hasattr(self, '_control_time'):
            self._control_time = 0.0
        self._control_time = max(self._control_time, time)
        ct = self._control_time

        # Check activation delay - controller activates AT the specified time
        if ct < self.activation_delay_seconds:
            # Controller is not yet active - force valve closed with smooth approach
            self._target_coefficient = 0.0
            self._previous_coefficient = self.current_flow_coefficient
            self._pid_update_time = ct
            self.is_active = False

            # Still store data for plotting (with zero values)
            self.time_history.append(time)
            self.required_pressure_history.append(0.0)  # No requirement yet
            self.activation_threshold_history.append(0.0)  # No threshold yet
            self.mission_flow_history.append(0.0)  # No mission flow consideration yet

            print(f"  Control t={time:.6f}s: Controller inactive (activation delay: {self.activation_delay_seconds:.1f}s, diff={time-self.activation_delay_seconds:.6f})")
        else:
            # 1) Interpolate toward the last PID target first (advance actuator state)
            if hasattr(self, '_target_coefficient') and hasattr(self, '_pid_update_time'):
                # Use monotonic control time deltas (clamped to >= 0)
                time_since_update = max(0.0, ct - self._pid_update_time)
                valve_response_time = max(1e-3, float(self.valve_response_time_s))

                # First-order response (exponential approach to target)
                response_factor = 1.0 - math.exp(-time_since_update / valve_response_time)

                # Smooth interpolation from previous to target coefficient
                interpolated_coefficient = (
                    self._previous_coefficient +
                    response_factor * (self._target_coefficient - self._previous_coefficient)
                )

                # Optional: Add rate limiting for extra realism
                max_rate = float(self.max_valve_rate_per_s)
                dt_interp = ct - self._last_interpolation_time if hasattr(self, '_last_interpolation_time') and self._last_interpolation_time > 0 else 0.01
                dt_interp = max(0.0, dt_interp)

                if dt_interp > 0.0:
                    max_change = max_rate * dt_interp
                    actual_change = interpolated_coefficient - self._last_coefficient
                    if abs(actual_change) > max_change:
                        interpolated_coefficient = self._last_coefficient + math.copysign(max_change, actual_change)

                self.current_flow_coefficient = interpolated_coefficient
                self._last_coefficient = self.current_flow_coefficient
                self._last_interpolation_time = ct

            # 2) Update the PID target on cadence (after interpolation)
            if ct - self._last_control_time >= 0.1:
                # Get target pressure based on mission requirements
                mission_flow_rate = self.get_mission_flow_rate(ct)

                # Calculate activation threshold with slight margin
                required_pressure_pa = self.calculate_minimum_discharge_pressure(mission_flow_rate, lh2_density)
                effective_target = required_pressure_pa + (self.pressure_margin_bar * 1e5)

                # Debug output to understand target calculation
                if abs(ct % 60) < 1.0:  # Every 60 seconds on control clock
                    print(f"  TARGET DEBUG t={ct:.0f}s: mission_flow={mission_flow_rate*1000:.1f}g/s, required_P={required_pressure_pa/1e5:.1f}bar, target_P={effective_target/1e5:.1f}bar")

                # Store data for plotting
                self.time_history.append(ct)
                self.required_pressure_history.append(required_pressure_pa)  # Store in Pa for consistency
                self.activation_threshold_history.append(effective_target)  # Store in Pa
                self.mission_flow_history.append(mission_flow_rate)

                # Use PID control for mission_adaptive_pressurization
                # Calculate desired flow rate using PID controller
                desired_flow_rate = self.calculate_pid_flow_rate(lh2_pressure, effective_target, ct)

                # Convert desired flow rate to valve coefficient (0.0 to 1.0)
                # Prefer mapping to the actual base flow capacity at current conditions (fully open valve)
                # to prevent under-delivery when capacity >> configured max, or over-demand when low.
                if base_flow_capacity is not None and base_flow_capacity > 1e-12:
                    denom = min(self.max_flow_rate, base_flow_capacity) if self.max_flow_rate > 0 else base_flow_capacity
                    target_coeff = max(0.0, min(1.0, desired_flow_rate / denom))
                else:
                    # Fallback to previous behavior (map to configured maximum)
                    if self.max_flow_rate > 0:
                        target_coeff = min(1.0, desired_flow_rate / self.max_flow_rate)
                    else:
                        target_coeff = 0.0

                # Store the target for next interpolation step
                self._previous_coefficient = self.current_flow_coefficient
                self._target_coefficient = target_coeff
                self._pid_update_time = ct
                self._last_control_time = ct

                # Debug output every 60 seconds (control clock)
                if abs(ct % 60) < 0.5:
                    current_bar = lh2_pressure / 1e5
                    target_bar = effective_target / 1e5
                    deficit_bar = (effective_target - lh2_pressure) / 1e5
                    cap_str = f", cap={base_flow_capacity*1000:.1f}g/s" if base_flow_capacity is not None else ""
                    print(f"  Control t={ct:.6f}s: P={current_bar:.1f}bar, target={target_bar:.1f}bar, deficit={deficit_bar:.2f}bar, target_coeff={target_coeff:.3f}, PID_flow={desired_flow_rate*1000:.1f}g/s{cap_str}")
                # Always log when a new PID target is set
                print(f"[PID SET] t={ct:.1f}s: target_coeff={target_coeff:.3f}, desired_flow={desired_flow_rate*1000:.1f} g/s")

        # Smooth interpolation between PID updates (realistic valve dynamics)
        if hasattr(self, '_target_coefficient') and hasattr(self, '_pid_update_time'):
            # Use monotonic control time deltas (clamped to >= 0)
            time_since_update = max(0.0, ct - self._pid_update_time)
            valve_response_time = max(1e-3, float(self.valve_response_time_s))
            
            # First-order response (exponential approach to target)
            response_factor = 1.0 - math.exp(-time_since_update / valve_response_time)
            
            # Smooth interpolation from previous to target coefficient
            interpolated_coefficient = (
                self._previous_coefficient + 
                response_factor * (self._target_coefficient - self._previous_coefficient)
            )
            
            # Optional: Add rate limiting for extra realism
            max_rate = float(self.max_valve_rate_per_s)
            dt = ct - self._last_interpolation_time if hasattr(self, '_last_interpolation_time') and self._last_interpolation_time > 0 else 0.01
            dt = max(0.0, dt)

            if dt > 0.0:
                max_change = max_rate * dt
                actual_change = interpolated_coefficient - self._last_coefficient
                if abs(actual_change) > max_change:
                    interpolated_coefficient = self._last_coefficient + math.copysign(max_change, actual_change)
            
            self.current_flow_coefficient = interpolated_coefficient
            self._last_coefficient = self.current_flow_coefficient
            self._last_interpolation_time = ct

        # Clamp to valid range
        self.current_flow_coefficient = max(0.0, min(1.0, self.current_flow_coefficient))

        # Valve is considered "active" if it has any opening
        self.is_active = self.current_flow_coefficient > 0.0

        # Throttled debug: show key control values every ~30s
        try:
            if abs(ct % 30.0) < 1.0:
                tgt = getattr(self, '_target_coefficient', 0.0)
                prev = getattr(self, '_previous_coefficient', 0.0)
                pidt = getattr(self, '_pid_update_time', 0.0)
                last_pid_flow = getattr(self, '_last_pid_output', 0.0)
                print(f"[VALVE CTRL] t={ct:.1f}s coeff={self.current_flow_coefficient:.3f} target={tgt:.3f} prev={prev:.3f} pid_flow={last_pid_flow*1000:.1f} g/s (ΔtPID={ct-pidt:.2f}s)")
        except Exception:
            pass



    def evaluate(self, t: float, tank_states: List) -> bool:
        """Determine if valve should be active using control system from t=0."""
        target_state = tank_states[self.target_idx]

        # Update dynamic thresholds based on current mission requirements
        if hasattr(target_state, 'density'):
            lh2_density = target_state.density
        else:
            lh2_density = target_state.fuel_mass / target_state.tank.volume

        self.update_dynamic_thresholds(t, lh2_density)

        # Check target tank pressure
        if target_state.pressure is None:
            target_state.compute_pressure()

        # Estimate current fully-open base flow capacity to improve coefficient mapping
        try:
            source_state = tank_states[self.source_idx]
            if source_state.pressure is None:
                source_state.compute_pressure()
            P1 = source_state.pressure
            P2 = target_state.pressure
            T1 = source_state.temperature
            rho1 = source_state.fuel_mass / source_state.tank.volume
            if self.flow_physics:
                base_capacity = self.flow_physics.calculate_orifice_flow_rate(
                    upstream_pressure=P1,
                    downstream_pressure=P2,
                    upstream_temperature=T1,
                    upstream_density=rho1,
                    orifice_diameter=self.orifice_diameter
                )
            else:
                gamma = 1.4
                R = 4124
                critical_pressure_ratio = (2/(gamma+1))**(gamma/(gamma-1))
                P_critical = P1 * critical_pressure_ratio
                if P2 <= P_critical:
                    rho_throat = rho1 * (2/(gamma+1))**(1/(gamma-1))
                    T_throat = T1 * (2/(gamma+1))
                    c_throat = (gamma * R * T_throat)**0.5
                    base_capacity = rho_throat * c_throat * self.effective_area
                else:
                    base_capacity = self.effective_area * (2 * rho1 * (P1 - P2))**0.5
            # Guard against negatives
            base_capacity = max(0.0, float(base_capacity))
        except Exception:
            base_capacity = None

        # Use continuous control to determine valve state and flow coefficient, using capacity if available
        self.update_continuous_control(t, target_state.pressure, lh2_density, base_flow_capacity=base_capacity)

        # Debug timing synchronization
        if abs(t % 60) < 0.1:  # Debug every 60s
            print(f"[SYNC DEBUG] t={t:.1f}s: evaluate() updated coeff to {self.current_flow_coefficient:.3f}")

        # Return True if valve is active (flow coefficient > 0)
        return self.current_flow_coefficient > 0.0

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate flow rate using time-based control with variable flow coefficient."""
        # print(f"[COUPLING DEBUG] t={t:.1f}s: Starting calculate_flow_rate")
        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        # Allow coupling flows from t=0 for discharge missions

        # Get current states
        if source_state.fuel_mass < 1.0:
            # print(f"[COUPLING DEBUG] t={t:.1f}s: Source mass too low ({source_state.fuel_mass:.1f}kg), returning 0")
            return 0.0

        # print(f"[COUPLING DEBUG] t={t:.1f}s: Computing pressures...")
        if source_state.pressure is None:
            source_state.compute_pressure()
        if target_state.pressure is None:
            target_state.compute_pressure()
        # print(f"[COUPLING DEBUG] t={t:.1f}s: Pressures computed - P1={source_state.pressure/1e5:.1f}bar, P2={target_state.pressure/1e5:.1f}bar")

        # Flow coefficient is updated by evaluate() method at proper intervals
        # DO NOT call update_continuous_control here - it causes multiple PID calls per timestep

        # print(f"[COUPLING DEBUG] t={t:.1f}s: Pressures computed - P1={source_state.pressure/1e5:.1f}bar, P2={target_state.pressure/1e5:.1f}bar")

        # If valve is completely closed, return zero flow
        # Note: flow coefficient is already updated by evaluate() method
        if self.current_flow_coefficient <= 0.0:
            if abs(t % 60) < 0.1:  # Debug every 60s
                print(f"[FLOW DEBUG] t={t:.1f}s: Valve closed (coeff={self.current_flow_coefficient:.3f}), returning 0")
            return 0.0

        P1, P2 = source_state.pressure, target_state.pressure

        # If no pressure differential, no flow possible regardless of valve position
        if P1 <= P2:
            # print(f"[COUPLING DEBUG] t={t:.1f}s: No pressure differential (P1={P1/1e5:.1f} <= P2={P2/1e5:.1f}), returning 0")
            return 0.0

        # print(f"[COUPLING DEBUG] t={t:.1f}s: Computing flow rate with coeff={self.current_flow_coefficient:.3f}")
        T1 = source_state.temperature
        rho1 = source_state.fuel_mass / source_state.tank.volume

        # Use configuration-driven flow physics with variable flow coefficient
        if self.flow_physics:
            # print(f"[COUPLING DEBUG] t={t:.1f}s: Using flow_physics.calculate_orifice_flow_rate...")
            # Calculate base flow rate
            flow_rate = self.flow_physics.calculate_orifice_flow_rate(
                upstream_pressure=P1,
                downstream_pressure=P2,
                upstream_temperature=T1,
                upstream_density=rho1,
                orifice_diameter=self.orifice_diameter
            )
            # print(f"[COUPLING DEBUG] t={t:.1f}s: Base flow rate calculated: {flow_rate:.6f} kg/s")

            # print(f"[COUPLING DEBUG] t={t:.1f}s: Base flow rate calculated: {flow_rate:.6f} kg/s")

            # Apply variable flow coefficient (acts like variable valve opening)
            base_flow_rate = flow_rate  # Store base flow rate for debug
            flow_rate *= self.current_flow_coefficient

            # Debug output for flow coefficient application
            if abs(t % 30) < 0.1:  # Debug every 30s
                print(f"[FLOW DEBUG] t={t:.1f}s: P1={P1/1e5:.1f}→P2={P2/1e5:.1f} bar, base={base_flow_rate*1000:.1f} g/s, coeff={self.current_flow_coefficient:.3f}, flow={flow_rate*1000:.1f} g/s")

            # Apply valve capacity limit
            flow_rate = min(flow_rate, self.max_flow_rate)

            # Apply safety limits
            flow_rate = self.flow_physics.apply_safety_limits(flow_rate, source_state.fuel_mass)
            # print(f"[COUPLING DEBUG] t={t:.1f}s: Final flow rate after limits: {flow_rate:.6f} kg/s")

        else:
            # print(f"[COUPLING DEBUG] t={t:.1f}s: Using fallback calculation...")
            # Fallback calculation with hardcoded values
            # Gas properties for hydrogen
            gamma = 1.4  # Heat capacity ratio for hydrogen
            R = 4124  # Specific gas constant for hydrogen [J/kg·K]

            # Critical pressure ratio for choked flow
            critical_pressure_ratio = (2/(gamma+1))**(gamma/(gamma-1))
            P_critical = P1 * critical_pressure_ratio

            if P2 <= P_critical:
                # Choked flow condition
                rho_throat = rho1 * (2/(gamma+1))**(1/(gamma-1))
                T_throat = T1 * (2/(gamma+1))
                c_throat = (gamma * R * T_throat)**0.5
                base_flow_rate = rho_throat * c_throat * self.effective_area
                # print(f"[COUPLING DEBUG] t={t:.1f}s: Choked flow: {flow_rate:.6f} kg/s")
            else:
                # Subsonic flow
                pressure_ratio = P2/P1
                base_flow_rate = self.effective_area * (2 * rho1 * (P1 - P2))**0.5
                # print(f"[COUPLING DEBUG] t={t:.1f}s: Subsonic flow: {flow_rate:.6f} kg/s")
            # Apply variable flow coefficient in fallback path as well
            flow_rate = base_flow_rate * self.current_flow_coefficient

            # Debug output for fallback coefficient application
            if abs(t % 30) < 0.1:
                print(f"[FLOW DEBUG] t={t:.1f}s (fallback): P1={P1/1e5:.1f}→P2={P2/1e5:.1f} bar, base={base_flow_rate*1000:.1f} g/s, coeff={self.current_flow_coefficient:.3f}, flow={flow_rate*1000:.1f} g/s")

            flow_rate = min(flow_rate, self.max_flow_rate)
            # print(f"[COUPLING DEBUG] t={t:.1f}s: Final fallback flow rate: {flow_rate:.6f} kg/s")

        # Minimal debug output every 60 seconds
        if abs(t % 60) < 0.1:
            print(f"  t={t:.0f}s: P2={P2/1e5:.1f}bar, flow={flow_rate:.3f}kg/s")

        return flow_rate

    def get_diagnostic_data(self) -> dict:
        """Get diagnostic data for plotting and analysis."""
        data = {
            'time_history': self.time_history.copy(),
            'required_pressure_history': self.required_pressure_history.copy(),
            'activation_threshold_history': self.activation_threshold_history.copy(),
            'mission_flow_history': self.mission_flow_history.copy(),
            'current_mission_flow_rate': self.current_mission_flow_rate,
            'current_activation_threshold_bar': self.current_activation_threshold / 1e5,
            'current_deactivation_threshold_bar': self.current_deactivation_threshold / 1e5,
            'last_required_pressure_bar': self.last_required_pressure / 1e5,
            'current_flow_coefficient': self.current_flow_coefficient,
            'first_timestep': self.first_timestep,
            'control_type': 'continuous'
        }

        # Add optional attributes if they exist
        if hasattr(self, 'previous_pressure'):
            data['previous_pressure'] = self.previous_pressure
        if hasattr(self, 'previous_time'):
            data['previous_time'] = self.previous_time

        return data

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Allow coupling flows from t=0 for discharge missions

        # Update pressure computations
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Create tank_states list for compatibility with evaluate method
        tank_states = [source_state, target_state]

        # Update valve state using PID controller (this sets current_flow_coefficient)
        self.evaluate(t, tank_states)

        # Use the time-based control flow rate calculation
        return self.calculate_flow_rate(t, tank_states)


class MassFlowPIDControlledValve(InterTankCoupling):
    """
    Mass flow PID controlled valve for direct flow-to-flow control.

    Instead of controlling pressure, this valve directly matches the target tank's
    outflow (mission demand) by adjusting the coupling inflow using PID control.
    This eliminates pressure dynamics from the control loop.
    """

    def __init__(self, source_idx: int, target_idx: int,
                 control_params: dict,
                 max_flow_rate: float = 0.005,
                 orifice_diameter: float = 0.001,
                 coupling_id: str = None,
                 flow_physics: Optional[FlowPhysics] = None,
                 mission_profile: dict = None):

        print(f"DEBUG: Creating MassFlowPIDControlledValve {coupling_id} from tank {source_idx} to tank {target_idx}")
        super().__init__(source_idx, target_idx, coupling_id)

        # Control parameters
        self.kp = control_params.get('pid_kp', 1.0)  # Proportional gain
        self.ki = control_params.get('pid_ki', 0.1)  # Integral gain
        self.kd = control_params.get('pid_kd', 0.01)  # Derivative gain

        # Flow parameters
        self.max_flow_rate = max_flow_rate
        self.orifice_diameter = orifice_diameter
        self.flow_physics = flow_physics

        # Mission profile for getting target outflow rates
        self.mission_profile = mission_profile

        # Add pipe parameters for compatibility with driver (default values)
        self.pipe_diameter = 0.005  # 5mm default
        self.pipe_length = 1.0      # 1m default
        self.pressure_margin_bar = 0.5  # Default pressure margin

        # PID state variables
        self.pid_integral = 0.0
        self.pid_previous_error = None
        self.pid_previous_time = 0.0
        self.integral_min = -1000.0  # Anti-windup limits
        self.integral_max = 1000.0

        # Calculate effective area for flow physics
        if self.flow_physics and not self.flow_physics.use_flow_coefficient:
            orifice_area = math.pi * (orifice_diameter / 2)**2
            self.effective_area = self.flow_physics.discharge_coefficient * orifice_area
        elif self.flow_physics and self.flow_physics.use_flow_coefficient:
            self.effective_area = self.flow_physics.flow_coefficient
        else:
            # Fallback for backward compatibility
            self.effective_area = 0.6 * math.pi * (orifice_diameter / 2)**2

        # State tracking
        self.is_active = False
        self.current_flow_coefficient = 0.0

        # Flow tracking for plotting
        self.outflow_history = {
            'time': [],
            'target_outflow': [],
            'actual_outflow': [],
            'coupling_inflow': []
        }

    def get_mission_outflow_rate(self, time: float) -> float:
        """Get the target outflow rate from mission profile at given time."""
        if not self.mission_profile or 'time_s' not in self.mission_profile:
            return 0.0

        times = self.mission_profile['time_s']
        flow_rates = self.mission_profile['flow_rate_kg_s']

        # Simple linear interpolation
        if time <= times[0]:
            return flow_rates[0]
        elif time >= times[-1]:
            return flow_rates[-1]
        else:
            # Find surrounding points and interpolate
            for i in range(len(times) - 1):
                if times[i] <= time <= times[i + 1]:
                    t_ratio = (time - times[i]) / (times[i + 1] - times[i])
                    return flow_rates[i] + t_ratio * (flow_rates[i + 1] - flow_rates[i])

        return 0.0

    def calculate_achievable_outflow(self, tank_state, target_outflow: float, time: float = 0.0) -> float:
        """
        Calculate the actual achievable outflow from target tank based on its pressure.

        This is the critical insight - the actual outflow depends on tank pressure
        and discharge characteristics, not just mission demand.
        """
        # Ensure pressure is computed
        if hasattr(tank_state, 'compute_pressure'):
            tank_state.compute_pressure()

        tank_pressure_bar = tank_state.pressure / 1e5  # Convert Pa to bar

        # Simplified discharge model: outflow depends on available pressure
        # This is where the physics comes in - tank can only discharge what pressure allows

        # Minimum pressure needed for any discharge (back-pressure, line losses, etc.)
        min_discharge_pressure = 2.0  # bar

        if tank_pressure_bar <= min_discharge_pressure:
            return 0.0  # Cannot discharge if pressure too low

        # Available pressure for discharge
        available_pressure = tank_pressure_bar - min_discharge_pressure

        # Discharge capacity based on available pressure (simplified linear model)
        # In reality this would involve choked/unchoked flow, orifice equations, etc.
        max_discharge_capacity = available_pressure * 0.010  # kg/s per bar of available pressure

        # Actual outflow is limited by both mission demand and tank discharge capacity
        achievable_outflow = min(target_outflow, max_discharge_capacity)

        # Add debug output to track this
        if target_outflow > 0.001:  # Only log when there's significant demand
            print(f"  OutflowCalc t={time:.1f}s: P={tank_pressure_bar:.1f}bar, target={target_outflow*1000:.1f}g/s, achievable={achievable_outflow*1000:.1f}g/s")

        return achievable_outflow

    def calculate_pid_flow_rate(self, target_outflow: float, actual_outflow: float, time: float) -> float:
        """
        Calculate desired coupling inflow using PID controller to match target outflow.

        Args:
            target_outflow: Desired outflow rate from target tank [kg/s]
            actual_outflow: Current outflow rate from target tank [kg/s]
            time: Current simulation time [s]

        Returns:
            Desired coupling inflow rate [kg/s]
        """
        # Calculate time step
        if self.pid_previous_time == 0.0:
            dt = 1.0
        else:
            dt = time - self.pid_previous_time

        if dt <= 1e-6:
            return getattr(self, '_last_pid_output', 0.0)

        # Flow error: positive when we need more coupling flow
        flow_error = target_outflow - actual_outflow

        # PID terms
        proportional = self.kp * flow_error

        # Integral with anti-windup
        if dt < 10.0:
            self.pid_integral += flow_error * dt
            self.pid_integral = max(self.integral_min, min(self.integral_max, self.pid_integral))
        integral = self.ki * self.pid_integral

        # Derivative
        if self.pid_previous_error is not None:
            derivative = self.kd * (flow_error - self.pid_previous_error) / dt
            derivative = max(-1e6, min(1e6, derivative))  # Limit spikes
        else:
            derivative = 0.0

        # PID output is desired coupling inflow
        desired_coupling_inflow = proportional + integral + derivative

        # Constrain to physical limits
        desired_coupling_inflow = max(0.0, min(self.max_flow_rate, desired_coupling_inflow))

        # Debug output
        if abs(time % 100) < 1.0:
            print(f"  FlowPID t={time:.1f}s: target={target_outflow*1000:.1f}g/s, actual={actual_outflow*1000:.1f}g/s, error={flow_error*1000:.1f}g/s, coupling={desired_coupling_inflow*1000:.1f}g/s")

        # Update state
        self.pid_previous_error = flow_error
        self.pid_previous_time = time
        self._last_pid_output = desired_coupling_inflow

        return desired_coupling_inflow

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate flow rate based on flow matching PID control."""
        if len(tank_states) <= max(self.source_idx, self.target_idx):
            return 0.0

        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        # Get target outflow from mission profile
        target_outflow = self.get_mission_outflow_rate(t)

        # Calculate actual achievable outflow based on target tank pressure
        actual_outflow = self.calculate_achievable_outflow(target_state, target_outflow, t)

        # Use PID to determine coupling inflow needed
        coupling_inflow = self.calculate_pid_flow_rate(target_outflow, actual_outflow, t)

        # Store data for plotting
        if t > 0 and len(self.outflow_history['time']) < 10000:  # Limit storage
            self.outflow_history['time'].append(t)
            self.outflow_history['target_outflow'].append(target_outflow)
            self.outflow_history['actual_outflow'].append(actual_outflow)
            self.outflow_history['coupling_inflow'].append(coupling_inflow)

        return coupling_inflow

    def set_mission_profile(self, mission_data: dict):
        """Set mission profile for getting target outflow rates."""
        self.mission_profile = mission_data
        print(f"DEBUG: MassFlowPIDControlledValve mission profile set with {len(mission_data.get('time_s', []))} time points")

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Update pressure computations
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Create tank_states list for compatibility
        tank_states = [source_state, target_state]

        return self.calculate_flow_rate(t, tank_states)