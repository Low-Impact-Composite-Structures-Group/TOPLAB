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
        # Ensure pressures are up to date before evaluating
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Build a temporary tank_states list so we can reuse evaluate()/calculate_flow_rate
        tank_states = [None] * max(self.source_idx + 1, self.target_idx + 1)
        tank_states[self.source_idx] = source_state
        tank_states[self.target_idx] = target_state

        # Always drive the hysteresis state machine explicitly
        self.evaluate(t, tank_states)

        # If closed, no flow
        if not self.is_active:
            return 0.0

        # Otherwise, compute flow with the selected physics and capacity limits
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
        # Optional shaping and cadence
        self.control_interval_s = float(control_params.get('control_interval_s', 0.1))
        self.target_filter_tau_s = float(control_params.get('target_pressure_filter_tau_s', 0.0))
        self.setpoint_bias_bar = float(control_params.get('setpoint_bias_bar', 0.0))
        self.control_deadband_bar = float(control_params.get('control_deadband_bar', 0.3))
        self.extended_deadband_factor = float(control_params.get('extended_deadband_factor', 2.0))
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
        # Target filtering state
        self._filtered_target_pressure = None
        self._last_target_update_time = None

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

        # Configurable control deadband (bar → Pa)
        control_deadband = max(0.0, self.control_deadband_bar) * 1e5
        extended_deadband = max(1.0, self.extended_deadband_factor) * control_deadband

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
        # Use configured integral gain modulated by how far below target we are
        # (previously had an extra 0.5 multiplier which made the integrator too sluggish
        #  and contributed to early-time undershoot). Removing that to improve tracking.
        ki_effective = self.ki * integral_factor
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
            if ct - self._last_control_time >= max(1e-3, float(self.control_interval_s)):
                # Get target pressure based on mission requirements
                mission_flow_rate = self.get_mission_flow_rate(ct)

                # Calculate activation threshold with slight margin
                required_pressure_pa = self.calculate_minimum_discharge_pressure(mission_flow_rate, lh2_density)
                # Add conservative bias on top of margin so we stay above requirement
                effective_target_raw = required_pressure_pa + ((self.pressure_margin_bar + self.setpoint_bias_bar) * 1e5)

                # Optional first-order low-pass filter on target to smooth jumps
                if self.target_filter_tau_s and self.target_filter_tau_s > 1e-6:
                    if self._filtered_target_pressure is None:
                        self._filtered_target_pressure = effective_target_raw
                        self._last_target_update_time = ct
                    else:
                        dtf = max(0.0, ct - (self._last_target_update_time or ct))
                        # Discrete-time first-order filter: y += (dt/tau)*(x - y)
                        alpha = dtf / max(1e-6, float(self.target_filter_tau_s))
                        # Limit alpha to [0,1] for stability
                        alpha = max(0.0, min(1.0, alpha))
                        self._filtered_target_pressure = (
                            self._filtered_target_pressure + alpha * (effective_target_raw - self._filtered_target_pressure)
                        )
                        self._last_target_update_time = ct
                    effective_target = self._filtered_target_pressure
                else:
                    effective_target = effective_target_raw

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


class PressureGovernorValve(InterTankCoupling):
    """
    Margin-free pressure governor for CH2 → LH2 pressurization (Option A).

    Tracks the minimum required LH2 pressure for mission discharge with a
    well-behaved first-order response. No hysteresis, no activation margin,
    and no PID. The desired CH2 flow is proportional to the pressure deficit
    (target − actual) with optional target low-pass filtering and realistic
    actuator dynamics and rate limits.

    Key ideas:
    - Compute required LH2 pressure from mission flow and discharge piping
    - Optionally smooth target pressure with a small LPF
    - Compute desired CH2 flow: mdot = gain_kg_s_per_bar · max(0, (P_req − P)/bar)
    - Map desired flow to valve coefficient using current base capacity
    - Apply first-order valve response and rate limiting for smooth behavior
    - No pressure margins or discrete activation thresholds
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
        print(f"DEBUG: Creating PressureGovernorValve {coupling_id} from tank {source_idx} to tank {target_idx}")
        super().__init__(source_idx, target_idx, coupling_id)

        # Mission profile parameters - can be injected later by system
        self.mission_profile = mission_profile or {}
        if mission_profile and 'time_s' in mission_profile:
            self.mission_times = mission_profile['time_s']
            self.mission_flow_rates = mission_profile['flow_rate_kg_s']
        else:
            self.mission_times = None
            self.mission_flow_rates = None

        # Discharge piping characteristics
        self.pipe_diameter = discharge_piping.get('diameter_m', 0.005)
        self.pipe_length = discharge_piping.get('length_m', 1.0)
        self.pipe_roughness = discharge_piping.get('roughness_m', 1.5e-6)
        self.loss_coefficient = discharge_piping.get('loss_coefficient', 1.0)
        self.choked_flow_enabled = discharge_piping.get('choked_flow_enabled', True)

        # Control parameters (margin-free)
        self.control_interval_s = float(control_params.get('control_interval_s', 1.0))
        self.target_filter_tau_s = float(control_params.get('target_pressure_filter_tau_s', 1.0))
        self.setpoint_bias_bar = float(control_params.get('setpoint_bias_bar', 0.0))
        self.min_source_pressure_bar = float(control_params.get('min_source_pressure_bar', 15.0))
        # New: measured pressure filter and soft-start controls
        self.measured_pressure_filter_tau_s = float(control_params.get('measured_pressure_filter_tau_s', 0.0))
        self.startup_delay_s = float(control_params.get('startup_delay_s', 0.0))
        self.startup_ramp_duration_s = float(control_params.get('startup_ramp_duration_s', 0.0))
        # New: explicit startup flow cap to limit delivered mass flow during ramp
        self.startup_flow_cap_kg_s = float(control_params.get('startup_flow_cap_kg_s', 0.0))

        # Optional: disable startup cap when pressure deficit is large
        self.disable_startup_cap_when_deficit = bool(control_params.get('disable_startup_cap_when_deficit', False))
        self.deficit_threshold_bar = float(control_params.get('deficit_threshold_bar', 0.0))

        # New: mission flow smoothing (EMA) to reduce jagged required-pressure signal
        self.mission_flow_filter_tau_s = float(control_params.get('mission_flow_filter_tau_s', 0.0))

        # First-order tracking – map pressure error to mass flow (kg/s per bar)
        self.pressure_gain_kg_s_per_bar = float(control_params.get('pressure_gain_kg_s_per_bar', 0.05))

        # Actuator dynamics
        self.valve_response_time_s = float(control_params.get('valve_response_time_s', 0.6))
        self.max_valve_rate_per_s = float(control_params.get('max_valve_rate_per_s', 1.2))
        # Optional lower opening rate during startup (defaults to normal rate)
        self.startup_max_valve_rate_per_s = float(control_params.get('startup_max_valve_rate_per_s', self.max_valve_rate_per_s))
        # New: directional (asymmetric) rate limits – allow faster closing to avoid overshoot
        self.max_valve_open_rate_per_s = float(control_params.get('max_valve_open_rate_per_s', self.max_valve_rate_per_s))
        self.max_valve_close_rate_per_s = float(control_params.get('max_valve_close_rate_per_s', self.max_valve_rate_per_s * 3.0))

        # New: target pressure slew-rate limiter (bar/s) to avoid aggressive chasing of fast-changing targets
        self.max_target_rise_bar_per_s = float(control_params.get('max_target_rise_bar_per_s', 0.0))
        self.max_target_fall_bar_per_s = float(control_params.get('max_target_fall_bar_per_s', 0.0))

        # Adaptive bias integrator (optional) to eliminate steady residual deficit
        self.bias_integrator_enabled = bool(control_params.get('bias_integrator_enabled', False))
        self.bias_gain_bar_per_s = float(control_params.get('bias_gain_bar_per_s', 0.0))
        self.bias_max_bar = float(control_params.get('bias_max_bar', 0.0))
        self.bias_decay_bar_per_s = float(control_params.get('bias_decay_bar_per_s', 0.0))

        # Target tank configuration (for diagnostics only)
        self.target_tank_config = target_tank_config or {}

        # Flow parameters
        self.max_flow_rate = max_flow_rate
        self.orifice_diameter = orifice_diameter
        self.flow_physics = flow_physics

        # Effective area for fallback physics
        if self.flow_physics and not self.flow_physics.use_flow_coefficient:
            orifice_area = math.pi * (orifice_diameter / 2)**2
            self.effective_area = self.flow_physics.discharge_coefficient * orifice_area
        elif self.flow_physics and self.flow_physics.use_flow_coefficient:
            self.effective_area = self.flow_physics.flow_coefficient
        else:
            self.effective_area = 0.6 * math.pi * (orifice_diameter / 2)**2

        # Internal state
        self.current_flow_coefficient = 0.0
        self._previous_coefficient = 0.0
        self._target_coefficient = 0.0
        self._last_interpolation_time = 0.0
        self._last_control_time = 0.0
        self._filtered_target_pressure = None
        self._last_target_update_time = None
        # Measurement filter state
        self._filtered_measured_pressure = None
        self._last_meas_update_time = None
        # Mission flow EMA state
        self._filtered_mission_flow = None
        self._last_flow_update_time = None
        # Target slew limiter state
        self._slew_limited_target = None
        self._last_slew_time = None
        # Track last effective signals and deficit for cap bypass
        self._last_effective_measured_pressure = None
        self._last_effective_target_pressure = None
        self._last_deficit_bar = 0.0
        self._last_cap_bypass = False
        # Adaptive bias state
        self._adaptive_bias_bar = 0.0
        self._last_bias_update_time = None

        # Diagnostics
        self.time_history = []
        self.required_pressure_history = []
        self.activation_threshold_history = []  # For plotting compatibility; equals target
        self.mission_flow_history = []
        self.last_required_pressure = 3.0e5
        self.current_mission_flow_rate = 0.0

    # Mission profile interface (same as adaptive valve)
    def set_mission_profile(self, mission_profile: dict):
        if 'time_s' in mission_profile and 'flow_rate_kg_s' in mission_profile:
            self.mission_times = mission_profile['time_s']
            self.mission_flow_rates = mission_profile['flow_rate_kg_s']
            self.mission_profile = mission_profile
            print(f"   ✓ Mission profile loaded into PressureGovernorValve: {len(self.mission_times)} points")
        else:
            print(f"   ⚠️ Mission profile missing keys for PressureGovernorValve: {list(mission_profile.keys())}")

    def get_mission_flow_rate(self, time: float) -> float:
        if not self.mission_times or not self.mission_flow_rates:
            return 0.0
        if time <= self.mission_times[0]:
            return self.mission_flow_rates[0]
        if time >= self.mission_times[-1]:
            return self.mission_flow_rates[-1]
        for i in range(len(self.mission_times) - 1):
            if self.mission_times[i] <= time <= self.mission_times[i + 1]:
                t1, t2 = self.mission_times[i], self.mission_times[i + 1]
                f1, f2 = self.mission_flow_rates[i], self.mission_flow_rates[i + 1]
                if t2 == t1:
                    return f1
                return f1 + (f2 - f1) * (time - t1) / (t2 - t1)
        return self.mission_flow_rates[-1]

    def calculate_minimum_discharge_pressure(self, flow_rate_kg_s: float, lh2_density: float) -> float:
        """Same method as adaptive valve (duplicated for isolation)."""
        if flow_rate_kg_s <= 0:
            return 1e5
        if self.flow_physics:
            props = self.flow_physics.get_fluid_properties(300000, 20.4)
            kinematic_viscosity = props['kinematic_viscosity']
            pressure_drop = self.flow_physics.calculate_pipe_pressure_drop(
                flow_rate=flow_rate_kg_s,
                density=lh2_density,
                viscosity=kinematic_viscosity * lh2_density,
                pipe_diameter=self.pipe_diameter,
                pipe_length=self.pipe_length,
                pipe_roughness=self.pipe_roughness,
                loss_coefficient=self.loss_coefficient
            )
            return self.flow_physics.atmospheric_pressure + pressure_drop
        # Fallback simplified calc
        volumetric_flow = flow_rate_kg_s / max(lh2_density, 1e-6)
        pipe_area = math.pi * (self.pipe_diameter / 2) ** 2
        velocity = volumetric_flow / max(pipe_area, 1e-9)
        reynolds = velocity * self.pipe_diameter / 1e-7
        friction_factor = 0.316 / (reynolds ** 0.25) if reynolds > 2300 else (64 / max(reynolds, 1e-6))
        friction_loss = friction_factor * (self.pipe_length / self.pipe_diameter) * (lh2_density * velocity**2 / 2)
        minor_loss = self.loss_coefficient * (lh2_density * velocity**2 / 2)
        total_pressure_drop = friction_loss + minor_loss
        if self.choked_flow_enabled and velocity > 500:
            total_pressure_drop *= 2.0
        return 1.01325e5 + total_pressure_drop

    def _startup_factor(self, t: float) -> float:
        """Linear ramp from 0→1 after an optional delay to soften initial actuation."""
        if t <= self.startup_delay_s:
            return 0.0
        if self.startup_ramp_duration_s <= 1e-9:
            return 1.0
        return max(0.0, min(1.0, (t - self.startup_delay_s) / self.startup_ramp_duration_s))

    def _current_max_rate(self, t: float) -> float:
        """Blend startup valve rate to normal rate according to startup ramp factor (legacy, opening only)."""
        s = self._startup_factor(t)
        return self.startup_max_valve_rate_per_s + (self.max_valve_open_rate_per_s - self.startup_max_valve_rate_per_s) * s

    def _current_directional_rate(self, t: float, delta: float) -> float:
        """Directional rate limit: slower opening (startup blended), faster closing to prevent spikes.

        Args:
            t: current time [s]
            delta: desired change in coefficient (positive = opening, negative = closing)
        Returns:
            Allowed coefficient rate [1/s]
        """
        if delta >= 0.0:
            # Opening – apply startup blended rate limit
            return self._current_max_rate(t)
        # Closing – use a (possibly much) higher close rate, not reduced during startup
        return float(self.max_valve_close_rate_per_s)

    def _compute_base_capacity(self, source_state, target_state) -> Optional[float]:
        try:
            if source_state.pressure is None:
                source_state.compute_pressure()
            if target_state.pressure is None:
                target_state.compute_pressure()
            P1, P2 = source_state.pressure, target_state.pressure
            T1 = source_state.temperature
            rho1 = source_state.fuel_mass / source_state.tank.volume
            if self.flow_physics:
                cap = self.flow_physics.calculate_orifice_flow_rate(
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
                    cap = rho_throat * c_throat * self.effective_area
                else:
                    cap = self.effective_area * (2 * rho1 * (P1 - P2))**0.5
            return max(0.0, float(cap))
        except Exception:
            return None

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Update valve coefficient using margin-free governor at a fixed cadence."""
        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        # Ensure pressures are available
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Extract LH2 density for required pressure computation
        if hasattr(target_state, 'density'):
            lh2_density = target_state.density
        else:
            lh2_density = target_state.fuel_mass / target_state.tank.volume

        # Compute mission-required pressure (Pa)
        mission_flow_rate = self.get_mission_flow_rate(t)
        # Optional EMA smoothing on mission flow to reduce target pressure jaggedness
        if self.mission_flow_filter_tau_s and self.mission_flow_filter_tau_s > 1e-6:
            if self._filtered_mission_flow is None:
                self._filtered_mission_flow = mission_flow_rate
                self._last_flow_update_time = t
            else:
                dtf = max(0.0, t - (self._last_flow_update_time or t))
                alpha_f = dtf / max(1e-6, float(self.mission_flow_filter_tau_s))
                alpha_f = max(0.0, min(1.0, alpha_f))
                self._filtered_mission_flow = self._filtered_mission_flow + alpha_f * (mission_flow_rate - self._filtered_mission_flow)
                self._last_flow_update_time = t
            smoothed_mission_flow = self._filtered_mission_flow
        else:
            smoothed_mission_flow = mission_flow_rate
        self.current_mission_flow_rate = smoothed_mission_flow
        required_pressure_pa = self.calculate_minimum_discharge_pressure(smoothed_mission_flow, lh2_density)

        # Optional LPF on measured pressure to reduce initial jitter (moved earlier for bias integration)
        measured_pressure = target_state.pressure
        if self.measured_pressure_filter_tau_s and self.measured_pressure_filter_tau_s > 1e-6:
            if self._filtered_measured_pressure is None:
                self._filtered_measured_pressure = measured_pressure
                self._last_meas_update_time = t
            else:
                dtm = max(0.0, t - (self._last_meas_update_time or t))
                alpha_m = dtm / max(1e-6, float(self.measured_pressure_filter_tau_s))
                alpha_m = max(0.0, min(1.0, alpha_m))
                self._filtered_measured_pressure = self._filtered_measured_pressure + alpha_m * (measured_pressure - self._filtered_measured_pressure)
                self._last_meas_update_time = t
            effective_measured = self._filtered_measured_pressure
        else:
            effective_measured = measured_pressure

        # Adaptive bias integration: integrate residual deficit to push target slightly above requirement
        if self.bias_integrator_enabled:
            dtb = max(0.0, t - (self._last_bias_update_time or t))
            self._last_bias_update_time = t
            deficit_bar_for_bias = max(0.0, (required_pressure_pa - effective_measured) / 1e5)
            if self.bias_gain_bar_per_s > 0.0:
                self._adaptive_bias_bar += self.bias_gain_bar_per_s * deficit_bar_for_bias * dtb
            # Decay when not needed
            if deficit_bar_for_bias <= 1e-6 and self.bias_decay_bar_per_s > 0.0:
                self._adaptive_bias_bar = max(0.0, self._adaptive_bias_bar - self.bias_decay_bar_per_s * dtb)
            # Clamp
            if self.bias_max_bar > 0.0:
                self._adaptive_bias_bar = min(self._adaptive_bias_bar, self.bias_max_bar)

        # Combine fixed bias and adaptive bias
        total_bias_bar = self.setpoint_bias_bar + (self._adaptive_bias_bar if self.bias_integrator_enabled else 0.0)
        target_pressure_pa = required_pressure_pa + total_bias_bar * 1e5

        # Optional LPF on target
        if self.target_filter_tau_s and self.target_filter_tau_s > 1e-6:
            if self._filtered_target_pressure is None:
                self._filtered_target_pressure = target_pressure_pa
                self._last_target_update_time = t
            else:
                dtf = max(0.0, t - (self._last_target_update_time or t))
                alpha = dtf / max(1e-6, float(self.target_filter_tau_s))
                alpha = max(0.0, min(1.0, alpha))
                self._filtered_target_pressure = self._filtered_target_pressure + alpha * (target_pressure_pa - self._filtered_target_pressure)
                self._last_target_update_time = t
            effective_target = self._filtered_target_pressure
        else:
            effective_target = target_pressure_pa

        # Optional slew-rate limiting on target pressure changes to avoid oscillations
        if (self.max_target_rise_bar_per_s and self.max_target_rise_bar_per_s > 1e-9) or \
           (self.max_target_fall_bar_per_s and self.max_target_fall_bar_per_s > 1e-9):
            if self._slew_limited_target is None:
                self._slew_limited_target = effective_target
                self._last_slew_time = t
            else:
                dts = max(0.0, t - (self._last_slew_time or t))
                prev = self._slew_limited_target
                delta = effective_target - prev
                limited = effective_target
                # Apply rise limit if configured
                if delta > 0.0 and (self.max_target_rise_bar_per_s and self.max_target_rise_bar_per_s > 1e-9):
                    max_rise = (self.max_target_rise_bar_per_s * 1e5) * dts
                    if delta > max_rise:
                        limited = prev + max_rise
                # Apply fall limit if configured
                if delta < 0.0 and (self.max_target_fall_bar_per_s and self.max_target_fall_bar_per_s > 1e-9):
                    max_fall = (self.max_target_fall_bar_per_s * 1e5) * dts
                    if -delta > max_fall:
                        limited = prev - max_fall
                self._slew_limited_target = limited
                self._last_slew_time = t
            effective_target = self._slew_limited_target

        # Store effective pressures for later logic
        self._last_effective_measured_pressure = effective_measured
        self._last_effective_target_pressure = effective_target

        # Diagnostics history (store in Pa)
        self.time_history.append(t)
        self.required_pressure_history.append(required_pressure_pa)
        self.activation_threshold_history.append(effective_target)
        # Store smoothed mission flow for diagnostics
        self.mission_flow_history.append(self.current_mission_flow_rate)
        self.last_required_pressure = required_pressure_pa

        # Only recompute target at control cadence
        if (t - self._last_control_time) >= max(1e-3, self.control_interval_s):
            # Source tank constraint
            if (source_state.pressure / 1e5) < self.min_source_pressure_bar or source_state.fuel_mass < 0.1:
                desired_flow_rate = 0.0
            else:
                # Margin-free error (bar) using filtered measurement if enabled
                error_bar = max(0.0, (effective_target - effective_measured) / 1e5)
                # Save deficit for downstream cap bypass logic
                self._last_deficit_bar = error_bar
                desired_flow_rate = self.pressure_gain_kg_s_per_bar * error_bar

                # Apply startup ramp to suppress initial spikes
                desired_flow_rate *= self._startup_factor(t)

            # Map desired flow to valve coefficient using current capacity
            base_capacity = self._compute_base_capacity(source_state, target_state)
            if base_capacity is not None and base_capacity > 1e-12:
                denom = min(self.max_flow_rate, base_capacity) if self.max_flow_rate > 0 else base_capacity
                target_coeff = max(0.0, min(1.0, desired_flow_rate / max(denom, 1e-12)))
            else:
                target_coeff = max(0.0, min(1.0, desired_flow_rate / max(self.max_flow_rate, 1e-12)))

            # Apply startup flow cap as an additional ceiling on valve coefficient
            if self.startup_flow_cap_kg_s > 0.0:
                allowed_startup_flow = self.startup_flow_cap_kg_s * self._startup_factor(t)
                # Determine if we should bypass the startup cap due to a large deficit
                cap_bypass = self.disable_startup_cap_when_deficit and (self._last_deficit_bar > self.deficit_threshold_bar)
                self._last_cap_bypass = bool(cap_bypass)
                if not cap_bypass:
                    if base_capacity is not None and base_capacity > 1e-12:
                        denom_cap = min(self.max_flow_rate, base_capacity) if self.max_flow_rate > 0 else base_capacity
                    else:
                        denom_cap = max(self.max_flow_rate, 1e-12)
                    coeff_ceiling = max(0.0, min(1.0, allowed_startup_flow / max(denom_cap, 1e-12)))
                    target_coeff = min(target_coeff, coeff_ceiling)

            # Update actuator target and timestamp
            self._previous_coefficient = self.current_flow_coefficient
            self._target_coefficient = target_coeff
            self._last_control_time = t

        # First-order valve response with rate limiting
        time_since_update = max(0.0, t - (self._last_interpolation_time or 0.0))
        response_factor = 1.0 - math.exp(-time_since_update / max(1e-3, self.valve_response_time_s))
        interpolated = self._previous_coefficient + response_factor * (self._target_coefficient - self._previous_coefficient)

        # Rate limit (directional)
        # Use a reduced opening rate during startup, but allow faster closing to avoid overshoot
        delta = interpolated - getattr(self, '_last_coefficient', 0.0)
        max_change = self._current_directional_rate(t, delta) * time_since_update
        if abs(delta) > max_change:
            interpolated = getattr(self, '_last_coefficient', 0.0) + math.copysign(max_change, delta)

        self.current_flow_coefficient = max(0.0, min(1.0, interpolated))
        self._last_coefficient = self.current_flow_coefficient
        self._last_interpolation_time = t

        # Active if any opening
        self.is_active = self.current_flow_coefficient > 0.0
        return self.is_active

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Scale base orifice flow by current valve coefficient with safety limits."""
        if self.current_flow_coefficient <= 0.0:
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

        if self.flow_physics:
            base_flow = self.flow_physics.calculate_orifice_flow_rate(
                upstream_pressure=P1,
                downstream_pressure=P2,
                upstream_temperature=T1,
                upstream_density=rho1,
                orifice_diameter=self.orifice_diameter
            )
            flow_rate = base_flow * self.current_flow_coefficient
            flow_rate = min(flow_rate, self.max_flow_rate)
            flow_rate = self.flow_physics.apply_safety_limits(flow_rate, source_state.fuel_mass)
        else:
            gamma = 1.4
            R = 4124
            critical_pressure_ratio = (2/(gamma+1))**(gamma/(gamma-1))
            P_critical = P1 * critical_pressure_ratio
            if P2 <= P_critical:
                rho_throat = rho1 * (2/(gamma+1))**(1/(gamma-1))
                T_throat = T1 * (2/(gamma+1))
                c_throat = (gamma * R * T_throat)**0.5
                base_flow = rho_throat * c_throat * self.effective_area
            else:
                base_flow = self.effective_area * (2 * rho1 * (P1 - P2))**0.5
            flow_rate = min(base_flow * self.current_flow_coefficient, self.max_flow_rate)

        # Safety: apply startup flow cap again to the final flow
        if self.startup_flow_cap_kg_s > 0.0:
            # Apply same bypass during final flow capping
            cap_bypass = self.disable_startup_cap_when_deficit and (self._last_deficit_bar > self.deficit_threshold_bar)
            if not cap_bypass:
                flow_rate = min(flow_rate, self.startup_flow_cap_kg_s * self._startup_factor(t))

        # Minimal periodic debug
        if abs(t % 60) < 0.1:
            print(f"  [GOV FLOW] t={t:.0f}s: P2={P2/1e5:.1f}bar, coeff={self.current_flow_coefficient:.3f}, flow={flow_rate*1000:.1f} g/s")

        return flow_rate

    def get_diagnostic_data(self) -> dict:
        return {
            'time_history': self.time_history.copy(),
            'required_pressure_history': self.required_pressure_history.copy(),
            'activation_threshold_history': self.activation_threshold_history.copy(),
            'mission_flow_history': self.mission_flow_history.copy(),
            'current_mission_flow_rate': self.current_mission_flow_rate,
            'current_activation_threshold_bar': (self.activation_threshold_history[-1]/1e5) if self.activation_threshold_history else 0.0,
            'current_deactivation_threshold_bar': (self.activation_threshold_history[-1]/1e5) if self.activation_threshold_history else 0.0,
            'last_required_pressure_bar': self.last_required_pressure / 1e5,
            'current_flow_coefficient': self.current_flow_coefficient,
            'last_deficit_bar': self._last_deficit_bar,
            'startup_cap_bypass': self._last_cap_bypass,
            'adaptive_bias_bar': self._adaptive_bias_bar,
            'control_type': 'governor'
        }

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows.

        Mirrors other coupling classes: ensure pressures, update valve via
        evaluate(), then compute flow using calculate_flow_rate().
        """
        # Ensure pressure computations are up to date
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Create tank_states list for compatibility with evaluate/calc methods
        tank_states = [None] * (max(self.source_idx, self.target_idx) + 1)
        tank_states[self.source_idx] = source_state
        tank_states[self.target_idx] = target_state

        # Update internal control state and valve coefficient
        self.evaluate(t, tank_states)

        # Compute and return the resulting flow rate
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


class FeedforwardPressureEnforcer(InterTankCoupling):
    """
    Feedforward pressure enforcer (three-step per-timestep controller).

    At each derivative evaluation (t):
      1) Compute required LH2 pressure P_req from mission outflow and discharge piping
      2) Predict the inflow mdot_in such that, over dt, P2_pred(m + (mdot_in - m_out)*dt) ≈ P_req
         - Use isochoric T-frozen predictor (EOS to compute P from [m,T])
         - Solve mdot_in by bisection within [0, capacity]
      3) Apply hydraulic capacity of the CH2→LH2 valve (orifice/pipe model) and source limits

    This avoids reactive oscillations by setting the algebraic flow needed to meet P_req.
    If capacity is insufficient, we saturate and record a deficit.
    """

    def __init__(self,
                 source_idx: int,
                 target_idx: int,
                 mission_profile: dict,
                 discharge_piping: dict,
                 control_params: dict,
                 max_flow_rate: float = 0.1,
                 orifice_diameter: float = 0.003,
                 coupling_id: str = None,
                 flow_physics: Optional[FlowPhysics] = None,
                 target_tank_config: dict = None):
        super().__init__(source_idx, target_idx, coupling_id)
        self.mission_profile = mission_profile or {}
        self.mission_times = mission_profile.get('time_s') if mission_profile else None
        self.mission_flow_rates = mission_profile.get('flow_rate_kg_s') if mission_profile else None

        # Piping and valve parameters
        self.pipe_diameter = discharge_piping.get('diameter_m', 0.005)
        self.pipe_length = discharge_piping.get('length_m', 1.0)
        self.pipe_roughness = discharge_piping.get('roughness_m', 1.5e-6)
        self.loss_coefficient = discharge_piping.get('loss_coefficient', 1.0)
        self.choked_flow_enabled = discharge_piping.get('choked_flow_enabled', True)

        self.max_flow_rate = max_flow_rate
        self.orifice_diameter = orifice_diameter
        self.flow_physics = flow_physics

    # Control tunables
        self.min_source_pressure_bar = float(control_params.get('min_source_pressure_bar', 15.0))
        self.safety_mass_fraction_per_s = float(control_params.get('safety_mass_fraction_per_s', 0.1))
        self.bracket_margin = float(control_params.get('bracket_margin', 1.2))  # capacity × margin as upper bisection bound
        self.max_bisection_iters = int(control_params.get('max_bisection_iters', 10))
        self.tol_pressure_bar = float(control_params.get('tol_pressure_bar', 0.02))  # 0.02 bar

        # Linearized predictor for mdot (faster than repeated EOS in bisection)
        self.use_linearized_predictor = bool(control_params.get('use_linearized_predictor', True))

        # Branch-free enforcement horizon (fixed, independent of solver dt)
        self.enforcement_horizon_s = float(control_params.get('enforcement_horizon_s', 0.5))

        # Overpressure margin: bias target upward slightly to avoid chatter at equality
        self.overpressure_margin_bar = float(control_params.get('overpressure_margin_bar', 0.5))

        # Smooth, always-open bias added algebraically (no branching)
        self.mdot_bias_kg_s = float(control_params.get('mdot_bias_kg_s', 0.0))

        # Optional soft saturation smoothness (fraction of capacity). If 0, use hard clamp.
        self.soft_saturation_rel = float(control_params.get('soft_saturation_rel', 0.0))

        # New: nonzero low-flow floor to prevent apparent valve "closures" from exact zeros
        # Kept branch-free by using it as the soft-clamp lower bound (scaled by capacity)
        self.min_flow_floor_kg_s = float(control_params.get('min_flow_floor_kg_s', 0.0))
        self.min_flow_floor_rel_cap = float(control_params.get('min_flow_floor_rel_cap', 0.0))

        # Diagnostics
        self._last_required_pressure = 0.0
        self._last_deficit_bar = 0.0
        self._last_capacity_kg_s = 0.0
        self._last_mission_outflow = 0.0
        self._last_mdot_in = 0.0

        # Histories for overlay plotting (aligned with MissionAdaptive/Governor)
        self.time_history = []
        self.required_pressure_history = []
        self.activation_threshold_history = []  # use effective target (req + margin)
        self.mission_flow_history = []

        # Internal state (kept minimal to stay memoryless across solver retries)
        self._last_time = None

    def set_mission_profile(self, mission_profile: dict):
        if 'time_s' in mission_profile and 'flow_rate_kg_s' in mission_profile:
            self.mission_times = mission_profile['time_s']
            self.mission_flow_rates = mission_profile['flow_rate_kg_s']
            self.mission_profile = mission_profile

    def _get_mission_outflow(self, time: float) -> float:
        if not self.mission_times or not self.mission_flow_rates:
            return 0.0
        if time <= self.mission_times[0]:
            return self.mission_flow_rates[0]
        if time >= self.mission_times[-1]:
            return self.mission_flow_rates[-1]
        for i in range(len(self.mission_times) - 1):
            t1, t2 = self.mission_times[i], self.mission_times[i+1]
            if t1 <= time <= t2:
                f1, f2 = self.mission_flow_rates[i], self.mission_flow_rates[i+1]
                if t2 == t1:
                    return f1
                return f1 + (f2 - f1) * (time - t1) / (t2 - t1)
        return self.mission_flow_rates[-1]

    def _required_pressure_pa(self, flow_rate_kg_s: float, lh2_density: float, temperature_K: float, pressure_Pa: float) -> float:
        # Reuse same method as governor for consistency
        if flow_rate_kg_s <= 0:
            return 1e5
        if self.flow_physics:
            # Prefer current LH2 state for viscosity; fall back to nominal if needed
            try:
                props = self.flow_physics.get_fluid_properties(pressure_Pa, temperature_K)
            except Exception:
                props = self.flow_physics.get_fluid_properties(300000, 20.4)
            kinematic_viscosity = props['kinematic_viscosity']
            pressure_drop = self.flow_physics.calculate_pipe_pressure_drop(
                flow_rate=flow_rate_kg_s,
                density=lh2_density,
                viscosity=kinematic_viscosity * lh2_density,
                pipe_diameter=self.pipe_diameter,
                pipe_length=self.pipe_length,
                pipe_roughness=self.pipe_roughness,
                loss_coefficient=self.loss_coefficient
            )
            try:
                atm = self.flow_physics.atmospheric_pressure
            except Exception:
                atm = 1.01325e5
            return atm + pressure_drop
        # Fallback
        volumetric_flow = flow_rate_kg_s / max(lh2_density, 1e-9)
        area = math.pi * (self.pipe_diameter/2)**2
        velocity = volumetric_flow / max(area, 1e-12)
        reynolds = velocity * self.pipe_diameter / 1e-7
        f = 0.316 / (reynolds**0.25) if reynolds > 2300 else 64/max(reynolds, 1e-6)
        dp = f * (self.pipe_length/self.pipe_diameter) * (lh2_density * velocity**2 / 2) + self.loss_coefficient * (lh2_density * velocity**2 / 2)
        if self.choked_flow_enabled and velocity > 500:
            dp *= 2.0
        return 1.01325e5 + dp

    def _capacity_kg_s(self, source_state, target_state) -> float:
        # Fully open valve capacity at current P1,P2,T1,ρ1
        try:
            if source_state.pressure is None:
                source_state.compute_pressure()
            if target_state.pressure is None:
                target_state.compute_pressure()
            P1, P2 = source_state.pressure, target_state.pressure
            T1 = source_state.temperature
            rho1 = source_state.fuel_mass / source_state.tank.volume
            if self.flow_physics:
                base = self.flow_physics.calculate_orifice_flow_rate(
                    upstream_pressure=P1, downstream_pressure=P2,
                    upstream_temperature=T1, upstream_density=rho1,
                    orifice_diameter=self.orifice_diameter
                )
            else:
                gamma, R = 1.4, 4124
                critical_ratio = (2/(gamma+1))**(gamma/(gamma-1))
                if P2 <= P1 * critical_ratio:
                    rho_th = rho1 * (2/(gamma+1))**(1/(gamma-1))
                    T_th = T1 * (2/(gamma+1))
                    c_th = (gamma * R * T_th)**0.5
                    area = 0.6 * math.pi * (self.orifice_diameter/2)**2
                    base = rho_th * c_th * area
                else:
                    area = 0.6 * math.pi * (self.orifice_diameter/2)**2
                    base = area * (2 * rho1 * max(P1 - P2, 0.0))**0.5
            base = max(0.0, float(base))
            base = min(base, self.max_flow_rate)
            # Safety limit relative to source mass
            base = min(base, self.safety_mass_fraction_per_s * max(source_state.fuel_mass, 0.0))
            return base
        except Exception:
            return 0.0

    def _predict_pressure_pa(self, mass: float, temperature: float, volume: float) -> float:
        # Isochoric EOS mapping: P = PropsSI("P", "T", T, "Dmass", rho, "hydrogen")
        try:
            from CoolProp.CoolProp import PropsSI
            density = mass / max(volume, 1e-9)
            if temperature <= 0:
                return 1e5
            return float(PropsSI("P", "T", temperature, "Dmass", density, "hydrogen"))
        except Exception:
            return 1e5

    def evaluate(self, t: float, tank_states: List) -> bool:
        # Stateless w.r.t valve position; compute at each call
        return True

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]
        # Debug gating
        try:
            import os
            debug_enabled = os.environ.get("H2_DEBUG", "0") == "1"
        except Exception:
            debug_enabled = False

        # Ensure pressures
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # If source constraints fail, no flow
        if (source_state.pressure or 0.0)/1e5 < self.min_source_pressure_bar:
            self._last_capacity_kg_s = 0.0
            self._last_deficit_bar = max(0.0, (target_state.pressure - 0.0)/1e5)  # meaningless but non-negative
            self._last_mdot_in = 0.0
            return 0.0

        # Step 1: Required pressure from mission outflow
        if hasattr(target_state, 'density'):
            lh2_density = target_state.density
        else:
            lh2_density = target_state.fuel_mass / target_state.tank.volume

        # Mission outflow (branch-free; no EMA/slew inside the solver)
        mission_outflow = self._get_mission_outflow(t)
        self._last_mission_outflow = mission_outflow
        P_now = target_state.pressure
        T = target_state.temperature
        P_req_raw = self._required_pressure_pa(mission_outflow, lh2_density, T, P_now)
        # Overpressure margin (Pa)
        P_req_eff = P_req_raw + self.overpressure_margin_bar * 1e5
        # Default to unfiltered target; can be filtered later if enabled
        P_target_ctrl = P_req_eff
        self._last_required_pressure = P_target_ctrl

        # Log diagnostics for overlay plot
        self.time_history.append(t)
        self.required_pressure_history.append(P_req_raw)
        self.activation_threshold_history.append(P_req_eff)
        self.mission_flow_history.append(mission_outflow)

        # Step 2: Invert mdot_in so that P_predicted ≈ P_req_eff over a fixed horizon
        dt_ctrl = max(1e-6, self.enforcement_horizon_s)

        m = target_state.fuel_mass
        V = target_state.tank.volume
        T = target_state.temperature

        # Helper to compute predicted pressure for a candidate mdot_in
        def pred_p(mdot_in: float) -> float:
            m_next = m + (mdot_in - mission_outflow) * dt_ctrl
            m_next = max(1e-9, m_next)
            return self._predict_pressure_pa(m_next, T, V)

        # Bisection bounds: [0, capacity×margin]
        capacity = self._capacity_kg_s(source_state, target_state)
        self._last_capacity_kg_s = capacity
        upper = max(0.0, min(self.max_flow_rate, capacity * self.bracket_margin))
        lower = 0.0

        # Current pressure (for linearization)
        err_bar = (P_target_ctrl - P_now) / 1e5

        # Always compute mdot_star by solving the algebraic equation (branch-free)
        if upper <= 1e-12:
            mdot_star = 0.0
        else:
            used_linear = False
            if self.use_linearized_predictor:
                try:
                    # Estimate dP/drho at current state via finite difference (isochoric, T-frozen)
                    rho = target_state.fuel_mass / max(V, 1e-12)
                    delta_rho = max(1e-6 * rho, 1e-7)
                    from CoolProp.CoolProp import PropsSI
                    P_plus = float(PropsSI("P", "T", T, "Dmass", rho + delta_rho, "hydrogen"))
                    dP_drho = (P_plus - P_now) / delta_rho
                    dP_dm = dP_drho / max(V, 1e-12)
                    if dP_dm > 1e-9:
                        deltaP = P_target_ctrl - P_now
                        mdot_star = mission_outflow + (deltaP / dP_dm) / dt_ctrl
                        used_linear = True
                except Exception:
                    used_linear = False

            if not used_linear:
                # Bisection for mdot_in where pred_p(mdot) ≈ P_target_ctrl
                mdot_lo, mdot_hi = lower, upper
                for _ in range(max(5, self.max_bisection_iters)):
                    mid = 0.5 * (mdot_lo + mdot_hi)
                    P_mid = pred_p(mid)
                    if P_mid < P_target_ctrl:
                        mdot_lo = mid
                    else:
                        mdot_hi = mid
                    if abs((P_mid - P_target_ctrl)/1e5) <= self.tol_pressure_bar:
                        break
                mdot_star = 0.5 * (mdot_lo + mdot_hi)
            # Discretize via simple explicit integration (sufficient under solver small dt)
        # Step 3: Apply capacity/safety limits with optional soft saturation and a tiny bias
        def _softplus(x: float, k: float) -> float:
            # Numerically stable softplus
            if k <= 0:
                return max(0.0, x)
            if x / k > 50:
                return x  # avoid overflow
            if x / k < -50:
                return 0.0
            return k * math.log1p(math.exp(x / k))

        def _soft_clamp(x: float, lo: float, hi: float, softness: float) -> float:
            if softness <= 0:
                return min(max(x, lo), hi)
            # shift to apply softplus at both bounds
            y = lo + _softplus(x - lo, softness)
            return hi - _softplus(hi - y, softness)

        mdot_cmd = mdot_star + max(0.0, self.mdot_bias_kg_s)

        cap = min(self.max_flow_rate, max(0.0, capacity))
        softness = self.soft_saturation_rel * max(cap, 1e-9)

        # Compute a soft, physics-consistent lower bound: don't exceed capacity,
        # and scale floor with capacity to avoid demanding unattainable flow at tiny caps
        if cap > 0.0:
            floor_cap_scaled = self.min_flow_floor_rel_cap * cap if self.min_flow_floor_rel_cap > 0.0 else 0.0
            lo_bound = max(0.0, min(self.min_flow_floor_kg_s, floor_cap_scaled))
        else:
            lo_bound = 0.0

        mdot_applied = _soft_clamp(mdot_cmd, lo_bound, cap, softness)

        # Record deficit relative to target if saturated
        P_pred_cap = pred_p(mdot_applied)
        self._last_deficit_bar = max(0.0, (P_target_ctrl - P_pred_cap) / 1e5)
        self._last_mdot_in = mdot_applied
        # Track last time for optional filtering/slew in future steps
        self._last_time = t

        if debug_enabled and (int(t) % 60 == 0 or t <= 5.0):
            try:
                print(f"[FF DEBUG] t={t:.1f}s: mdot={mdot_applied*1000:.2f}g/s, cap={cap*1000:.1f}g/s, deficit={self._last_deficit_bar:.3f}bar")
            except Exception:
                pass

        return mdot_applied

    def get_diagnostic_data(self) -> dict:
        """Expose histories so the pressure requirements overlay plot can use them."""
        return {
            'time_history': self.time_history.copy(),
            'required_pressure_history': self.required_pressure_history.copy(),
            'activation_threshold_history': self.activation_threshold_history.copy(),
            'mission_flow_history': self.mission_flow_history.copy(),
            'current_mission_flow_rate': self._last_mission_outflow,
            'current_activation_threshold_bar': (self.activation_threshold_history[-1]/1e5) if self.activation_threshold_history else 0.0,
            'current_deactivation_threshold_bar': (self.activation_threshold_history[-1]/1e5) if self.activation_threshold_history else 0.0,
            'last_required_pressure_bar': (self.required_pressure_history[-1]/1e5) if self.required_pressure_history else 0.0,
            'current_flow_coefficient': 1.0,  # conceptual (always-open, algebraic)
            'control_type': 'feedforward'
        }

    def calculate_flow(self, source_state, target_state, t):
        # Ensure pressure updates and delegate
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()

        # Construct tank_states list and delegate to calculate_flow_rate
        max_idx = max(self.source_idx, self.target_idx)
        tank_states = [None]*(max_idx+1)
        tank_states[self.source_idx] = source_state
        tank_states[self.target_idx] = target_state
        return self.calculate_flow_rate(t, tank_states)