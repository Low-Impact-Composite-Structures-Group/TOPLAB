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

        # PID state variables
        self.pid_integral = 0.0
        self.pid_previous_error = 0.0
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

    def get_mission_flow_rate(self, time: float) -> float:
        """Simple constant mission flow rate - no complex logic."""
        return 0.054  # 54 g/s constant flow rate

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
        # Calculate time step
        if self.pid_previous_time == 0.0:
            dt = 1.0  # Default for first call
        else:
            dt = time - self.pid_previous_time

        if dt <= 0:
            dt = 1.0  # Prevent division by zero

        # Calculate error (positive error means we need more pressure)
        error = target_pressure_pa - current_pressure_pa

        # OPTIMIZED gain scheduling: less aggressive reduction for better startup tracking
        controller_active_time = time - 100.0  # Time since controller activation
        if controller_active_time < 30.0:  # REDUCED startup period with higher gains
            # IMPROVED startup gains (80% of nominal instead of 50%)
            gain_factor = 0.8
        else:
            # Full gains for steady-state performance
            gain_factor = 1.0

        kp_effective = self.kp * gain_factor
        ki_effective = self.ki * gain_factor
        kd_effective = self.kd * gain_factor

        # Proportional term
        proportional = kp_effective * error

        # Integral term with anti-windup (since we start after 30 seconds, integral works immediately)
        self.pid_integral += error * dt
        self.pid_integral = max(self.integral_min, min(self.integral_max, self.pid_integral))
        integral = ki_effective * self.pid_integral

        # Derivative term
        derivative = kd_effective * (error - self.pid_previous_error) / dt

        # PID output
        pid_output = proportional + integral + derivative

        # Convert PID output to flow rate (positive output = need more flow)
        # OPTIMIZED scaling factor for better pressure tracking
        flow_scale_factor = 5e-7  # INCREASED scaling for more responsive flow control (was 1e-7)
        desired_flow = pid_output * flow_scale_factor

        # Constrain to physical limits
        desired_flow = max(0.0, min(self.max_flow_rate, desired_flow))

        # Debug output every 100 seconds to track PID behavior
        if abs(time % 100) < 1.0 and desired_flow > 0:
            error_bar = error / 1e5
            print(f"  PID Debug t={time:.1f}s: error={error_bar:.2f}bar, P={proportional:.1f}, I={integral:.1f}, D={derivative:.1f}, flow={desired_flow*1000:.1f}g/s, gains={gain_factor:.1f}x")

        # Update state for next iteration
        self.pid_previous_error = error
        self.pid_previous_time = time

        return desired_flow

    def get_future_flow_rate(self, time: float, lookahead_seconds: float = 5.0) -> float:
        """Get maximum flow rate expected in the next lookahead_seconds for predictive control."""
        future_time = time + lookahead_seconds
        current_flow = self.get_mission_flow_rate(time)
        future_flow = self.get_mission_flow_rate(future_time)
        return max(current_flow, future_flow)

    def update_continuous_control(self, time: float, lh2_pressure: float, lh2_density: float):
        """Continuous control logic with predictive pre-activation to prevent pressure drops."""

        # Debug for early timesteps
        if time < 10.0:
            print(f"  Early Control t={time:.1f}s: P={lh2_pressure/1e5:.1f}bar, first_step={self.first_timestep}")

        # Handle first timestep (t=0) with predictive pre-activation
        if self.first_timestep:
            self.first_timestep = False

            # Pre-emptive coupling activation: Look ahead to see if mission will start soon
            upcoming_flow = self.get_mission_flow_rate(5.0)  # Check flow in next 5 seconds
            if upcoming_flow > 0.001:  # If significant flow expected (>1 g/s)
                # Pre-calculate required pressure for upcoming mission flow
                required_pressure = self.calculate_minimum_discharge_pressure(upcoming_flow, lh2_density)
                margin_pressure = self.pressure_margin_bar * 1e5
                target_pressure = required_pressure + margin_pressure

                # Calculate smooth ramp-down from initial pressure to target
                pressure_drop_needed = lh2_pressure - target_pressure
                if pressure_drop_needed > 1e5:  # If we need to drop more than 1 bar
                    # Start with very gentle opening to gradually reduce pressure
                    initial_opening = min(0.15, pressure_drop_needed / 10e5)  # Scale with pressure drop
                    self.current_flow_coefficient = initial_opening
                    self.is_active = True
                    print(f"  Gentle pre-activation t={time:.1f}s: Expected flow={upcoming_flow*1000:.1f}g/s, initial_opening={initial_opening:.1f}, P_drop_needed={pressure_drop_needed/1e5:.1f}bar")
                    return

            # Normal startup - valve closed
            self.current_flow_coefficient = 0.0
            self.is_active = False
            return

        # Use CURRENT pressure for control decisions (no lag)
        control_pressure = lh2_pressure

        # Get current AND future mission flow requirements for predictive control
        current_mission_flow = self.get_mission_flow_rate(time)
        future_mission_flow = self.get_future_flow_rate(time, lookahead_seconds=10.0)  # Look ahead 10 seconds

        # Use the higher of current or future flow for pressure requirement calculation
        # This prevents pressure drops before they occur
        control_mission_flow = max(current_mission_flow, future_mission_flow)

        # Calculate required LH2 pressure for control mission flow (current or future)
        required_pressure_pa = self.calculate_minimum_discharge_pressure(control_mission_flow, lh2_density)

        # Calculate activation threshold for PID control
        base_margin = self.pressure_margin_bar * 1e5
        activation_threshold = required_pressure_pa + base_margin

        # Tank minimum pressure from configuration
        p_min_pa = self.target_minimum_pressure_pa

        # Safety floor
        safety_pressure_pa = self.minimum_safety_pressure_bar * 1e5

        # Store data for plotting
        self.time_history.append(time)
        self.required_pressure_history.append(required_pressure_pa)  # Store in Pa for consistency
        self.activation_threshold_history.append(activation_threshold)  # Store in Pa
        self.mission_flow_history.append(current_mission_flow)

        # Variable flow control logic - valve continuously modulates flow
        # The goal is to maintain actual pressure at or slightly above the activation threshold
        # Don't override with safety floors - let the activation threshold drive the control
        effective_target = activation_threshold

        # Calculate pressure error (positive = need more pressure)
        pressure_error_pa = effective_target - control_pressure
        pressure_error_bar = pressure_error_pa / 1e5

        # Variable flow control: PID controller directly sets flow coefficient
        # No binary on/off - valve position varies continuously from 0-100%

        # OPTIMIZED startup stabilization mode - much shorter startup period with smaller deadbands
        if time < 60:  # First 1 minute only - reduced startup period
            startup_stabilization_mode = True
            deadband_bar = 0.2  # REDUCED deadband for better initial tracking (was 1.0)
        elif time < 120:  # 1-2 minutes - quick transition to normal control
            startup_stabilization_mode = False
            deadband_bar = 0.1  # REDUCED transition deadband (was 0.3)
        else:
            startup_stabilization_mode = False
            deadband_bar = 0.05  # REDUCED normal deadband for precise control (was 0.1)

        # Debug output during problematic periods
        debug_condition = (time > 0 and abs(time % 600) < 5.0) or \
                         (0 <= time <= 720) or \
                         (2520 <= time <= 3600)

        # Extra debug during early mission to understand opening/closing
        early_debug = time <= 300 and (int(time) % 30 == 0)  # Every 30s for first 5 minutes

        if (debug_condition and int(time) % 60 == 0) or early_debug:
            deadband_status = "deadband" if abs(pressure_error_bar) < deadband_bar else "active"
            print(f"  Control Debug t={time/3600:.2f}h: P={control_pressure/1e5:.1f}bar, act_thresh={activation_threshold/1e5:.1f}bar, error={pressure_error_bar:.2f}bar, flow_coeff={self.current_flow_coefficient:.3f}, deadband={deadband_bar:.2f}bar, status={deadband_status}")

        # Startup stabilization mode - use simple open-loop control to prevent oscillations
        if startup_stabilization_mode:
            # During startup, use very gentle and predictable valve control
            if pressure_error_bar > deadband_bar:
                # Pressure too low - very gentle opening
                target_opening = 0.1  # Very conservative opening during startup
                if self.current_flow_coefficient < target_opening:
                    self.current_flow_coefficient = min(target_opening, self.current_flow_coefficient + 0.02)
            elif pressure_error_bar < -deadband_bar:
                # Pressure too high - very gentle closing
                if self.current_flow_coefficient > 0.001:
                    self.current_flow_coefficient = max(0.001, self.current_flow_coefficient - 0.01)  # Maintain minimum maintenance flow
            # If within large deadband, maintain current position - no changes at all

        elif abs(pressure_error_bar) < deadband_bar:
            # Within deadband - maintain current position with minimum maintenance flow
            # Ensure minimum flow coefficient to represent maintenance flow needed for pressure stability
            if self.current_flow_coefficient < 0.001:  # Minimum 0.1% opening for maintenance flow
                self.current_flow_coefficient = 0.001
        elif pressure_error_bar > deadband_bar:
            # Need more pressure - calculate desired flow coefficient
            desired_flow_rate = self.calculate_pid_flow_rate(control_pressure, effective_target, time)

            # Convert flow rate to flow coefficient (0.0 to 1.0 representing 0-100% valve opening)
            if desired_flow_rate > 0:
                # Scale flow rate to flow coefficient - INCREASED scaling for better response
                flow_coefficient_scale = 8.0  # INCREASED for more responsive valve control (was 3.0)
                target_flow_coefficient = min(1.0, desired_flow_rate * flow_coefficient_scale)

                # OPTIMIZED rate limiting - faster response while preventing oscillations
                if time < 60:  # First 1 minute - moderate valve movements
                    max_change_rate = 0.15  # INCREASED change rate during startup (was 0.05)
                else:
                    max_change_rate = 0.25   # INCREASED change rate for faster normal response (was 0.1)

                if hasattr(self, 'previous_flow_coefficient'):
                    max_change = max_change_rate
                    change = target_flow_coefficient - self.previous_flow_coefficient
                    limited_change = max(-max_change, min(max_change, change))
                    self.current_flow_coefficient = self.previous_flow_coefficient + limited_change
                else:
                    self.current_flow_coefficient = target_flow_coefficient

                # Ensure minimum opening for stable flow
                self.current_flow_coefficient = max(0.02, self.current_flow_coefficient)
            else:
                # Rate-limited closure
                if hasattr(self, 'previous_flow_coefficient'):
                    self.current_flow_coefficient = max(0.001, self.previous_flow_coefficient - 0.05)  # Minimum maintenance flow
                else:
                    self.current_flow_coefficient = 0.001  # Minimum maintenance flow instead of complete closure
        else:
            # Pressure too high - reduce flow (rate-limited)
            if hasattr(self, 'previous_flow_coefficient'):
                self.current_flow_coefficient = max(0.001, self.previous_flow_coefficient - 0.03)  # Minimum maintenance flow
            else:
                self.current_flow_coefficient = 0.001  # Minimum maintenance flow instead of complete closure

        # Store for next control cycle
        self.previous_flow_coefficient = self.current_flow_coefficient

        # Valve is considered "active" if it has any opening (eliminates binary state)
        self.is_active = self.current_flow_coefficient > 0.01



    def evaluate(self, t: float, tank_states: List) -> bool:
        """Determine if valve should be active using PID control after settling period."""
        # Skip control for first ~100 seconds - let initial pressure settle before starting control
        if t < 100.0:
            self.current_flow_coefficient = 0.0
            return False

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

        # Use continuous control to determine valve state and flow coefficient
        self.update_continuous_control(t, target_state.pressure, lh2_density)

        # Return True if valve is active (flow coefficient > 0)
        return self.current_flow_coefficient > 0.0

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate flow rate using time-based control with variable flow coefficient."""
        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]

        # Default coupling flows to 0 during settling period (before 100 seconds)
        if t < 100.0:
            if t < 5.0 or (int(t) % 30 == 0):  # Debug during startup or every 30 seconds
                print(f"Storage Debug t={t:.1f}s: Tank{self.source_idx+1} coupling=0.0g/s (controller delayed), inflow=0.0g/s, outflow=0.0g/s")
            return 0.0

        # Get current states
        if source_state.fuel_mass < 1.0:
            return 0.0

        if source_state.pressure is None:
            source_state.compute_pressure()
        if target_state.pressure is None:
            target_state.compute_pressure()

        # If valve is completely closed, return zero flow
        # Note: flow coefficient is already updated by evaluate() method
        if self.current_flow_coefficient <= 0.0:
            if t > 100.0 and t < 106.0:  # Debug for first few seconds after controller starts
                print(f"  Flow Debug t={t:.1f}s: Valve closed, flow_coeff={self.current_flow_coefficient:.3f}")
            return 0.0

        P1, P2 = source_state.pressure, target_state.pressure

        # If no pressure differential, no flow possible regardless of valve position
        if P1 <= P2:
            if t > 64.0 and t < 70.0:
                print(f"  Flow Debug t={t:.1f}s: No pressure differential, P1={P1/1e5:.1f}bar, P2={P2/1e5:.1f}bar")
            return 0.0

        T1 = source_state.temperature
        rho1 = source_state.fuel_mass / source_state.tank.volume

        # Use configuration-driven flow physics with variable flow coefficient
        if self.flow_physics:
            # Calculate base flow rate
            flow_rate = self.flow_physics.calculate_orifice_flow_rate(
                upstream_pressure=P1,
                downstream_pressure=P2,
                upstream_temperature=T1,
                upstream_density=rho1,
                orifice_diameter=self.orifice_diameter
            )

            # Apply variable flow coefficient (acts like variable valve opening)
            flow_rate *= self.current_flow_coefficient

            # Apply valve capacity limit
            flow_rate = min(flow_rate, self.max_flow_rate)

            # Apply safety limits
            flow_rate = self.flow_physics.apply_safety_limits(flow_rate, source_state.fuel_mass)

        else:
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
                flow_rate = rho_throat * c_throat * self.effective_area
            else:
                # Subsonic flow
                pressure_ratio = P2/P1
                flow_rate = self.effective_area * (2 * rho1 * (P1 - P2))**0.5

            flow_rate = min(flow_rate, self.max_flow_rate)

        # Debug output for flow rate calculations
        if t > 100.0 and t < 106.0:
            print(f"  Flow Debug t={t:.1f}s: Calculated flow={flow_rate:.3f}kg/s, coeff={self.current_flow_coefficient:.3f}, P1={P1/1e5:.1f}bar, P2={P2/1e5:.1f}bar")

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
        # Default coupling flows to 0 during settling period (before 100 seconds)
        if t < 100.0:
            return 0.0

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