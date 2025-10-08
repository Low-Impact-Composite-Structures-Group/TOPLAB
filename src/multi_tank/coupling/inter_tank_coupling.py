"""
Inter-tank coupling mechanisms for multi-tank systems.

This module provides base classes and specific implementations for mass transfer
between tanks in a multi-tank hydrogen storage system.
"""

import math
from typing import List


class InterTankCoupling:
    """Base class for inter-tank mass transfer mechanisms."""

    def __init__(self, source_idx: int, target_idx: int, coupling_id: str = None):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.coupling_id = coupling_id or f"Coupling_{source_idx}→{target_idx}"
        self.is_active = False

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Determine if coupling should be active at current conditions."""
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
                 coupling_id: str = None):
        super().__init__(source_idx, target_idx, coupling_id)
        # Correct pressure logic: p_open is activation threshold, p_close is deactivation threshold
        self.p_open = p_open     # Valve opens when target pressure <= p_open
        self.p_close = p_close   # Valve closes when target pressure >= p_close
        self.max_flow_rate = max_flow_rate
        # Hysteresis thresholds for clear logic
        self.activation_threshold = p_open    # Open valve when P_target ≤ this
        self.deactivation_threshold = p_close # Close valve when P_target ≥ this
        self.effective_area = math.pi * (orifice_diameter / 2)**2

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
        """Calculate choked flow rate using compressible gas physics."""
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

        # Gas properties
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

        # Check if valve should open/close
        self.update_valve_state(target_state.pressure, t)

        # Calculate flow rate if valve is active
        if not self.is_active:
            return 0.0

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
                 coupling_id: str = None):
        super().__init__(source_idx, target_idx, coupling_id)

        # Mission profile parameters - store full dict for get_mission_flow_rate method
        self.mission_profile = mission_profile
        self.mission_times = mission_profile['time_s']
        self.mission_flow_rates = mission_profile['flow_rate_kg_s']

        # Discharge piping characteristics
        self.pipe_diameter = discharge_piping['diameter_m']
        self.pipe_length = discharge_piping['length_m']
        self.pipe_roughness = discharge_piping['roughness_m']
        self.loss_coefficient = discharge_piping['loss_coefficient']
        self.choked_flow_enabled = discharge_piping.get('choked_flow_enabled', True)

        # Control parameters
        self.pressure_margin_bar = control_params['pressure_margin_bar']
        self.deactivation_margin_bar = control_params['deactivation_margin_bar']

        # Flow parameters
        self.max_flow_rate = max_flow_rate
        self.effective_area = math.pi * (orifice_diameter / 2)**2

        # Dynamic threshold tracking
        self.current_activation_threshold = 3.0e5  # Default 3 bar
        self.current_deactivation_threshold = 4.0e5  # Default 4 bar
        self.current_mission_flow_rate = 0.0
        self.last_required_pressure = 3.0e5

        # Data collection for plotting
        self.time_history = []
        self.required_pressure_history = []
        self.activation_threshold_history = []
        self.mission_flow_history = []

    def get_mission_flow_rate(self, time: float) -> float:
        """Get required mission flow rate at current time from profile."""
        if time <= self.mission_times[0]:
            return self.mission_flow_rates[0]
        elif time >= self.mission_times[-1]:
            return self.mission_flow_rates[-1]
        else:
            # Linear interpolation between mission profile points
            for i in range(len(self.mission_times) - 1):
                if self.mission_times[i] <= time <= self.mission_times[i + 1]:
                    t1, t2 = self.mission_times[i], self.mission_times[i + 1]
                    f1, f2 = self.mission_flow_rates[i], self.mission_flow_rates[i + 1]
                    return f1 + (f2 - f1) * (time - t1) / (t2 - t1)
            return 0.0

    def calculate_minimum_discharge_pressure(self, flow_rate_kg_s: float, lh2_density: float) -> float:
        """Calculate minimum tank pressure required to achieve target flow rate through discharge piping."""
        if flow_rate_kg_s <= 0:
            return 1e5  # 1 bar minimum for no flow

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

        # Set deactivation threshold with additional margin to prevent oscillation
        deactivation_pressure_bar = activation_pressure_bar + self.deactivation_margin_bar
        self.current_deactivation_threshold = deactivation_pressure_bar * 1e5

        # Store for logging/debugging
        self.last_required_pressure = min_pressure_pa

        # Store data for plotting
        self.time_history.append(time)
        self.required_pressure_history.append(min_pressure_pa)  # Store in Pa for consistency
        self.activation_threshold_history.append(activation_pressure_bar * 1e5)  # Store in Pa
        self.mission_flow_history.append(self.current_mission_flow_rate)

    def evaluate(self, t: float, tank_states: List) -> bool:
        """Determine if valve should be active with dynamic threshold calculation."""
        target_state = tank_states[self.target_idx]

        # Update dynamic thresholds based on current mission requirements
        if hasattr(target_state, 'density'):
            lh2_density = target_state.density
        else:
            lh2_density = target_state.fuel_mass / target_state.tank.volume

        self.update_dynamic_thresholds(t, lh2_density)

        # Check target tank pressure against dynamic thresholds
        if target_state.pressure is None:
            target_state.compute_pressure()

        target_pressure = target_state.pressure

        # Use hysteresis: different thresholds for activation vs deactivation
        if not self.is_active:
            # Activation condition: target pressure below activation threshold
            should_activate = target_pressure < self.current_activation_threshold
            if should_activate:
                self.is_active = True
            return should_activate
        else:
            # Already active - use deactivation threshold to prevent oscillation
            should_deactivate = target_pressure >= self.current_deactivation_threshold
            if should_deactivate:
                self.is_active = False
                return False
            return True

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        """Calculate choked flow rate using compressible gas physics."""
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

        return min(flow_rate, self.max_flow_rate)

    def get_diagnostic_data(self) -> dict:
        """Get diagnostic data for plotting and analysis."""
        return {
            'time_history': self.time_history.copy(),
            'required_pressure_history': self.required_pressure_history.copy(),
            'activation_threshold_history': self.activation_threshold_history.copy(),
            'mission_flow_history': self.mission_flow_history.copy(),
            'current_mission_flow_rate': self.current_mission_flow_rate,
            'current_activation_threshold_bar': self.current_activation_threshold / 1e5,
            'current_deactivation_threshold_bar': self.current_deactivation_threshold / 1e5,
            'last_required_pressure_bar': self.last_required_pressure / 1e5
        }

    def calculate_flow(self, source_state, target_state, t):
        """Interface method expected by TankSystem._calculate_coupling_flows"""
        # Update pressure computations
        if hasattr(source_state, 'compute_pressure'):
            source_state.compute_pressure()
        if hasattr(target_state, 'compute_pressure'):
            target_state.compute_pressure()



        # Update dynamic thresholds based on current mission time
        self.update_dynamic_thresholds(t, target_state.density)

        # Create mock tank_states list for compatibility with calculate_flow_rate
        tank_states = [source_state, target_state]

        # Update valve state based on current conditions
        self.evaluate(t, tank_states)

        # Use the existing calculate_flow_rate method
        return self.calculate_flow_rate(t, tank_states)