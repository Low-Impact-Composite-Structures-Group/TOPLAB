"""Feedforward pressure enforcer: sets the algebraic inflow that holds a target tank at its required pressure."""

import math
from typing import Any, Dict, List, Optional

from CoolProp.CoolProp import PropsSI

from toplab.config.loader import ConfigError
from toplab.coupling.base import EdgeContext, InterTankCoupling, as_float
from toplab.coupling.registry import register_edge
from toplab.network.network import CouplingEdge
from toplab.fluids.fluid import DEFAULT_COOLPROP_FLUID
from toplab.legacy.flow_physics import FlowPhysics
from toplab.peripheral_components.base import PeripheralFlowState


@register_edge("feedforward_pressure_enforcer")
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
                 source_tank_config: dict = None,
                 target_tank_config: dict = None,
                 fluid: str = DEFAULT_COOLPROP_FLUID):
        super().__init__(source_idx, target_idx, coupling_id)
        self.fluid = fluid
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

        # Source tank configuration (for minimum operating pressure)
        self.source_tank_config = source_tank_config or {}
        source_min_pressure_pa = self.source_tank_config.get('minimum_pressure', None)

        # Target tank configuration (for minimum pressure)
        self.target_tank_config = target_tank_config or {}
        self.target_minimum_pressure_pa = self.target_tank_config.get('minimum_pressure', None)

        # FAIL-FAST VALIDATION: Required parameters must be present
        if self.target_minimum_pressure_pa is None:
            raise ValueError(
                "FeedforwardPressureEnforcer requires target tank 'minimum_pressure' "
                "in operating_limits. This is needed to compute required pressure for mission flow."
            )

    # Control tunables - Pull min_source_pressure from tank config if not in control_params
        if 'min_source_pressure_bar' in control_params:
            # Explicit override in control_params takes precedence
            self.min_source_pressure_bar = float(control_params['min_source_pressure_bar'])
        elif source_min_pressure_pa is not None:
            # Use source tank's minimum_pressure from operating_limits
            self.min_source_pressure_bar = source_min_pressure_pa / 1e5
        else:
            raise ValueError(
                "FeedforwardPressureEnforcer requires either 'min_source_pressure_bar' in control_parameters "
                "or source tank 'minimum_pressure' in operating_limits."
            )

        if 'safety_mass_fraction_per_s' not in control_params:
            raise ValueError(
                "FeedforwardPressureEnforcer requires 'safety_mass_fraction_per_s' in control_parameters. "
                "This sets the maximum fraction of source tank mass that can flow per second."
            )
        self.safety_mass_fraction_per_s = float(control_params['safety_mass_fraction_per_s'])

        if 'enforcement_horizon_s' not in control_params:
            raise ValueError(
                "FeedforwardPressureEnforcer requires 'enforcement_horizon_s' in control_parameters. "
                "This sets the prediction horizon for pressure enforcement (typically 2× solver timestep)."
            )
        self.enforcement_horizon_s = float(control_params['enforcement_horizon_s'])

        # Optional parameters with reasonable defaults
        self.bracket_margin = float(control_params.get('bracket_margin', 1.2))  # capacity × margin as upper bisection bound

        # Overpressure margin: bias target upward slightly to avoid chatter at equality
        self.overpressure_margin_bar = float(control_params.get('overpressure_margin_bar', 0.5))

        # Maximum target pressure ceiling to prevent overshoot during mission transitions
        self.max_target_pressure_bar = float(control_params.get('max_target_pressure_bar', float('inf')))

        # Physical actuator dynamics (to reduce chatter without hiding physics):
        # - First-order valve response with time constant tau (faster for responsiveness)
        self.valve_time_constant_s = float(control_params.get('valve_time_constant_s', 0.5))


        # Smooth activation curve around the target to prevent on/off behavior
        self.activation_softness_bar = float(control_params.get('activation_softness_bar', 0.2))

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
        # Valve coefficient state for smooth modulation against changing capacity
        self._coeff = 0.0
        self._last_coeff = 0.0
        self._last_coeff_time = None

    def set_mission_profile(self, mission_profile: dict):
        if 'time_s' in mission_profile and 'flow_rate_kg_s' in mission_profile:
            self.mission_times = mission_profile['time_s']
            self.mission_flow_rates = mission_profile['flow_rate_kg_s']
            self.mission_profile = mission_profile

    def _get_mission_outflow(self, time: float) -> float:
        if not self.mission_times or not self.mission_flow_rates:
            # Debug: Log when mission data is missing
            if not hasattr(self, '_warned_no_mission'):
                print(f"WARNING [FF] No mission data available! mission_times={self.mission_times}, mission_flow_rates={self.mission_flow_rates}")
                self._warned_no_mission = True
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
        """Calculate required LH2 pressure for given mission flow rate.

        Conservative assumption: discharge is always saturated liquid at current tank temperature.
        This is reasonable because:
        - Tank outlets are designed to draw liquid (baffles, bottom positioning)
        - Saturated liquid gives minimum pressure drop (highest density)
        - Eliminates oscillations from phase transition predictions

        Args:
            flow_rate_kg_s: Mission outflow rate [kg/s]
            lh2_density: LH2 density [kg/m³] (not used - we use saturated liquid)
            temperature_K: LH2 temperature [K]
            pressure_Pa: Current LH2 pressure [Pa] (not used)

        Returns:
            Required pressure [Pa] including atmospheric and piping losses
        """
        import os
        from CoolProp.CoolProp import PropsSI
        debug_enabled = os.environ.get("H2_DEBUG", "0") == "1"

        import os
        from CoolProp.CoolProp import PropsSI
        debug_enabled = os.environ.get("H2_DEBUG", "0") == "1"

        if flow_rate_kg_s <= 0:
            # Zero flow requires only atmospheric pressure - use from flow_physics
            if self.flow_physics:
                return self.flow_physics.atmospheric_pressure
            else:
                raise RuntimeError(
                    "FeedforwardPressureEnforcer requires flow_physics to be configured. "
                    "Cannot compute required pressure without atmospheric_pressure."
                )

        if not self.flow_physics:
            raise RuntimeError(
                "FeedforwardPressureEnforcer requires flow_physics to be configured. "
                "Cannot compute pipe pressure drop without flow physics."
            )

        # Always use saturated liquid properties at current tank temperature
        rho_sat_liq = PropsSI("D", "T", temperature_K, "Q", 0, self.fluid)
        mu_sat_liq = PropsSI("V", "T", temperature_K, "Q", 0, self.fluid) * rho_sat_liq

        # Calculate pressure drop using saturated liquid properties
        pressure_drop = self.flow_physics.calculate_pipe_pressure_drop(
            flow_rate=as_float(flow_rate_kg_s),
            density=rho_sat_liq,
            viscosity=mu_sat_liq,
            pipe_diameter=self.pipe_diameter,
            pipe_length=self.pipe_length,
            pipe_roughness=self.pipe_roughness,
            loss_coefficient=self.loss_coefficient
        )

        atm = self.flow_physics.atmospheric_pressure

        # DIAGNOSTICS: Log pressure drop calculation when flow changes significantly
        if debug_enabled and hasattr(self, '_last_logged_flow_rate'):
            flow_change_pct = abs(flow_rate_kg_s - self._last_logged_flow_rate) / max(self._last_logged_flow_rate, 1e-6) * 100
            if flow_change_pct > 10.0:
                print(f"[REQ_P] flow={flow_rate_kg_s*1000:.1f} g/s | T={temperature_K:.1f}K | "
                      f"sat_liq: rho={rho_sat_liq:.1f} kg/m³, mu={mu_sat_liq:.2e} Pa·s | "
                      f"ΔP={pressure_drop/1e5:.3f} bar → P_req={(atm+pressure_drop)/1e5:.3f} bar")
                self._last_logged_flow_rate = flow_rate_kg_s
        elif debug_enabled and not hasattr(self, '_last_logged_flow_rate'):
            self._last_logged_flow_rate = flow_rate_kg_s

        return atm + pressure_drop

    def _capacity_kg_s(self, source_state, target_state) -> float:
        """Calculate fully-open valve capacity at current conditions.

        Returns:
            Flow capacity [kg/s] including safety limits
        """
        if source_state.pressure is None:
            source_state.compute_pressure()
        if target_state.pressure is None:
            target_state.compute_pressure()
        P1, P2 = source_state.pressure, target_state.pressure
        T1 = source_state.h2_temperature
        rho1 = source_state.fuel_mass / source_state.tank.volume

        # Use flow_physics for capacity (required - no fallback)
        base = self.flow_physics.calculate_orifice_flow_rate(
            upstream_pressure=P1, downstream_pressure=P2,
            upstream_temperature=T1, upstream_density=rho1,
            orifice_diameter=self.orifice_diameter
        )
        base = max(0.0, float(base))
        base = min(base, self.max_flow_rate)
        # Safety limit relative to source mass
        base = min(base, self.safety_mass_fraction_per_s * max(source_state.fuel_mass, 0.0))
        return base

    def _predict_pressure_pa(self, mass: float, temperature: float, volume: float, fluid: str) -> float:
        """Predict pressure using isochoric EOS: P = PropsSI("P", "T", T, "Dmass", ρ, fluid)

        Args:
            mass: Tank fuel mass [kg]
            temperature: Temperature [K]
            volume: Tank volume [m³]

        Returns:
            Predicted pressure [Pa]
        """
        from CoolProp.CoolProp import PropsSI
        density = mass / max(volume, 1e-9)
        if temperature <= 0:
            raise ValueError(f"Invalid temperature for pressure prediction: {temperature} K")
        return float(PropsSI("P", "T", temperature, "Dmass", density, fluid))

    def evaluate(self, t: float, tank_states: List) -> bool:
        # Stateless w.r.t valve position; compute at each call
        return True

    def calculate_flow_rate(self, t: float, tank_states: List) -> float:
        source_state = tank_states[self.source_idx]
        target_state = tank_states[self.target_idx]
        # Debug gating
        import os
        debug_enabled = os.environ.get("H2_DEBUG", "0") == "1"

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

        # Use measured pressure directly (no LPF by default to avoid phase lag)
        P_now = target_state.pressure
        T = target_state.h2_temperature
        P_req_raw = self._required_pressure_pa(mission_outflow, lh2_density, T, P_now)

        # Apply minimum pressure floor: don't target below the tank's minimum allowed pressure
        P_req_floored = max(P_req_raw, self.target_minimum_pressure_pa)        # Overpressure margin (Pa)
        P_req_eff = P_req_floored + self.overpressure_margin_bar * 1e5
        # Apply maximum pressure ceiling to prevent overshoot during mission transitions
        P_target_ctrl = min(P_req_eff, self.max_target_pressure_bar * 1e5)
        self._last_required_pressure = P_target_ctrl

        # EARLY DIAGNOSTICS: Track required pressure fluctuations in first 15 minutes
        if debug_enabled and t <= 900.0 and t % 5.0 < 0.1:  # Every 5s for first 15 min
            print(f"[EARLY REQ_P] t={t:.1f}s | mission_out={mission_outflow*1000:.1f} g/s | "
                  f"rho_LH2={lh2_density:.1f} kg/m³ | T_LH2={T:.1f} K | "
                  f"P_req_raw={P_req_raw/1e5:.3f} bar | P_target={P_target_ctrl/1e5:.3f} bar | "
                  f"P_now={P_now/1e5:.3f} bar | deficit={(P_target_ctrl-P_now)/1e5:.3f} bar")

        # DEBUG: Log pressure calculation details during overshoot period
        # Log at problem areas: 0.8 hrs (2880s) and 1.0 hrs (3500-3700s)
        if debug_enabled and (3500.0 <= t <= 3700.0):
            floor_applied = "YES" if P_req_raw < self.target_minimum_pressure_pa else "NO"
            print(f"[PRESSURE CALC] t={t:.1f}s | "
                  f"mission_out={mission_outflow*1000:.1f} g/s | "
                  f"P_req_raw={P_req_raw/1e5:.2f} bar (floor={floor_applied}), "
                  f"P_req_floored={P_req_floored/1e5:.2f} bar, "
                  f"P_min={self.target_minimum_pressure_pa/1e5:.2f} bar, "
                  f"margin={self.overpressure_margin_bar:.2f} bar, "
                  f"P_req_eff={P_req_eff/1e5:.2f} bar, "
                  f"P_target={P_target_ctrl/1e5:.2f} bar, "
                  f"P_now={P_now/1e5:.2f} bar")

        # Log diagnostics for overlay plot
        self.time_history.append(t)
        self.required_pressure_history.append(P_req_raw)
        self.activation_threshold_history.append(P_req_eff)
        self.mission_flow_history.append(mission_outflow)

        # Step 2: Invert mdot_in so that P_predicted ≈ P_req_eff over a fixed horizon
        # Update mdot command only on a fixed cadence and hold between updates to eliminate chatter
        dt_ctrl = max(1e-6, self.enforcement_horizon_s)

        m = target_state.fuel_mass
        V = target_state.tank.volume
        T = target_state.h2_temperature

        # Helper to compute predicted pressure for a candidate mdot_in
        # CRITICAL: Must account for temperature change due to warm CH2 injection
        def pred_p(mdot_in: float) -> float:
            m_next = m + (mdot_in - mission_outflow) * dt_ctrl
            m_next = max(1e-9, m_next)

            # Predict temperature change from energy balance
            # Q_in = mdot_in * (h_source - h_target) * dt
            # T_next = T + Q_in / (m * c_v)
            if mdot_in > 1e-6:
                from CoolProp.CoolProp import PropsSI
                # Source enthalpy (warm CH2 at ~288K)
                h_source = PropsSI("Hmass", "T", source_state.h2_temperature,
                                  "Dmass", source_state.fuel_mass / source_state.tank.volume, source_state.fluid)
                # Target enthalpy (current state)
                h_target = PropsSI("Hmass", "T", T, "Dmass", m / V, target_state.fluid)
                # Specific heat at constant volume
                c_v = PropsSI("Cvmass", "T", T, "Dmass", m / V, target_state.fluid)

                # Energy added over dt_ctrl
                delta_h = h_source - h_target
                Q_in = mdot_in * delta_h * dt_ctrl

                # Predicted temperature rise
                delta_T = Q_in / (m * c_v) if m * c_v > 0 else 0.0
                T_next = T + delta_T
            else:
                T_next = T

            return self._predict_pressure_pa(m_next, T_next, V, target_state.fluid)

        # Bisection bounds: [0, capacity×margin]
        capacity = self._capacity_kg_s(source_state, target_state)
        self._last_capacity_kg_s = capacity
        upper = max(0.0, min(self.max_flow_rate, capacity * self.bracket_margin))
        lower = 0.0

        # Current pressure (for linearization)
        err_bar = (P_target_ctrl - P_now) / 1e5

        # Compute mdot_star by solving the algebraic equation
        if upper <= 1e-12:
            mdot_star = 0.0
        else:
            # Temperature-aware linearized predictor with two-phase heat capacity
            # Estimate dP/dm accounting for both density and temperature changes

            # Get source and target enthalpies
            h_source = PropsSI("Hmass", "T", source_state.h2_temperature,
                              "Dmass", source_state.fuel_mass / source_state.tank.volume, source_state.fluid)
            h_target = PropsSI("Hmass", "T", T, "Dmass", m / V, target_state.fluid)
            delta_h = h_source - h_target

            # Use two-phase isochoric heat capacity for accurate thermal prediction
            rho = m / V
            # Check if we're in two-phase region
            try:
                rho_sat_liq = PropsSI("D", "T", T, "Q", 0, target_state.fluid)
                rho_sat_vap = PropsSI("D", "T", T, "Q", 1, target_state.fluid)
                in_two_phase = rho_sat_vap <= rho <= rho_sat_liq
            except Exception as e:
                # FAIL-FAST: Don't silently assume single-phase if CoolProp fails
                raise RuntimeError(
                    f"CoolProp saturation property query failed at T={T:.2f}K, rho={rho:.2f} kg/m³. "
                    f"Cannot determine two-phase region. Error: {e}"
                )

            if in_two_phase:
                # Two-phase region: use cv2p for accurate latent heat effects
                alpha = (1.0/rho - 1.0/rho_sat_liq) / (1.0/rho_sat_vap - 1.0/rho_sat_liq)
                alpha = max(0.0, min(1.0, alpha))

                # Single-phase cv at saturation states
                cv_g = PropsSI("CVMASS", "T", T, "Q", 1, target_state.fluid)
                cv_l = PropsSI("CVMASS", "T", T, "Q", 0, target_state.fluid)

                # Saturated enthalpies
                h_g = PropsSI("Hmass", "T", T, "Q", 1, target_state.fluid)
                h_l = PropsSI("Hmass", "T", T, "Q", 0, target_state.fluid)

                # Derivatives of saturation densities wrt T
                dT = 0.1
                try:
                    rho_g_plus = PropsSI("D", "T", T + dT, "Q", 1, target_state.fluid)
                    rho_l_plus = PropsSI("D", "T", T + dT, "Q", 0, target_state.fluid)
                    drho_g_dT = (rho_g_plus - rho_sat_vap) / dT
                    drho_l_dT = (rho_l_plus - rho_sat_liq) / dT
                except Exception as e:
                    # FAIL-FAST: Derivative calculation is critical for two-phase cv
                    raise RuntimeError(
                        f"CoolProp saturation density derivative failed at T={T:.2f}K. "
                        f"Cannot compute two-phase heat capacity correction. Error: {e}"
                    )

                # Two-phase correction term
                denom = (1.0 / rho_sat_vap - 1.0 / rho_sat_liq)
                if abs(denom) > 1e-10 and abs(drho_g_dT) > 1e-10 and abs(drho_l_dT) > 1e-10:
                    dvg_dT = -drho_g_dT / (rho_sat_vap**2)
                    dvl_dT = -drho_l_dT / (rho_sat_liq**2)
                    correction = (h_g - h_l) / denom * (dvg_dT - dvl_dT)
                else:
                    correction = 0.0

                c_v = alpha * cv_g + (1.0 - alpha) * cv_l + correction
            else:
                # Single-phase: use standard cv
                c_v = PropsSI("Cvmass", "T", T, "Dmass", rho, target_state.fluid)

            # Compute dP/dm with thermal effects
            dP_drho = PropsSI('d(P)/d(D)|T', 'T', T, 'Dmass', rho, 'hydrogen')
            dP_dT = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, 'hydrogen')
            dP_dm_net = dP_drho / V + dP_dT * delta_h / c_v

            if abs(dP_dm_net) > 1e-9:
                deltaP = P_target_ctrl - P_now
                mdot_star = mission_outflow + (deltaP / dP_dm_net) / dt_ctrl
                # Clamp to valid bounds
                mdot_star = max(0.0, min(mdot_star, upper))
            else:
                # Fallback if derivative is too small
                mdot_star = 0.5 * upper

        # Landing zero-flow rule
        if mission_outflow < 0.005:  # Below 5 g/s
            mdot_star = 0.0


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

        # Smoothly gate the command based on pressure deficit to avoid on/off around the target
        deficit_pa = P_target_ctrl - P_now
        if self.activation_softness_bar > 1e-9:
            # Symmetric smooth gate centered at zero deficit; 0.5 at equality, →1 for positive deficit, →0 for negative
            k_pa = max(1e2, self.activation_softness_bar * 1e5)
            gate = 0.5 * (1.0 + math.tanh(deficit_pa / k_pa))
        else:
            gate = 1.0 if deficit_pa >= 0.0 else 0.0

        mdot_cmd = gate * mdot_star

        cap = min(self.max_flow_rate, max(0.0, capacity))

        # Translate desired mass flow into a target opening coefficient
        if cap > 1e-12:
            target_coeff = max(0.0, min(1.0, mdot_cmd / cap))
        else:
            target_coeff = 0.0

        # Smooth coefficient with first-order lag
        dt_act = max(0.0, t - (self._last_coeff_time or t))
        coeff_prev = self._coeff if (self._last_coeff_time is not None) else target_coeff
        coeff_alpha = 0.0
        if self.valve_time_constant_s > 1e-9 and dt_act > 0.0:
            coeff_alpha = max(0.0, min(1.0, dt_act / max(self.valve_time_constant_s, 1e-9)))
        coeff_smooth = coeff_prev + coeff_alpha * (target_coeff - coeff_prev)


        # Clamp and compute final flow
        self._coeff = max(0.0, min(1.0, coeff_smooth))
        self._last_coeff = self._coeff
        self._last_coeff_time = t
        mdot_final = self._coeff * cap

        # CONSOLIDATED DEBUG LOGGING: Single comprehensive diagnostic message
        if debug_enabled:
            # Early dynamics (0-15 min): Track controller behavior during warmup
            if t <= 900.0 and t % 5.0 < 0.1:
                print(f"[FF_EARLY] t={t:.1f}s | P_now={P_now/1e5:.2f} bar, P_target={P_target_ctrl/1e5:.2f} bar | "
                      f"mdot_star={mdot_star*1000:.1f} g/s, gate={gate:.3f}, coeff={self._coeff:.3f} | "
                      f"mdot_final={mdot_final*1000:.1f} g/s, capacity={capacity*1000:.1f} g/s")
            # Saturation period (0.2-0.4 hrs = 720-1440s) and low-demand (2850-3000s)
            elif (720.0 <= t <= 1440.0 and t % 30.0 < 0.6) or (2850.0 <= t <= 3000.0 and t % 10.0 < 0.6):
                print(f"[FF_SAT] t={t:.1f}s | P_now={P_now/1e5:.2f} bar, P_target={P_target_ctrl/1e5:.2f} bar | "
                      f"mission_out={mission_outflow*1000:.1f} g/s, mdot_final={mdot_final*1000:.1f} g/s | "
                      f"P1={source_state.pressure/1e5:.1f} bar, P2={target_state.pressure/1e5:.2f} bar")
            # End-of-mission (3500-3700s): Track controller shutdown behavior
            elif 3500.0 <= t <= 3700.0 and t % 5.0 < 0.1:
                print(f"[FF_END] t={t:.1f}s | mdot_cmd={mdot_cmd*1000:.1f} g/s, mdot_final={mdot_final*1000:.1f} g/s | "
                      f"P_now={P_now/1e5:.2f} bar vs P_target={P_target_ctrl/1e5:.2f} bar")


        # Record deficit relative to target if saturated
        P_pred_cap = pred_p(mdot_final)
        self._last_deficit_bar = max(0.0, (P_target_ctrl - P_pred_cap) / 1e5)
        self._last_mdot_in = mdot_final
        # Track last time for optional filtering/slew in future steps
        self._last_time = t

        return mdot_final

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

    @classmethod
    def from_edge(cls, edge: CouplingEdge, ctx: EdgeContext) -> "FeedforwardPressureEnforcer":
        if ctx.flow_physics is None:
            raise ConfigError(
                f"Edge {edge.edge_id!r}: feedforward_pressure_enforcer needs a top-level 'physics:' section")
        o = edge.options
        flow = o.get("flow_parameters", {})
        coupling = cls(
            source_idx=edge.source,
            target_idx=edge.target,
            mission_profile={},
            discharge_piping=o.get("discharge_piping", {}),
            control_params=o.get("control_parameters", {}),
            max_flow_rate=flow.get("max_flow_rate_kg_s", 0.005),
            orifice_diameter=flow.get("orifice_diameter_m", 0.001),
            coupling_id=edge.edge_id,
            flow_physics=ctx.flow_physics,
            source_tank_config={"minimum_pressure": ctx.minimum_pressures[edge.source]},
            target_tank_config={"minimum_pressure": ctx.minimum_pressures[edge.target]},
            fluid=ctx.fluids[edge.source],
        )
        if ctx.mission_data:
            coupling.set_mission_profile(ctx.mission_data)
        return coupling
