import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os

from CoolProp.CoolProp import PropsSI
import CoolProp as cp

# Add the hydrogen_fuel_tank directory to the path to import NIST materials
hydrogen_tank_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, hydrogen_tank_path)
from src.materials.nist_materials import NISTMetal, NISTComposite
from plotting.sb_plotting import SeabornPlotter

# ----------------- Global Variables -----------------
global_scenario_manager = None  # Global reference for scenario access

# ----------------- Scenario and Configuration Framework -----------------
def calculate_thermal_equilibrium_Ts(T_fluid, T_amb=298.15, k_amb=0.025, A_out=5.0, alpha_s_approx=150.0, A_in=4.0):
    """
    Calculate initial solid temperature for thermal equilibrium.

    At equilibrium: dTs/dt = 0, so Qdot_amb = Qdot_s
    k_amb * A_out * (T_amb - Ts) = alpha_s * A_in * (Ts - T_fluid)

    Solving for Ts:
    Ts = (k_amb*A_out*T_amb + alpha_s*A_in*T_fluid) / (k_amb*A_out + alpha_s*A_in)
    """
    numerator = k_amb * A_out * T_amb + alpha_s_approx * A_in * T_fluid
    denominator = k_amb * A_out + alpha_s_approx * A_in

    Ts_equilibrium = numerator / denominator

    print(f"Thermal equilibrium calculation:")
    print(f"  T_fluid = {T_fluid:.2f}K, T_amb = {T_amb:.2f}K")
    print(f"  Equilibrium Ts = {Ts_equilibrium:.2f}K")
    print(f"  This gives Qdot_amb = Qdot_s = {k_amb * A_out * (T_amb - Ts_equilibrium):.2f}W")

    return Ts_equilibrium


class ScenarioManager:
    """
    Manages scenario-specific parameters and initial conditions.
    Scenarios: REFUEL, DISCHARGE, DORMANCY
    """

    def __init__(self):
        self.scenarios = {
            'REFUEL': {
                'initial_conditions': {
                    'p0': 15e5,      # Initial pressure [Pa]
                    'T0': 65.5,        # Initial temperature [K]
                    'Ts0': "thermal_equilibrium",   # Initial solid temperature [K]
                },
                'rho_stop': 78.0,    # Stopping density [kg/m³]
                'max_time': 700.0,   # Maximum simulation time [s]
                'solver_settings': {
                    'method': 'Radau',
                    'atol': 1e-10,
                    'rtol': 1e-8,
                    'max_step': 0.5,
                    'min_step': None,
                    'first_step': None,
                    'dense_output': True,
                },
                'mass_flow_functions': {
                    'mdot_fuel': lambda t: 0.07,  # [kg/s]
                    'mdot_disch': lambda t: 0.0,
                    'mdot_vent': lambda t: 0.0,
                },
                'Qdot_disch': lambda t: 0.0,
                'description': 'Tank refueling scenario'
            },
            'DISCHARGE': {
                'initial_conditions': {
                    'p0': 400e5,      # Initial pressure [Pa]
                    'T0': 53.25,      # Initial temperature [K]
                    'Ts0': 'thermal_equilibrium',    # Initial solid temperature [K]
                },
                'rho_stop': 5.8,     # Stopping density [kg/m³]
                'max_time': 40000.0, # Maximum simulation time [s]
                'solver_settings': {
                    'method': 'RK23',
                    'atol': 1e-9,
                    'rtol': 1e-7,
                    'max_step': 10.0,
                    'min_step': None,
                    'first_step': None,
                    'dense_output': True,
                },
                'mass_flow_functions': {
                    'mdot_fuel': lambda t: 0.0,
                    'mdot_disch': lambda t: 0.001,  # [kg/s]
                    'mdot_vent': lambda t: 0.0,
                },
                'Qdot_disch': lambda t: 1000.0,  # [W]
                'description': 'Tank discharge scenario'
            },
            'DORMANCY': {
                'initial_conditions': {
                    'p0': 400e5,     # Initial pressure [Pa] - 400 bar
                    'T0': 53.25,        # Initial temperature [K]
                    'Ts0': "thermal_equilibrium",   # Initial solid temperature [K]
                },
                'rho_stop': 70.0,    # Stopping density [kg/m³]
                'max_time': 216000.0,  # Maximum simulation time [s]
                'solver_settings': {
                    'method': 'RK45',
                    'atol': 1e-8,
                    'rtol': 1e-6,
                    'max_step': 100.0,
                    'min_step': None,
                    'first_step': None,
                    'dense_output': True,
                },
                'mass_flow_functions': {
                    'mdot_fuel': lambda t: 0.0,
                    'mdot_disch': lambda t: 0.0,
                    'mdot_vent': lambda t: 0.0,  # Will be calculated by Config C
                },
                'Qdot_disch': lambda t: 0.0,
                'description': 'Tank dormancy scenario with venting (Config C)'
            }
        }
        self.current_scenario = None

    def set_scenario(self, scenario_name):
        """Set the current scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.scenarios.keys())}")
        self.current_scenario = scenario_name

    def get_scenario_config(self):
        """Get the current scenario configuration."""
        if self.current_scenario is None:
            raise ValueError("No scenario selected")
        return self.scenarios[self.current_scenario]

    def get_solver_settings(self):
        """Get the solver settings for the current scenario."""
        if self.current_scenario is None:
            raise ValueError("No scenario selected")
        return self.scenarios[self.current_scenario]['solver_settings']

class ConfigurationManager:
    """
    Manages configuration switching based on pressure thresholds.
    Configurations: A (normal), B (minimum pressure), C (maximum pressure)
    Handles the three configuration-dependent algebraic equations.
    """

    def __init__(self, fluid, p_min=15e5, p_vent=450e5):
        self.fluid = fluid
        self.p_min = p_min    # Minimum pressure threshold [Pa]
        self.p_vent = p_vent  # Venting pressure threshold [Pa]
        self.current_config = None

        # Configuration definitions
        self.configurations = {
            'A': {
                'name': 'Normal Operating Mode',
                'description': 'p = p(T,rho), qdot_dis = 0, mdot_vent = 0'
            },
            'B': {
                'name': 'Minimum Pressure Mode',
                'description': 'p = p_min, qdot_dis = config_B_value, mdot_vent = 0'
            },
            'C': {
                'name': 'Maximum Pressure Mode',
                'description': 'p = p_vent, qdot_dis = 0, mdot_vent = config_C_value'
            }
        }

    def select_configuration(self, p, is_two_phase=False):
        """
        Select configuration based on pressure following flowchart logic.

        Args:
            p: Current pressure [Pa]
            is_two_phase: Whether system is in two-phase region

        Returns:
            str: Configuration name ('A', 'B', or 'C')
        """
        # Following flowchart logic
        if p <= self.p_min:
            return 'B'
        elif p >= self.p_vent:
            selected_config = 'C'  # Maximum pressure mode (venting)
        else:
            selected_config = 'A'  # Normal operation

        # Update current configuration
        self.current_config = selected_config

        return selected_config

    def get_algebraic_equations(self, config, T, rho, is_two_phase=False, t=None, mdot_disch_func=None, Ts=None, scenario_manager=None):
        """
        Compute the three configuration-dependent algebraic equations.

        Args:
            config: Configuration name ('A', 'B', or 'C')
            T: Temperature [K]
            rho: Density [kg/m³]
            is_two_phase: Whether system is in two-phase region
            t: Current time [s] (needed for Configuration B)
            mdot_disch_func: Discharge mass flow function (needed for Configuration B)
            Ts: Solid temperature [K] (needed for heat transfer calculations)

        Returns:
            dict: Contains 'pressure', 'qdot_disch', 'mdot_vent'
        """
        if config == 'A':
            # Normal operation: p = p(T,rho), qdot_dis = 0, mdot_vent = 0
            if is_two_phase:
                p = PropsSI("P", "T", T, "Q", 0, self.fluid)  # Saturation pressure
            else:
                p = PropsSI("P", "T", T, "Dmass", rho, self.fluid)
            return {
                'pressure': p,
                'qdot_disch': 0.0,
                'mdot_vent': 0.0
            }

        elif config == 'B':
            # Minimum pressure mode: p = p_min, qdot_dis = config_B_value, mdot_vent = 0
            # Calculate the required discharge heat to maintain minimum pressure

            # Calculate Configuration B discharge heat
            qdot_disch = self._calculate_config_B_qdot_disch(T, rho, is_two_phase, t, mdot_disch_func, Ts)

            return {
                'pressure': self.p_min,
                'qdot_disch': qdot_disch,
                'mdot_vent': 0.0
            }

        elif config == 'C':
            # Maximum pressure mode: p = p_vent, qdot_dis = 0, mdot_vent = config_C_value
            # Calculate the required venting mass flow to maintain maximum pressure
            mdot_vent = self._calculate_config_C_mdot_vent(T, rho, is_two_phase, t, Ts)
            return {
                'pressure': self.p_vent,
                'qdot_disch': 0.0,
                'mdot_vent': mdot_vent
            }

        else:
            raise ValueError(f"Unknown configuration: {config}")

    def _calculate_config_B_qdot_disch(self, T, rho, is_two_phase, t, mdot_disch_func, Ts):
        """
        Calculate the discharge heat for Configuration B (minimum pressure mode).

        Configuration B equation: Q_disch = M_disch · [T/ρ·(∂p/∂T)_ρ - ρ·cv·(∂T/∂ρ)_p] - Q_s

        TODO: Investigate why we need ideal gas for term1 but CoolProp for term2 for numerical stability.
        Mixed approach is required: ideal gas for term1, real thermodynamic derivatives for term2.

        Args:
            T: Temperature [K]
            rho: Density [kg/m³]
            is_two_phase: Whether the state is two-phase
            t: Current time [s]
            mdot_disch_func: Discharge mass flow rate function
            Ts: Solid temperature [K]

        Returns:
            qdot_disch: Discharge heat rate [W]
        """
        import CoolProp.CoolProp as cp

        # Get mass flow rate from scenario
        mdot_disch = mdot_disch_func(t) if mdot_disch_func else 0.0

        # For Configuration B, pressure is constrained to p_min
        p = self.p_min
        nu = 1.0 / rho  # Specific volume [m³/kg]

        # Get thermodynamic properties at current state
        if is_two_phase:
            # For two-phase, use average cv
            cv_liquid = cp.PropsSI("Cvmass", "T", T, "Q", 0, self.fluid)
            cv_vapor = cp.PropsSI("Cvmass", "T", T, "Q", 1, self.fluid)
            try:
                x = cp.PropsSI("Q", "T", T, "Dmass", rho, self.fluid)
                x = max(0.0, min(1.0, x))
            except:
                x = 0.5
            cv = x * cv_vapor + (1.0 - x) * cv_liquid
        else:
            cv = cp.PropsSI("Cvmass", "P", p, "T", T, self.fluid)

        # Calculate alpha_s for heat transfer
        alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)
        # alpha_s = get_alpha_s_internal_model(T,p, Ts, None, 1.0, A_in)
        Qdot_s = alpha_s * A_in * (Ts - T)  # Environmental heat leak [W]

        # Term 1: T/ρ · (∂p/∂T)_ρ using real gas relationship
        term1 = get_real_gas_work_term(T, rho, self.fluid)

        # Term 2: ρ·cv·(∂T/∂ρ)_p using real CoolProp derivatives for accuracy
        try:
            dT_drho_p = cp.PropsSI('d(T)/d(D)|P', 'P', p, 'T', T, self.fluid)
        except:
            # Fallback to ideal gas relationship if CoolProp fails
            dT_drho_p = -T / rho
        term2 = rho * cv * dT_drho_p

        qdot_disch = mdot_disch * (term1 - term2) - Qdot_s

        return qdot_disch

    def _calculate_config_C_mdot_vent(self, T, rho, is_two_phase, t, Ts):
        """
        Calculate the venting mass flow for Configuration C (maximum pressure mode).

        Configuration C equation: M_vent = Q_s / [T/ρ·(∂p/∂T)_ρ - ρ·cv·(∂T/∂ρ)_p + h_vent - h]

        Uses the same mixed approach as Configuration B: ideal gas for term1, CoolProp for term2.

        Args:
            T: Temperature [K]
            rho: Density [kg/m³]
            is_two_phase: Whether the state is two-phase
            t: Current time [s]
            Ts: Solid temperature [K]

        Returns:
            mdot_vent: Venting mass flow rate [kg/s]
        """
        import CoolProp.CoolProp as cp

        # For Configuration C, pressure is constrained to p_vent
        p = self.p_vent
        nu = 1.0 / rho  # Specific volume [m³/kg]

        # Get thermodynamic properties at current state
        if is_two_phase:
            # For two-phase, use average cv and enthalpy
            cv_liquid = cp.PropsSI("Cvmass", "T", T, "Q", 0, self.fluid)
            cv_vapor = cp.PropsSI("Cvmass", "T", T, "Q", 1, self.fluid)
            h_liquid = cp.PropsSI("Hmass", "T", T, "Q", 0, self.fluid)
            h_vapor = cp.PropsSI("Hmass", "T", T, "Q", 1, self.fluid)
            try:
                x = cp.PropsSI("Q", "T", T, "Dmass", rho, self.fluid)
                x = max(0.0, min(1.0, x))
            except:
                x = 0.5
            cv = x * cv_vapor + (1.0 - x) * cv_liquid
            h = x * h_vapor + (1.0 - x) * h_liquid
            h_vent = h_vapor  # Venting vapor preferentially
        else:
            cv = cp.PropsSI("Cvmass", "P", p, "T", T, self.fluid)
            h = cp.PropsSI("Hmass", "P", p, "T", T, self.fluid)
            h_vent = h  # Single-phase venting

        # Calculate alpha_s for heat transfer
        alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)
        # alpha_s = get_alpha_s_internal_model(T,p, Ts, None, 1.0, A_in)
        Qdot_s = alpha_s * A_in * (Ts - T)  # Environmental heat leak [W]

        # Term 1: T/ρ · (∂p/∂T)_ρ using real gas relationship
        term1 = get_real_gas_work_term(T, rho, self.fluid)

        # Term 2: ρ·cv·(∂T/∂ρ)_p using real CoolProp derivatives for accuracy
        try:
            dT_drho_p = cp.PropsSI('d(T)/d(D)|P', 'P', p, 'T', T, self.fluid)
        except:
            # Fallback to ideal gas relationship if CoolProp fails
            dT_drho_p = -T / rho
        term2 = rho * cv * dT_drho_p

        # Calculate denominator
        denominator = term1 - term2 + h_vent - h

        # Avoid division by zero
        if abs(denominator) < 1e-12:
            return 0.0

        mdot_vent = Qdot_s / denominator

        # Ensure non-negative venting (can't vent negative mass)
        mdot_vent = max(0.0, mdot_vent)

        return mdot_vent


class ModelSwitcher:
    """
    Handles single-phase vs two-phase model switching.
    This is separate from configuration switching.
    """

    def __init__(self, fluid, config_manager=None):
        self.fluid = fluid
        self.config_manager = config_manager
        self.models = {}
        self.current_model = None
        self.alpha_s_last = 0.0  # For debugging

    def register_model(self, name, condition_func, ode_func):
        """
        Register a new model with its condition and ODE function.

        Args:
            name: String identifier for the model
            condition_func: Function that takes (T, p, rho) and returns True if this model should be used
            ode_func: Function that computes dT_dt for this model
        """
        self.models[name] = {
            'condition': condition_func,
            'ode_func': ode_func
        }

    def compute_dT_dt(self, t, y, Qdot_s, Qdot_disch, *args):
        """
        Compute dT/dt using the currently selected model.

        Args:
            t: Time
            y: State vector [m, T, Ts]
            Qdot_s: Heat flow from solid to fluid [W]
            Qdot_disch: Discharge heat flow [W]
            *args: Additional arguments passed to the ODE function

        Returns:
            dT/dt value
        """
        if self.current_model is None:
            raise ValueError("No model selected. Call select_model first.")

        return self.models[self.current_model]['ode_func'](t, y, Qdot_s, Qdot_disch)

    def solve(self, t, y, mdot_fuel_func, mdot_disch_func, scenario_manager=None, *args):
        """
        Solve the coupled DAE system with pre-calculated thermodynamic derivatives.

        Approach:
        1. Pre-calculate thermodynamic derivatives (∂P/∂T)_ρ, (dP_sat/dT) at current state
        2. Solve remaining coupled algebraic + differential equations simultaneously

        This follows the 10-equation DAE system from the mathematical notes.
        """
        m, T, Ts = y

        # Bounds enforcement to prevent unphysical states
        m = max(m, 1e-12)
        T = max(T, 14.0)    # Just above hydrogen triple point (13.8K)
        T = min(T, 1000.0)  # Reasonable upper bound
        Ts = max(Ts, 1.0)

        # Algebraic equation (6): ρ = m/V_tank
        rho = m / V_t

        try:
            # Algebraic equation (7): h = h(T, ρ)
            h = PropsSI("Hmass", "T", T, "Dmass", rho, self.fluid)
            p = PropsSI("P", "T", T, "Dmass", rho, self.fluid)

            # Pre-calculate thermodynamic derivatives at current state
            is_two_phase = is_near_saturation(T, p, self.fluid)

            if is_two_phase:
                # Two-phase: pre-calculate (T/ρ) * (dP_sat/dT)
                h_vapor = PropsSI("Hmass", "T", T, "Q", 1, self.fluid)
                h_liquid = PropsSI("Hmass", "T", T, "Q", 0, self.fluid)
                rho_vapor = PropsSI("Dmass", "T", T, "Q", 1, self.fluid)
                rho_liquid = PropsSI("Dmass", "T", T, "Q", 0, self.fluid)

                L_v = h_vapor - h_liquid  # Latent heat
                delta_v = (1.0/rho_vapor) - (1.0/rho_liquid)  # Specific volume difference
                dp_sat_dT = L_v / (T * delta_v)  # Clausius-Clapeyron

                thermo_coeff = (T / rho) * dp_sat_dT  # Pre-calculated coefficient
            else:
                # Single-phase: pre-calculate (∂P/∂T)_ρ
                # Use finite difference approximation
                dT = 0.01  # Small temperature perturbation
                try:
                    p_plus = PropsSI("P", "T", T + dT, "Dmass", rho, self.fluid)
                    dp_dT = (p_plus - p) / dT
                except:
                    # Fallback using ideal gas approximation
                    dp_dT = p / T

                nu = 1.0 / rho  # Specific volume
                thermo_coeff = nu * dp_dT  # Pre-calculated coefficient

        except Exception as e:
            print(f"CoolProp error at t={t:.3f}s: {e}")
            print(f"  State: m={m:.3f}, T={T:.3f}, rho={rho:.3f}")
            # Use fallback values to continue simulation
            h = 500000.0  # Reasonable enthalpy fallback
            is_two_phase = False
            thermo_coeff = 0.1  # Small fallback coefficient
            p = 1e5  # 1 bar fallback pressure

        # Now solve the coupled system with pre-calculated thermodynamic coefficient

        # Select configuration for algebraic equations (8,9,10)
        current_config = self.config_manager.select_configuration(p, is_two_phase)
        config_eqs = self.config_manager.get_algebraic_equations(
            current_config, T, rho, is_two_phase=is_two_phase,
            t=t, mdot_disch_func=mdot_disch_func, Ts=Ts, scenario_manager=scenario_manager
        )

        p_config = config_eqs['pressure']      # May override p for configs B,C
        Qdot_disch = config_eqs['qdot_disch']  # Config-dependent discharge heat
        mdot_vent = config_eqs['mdot_vent']    # Config-dependent venting

        # Algebraic equations (4,5): Heat transfer (coupled with temperatures)
        alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p_config)
        self.alpha_s_last = alpha_s  # Store for debugging
        Qdot_s = alpha_s * A_in * (Ts - T)     # Equation (4)
        Qdot_amb = k_amb * A_out * (T_amb - Ts) # Equation (5)

        # Mass flow rates
        mdot_f = mdot_fuel_func(t)
        mdot_d = mdot_disch_func(t)

        # Now solve the 3 coupled ODEs simultaneously:

        # ODE (1): dm/dt = ṁ_fuel - ṁ_disch - ṁ_vent
        dm_dt = mdot_f - mdot_d - mdot_vent

        # ODE (3): dTs/dt = (Q̇_amb - Q̇_s) / (m_s c_s)
        c_liner = float(c_liner_func(Ts))
        c_wall = float(c_wall_func(Ts))
        denom_Ts = m_liner * c_liner + m_wall * c_wall
        dTs_dt = (Qdot_amb - Qdot_s) / denom_Ts

        # ODE (2): dT/dt = energy balance with pre-calculated thermo coefficient
        # This uses the current model (single_phase vs two_phase) with proper coupling
        self.current_model = "two_phase" if is_two_phase else "single_phase"
        dT_dt = self.compute_dT_dt(t, [m, T, Ts], Qdot_s, Qdot_disch)

        return [dm_dt, dT_dt, dTs_dt]

# Simulation Setup and User Parameters
fluid = "Hydrogen"
V_t = 0.5          # Vessel volume [m³]

# Heat transfer constants
A_in = 4.0         # Inner surface area [m²]
A_out = 4.1        # Outer surface area [m²]
k_amb = 0.025      # Ambient heat transfer coefficient [W/m²K]
T_amb = 298.15     # Ambient temperature [K]
m_liner = 100.0    # Liner mass [kg]
m_wall = 150.0     # Wall mass [kg]

# Progress printing parameters
PRINT_EVERY_N_STEPS = 50  # Print progress every N steps
step_counter = 0  # Global counter

# Geometric parameters for alpha_s calculation
D_inner = 1.0      # Inner diameter [m]
D_outer = 1.0     # Outer diameter [m]
annular_fluid = "Hydrogen"  # Fluid in the gap

# Configuration pressure thresholds
p_min = 15e5       # Minimum pressure threshold for configuration B [Pa]
p_vent = 450e5     # Venting pressure threshold for configuration C [Pa]

# Initialize global framework managers
scenario_manager = ScenarioManager()
config_manager = ConfigurationManager(fluid, p_min, p_vent)
model_switcher = ModelSwitcher(fluid, config_manager)

# Initialize NIST materials
liner_material = NISTMetal.aluminum_6061T6_nist()
wall_material = NISTComposite.g10_nist(winding_angle=0.0)

def c_liner_func(Ts):
    """Get specific heat of aluminum 6061-T6 liner using NIST data"""
    return liner_material.determine_specific_heat(Ts)

def c_wall_func(Ts):
    """Get specific heat of G10 wall using NIST data"""
    return wall_material.determine_specific_heat(Ts)

# ----------------- Real Gas Work Term Function -----------------
def get_real_gas_work_term(T, rho, fluid):
    """
    Calculate the real gas work term (T/ρ) * (∂p/∂T)_ρ using CoolProp.

    This replaces the ideal gas assumption p*v with proper real gas behavior.

    Args:
        T: Temperature [K]
        rho: Density [kg/m³]
        fluid: Fluid name for CoolProp

    Returns:
        work_term: (T/ρ) * (∂p/∂T)_ρ [Pa·m³/kg]
    """
    try:
        # Calculate (∂p/∂T)_ρ using CoolProp
        dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, fluid)

        # Calculate (T/ρ) * (∂p/∂T)_ρ
        work_term = (T / rho) * dp_dT_rho

        return work_term

    except Exception as e:
        # Fallback to ideal gas if CoolProp fails
        print(f"Warning: CoolProp failed for real gas work term at T={T:.2f}K, rho={rho:.2f}kg/m³: {e}")
        print("  Falling back to ideal gas approximation p/rho")

        # Use ideal gas: p*v = p/rho (approximately)
        p = PropsSI("P", "T", T, "Dmass", rho, fluid)
        return p / rho

# ----------------- Heat Transfer Coefficient Function -----------------
def get_alpha_s(T, Ts, D, D_o, fluid="Air", p=101325):
    """
    Compute equivalent inner convective heat transfer coefficient (h_i)
    for a horizontal concentric cylindrical annulus using
    Kuehn & Goldstein correlation (1976).

    This function implements equations (1a)-(1g) from the Kuehn & Goldstein
    correlation for natural convection in horizontal concentric cylindrical annuli.

    Parameters
    ----------
    T_i : float
        Inner cylinder surface temperature [K]
    T_o : float
        Outer cylinder surface temperature [K]
    D_i : float
        Inner cylinder diameter [m]
    D_o : float
        Outer cylinder diameter [m]
    fluid : str, optional
        Fluid name for CoolProp (default = 'Air')
    p : float, optional
        Pressure [Pa] (default = 101325 Pa)

    Returns
    -------
    h_i : float
        Equivalent inner convective coefficient [W/m2-K]

    Notes
    -----
    The correlation is valid for:
    - Horizontal concentric cylinders
    - Natural convection
    - Rayleigh numbers in the range applicable to the original correlation

    References
    ----------
    Kuehn, T.H., & Goldstein, R.J. (1976). Correlating equations for natural
    convection heat transfer between horizontal circular cylinders.
    """

    # Film temperature
    T_film = 0.5 * (T + Ts)

    # Fluid properties at film temperature
    k = PropsSI('L', 'T', T_film, 'P', p, fluid)    # W/m-K
    mu = PropsSI('V', 'T', T_film, 'P', p, fluid)   # Pa·s
    rho = PropsSI('D', 'T', T_film, 'P', p, fluid)  # kg/m^3
    cp = PropsSI('C', 'T', T_film, 'P', p, fluid)   # J/kg-K

    # Derived properties
    nu = mu / rho                  # kinematic viscosity [m^2/s]
    alpha = k / (rho * cp)         # thermal diffusivity [m^2/s]
    Pr = nu / alpha                # Prandtl number
    beta = 1.0 / T_film            # thermal expansion coeff (ideal gas approx)

    # Rayleigh number (based on diameter for horizontal cylinder)
    Ra_D = 9.81 * beta * abs(Ts - T) * D**3 / (nu * alpha)

    # Churchill & Chu correlation for horizontal cylinder (valid 10^-5 < Ra < 10^12)
    Nu_D = (0.60 + (0.387 * Ra_D**(1/6)) /
           ( (1 + (0.559/Pr)**(9/16))**(8/27) ))**2
    L = 2.0
    # Rayleigh number (based on plate height L)
    Ra_L = 9.81 * beta * abs(Ts - T) * L**3 / (nu * alpha)

    # Churchill & Chu correlation for vertical plate (valid ~10^-1 < Ra < 10^12)
    Nu_L = (0.825 + (0.387 * Ra_L**(1/6)) /
            ( (1 + (0.492/Pr)**(9/16))**(8/27) ))**2

     # Heat transfer coefficient
    h = Nu_L * k / D
    return h


def get_alpha_s_internal_model(T_hydrogen, p_hydrogen, T_surface, fuel_height=None, characteristic_height=1.0, surface_area=None):
    """
    Compute heat transfer coefficient using the SingleZoneModel approach from internal_models.py.

    This function emulates the logic from SingleZoneModel.get_thermal_resistances() to compute
    the equivalent heat transfer coefficient for hydrogen convection based on phase state.

    Parameters
    ----------
    T_hydrogen : float
        Hydrogen temperature [K]
    p_hydrogen : float
        Hydrogen pressure [Pa]
    T_surface : float
        Surface temperature [K]
    fuel_height : float, optional
        Height of liquid fuel if two-phase (default: None, auto-detect phase)
    characteristic_height : float, optional
        Characteristic height for convection calculations [m] (default: 1.0)
    surface_area : float, optional
        Surface area [m²] (default: 1.0, returns coefficient per unit area)

    Returns
    -------
    alpha_s : float
        Heat transfer coefficient [W/m²·K]

    Notes
    -----
    This function uses the same correlations as the SingleZoneModel:
    - LiquidPhaseConvection: Nu = 0.0605 * Ra^(1/3) (Hochstein et al. 1986)
    - GasPhaseConvection: Nu = 17 (Brewer 1991)
    - For two-phase: parallel resistance combination

    References
    ----------
    - Hochstein et al. (1986), Verstraete (2009)
    - Brewer (1991)
    - SingleZoneModel in internal_models.py
    """

    # Set default surface area if not provided
    if surface_area is None:
        surface_area = 1.0  # Return coefficient per unit area

    # Determine phase state of hydrogen
    try:
        # Get critical properties for hydrogen
        T_critical = PropsSI("TCRIT", "Hydrogen")  # ~33.2 K
        p_critical = PropsSI("PCRIT", "Hydrogen")  # ~1.297 MPa

        # Check if we're above critical conditions
        if p_hydrogen > p_critical or T_hydrogen > T_critical:
            # Supercritical - treat as gas
            phase = "supercritical"
            is_two_phase = False
        else:
            # Check if we're near saturation (two-phase)
            try:
                T_sat = PropsSI("T", "P", p_hydrogen, "Q", 0, "Hydrogen")
                is_two_phase = abs(T_hydrogen - T_sat) < 1.0  # Within 1K of saturation
            except:
                # If saturation lookup fails, assume single phase
                is_two_phase = False
                # Determine phase based on temperature relative to saturation
                try:
                    T_sat_1bar = PropsSI("T", "P", 101325, "Q", 0, "Hydrogen")  # ~20.4 K
                    if T_hydrogen < T_sat_1bar + 10:  # rough liquid estimate
                        phase = "liquid"
                    else:
                        phase = "gas"
                except:
                    phase = "gas"  # default to gas if all else fails

        if is_two_phase:
            # Two-phase: get both liquid and gas properties
            rho_liquid = PropsSI("D", "P", p_hydrogen, "Q", 0, "Hydrogen")
            rho_gas = PropsSI("D", "P", p_hydrogen, "Q", 1, "Hydrogen")

            # For two-phase, estimate fill fraction if not provided
            if fuel_height is None:
                fill_fraction = 0.5  # Default 50% liquid fill
            else:
                fill_fraction = fuel_height / characteristic_height
                fill_fraction = max(0.0, min(1.0, fill_fraction))


            # Get liquid properties at saturation
            mu_liquid = PropsSI("V", "P", p_hydrogen, "Q", 0, "Hydrogen")
            k_liquid = PropsSI("CONDUCTIVITY", "P", p_hydrogen, "Q", 0, "Hydrogen")
            cp_liquid = PropsSI("C", "P", p_hydrogen, "Q", 0, "Hydrogen")
            beta_liquid = PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "P", p_hydrogen, "Q", 0, "Hydrogen")

            # Get gas properties at saturation
            mu_gas = PropsSI("V", "P", p_hydrogen, "Q", 1, "Hydrogen")
            k_gas = PropsSI("CONDUCTIVITY", "P", p_hydrogen, "Q", 1, "Hydrogen")
            cp_gas = PropsSI("C", "P", p_hydrogen, "Q", 1, "Hydrogen")
            beta_gas = PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "P", p_hydrogen, "Q", 1, "Hydrogen")

            # Calculate liquid phase convection (Hochstein correlation)
            nu_liquid = mu_liquid / rho_liquid
            alpha_liquid = k_liquid / (rho_liquid * cp_liquid)
            Pr_liquid = nu_liquid / alpha_liquid

            # Rayleigh number for liquid phase
            g = 9.81
            Delta_T = abs(T_surface - T_hydrogen)
            L_char_liquid = fuel_height if fuel_height is not None else characteristic_height * fill_fraction
            L_char_liquid = max(L_char_liquid, 0.01)  # Minimum characteristic length

            Ra_liquid = g * beta_liquid * Delta_T * L_char_liquid**3 * Pr_liquid / nu_liquid

            # Liquid phase Nusselt number (Hochstein et al. 1986)
            Nu_liquid = 0.0605 * Ra_liquid**(1/3)
            h_liquid = Nu_liquid * k_liquid / L_char_liquid

            # Calculate gas phase convection (Brewer correlation)
            nu_gas = mu_gas / rho_gas
            alpha_gas = k_gas / (rho_gas * cp_gas)

            # Gas phase characteristic length
            L_char_gas = characteristic_height - L_char_liquid
            L_char_gas = max(L_char_gas, 0.01)  # Minimum characteristic length

            # Gas phase Nusselt number (Brewer 1991)
            Nu_gas = 17.0
            h_gas = Nu_gas * k_gas / L_char_gas

            # Compute thermal resistances
            A_liquid = surface_area * fill_fraction
            A_gas = surface_area * (1.0 - fill_fraction)

            R_liquid = 1.0 / (h_liquid * A_liquid) if A_liquid > 0 else float('inf')
            R_gas = 1.0 / (h_gas * A_gas) if A_gas > 0 else float('inf')

            # Parallel resistance combination
            if R_liquid == float('inf'):
                R_total = R_gas
            elif R_gas == float('inf'):
                R_total = R_liquid
            else:
                R_total = 1.0 / (1.0/R_liquid + 1.0/R_gas)

            # Convert back to heat transfer coefficient
            alpha_s = 1.0 / (R_total * surface_area)

        else:
            # Single phase - determine if liquid or gas
            if not locals().get('phase'):  # Only determine if not already set above
                try:
                    if T_hydrogen < T_sat:
                        # Subcooled liquid
                        phase = "liquid"
                    else:
                        # Superheated gas
                        phase = "gas"
                except:
                    # If T_sat not available, default to gas
                    phase = "gas"

            if phase == "liquid":
                # Liquid phase properties
                rho = PropsSI("D", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")
                mu = PropsSI("V", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")
                k = PropsSI("CONDUCTIVITY", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")
                cp = PropsSI("C", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")
                beta = PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")

                # Liquid phase convection calculation
                nu = mu / rho
                alpha = k / (rho * cp)
                Pr = nu / alpha

                g = 9.81
                Delta_T = abs(T_surface - T_hydrogen)
                L_char = characteristic_height

                Ra = g * beta * Delta_T * L_char**3 * Pr / nu

                # Liquid phase Nusselt number (Hochstein et al. 1986)
                Nu = 0.0605 * Ra**(1/3)
                alpha_s = Nu * k / L_char

            else:  # gas phase or supercritical
                # Gas phase properties
                rho = PropsSI("D", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")
                k = PropsSI("CONDUCTIVITY", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")

                # Gas phase convection calculation
                # Use gas height (assume full tank height for single-phase gas)
                L_char = characteristic_height

                # Gas phase Nusselt number (Brewer 1991)
                Nu = 17.0
                alpha_s = Nu * k / L_char

    except Exception as e:
        print(f"Warning: Error in get_alpha_s_internal_model: {e}")
        # Fallback to simple calculation
        try:
            k = PropsSI("CONDUCTIVITY", "T", T_hydrogen, "P", p_hydrogen, "Hydrogen")
            alpha_s = 10.0 * k / characteristic_height  # Simple fallback
        except:
            alpha_s = 50.0  # Final fallback value

    return alpha_s


# Remove the test function as it's not needed in production code

# ----------------- Helper Functions -----------------
def is_near_saturation(T, p, fluid):
    """
    Single unified phase checker function.

    Returns:
        True if P ≈ P_sat (use two-phase model)
        False if P is not close to P_sat (use single-phase model)
    """
    try:
        # Get saturation pressure at this temperature
        p_sat = PropsSI("P", "T", T, "Q", 0, fluid)

        # Simple tolerance check: if P is very close to P_sat, use two-phase
        tolerance = 1e-6  # Very small tolerance
        return abs(p - p_sat) < tolerance * p_sat

    except:
        # If saturation pressure calculation fails (e.g., above critical point),
        # assume single-phase
        return False

def compute_pump_outlet_hydrogen(tank_pressure: float, tank_temperature: float):
    """
    Calculate hydrogen enthalpy after cryogenic pump compression.

    Parameters
    ----------
    tank_pressure : float
        Target tank pressure [Pa]
    tank_temperature : float
        Current tank temperature [K]

    Returns
    -------
    h2 : float
        Enthalpy at pump outlet [J/kg]
    """
    fluid_local = "Hydrogen"
    P1 = 3e5       # Dewar pressure [Pa]
    P2 = tank_pressure  # Target pressure [Pa]
    eta_p = 0.78   # Pump isentropic efficiency

    # Inlet state: saturated liquid at P1
    h1 = PropsSI("H", "P", P1, "Q", 0, fluid_local)
    s1 = PropsSI("S", "P", P1, "Q", 0, fluid_local)

    # Ideal isentropic outlet at P2
    h2s = PropsSI("H", "P", P2, "S", s1, fluid_local)

    # Actual outlet enthalpy with efficiency
    h2 = h1 + (h2s - h1)/eta_p

    return h2

# ----------------- Model Implementations -----------------
def single_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager, Qdot_s, Qdot_disch):
    """
    Single-phase energy balance.
    Uses cv for temperature derivative calculation.

    Includes scenario-specific enthalpy calculations (e.g., cryopump for refueling).

    Args:
        Qdot_s: Heat flow from solid to fluid [W] (pre-calculated in solve method)
        Qdot_disch: Discharge heat flow [W] (pre-calculated in solve method)
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    T = max(T, 1.0)

    rho = m / V_t

    # Get configuration-dependent algebraic equations
    config_eqs = config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=False, t=t, mdot_disch_func=mdot_disch_func, Ts=Ts, scenario_manager=scenario_manager)
    p = config_eqs['pressure']

    h = PropsSI("Hmass", "T", T, "Dmass", rho, fluid)
    c_v = PropsSI("Cvmass", "T", T, "Dmass", rho, fluid)
    nu = 1.0 / rho

    # Get mass flow rates (may be config-dependent)
    mdot_f = mdot_fuel_func(t)
    mdot_d = mdot_disch_func(t)
    mdot_v = config_eqs['mdot_vent']  # Configuration-dependent

    # Scenario-specific enthalpy calculation
    current_scenario = scenario_manager.current_scenario
    if current_scenario == 'REFUEL':
        # Use pump outlet enthalpy for refueling (includes cryopump compression work)
        h_fuel = compute_pump_outlet_hydrogen(p, T)
    else:
        # For discharge and dormancy, use tank enthalpy (no cryopump)
        h_fuel = 0.0

    h_dich = h
    h_vent = h

    # Use pre-calculated heat flows instead of computing them again
    # (Qdot_s and Qdot_disch are passed as parameters)

    # Energy balance terms
    h_term = mdot_f * (h_fuel - h) - mdot_d * (h_dich - h) - mdot_v * (h_vent - h)

    # PV work term - EXPERIMENTAL: Disable during refueling to avoid double-counting compression work
    net_mass_flow = mdot_f - mdot_d - mdot_v
    if mdot_f > 0 and mdot_d == 0 and mdot_v == 0:
        # Pure refueling case - compression work already included in h_fuel from cryopump
        pv_work = 0.0
    else:
        # Discharge/venting case - use normal PV work term
        pv_work = get_real_gas_work_term(T, rho, fluid) * net_mass_flow

    numerator_T = h_term + pv_work + Qdot_s + Qdot_disch
    dT_dt = numerator_T / (m * c_v)

    return dT_dt

def two_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager, Qdot_s, Qdot_disch):
    """
    Two-phase energy balance implementation.
    Uses c_v2P (two-phase specific heat capacity).

    Includes scenario-specific enthalpy calculations (e.g., cryopump for refueling).

    Args:
        Qdot_s: Heat flow from solid to fluid [W] (pre-calculated in solve method)
        Qdot_disch: Discharge heat flow [W] (pre-calculated in solve method)
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    T = max(T, 1.0)

    rho = m / V_t

    # Get configuration-dependent algebraic equations for two-phase
    config_eqs = config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=True, t=t, mdot_disch_func=mdot_disch_func, Ts=Ts, scenario_manager=scenario_manager)
    p = config_eqs['pressure']

    h = PropsSI("Hmass", "T", T, "Dmass", rho, fluid)

    # Calculate vapor fraction
    try:
        x = PropsSI("Q", "T", T, "Dmass", rho, fluid)  # Quality (vapor fraction)
        x = max(0.0, min(1.0, x))  # Clamp between 0 and 1
    except:
        x = 0.5  # Default to 50% if calculation fails

    # Get saturated liquid and vapor properties
    c_v_liquid = PropsSI("Cvmass", "T", T, "Q", 0, fluid)  # Saturated liquid cv
    c_v_vapor = PropsSI("Cvmass", "T", T, "Q", 1, fluid)   # Saturated vapor cv

    # Calculate two-phase specific heat capacity (c_v2P)
    c_v2P = x * c_v_vapor + (1.0 - x) * c_v_liquid

    # Calculate saturation pressure derivative (dp_sat/dT) using Clausius-Clapeyron
    h_vapor = PropsSI("Hmass", "T", T, "Q", 1, fluid)    # Saturated vapor enthalpy
    h_liquid = PropsSI("Hmass", "T", T, "Q", 0, fluid)   # Saturated liquid enthalpy
    rho_vapor = PropsSI("Dmass", "T", T, "Q", 1, fluid)  # Saturated vapor density
    rho_liquid = PropsSI("Dmass", "T", T, "Q", 0, fluid) # Saturated liquid density

    L_v = h_vapor - h_liquid  # Latent heat of vaporization [J/kg]
    delta_v = (1.0/rho_vapor) - (1.0/rho_liquid)  # Specific volume difference [m³/kg]

    # Clausius-Clapeyron: dp_sat/dT = L_v / (T * Δv)
    dp_sat_dT = L_v / (T * delta_v)

    nu = 1.0 / rho

    # Get mass flow rates (may be config-dependent)
    mdot_f = mdot_fuel_func(t)
    mdot_d = mdot_disch_func(t)
    mdot_v = config_eqs['mdot_vent']  # Configuration-dependent

    # Scenario-specific enthalpy calculation
    current_scenario = scenario_manager.current_scenario
    if current_scenario == 'REFUEL':
        # Use pump outlet enthalpy for refueling (includes cryopump compression work)
        h_fuel = compute_pump_outlet_hydrogen(p, T)
    else:
        # For discharge and dormancy, use tank enthalpy (no cryopump)
        h_fuel = h

    h_dich = h
    h_vent = h
    Qdot_disch = config_eqs['qdot_disch']  # Configuration-dependent

    # # Configuration A and C need explicit environmental heat transfer
    # alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)
    # Qdot_s = alpha_s * A_in * (Ts - T)

    # Two-phase energy balance numerator
    numerator_T = (mdot_f * (h_fuel - h)
                   - mdot_d * (h_dich - h)
                   - mdot_v * (h_vent - h)
                   + (T / rho) * dp_sat_dT * (mdot_f - mdot_d - mdot_v)
                   + Qdot_s
                   + Qdot_disch)

    return numerator_T / (m * c_v2P)

# Unified condition functions for model registration
def is_two_phase_condition(T, p, rho):
    """Two-phase condition: P ≈ P_sat"""
    return is_near_saturation(T, p, fluid)

def is_single_phase_condition(T, p, rho):
    """Single-phase condition: P not close to P_sat"""
    return not is_near_saturation(T, p, fluid)

# Wrapper functions to handle the new signature
def single_phase_wrapper(t, y, Qdot_s, Qdot_disch):
    """Wrapper for single phase model to handle configuration management."""
    # Get current state for configuration selection
    m, T, Ts = y
    rho = m / V_t

    # Get preliminary pressure for configuration selection
    try:
        p_prelim = PropsSI("P", "T", T, "Dmass", rho, fluid)
    except:
        p_prelim = 15e5  # Fallback to minimum pressure

    # Select configuration
    is_two_phase = is_near_saturation(T, p_prelim, fluid)
    current_config = config_manager.select_configuration(p_prelim, is_two_phase)

    return single_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager, Qdot_s, Qdot_disch)

def two_phase_wrapper(t, y, Qdot_s, Qdot_disch):
    """Wrapper for two phase model to handle configuration management."""
    # Get current state for configuration selection
    m, T, Ts = y
    rho = m / V_t

    # Get preliminary pressure for configuration selection (saturation pressure)
    try:
        p_prelim = PropsSI("P", "T", T, "Q", 0, fluid)
    except:
        p_prelim = 15e5  # Fallback to minimum pressure

    # Select configuration
    is_two_phase = True  # We're in two-phase wrapper
    current_config = config_manager.select_configuration(p_prelim, is_two_phase)

    return two_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager, Qdot_s, Qdot_disch)

def odes(t, y):
    global step_counter  # Access the global step counter

    # Print state every N steps if enabled
    if PRINT_EVERY_N_STEPS > 0:
        step_counter += 1
        if step_counter % PRINT_EVERY_N_STEPS == 0:
            m, T, Ts = y
            rho = m / V_t
            try:
                p = PropsSI("P", "T", T, "Dmass", rho, fluid)
                selected_model = "two_phase" if is_near_saturation(T, p, fluid) else "single_phase"
                print(f"Step {step_counter:4d} | t={t:7.2f}s | "
                      f"m={m:6.2f}kg | ρ={rho:6.2f}kg/m³ | "
                      f"T={T:6.2f}K | Ts={Ts:6.2f}K | "
                      f"P={p/1e5:6.2f}bar | alpha_s={getattr(model_switcher, 'alpha_s_last', 0.0):.4f} | {selected_model}")
            except:
                print(f"Step {step_counter:4d} | t={t:7.2f}s | "
                      f"m={m:6.2f}kg | ρ={rho:6.2f}kg/m³ | "
                      f"T={T:6.2f}K | Ts={Ts:6.2f}K | "
                      f"alpha_s={getattr(model_switcher, 'alpha_s_last', 0.0):.4f} | (CoolProp error)")

    return model_switcher.solve(t, y, mdot_fuel_func, mdot_disch_func)

# Register the models with wrappers - this needs to be done after model_switcher is created
model_switcher.register_model("single_phase", is_single_phase_condition, single_phase_wrapper)
model_switcher.register_model("two_phase", is_two_phase_condition, two_phase_wrapper)

# ----------------- Simulation Runner Function -----------------
def run_hydrogen_tank_simulation(scenario_name, verbose=True, t_offset=0.0):
    """
    Run a hydrogen tank simulation for a given scenario.

    Parameters
    ----------
    scenario_name : str
        Name of scenario to run: 'REFUEL', 'DISCHARGE', or 'DORMANCY'
    verbose : bool, optional
        Whether to print progress information (default: True)
    t_offset : float, optional
        Time offset to add to all time values for chaining scenarios (default: 0.0)

    Returns
    -------
    dict
        Dictionary containing:
        - 'sol': scipy ODE solution object
        - 'scenario': scenario name
        - 'success': whether simulation completed successfully
        - 'stop_info': information about stopping condition if triggered
        - 't_offset': time offset used
        - 'metadata': additional simulation metadata
    """
    global step_counter, mdot_fuel_func, mdot_disch_func, rho_stop, global_scenario_manager

    # Set scenario
    scenario_manager.set_scenario(scenario_name)
    global_scenario_manager = scenario_manager  # Set global reference for configuration functions
    scenario_config = scenario_manager.get_scenario_config()

    # Get scenario-specific parameters
    initial_conditions = scenario_config['initial_conditions']
    p0 = initial_conditions['p0']
    T0 = initial_conditions['T0']
    Ts0 = initial_conditions['Ts0']

    # Handle thermal equilibrium calculation for Ts0
    if Ts0 == 'thermal_equilibrium':
        Ts0 = calculate_thermal_equilibrium_Ts(T0)
        print(f"Using thermal equilibrium Ts0 = {Ts0:.2f}K instead of arbitrary high temperature")

    rho_stop = scenario_config['rho_stop']
    max_time = scenario_config['max_time']
    solver_settings = scenario_config['solver_settings']

    # Get scenario-specific mass flow functions (make them global for ODE access)
    mdot_fuel_func = scenario_config['mass_flow_functions']['mdot_fuel']
    mdot_disch_func = scenario_config['mass_flow_functions']['mdot_disch']
    mdot_vent_func = scenario_config['mass_flow_functions']['mdot_vent']
    Qdot_disch_func = scenario_config['Qdot_disch']

    # Calculate initial density from given pressure and temperature
    rho0 = PropsSI("Dmass", "P", p0, "T", T0, fluid)
    m0 = rho0 * V_t
    t_span = (t_offset, t_offset + max_time)

    # Create time-offset adjusted density event function
    def density_event_with_offset(t, y):
        """Event function to detect when density reaches stopping threshold."""
        m, T, Ts = y
        m = max(m, 1e-12)
        current_density = m / V_t
        return current_density - rho_stop

    # Configure the event to stop integration when density threshold is reached
    density_event_with_offset.terminal = True

    # Set direction based on scenario:
    # REFUEL: +1 (density increasing), DISCHARGE/DORMANCY: -1 (density decreasing)
    if scenario_name == 'REFUEL':
        density_event_with_offset.direction = +1  # Detect increasing density for refuel scenarios
    else:
        density_event_with_offset.direction = -1  # Detect decreasing density for discharge/dormancy scenarios

    # Initial state
    y0 = [m0, T0, Ts0]

    if verbose:
        print(f"\n=== Starting {scenario_name} simulation ===")
        print(f"Initial conditions: T={T0:.2f}K, P={p0/1e5:.2f}bar")
        print(f"Stopping condition: density {rho_stop:.1f}kg/m³")
        print(f"Time span: {t_span[0]:.1f} - {t_span[1]:.1f} seconds")

    # Check initial model selection
    initial_p = PropsSI('P', 'T', T0, 'Dmass', rho0, fluid)
    if is_near_saturation(T0, initial_p, fluid):
        initial_model = "two_phase"
    else:
        initial_model = "single_phase"

    if verbose:
        print(f"Initial model: {initial_model}")
        print("Solving ODEs...")

    # Reset step counter before integration
    step_counter = 0

    # Build solver arguments from scenario settings
    solver_kwargs = {
        'method': solver_settings['method'],
        'atol': solver_settings['atol'],
        'rtol': solver_settings['rtol'],
        'dense_output': solver_settings['dense_output'],
        'events': density_event_with_offset
    }

    # Add optional parameters only if they are not None
    if solver_settings['max_step'] is not None:
        solver_kwargs['max_step'] = solver_settings['max_step']
    if solver_settings['min_step'] is not None:
        solver_kwargs['min_step'] = solver_settings['min_step']
    if solver_settings['first_step'] is not None:
        solver_kwargs['first_step'] = solver_settings['first_step']

    # Solve the ODE system
    sol = solve_ivp(odes, t_span, y0, **solver_kwargs)

    # Prepare stop information
    stop_info = None
    if sol.success:
        if hasattr(sol, 't_events') and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
            # Simulation stopped by density event
            stop_time = sol.t_events[0][0]
            stop_state = sol.y_events[0][0]
            stop_mass, stop_temp, stop_temp_s = stop_state
            stop_density = stop_mass / V_t
            stop_pressure = PropsSI("P", "T", stop_temp, "Dmass", stop_density, fluid)

            stop_info = {
                'stopped_by_event': True,
                'stop_time': stop_time,
                'final_mass': stop_mass,
                'final_temperature': stop_temp,
                'final_solid_temperature': stop_temp_s,
                'final_density': stop_density,
                'final_pressure': stop_pressure
            }

            if verbose:
                print(f"\n*** DENSITY STOPPING CONDITION TRIGGERED ***")
                print(f"Simulation stopped at t = {stop_time:.2f} seconds")
                print(f"Final density: {stop_density:.2f} kg/m³ ({stop_density:.2f} g/L)")
                print(f"Final conditions:")
                print(f"  Mass: {stop_mass:.2f} kg")
                print(f"  Temperature: {stop_temp:.2f} K")
                print(f"  Solid Temperature: {stop_temp_s:.2f} K")
                print(f"  Pressure: {stop_pressure/1e5:.2f} bar")
        else:
            # Simulation completed full time span
            final_density = sol.y[0, -1] / V_t
            stop_info = {
                'stopped_by_event': False,
                'final_density': final_density
            }

            if verbose:
                print(f"Simulation completed without reaching density threshold.")
                print(f"Final density: {final_density:.2f} kg/m³ ({final_density:.2f} g/L)")

    if verbose:
        print("Integration completed!")
        if not sol.success:
            print(f"Solution message: {sol.message}")

    # Prepare metadata
    metadata = {
        'initial_conditions': initial_conditions,
        'rho_stop': rho_stop,
        'max_time': max_time,
        'solver_settings': solver_settings,
        'initial_model': initial_model,
        'V_t': V_t
    }

    return {
        'sol': sol,
        'scenario': scenario_name,
        'success': sol.success,
        'stop_info': stop_info,
        't_offset': t_offset,
        'metadata': metadata
    }

def postprocess_simulation_result(result, num_points=400):
    """
    Postprocess a simulation result to extract time series data.

    Parameters
    ----------
    result : dict
        Result dictionary from run_hydrogen_tank_simulation()
    num_points : int, optional
        Number of points for interpolated time series (default: 400)

    Returns
    -------
    dict
        Dictionary containing:
        - 't': time array
        - 'm': mass array
        - 'T': temperature array
        - 'Ts': solid temperature array
        - 'rho': density array
        - 'p': pressure array
        - 'model_used': array of model types used
        - 'stats': summary statistics
    """
    sol = result['sol']
    scenario = result['scenario']
    t_offset = result['t_offset']
    metadata = result['metadata']
    V_t = metadata['V_t']

    # Determine the actual time span for evaluation
    if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
        actual_t_final = result['stop_info']['stop_time']
    else:
        actual_t_final = sol.t[-1]

    # Create evaluation time points
    t_eval = np.linspace(sol.t[0], actual_t_final, num_points)
    y_eval = sol.sol(t_eval)
    m_sol, T_sol, Ts_sol = y_eval
    rho_sol = m_sol / V_t

    # Calculate pressure and track models used
    p_sol = []
    model_used = []
    for i, (T, rho) in enumerate(zip(T_sol, rho_sol)):
        p = PropsSI("P", "T", T, "Dmass", rho, fluid)
        p_sol.append(p)

        # Determine which model would be used at this state
        if is_near_saturation(T, p, fluid):
            selected_model = "two_phase"
        else:
            selected_model = "single_phase"
        model_used.append(selected_model)

    p_sol = np.array(p_sol)
    model_used = np.array(model_used)

    # Calculate summary statistics
    single_phase_count = np.sum(model_used == 'single_phase')
    two_phase_count = np.sum(model_used == 'two_phase')
    total_points = len(model_used)

    stats = {
        'total_points': total_points,
        'single_phase_count': single_phase_count,
        'two_phase_count': two_phase_count,
        'single_phase_percentage': 100 * single_phase_count / total_points,
        'two_phase_percentage': 100 * two_phase_count / total_points,
        'final_density': rho_sol[-1],
        'density_range': [rho_sol.min(), rho_sol.max()],
        'density_threshold_reached': rho_sol[-1] >= metadata['rho_stop'] * 0.99
    }

    # Add two-phase region statistics if applicable
    if two_phase_count > 0:
        two_phase_indices = model_used == 'two_phase'
        p_two_phase = p_sol[two_phase_indices]
        T_two_phase = T_sol[two_phase_indices]
        stats['two_phase_pressure_range'] = [p_two_phase.min(), p_two_phase.max()]
        stats['two_phase_temperature_range'] = [T_two_phase.min(), T_two_phase.max()]

    return {
        't': t_eval,
        'm': m_sol,
        'T': T_sol,
        'Ts': Ts_sol,
        'rho': rho_sol,
        'p': p_sol,
        'model_used': model_used,
        'stats': stats,
        'scenario': scenario,
        't_offset': t_offset
    }

def run_chained_scenarios(scenarios=['DISCHARGE', 'REFUEL', 'DORMANCY'], verbose=True):
    """
    Run multiple scenarios in sequence, using the final state of one as initial state of the next.

    Parameters
    ----------
    scenarios : list, optional
        List of scenario names to run in order (default: ['DISCHARGE', 'REFUEL', 'DORMANCY'])
    verbose : bool, optional
        Whether to print progress information (default: True)

    Returns
    -------
    list
        List of simulation result dictionaries
    """
    results = []
    current_time_offset = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running chained scenarios: {' → '.join(scenarios)}")
        print(f"{'='*60}")

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"\n--- Scenario {i+1}/{len(scenarios)}: {scenario} ---")

        # Run the scenario
        result = run_hydrogen_tank_simulation(scenario, verbose=verbose, t_offset=current_time_offset)
        results.append(result)

        # Update time offset for next scenario
        if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
            current_time_offset = result['stop_info']['stop_time']
        else:
            current_time_offset = result['sol'].t[-1]

        if verbose and result['success']:
            print(f"✓ {scenario} completed successfully")
            if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
                print(f"  Stopped at density threshold: {result['stop_info']['final_density']:.2f} kg/m³")
            else:
                print(f"  Completed full time span")

    if verbose:
        print(f"\n{'='*60}")
        print(f"All scenarios completed! Total simulation time: {current_time_offset:.2f} seconds")
        print(f"{'='*60}")

    return results

def plot_chained_scenarios(results, postprocessed_data=None):
    """
    Plot results from multiple scenarios with different colors.

    Parameters
    ----------
    results : list
        List of result dictionaries from run_hydrogen_tank_simulation()
    postprocessed_data : list, optional
        List of postprocessed data dictionaries. If None, will postprocess automatically.
    """
    if postprocessed_data is None:
        postprocessed_data = [postprocess_simulation_result(result) for result in results]

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    scenario_colors = {data['scenario']: colors[i % len(colors)] for i, data in enumerate(postprocessed_data)}

    plt.figure(figsize=(16, 12))

    # Mass vs time
    plt.subplot(3, 4, 1)
    for data in postprocessed_data:
        plt.plot(data['t'], data['m'], color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.title("Mass vs Time")
    plt.legend()
    plt.grid(True)

    # Gas Temperature vs time
    plt.subplot(3, 4, 2)
    for data in postprocessed_data:
        plt.plot(data['t'], data['T'], color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Gas Temperature vs Time")
    plt.legend()
    plt.grid(True)

    # Liner/Wall Temperature vs time
    plt.subplot(3, 4, 3)
    for data in postprocessed_data:
        plt.plot(data['t'], data['Ts'], color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Solid Temperature (K)")
    plt.title("Liner/Wall Temperature vs Time")
    plt.legend()
    plt.grid(True)

    # Pressure vs time
    plt.subplot(3, 4, 4)
    for data in postprocessed_data:
        plt.plot(data['t'], data['p']/1e5, color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure vs Time")
    plt.legend()
    plt.grid(True)

    # Density vs time
    plt.subplot(3, 4, 5)
    for i, data in enumerate(postprocessed_data):
        plt.plot(data['t'], data['rho'], color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
        # Add density stopping threshold for reference
        result = results[i]
        rho_stop = result['metadata']['rho_stop']
        plt.axhline(y=rho_stop, color=scenario_colors[data['scenario']],
                   linestyle='--', alpha=0.5, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Density (kg/m³)")
    plt.title("Density vs Time")
    plt.legend()
    plt.grid(True)

    # Model usage vs time
    plt.subplot(3, 4, 6)
    for data in postprocessed_data:
        model_numeric = np.where(data['model_used'] == 'single_phase', 0, 1)
        plt.plot(data['t'], model_numeric, color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Model Type")
    plt.title("Model Usage vs Time")
    plt.yticks([0, 1], ['Single Phase', 'Two Phase'])
    plt.legend()
    plt.grid(True)

    # Pressure vs Temperature
    plt.subplot(3, 4, 7)
    for data in postprocessed_data:
        single_phase_mask = data['model_used'] == 'single_phase'
        two_phase_mask = data['model_used'] == 'two_phase'

        base_color = scenario_colors[data['scenario']]
        if np.any(single_phase_mask):
            plt.scatter(data['T'][single_phase_mask], data['p'][single_phase_mask]/1e5,
                       c=base_color, alpha=0.6, s=10, marker='o',
                       label=f'{data["scenario"]} (Single)')
        if np.any(two_phase_mask):
            plt.scatter(data['T'][two_phase_mask], data['p'][two_phase_mask]/1e5,
                       c=base_color, alpha=0.6, s=10, marker='s',
                       label=f'{data["scenario"]} (Two)')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure vs Temperature")
    plt.legend()
    plt.grid(True)

    # Density vs Temperature
    plt.subplot(3, 4, 8)
    for i, data in enumerate(postprocessed_data):
        plt.plot(data['T'], data['rho'], color=scenario_colors[data['scenario']],
                label=data['scenario'], linewidth=2)
        # Add density stopping threshold for reference
        result = results[i]
        rho_stop = result['metadata']['rho_stop']
        plt.axhline(y=rho_stop, color=scenario_colors[data['scenario']],
                   linestyle='--', alpha=0.5, linewidth=1)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Density (kg/m³)")
    plt.title("Density vs Temperature")
    plt.legend()
    plt.grid(True)

    # Summary statistics
    plt.subplot(3, 4, 9)
    scenarios = [data['scenario'] for data in postprocessed_data]
    single_percentages = [data['stats']['single_phase_percentage'] for data in postprocessed_data]
    two_percentages = [data['stats']['two_phase_percentage'] for data in postprocessed_data]

    x = np.arange(len(scenarios))
    width = 0.35

    plt.bar(x - width/2, single_percentages, width, label='Single Phase', alpha=0.7)
    plt.bar(x + width/2, two_percentages, width, label='Two Phase', alpha=0.7)
    plt.xlabel('Scenario')
    plt.ylabel('Percentage (%)')
    plt.title('Model Usage Summary')
    plt.xticks(x, scenarios, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Timeline overview
    plt.subplot(3, 4, 10)
    for i, data in enumerate(postprocessed_data):
        scenario_duration = data['t'][-1] - data['t'][0]
        plt.barh(i, scenario_duration, left=data['t'][0],
                color=scenario_colors[data['scenario']], alpha=0.7,
                label=data['scenario'])
        plt.text(data['t'][0] + scenario_duration/2, i,
                f'{scenario_duration:.0f}s', ha='center', va='center')
    plt.xlabel('Time (s)')
    plt.ylabel('Scenario')
    plt.title('Scenario Timeline')
    plt.yticks(range(len(postprocessed_data)), [data['scenario'] for data in postprocessed_data])
    plt.grid(True, alpha=0.3)

    # Final density comparison
    plt.subplot(3, 4, 11)
    final_densities = [data['rho'][-1] for data in postprocessed_data]
    target_densities = [results[i]['metadata']['rho_stop'] for i in range(len(results))]

    x = np.arange(len(scenarios))
    plt.bar(x, final_densities, alpha=0.7, label='Final Density')
    plt.scatter(x, target_densities, color='red', s=50, label='Target Density', zorder=5)
    plt.xlabel('Scenario')
    plt.ylabel('Density (kg/m³)')
    plt.title('Final vs Target Density')
    plt.xticks(x, scenarios, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Print summary table
    plt.subplot(3, 4, 12)
    plt.axis('off')
    summary_text = "Simulation Summary:\n\n"
    for i, data in enumerate(postprocessed_data):
        result = results[i]
        stats = data['stats']
        summary_text += f"{data['scenario']}:\n"
        summary_text += f"  Duration: {data['t'][-1] - data['t'][0]:.1f}s\n"
        summary_text += f"  Final ρ: {stats['final_density']:.1f} kg/m³\n"
        summary_text += f"  Two-phase: {stats['two_phase_percentage']:.1f}%\n"
        if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
            summary_text += f"  ✓ Stopped at threshold\n"
        else:
            summary_text += f"  ○ Completed time span\n"
        summary_text += "\n"

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9)

    plt.tight_layout()

    # Print detailed statistics
    print(f"\n{'='*80}")
    print("DETAILED SIMULATION STATISTICS")
    print(f"{'='*80}")

    for i, (result, data) in enumerate(zip(results, postprocessed_data)):
        print(f"\n{i+1}. {data['scenario']} SCENARIO:")
        print(f"   Time range: {data['t'][0]:.1f} - {data['t'][-1]:.1f} seconds ({data['t'][-1] - data['t'][0]:.1f}s duration)")
        print(f"   Final density: {data['stats']['final_density']:.2f} kg/m³ (target: {result['metadata']['rho_stop']:.1f} kg/m³)")
        print(f"   Density range: {data['stats']['density_range'][0]:.2f} - {data['stats']['density_range'][1]:.2f} kg/m³")
        print(f"   Model usage: {data['stats']['single_phase_percentage']:.1f}% single-phase, {data['stats']['two_phase_percentage']:.1f}% two-phase")

        if result['stop_info'] and result['stop_info'].get('stopped_by_event', False):
            print(f"   ✓ Stopped by density threshold at t={result['stop_info']['stop_time']:.2f}s")
        else:
            print(f"   ○ Completed full time span")

        if 'two_phase_pressure_range' in data['stats']:
            p_range = data['stats']['two_phase_pressure_range']
            T_range = data['stats']['two_phase_temperature_range']
            print(f"   Two-phase region: P={p_range[0]/1e5:.2f}-{p_range[1]/1e5:.2f} bar, T={T_range[0]:.2f}-{T_range[1]:.2f} K")

def plot_combined_density_temperature(results, postprocessed_data=None):
    """
    Create a combined density-temperature plot using SeabornPlotter.

    This function transforms the simulation results into the format expected by
    plot_density_temperature_combined and creates a professional density-temperature plot.

    Parameters
    ----------
    results : list
        List of result dictionaries from run_hydrogen_tank_simulation()
    postprocessed_data : list, optional
        List of postprocessed data dictionaries. If None, will postprocess automatically.
    """
    if postprocessed_data is None:
        postprocessed_data = [postprocess_simulation_result(result) for result in results]

    # Initialize scenario data structure expected by SeabornPlotter
    scenario_data = {
        'discharge': {'temperatures': [], 'densities': [], 'pressures': []},
        'refuel': {'temperatures': [], 'densities': [], 'pressures': []},
        'dormancy': {'temperatures': [], 'densities': [], 'pressures': []}
    }

    # Map scenario names to consistent keys
    scenario_mapping = {
        'DISCHARGE': 'discharge',
        'REFUEL': 'refuel',
        'DORMANCY': 'dormancy'
    }

    print("\n==== PREPARING DATA FOR DENSITY-TEMPERATURE PLOT ====")

    # Extract data from each scenario
    for data in postprocessed_data:
        scenario_name = data['scenario']

        # Map to consistent naming
        plot_key = scenario_mapping.get(scenario_name.upper(), scenario_name.lower())

        if plot_key in scenario_data:
            print(f"Processing {scenario_name} data...")

            # Extract temperatures (K)
            temperatures = data['T']
            scenario_data[plot_key]['temperatures'] = list(temperatures)

            # Extract pressures (Pa)
            pressures = data['p']
            scenario_data[plot_key]['pressures'] = list(pressures)

            # Extract densities (kg/m³) and convert to g/L
            densities_kg_m3 = data['rho']
            densities_g_L = [rho for rho in densities_kg_m3]  # 1 kg/m³ = 1 g/L
            scenario_data[plot_key]['densities'] = densities_g_L

            print(f"  {scenario_name}: {len(temperatures)} data points")
            print(f"    Temperature range: {min(temperatures):.1f} - {max(temperatures):.1f} K")
            print(f"    Density range: {min(densities_g_L):.1f} - {max(densities_g_L):.1f} g/L")
            print(f"    Pressure range: {min(pressures)/1e5:.1f} - {max(pressures)/1e5:.1f} bar")
        else:
            print(f"Warning: Unknown scenario '{scenario_name}', skipping...")

    # Create the SeabornPlotter instance
    print("\n==== CREATING COMBINED DENSITY-TEMPERATURE PLOT ====")
    plotter = SeabornPlotter(font="Cambria", palette="delft")

    # Create the plot with appropriate settings
    fig = plotter.plot_density_temperature_combined(
        scenario_data=scenario_data,
        include_saturation_line=True,
        include_isobars=True,
        include_ref_data=True,
        figsize=(12, 8),
        temperature_range=(15, 80),  # Adjust based on your data range
        density_range=(0, 80)        # Adjust based on your data range
    )


    return fig

# ----------------- Main Execution (Example Usage) -----------------
if __name__ == "__main__":
    # Example: Run single scenario
    # print("Example 1: Running single REFUEL scenario")
    # result = run_hydrogen_tank_simulation('REFUEL', verbose=True)

    # if result['success']:
    #     # Postprocess and plot single scenario
    #     data = postprocess_simulation_result(result)
    #     plot_chained_scenarios([result], [data])

    # Example: Run chained scenarios
    print("\n" + "="*80)
    print("Example 2: Running chained scenarios")
    chained_results = run_chained_scenarios(['DISCHARGE', 'REFUEL', 'DORMANCY'], verbose=True)

    # Postprocess all results
    chained_data = [postprocess_simulation_result(result) for result in chained_results]

    # Plot all scenarios together
    plot_chained_scenarios(chained_results, chained_data)

    # Create the combined density-temperature plot using SeabornPlotter
    plot_combined_density_temperature(chained_results, chained_data)

    plt.show()

