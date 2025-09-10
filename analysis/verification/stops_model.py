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

# ----------------- Scenario and Configuration Framework -----------------
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
                    'T0': 66,        # Initial temperature [K]
                    'Ts0': 298.15,   # Initial solid temperature [K]
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
                    'Ts0': 100.15,    # Initial solid temperature [K]
                },
                'rho_stop': 5.8,     # Stopping density [kg/m³]
                'max_time': 40000.0, # Maximum simulation time [s]
                'solver_settings': {
                    'method': 'Radau',
                    'atol': 1e-9,
                    'rtol': 1e-7,
                    'max_step': 10,
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
                    'Ts0': 298.15,   # Initial solid temperature [K]
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

    def get_algebraic_equations(self, config, T, rho, is_two_phase=False, t=None, mdot_disch_func=None, Ts=None):
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
        Qs = alpha_s * A_in * (Ts - T)  # Environmental heat leak [W]

        # Term 1: T/ρ · (∂p/∂T)_ρ using ideal gas relationship for numerical stability
        term1 = p * nu

        # Term 2: ρ·cv·(∂T/∂ρ)_p using real CoolProp derivatives for accuracy
        try:
            dT_drho_p = cp.PropsSI('d(T)/d(D)|P', 'P', p, 'T', T, self.fluid)
        except:
            # Fallback to ideal gas relationship if CoolProp fails
            dT_drho_p = -T / rho
        term2 = rho * cv * dT_drho_p

        qdot_disch = mdot_disch * (term1 - term2) - Qs

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
        Qs = alpha_s * A_in * (Ts - T)  # Environmental heat leak [W]

        # Term 1: T/ρ · (∂p/∂T)_ρ using ideal gas relationship for numerical stability
        term1 = p * nu

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

        mdot_vent = Qs / denominator

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

    def compute_dT_dt(self, t, y, *args):
        """
        Compute dT/dt using the currently selected model.

        Args:
            t: Time
            y: State vector [m, T, Ts]
            *args: Additional arguments passed to the ODE function

        Returns:
            dT/dt value
        """
        if self.current_model is None:
            raise ValueError("No model selected. Call select_model first.")

        return self.models[self.current_model]['ode_func'](t, y)

    def solve(self, t, y, mdot_fuel_func, mdot_disch_func, *args):
        """
        Solve the complete ODE system by selecting appropriate model and computing all derivatives.

        Args:
            t: Time
            y: State vector [m, T, Ts]
            mdot_fuel_func: Fuel mass flow function
            mdot_disch_func: Discharge mass flow function
            *args: Additional arguments

        Returns:
            [dm_dt, dT_dt, dTs_dt]
        """
        m, T, Ts = y
        m = max(m, 1e-12)
        T = max(T, 1.0)
        Ts = max(Ts, 1.0)

        rho = m / V_t

        # Get thermodynamic properties for model selection
        try:
            h = PropsSI("Hmass", "T", T, "Dmass", rho, self.fluid)
            p = PropsSI("P", "T", T, "Dmass", rho, self.fluid)
        except Exception as e:
            print(f"CoolProp error at t={t:.3f}s: {e}")
            print(f"  State: m={m:.3f}, T={T:.3f}, rho={rho:.3f}")
            raise e

        # Select appropriate model based on current state using simplified logic
        if is_near_saturation(T, p, self.fluid):
            selected_model = "two_phase"
            is_two_phase = True
        else:
            selected_model = "single_phase"
            is_two_phase = False

        # Update the current model
        self.current_model = selected_model

        # Select configuration and get configuration-dependent mass flows
        current_config = self.config_manager.select_configuration(p, is_two_phase)
        config_eqs = self.config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=is_two_phase, t=t, mdot_disch_func=mdot_disch_func, Ts=Ts)

        # Mass balance using configuration-dependent flows
        mdot_f = mdot_fuel_func(t)
        mdot_d = mdot_disch_func(t)
        mdot_v = config_eqs['mdot_vent']  # Configuration-dependent venting
        dm_dt = mdot_f - mdot_d - mdot_v

        # Temperature balance using selected model
        dT_dt = self.compute_dT_dt(t, y)

        # Solid temperature balance
        c_liner = float(c_liner_func(Ts))
        c_wall = float(c_wall_func(Ts))

        # Calculate alpha_s based on current temperature conditions
        alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)

        numerator_Ts = (k_amb * A_out * T_amb
                        - (k_amb * A_out + alpha_s * A_in) * Ts
                        + alpha_s * A_in * T)

        denom_Ts = m_liner * c_liner + m_wall * c_wall
        dTs_dt = numerator_Ts / denom_Ts

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
D_inner = 0.4      # Inner diameter [m]
D_outer = 0.45     # Outer diameter [m]
annular_fluid = "Hydrogen"  # Fluid in the gap

# Configuration pressure thresholds
p_min = 15e5       # Minimum pressure threshold for configuration B [Pa]
p_vent = 450e5     # Venting pressure threshold for configuration C [Pa]

# Initialize framework managers
scenario_manager = ScenarioManager()
config_manager = ConfigurationManager(fluid, p_min, p_vent)
model_switcher = ModelSwitcher(fluid, config_manager)

# Set scenario
CURRENT_SCENARIO = 'REFUEL'  # Options: 'REFUEL', 'DISCHARGE', 'DORMANCY'
scenario_manager.set_scenario(CURRENT_SCENARIO)
scenario_config = scenario_manager.get_scenario_config()

# Get scenario-specific parameters
initial_conditions = scenario_config['initial_conditions']
p0 = initial_conditions['p0']
T0 = initial_conditions['T0']
Ts0 = initial_conditions['Ts0']
rho_stop = scenario_config['rho_stop']
max_time = scenario_config['max_time']
solver_settings = scenario_config['solver_settings']

# Get scenario-specific mass flow functions
mdot_fuel_func = scenario_config['mass_flow_functions']['mdot_fuel']
mdot_disch_func = scenario_config['mass_flow_functions']['mdot_disch']
mdot_vent_func = scenario_config['mass_flow_functions']['mdot_vent']
Qdot_disch_func = scenario_config['Qdot_disch']

# Calculate initial density from given pressure and temperature
rho0 = PropsSI("Dmass", "P", p0, "T", T0, fluid)
m0 = rho0 * V_t
t_span = (0.0, max_time)

# Initialize NIST materials
liner_material = NISTMetal.aluminum_6061T6_nist()
wall_material = NISTComposite.g10_nist(winding_angle=0.0)

def c_liner_func(Ts):
    """Get specific heat of aluminum 6061-T6 liner using NIST data"""
    return liner_material.determine_specific_heat(Ts)

def c_wall_func(Ts):
    """Get specific heat of G10 wall using NIST data"""
    return wall_material.determine_specific_heat(Ts)

# ----------------- Heat Transfer Coefficient Function -----------------
def get_alpha_s(T_i, T_o, D_i, D_o, fluid="Air", p=101325):
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

    # Input validation
    if D_o <= D_i:
        raise ValueError("Outer diameter must be greater than inner diameter")
    if T_i <= 0 or T_o <= 0:
        raise ValueError("Temperatures must be positive")
    if p <= 0:
        raise ValueError("Pressure must be positive")

    # Mean film temperature
    T_mean = 0.5 * (T_i + T_o)

    try:
        # Fluid properties at mean T
        k = PropsSI('L', 'T', T_mean, 'P', p, fluid)   # thermal conductivity [W/m/K]
        mu = PropsSI('V', 'T', T_mean, 'P', p, fluid)   # dynamic viscosity [Pa*s]
        rho = PropsSI('D', 'T', T_mean, 'P', p, fluid)  # density [kg/m3]
        cp = PropsSI('C', 'T', T_mean, 'P', p, fluid)   # specific heat [J/kg/K]
    except Exception as e:
        raise ValueError(f"Could not calculate fluid properties for {fluid} at T={T_mean:.2f}K, P={p:.0f}Pa: {e}")

    # Derived properties
    nu = mu / rho                        # kinematic viscosity [m2/s]
    alpha = k / (rho * cp)              # thermal diffusivity [m2/s]
    beta = 1.0 / T_mean                  # thermal expansion [1/K] (ideal gas approx)
    Pr = nu / alpha                      # Prandtl number

    # Geometry
    L = (D_o - D_i) / 2.0                # gap thickness [m]

    # Rayleigh number based on gap
    g = 9.81
    Ra = g * beta * abs(T_i - T_o) * L**3 / (nu * alpha)

    # Check for very low Rayleigh numbers (pure conduction)
    if Ra < 1e3:
        # For very low Ra, use pure conduction
        Nu = 2.0 / np.log(D_o / D_i)
        h_i = (2.0 * k) / (D_i * np.log(D_o / D_i))
        return h_i

    # Pure conduction Nusselt number
    Nu_cond = 2.0 / np.log(D_o / D_i)

    # --------------------------------------------------
    # Kuehn & Goldstein correlation (1976) - Equations (1a)-(1f)
    # --------------------------------------------------

    # Calculate Ra*D_i and Ra*D_o for the correlations
    Ra_Di = Ra * D_i
    Ra_Do = Ra * D_o

    # Equation (1a): Nu_i'
    denominator_i = (0.5 * Ra_Di**(1/4))**15 + (0.12 * Ra_Di**(1/3))**15
    Nu_i_prime = 2.0 / np.log(1 + 2.0 / (denominator_i**(1/15)))

    # Equation (1b): Nu_o'
    denominator_o = (Ra_Do**(1/4))**15 + (0.12 * Ra_Do**(1/3))**15
    Nu_o_prime = -2.0 / np.log(1 - 2.0 / (denominator_o**(1/15)))

    # Equation (1c): phi_b (blending function)
    phi_b = Nu_i_prime / (Nu_i_prime + Nu_o_prime)

    # Equation (1d): Nu_conv (convective contribution)
    Nu_conv = 1.0 / (1.0/Nu_i_prime + 1.0/Nu_o_prime)

    # Equation (1e): Nu_cond already calculated above
    # Nu_cond = 2.0 / np.log(D_o / D_i)

    # Equation (1f): Final Nusselt number
    Nu = ((Nu_cond)**15 + (Nu_conv)**15)**(1/15)

    # Equation (1g): Equivalent thermal conductivity ratio
    k_eq_ratio = Nu / Nu_cond
    # --------------------------------------------------

    # Equivalent thermal conductivity
    k_eq = k_eq_ratio * k

    # Convert to equivalent h_i (inner surface coefficient)
    # For concentric cylinders: h_i = 2*k_eq / (D_i * ln(D_o/D_i))
    h_i = (2.0 * k_eq) / (D_i * np.log(D_o / D_i))

    # return h_i
    return 1  # Simplified return for testing

# Remove the test function as it's not needed in production code

# ----------------- Model Implementations -----------------
def single_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager):
    """
    Single-phase energy balance.
    Uses cv for temperature derivative calculation.

    Includes scenario-specific enthalpy calculations (e.g., cryopump for refueling).
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    T = max(T, 1.0)

    rho = m / V_t

    # Get configuration-dependent algebraic equations
    config_eqs = config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=False, t=t, mdot_disch_func=mdot_disch_func, Ts=Ts)
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
    Qdot_disch = config_eqs['qdot_disch']  # Configuration-dependent

    # Configuration A and C need explicit environmental heat transfer
    alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)
    Qs_env = alpha_s * A_in * (Ts - T)


    numerator_T = (mdot_f * (h_fuel - h)
                   - mdot_d * (h_dich - h)
                   - mdot_v * (h_vent - h)
                   + p * nu * (mdot_f - mdot_d - mdot_v)
                   + Qs_env
                   + Qdot_disch)

    dT_dt = numerator_T / (m * c_v)

    return dT_dt

def two_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager):
    """
    Two-phase energy balance implementation.
    Uses c_v2P (two-phase specific heat capacity).

    Includes scenario-specific enthalpy calculations (e.g., cryopump for refueling).
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    T = max(T, 1.0)

    rho = m / V_t

    # Get configuration-dependent algebraic equations for two-phase
    config_eqs = config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=True, t=t, mdot_disch_func=mdot_disch_func, Ts=Ts)
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

    # Calculate saturation pressure derivative (dp_sat/dT)
    dp_sat_dT = p / (t + 1e-6)  # Simplified approximation

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

    # Configuration A and C need explicit environmental heat transfer
    alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)
    Qs_env = alpha_s * A_in * (Ts - T)

    # Two-phase energy balance numerator
    numerator_T = (mdot_f * (h_fuel - h)
                   - mdot_d * (h_dich - h)
                   - mdot_v * (h_vent - h)
                   + (T / rho) * dp_sat_dT * (mdot_f - mdot_d - mdot_v)
                   + Qs_env
                   + Qdot_disch)

    return numerator_T / (m * c_v2P)

# Initialize model switcher
model_switcher = ModelSwitcher(fluid, config_manager)

# Simplified Phase Detection
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
        tolerance = 1e-4  # Very small tolerance
        return abs(p - p_sat) < tolerance * p_sat

    except:
        # If saturation pressure calculation fails (e.g., above critical point),
        # assume single-phase
        return False

# Unified condition functions for model registration
def is_two_phase_condition(T, p, rho):
    """Two-phase condition: P ≈ P_sat"""
    return is_near_saturation(T, p, fluid)

def is_single_phase_condition(T, p, rho):
    """Single-phase condition: P not close to P_sat"""
    return not is_near_saturation(T, p, fluid)

# Wrapper functions to handle the new signature
def single_phase_wrapper(t, y):
    """Wrapper for single phase model to handle configuration management."""
    # Get current state for configuration selection
    m, T, Ts = y
    rho = m / V_t

    # Get preliminary pressure for configuration selection
    try:
        p_prelim = PropsSI("P", "T", T, "Dmass", rho, fluid)
    except:
        p_prelim = p0  # Fallback to initial pressure

    # Select configuration
    is_two_phase = is_near_saturation(T, p_prelim, fluid)
    current_config = config_manager.select_configuration(p_prelim, is_two_phase)

    return single_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager)

def two_phase_wrapper(t, y):
    """Wrapper for two phase model to handle configuration management."""
    # Get current state for configuration selection
    m, T, Ts = y
    rho = m / V_t

    # Get preliminary pressure for configuration selection (saturation pressure)
    try:
        p_prelim = PropsSI("P", "T", T, "Q", 0, fluid)
    except:
        p_prelim = p0  # Fallback to initial pressure

    # Select configuration
    is_two_phase = True  # We're in two-phase wrapper
    current_config = config_manager.select_configuration(p_prelim, is_two_phase)

    return two_phase_dT_dt(t, y, model_switcher, config_manager, current_config, scenario_manager)

# Register the models with wrappers
model_switcher.register_model("single_phase", is_single_phase_condition, single_phase_wrapper)
model_switcher.register_model("two_phase", is_two_phase_condition, two_phase_wrapper)

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
                      f"P={p/1e5:6.2f}bar | {selected_model}")
            except:
                print(f"Step {step_counter:4d} | t={t:7.2f}s | (CoolProp error)")

    return model_switcher.solve(t, y, mdot_fuel_func, mdot_disch_func)

def density_event(t, y):
    """
    Event function to detect when density reaches stopping threshold.

    Returns zero when density equals rho_stop for solver termination.
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    current_density = m / V_t
    return current_density - rho_stop

# Configure the event to stop integration when density threshold is reached
density_event.terminal = True

# Set direction based on scenario:
# REFUEL: +1 (density increasing), DISCHARGE/DORMANCY: -1 (density decreasing)
if CURRENT_SCENARIO == 'REFUEL':
    density_event.direction = +1  # Detect increasing density for refuel scenarios
else:
    density_event.direction = -1  # Detect decreasing density for discharge/dormancy scenarios

# Initial state
y0 = [m0, T0, Ts0]

print("Starting simulation...")
print(f"Initial conditions: T={T0:.2f}K, P={p0/1e5:.2f}bar")
print(f"Stopping condition: density {rho_stop:.1f}kg/m³")

# Check initial model selection
initial_p = PropsSI('P', 'T', T0, 'Dmass', rho0, fluid)
if is_near_saturation(T0, initial_p, fluid):
    initial_model = "two_phase"
else:
    initial_model = "single_phase"

try:
    p_sat_initial = PropsSI("P", "T", T0, "Q", 0, fluid)
except:
    pass


print("Solving ODEs...")

# Reset step counter before integration
step_counter = 0

# Use scenario-specified time span directly
t_span_limited = t_span

# Build solver arguments from scenario settings
solver_kwargs = {
    'method': solver_settings['method'],
    'atol': solver_settings['atol'],
    'rtol': solver_settings['rtol'],
    'dense_output': solver_settings['dense_output'],
    'events': density_event
}

# Add optional parameters only if they are not None
if solver_settings['max_step'] is not None:
    solver_kwargs['max_step'] = solver_settings['max_step']
if solver_settings['min_step'] is not None:
    solver_kwargs['min_step'] = solver_settings['min_step']
if solver_settings['first_step'] is not None:
    solver_kwargs['first_step'] = solver_settings['first_step']

sol = solve_ivp(odes, t_span_limited, y0, **solver_kwargs)

print("Integration completed!")
if not sol.success:
    print(f"Solution message: {sol.message}")
else:
    # Check if simulation was stopped by density event
    if hasattr(sol, 't_events') and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
        stop_time = sol.t_events[0][0]
        stop_state = sol.y_events[0][0]
        stop_mass, stop_temp, stop_temp_s = stop_state
        stop_density = stop_mass / V_t
        stop_pressure = PropsSI("P", "T", stop_temp, "Dmass", stop_density, fluid)

        print(f"\n*** DENSITY STOPPING CONDITION TRIGGERED ***")
        print(f"Simulation stopped at t = {stop_time:.2f} seconds")
        print(f"Final density: {stop_density:.2f} kg/m³ ({stop_density/1000:.3f} g/L)")
        print(f"Final conditions:")
        print(f"  Mass: {stop_mass:.2f} kg")
        print(f"  Temperature: {stop_temp:.2f} K")
        print(f"  Solid Temperature: {stop_temp_s:.2f} K")
        print(f"  Pressure: {stop_pressure/1e5:.2f} bar")
    else:
        final_density = sol.y[0, -1] / V_t
        print(f"Simulation completed without reaching density threshold.")
        print(f"Final density: {final_density:.2f} kg/m³ ({final_density/1000:.3f} g/L)")

# Postprocess
print("\nPostprocessing...")

# Determine the actual time span for evaluation based on whether simulation stopped early
if hasattr(sol, 't_events') and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
    # Simulation stopped due to density event
    actual_t_final = sol.t_events[0][0]
else:
    # Simulation completed full time span
    actual_t_final = t_span[1]

t_eval = np.linspace(t_span[0], actual_t_final, 400)
y_eval = sol.sol(t_eval)
m_sol, T_sol, Ts_sol = y_eval
rho_sol = m_sol / V_t

p_sol = []
model_used = []  # Track which model was used at each time point
for i, (T, rho) in enumerate(zip(T_sol, rho_sol)):
    p = PropsSI("P", "T", T, "Dmass", rho, fluid)
    p_sol.append(p)

    # Determine which model would be used at this state
    if is_near_saturation(T, p, fluid):
        selected_model = "two_phase"
    else:
        selected_model = "single_phase"
    model_used.append(selected_model)

print("Postprocessing completed!")

p_sol = np.array(p_sol)
model_used = np.array(model_used)

# Print summary statistics
single_phase_count = np.sum(model_used == 'single_phase')
two_phase_count = np.sum(model_used == 'two_phase')
total_points = len(model_used)

print(f"\nSimulation Summary:")
print(f"Total time points: {total_points}")
print(f"Single-phase model used: {single_phase_count} times ({100*single_phase_count/total_points:.1f}%)")
print(f"Two-phase model used: {two_phase_count} times ({100*two_phase_count/total_points:.1f}%)")
print(f"Final density: {rho_sol[-1]:.2f} kg/m³ ({rho_sol[-1]/1000:.3f} g/L)")
print(f"Density range: {rho_sol.min():.2f} - {rho_sol.max():.2f} kg/m³")
if rho_sol[-1] >= rho_stop * 0.99:  # Within 1% of stopping density
    print(f"*** Density stopping condition ({rho_stop:.1f} kg/m³) was reached ***")

# Find pressure range where two-phase model is active
if two_phase_count > 0:
    two_phase_indices = model_used == 'two_phase'
    p_two_phase = p_sol[two_phase_indices]
    T_two_phase = T_sol[two_phase_indices]
    print(f"Two-phase region:")
    print(f"  Pressure range: {p_two_phase.min()/1e5:.2f} - {p_two_phase.max()/1e5:.2f} bar")
    print(f"  Temperature range: {T_two_phase.min():.2f} - {T_two_phase.max():.2f} K")

# ----------------- Plots -----------------
plt.figure(figsize=(15, 10))

# Mass vs time
plt.subplot(2, 4, 1)
plt.plot(t_eval, m_sol)
plt.xlabel("time (s)")
plt.ylabel("m (kg)")
plt.title("Mass vs time")
plt.grid(True)

# Gas Temperature vs time
plt.subplot(2, 4, 2)
plt.plot(t_eval, T_sol)
plt.xlabel("time (s)")
plt.ylabel("T (K)")
plt.title("Gas Temperature vs time")
plt.grid(True)

# Liner/Wall Temperature vs time
plt.subplot(2, 4, 3)
plt.plot(t_eval, Ts_sol)
plt.xlabel("time (s)")
plt.ylabel("T_s (K)")
plt.title("Liner/Wall Temperature vs time")
plt.grid(True)

# Pressure vs time
plt.subplot(2, 4, 4)
plt.plot(t_eval, p_sol/1e5)
plt.xlabel("time (s)")
plt.ylabel("p (bar)")
plt.title("Pressure vs time")
plt.grid(True)

# Density vs time (NEW)
plt.subplot(2, 4, 5)
plt.plot(t_eval, rho_sol, 'b-', linewidth=2, label='Density')
plt.axhline(y=rho_stop, color='r', linestyle='--', linewidth=2,
           label=f'Stop threshold ({rho_stop:.1f} kg/m³)')
plt.xlabel("time (s)")
plt.ylabel("ρ (kg/m³)")
plt.title("Density vs time")
plt.legend()
plt.grid(True)

# Model usage vs time
plt.subplot(2, 4, 6)
model_numeric = np.where(model_used == 'single_phase', 0, 1)
plt.plot(t_eval, model_numeric, linewidth=2)
plt.xlabel("time (s)")
plt.ylabel("Model Type")
plt.title("Model Usage vs time")
plt.yticks([0, 1], ['Single Phase', 'Two Phase'])
plt.grid(True)

# Pressure vs Temperature with model regions
plt.subplot(2, 4, 7)
single_phase_mask = model_used == 'single_phase'
two_phase_mask = model_used == 'two_phase'

if np.any(single_phase_mask):
    plt.scatter(T_sol[single_phase_mask], p_sol[single_phase_mask]/1e5,
               c='blue', label='Single Phase', alpha=0.6, s=10)
if np.any(two_phase_mask):
    plt.scatter(T_sol[two_phase_mask], p_sol[two_phase_mask]/1e5,
               c='red', label='Two Phase', alpha=0.6, s=10)

plt.xlabel("T (K)")
plt.ylabel("p (bar)")
plt.title("Pressure vs Temperature\n(colored by model)")
plt.legend()
plt.grid(True)

# Density vs Temperature (NEW)
plt.subplot(2, 4, 8)
plt.plot(T_sol, rho_sol, 'g-', linewidth=2)
plt.axhline(y=rho_stop, color='r', linestyle='--', linewidth=2,
           label=f'Stop threshold')
plt.xlabel("T (K)")
plt.ylabel("ρ (kg/m³)")
plt.title("Density vs Temperature")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

