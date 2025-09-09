import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os

from CoolProp.CoolProp import PropsSI

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
                    'p0': 15e5,      # Pa
                    'T0': 66,        # K
                    'Ts0': 298.15,   # K
                },
                'rho_stop': 78.0,    # kg/m³ - stopping density
                'max_time': 700.0,   # seconds - maximum simulation time
                'mass_flow_functions': {
                    'mdot_fuel': lambda t: 0.07,  # kg/s
                    'mdot_disch': lambda t: 0.0,
                    'mdot_vent': lambda t: 0.0,
                },
                'Qdot_disch': lambda t: 0.0,
                'description': 'Tank refueling scenario'
            },
            'DISCHARGE': {
                'initial_conditions': {
                    'p0': 50e5,      # Pa - placeholder
                    'T0': 30,        # K - placeholder
                    'Ts0': 298.15,   # K
                },
                'rho_stop': 5.0,     # kg/m³ - placeholder
                'max_time': 600.0,   # seconds - maximum simulation time (placeholder)
                'mass_flow_functions': {
                    'mdot_fuel': lambda t: 0.0,
                    'mdot_disch': lambda t: 0.05,  # kg/s - placeholder
                    'mdot_vent': lambda t: 0.0,
                },
                'Qdot_disch': lambda t: 1000.0,  # W - placeholder
                'description': 'Tank discharge scenario'
            },
            'DORMANCY': {
                'initial_conditions': {
                    'p0': 25e5,      # Pa - placeholder
                    'T0': 25,        # K - placeholder
                    'Ts0': 298.15,   # K
                },
                'rho_stop': 40.0,    # kg/m³ - placeholder
                'max_time': 3600.0,  # seconds - maximum simulation time (1 hour, placeholder)
                'mass_flow_functions': {
                    'mdot_fuel': lambda t: 0.0,
                    'mdot_disch': lambda t: 0.0,
                    'mdot_vent': lambda t: 0.0,  # Will be config-dependent
                },
                'Qdot_disch': lambda t: 0.0,
                'description': 'Tank dormancy scenario'
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

class ConfigurationManager:
    """
    Manages configuration switching based on pressure thresholds.
    Configurations: A (normal), B (minimum pressure), C (maximum pressure)
    Handles the three configuration-dependent algebraic equations.
    """

    def __init__(self, fluid, p_min=15e5, p_vent=450e5):
        self.fluid = fluid
        self.p_min = p_min    # Pa - minimum pressure threshold
        self.p_vent = p_vent  # Pa - venting pressure threshold
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
            return 'C'
        else:
            return 'A'

    def get_algebraic_equations(self, config, T, rho, is_two_phase=False):
        """
        Compute the three configuration-dependent algebraic equations.

        Args:
            config: Configuration name ('A', 'B', or 'C')
            T: Temperature [K]
            rho: Density [kg/m³]
            is_two_phase: Whether system is in two-phase region

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
            # TODO: Implement config_B_value calculation
            return {
                'pressure': self.p_min,
                'qdot_disch': 0.0,  # Placeholder - will be implemented later
                'mdot_vent': 0.0
            }

        elif config == 'C':
            # Maximum pressure mode: p = p_vent, qdot_dis = 0, mdot_vent = config_C_value
            # TODO: Implement config_C_value calculation
            return {
                'pressure': self.p_vent,
                'qdot_disch': 0.0,
                'mdot_vent': 0.0  # Placeholder - will be implemented later
            }

        else:
            raise ValueError(f"Unknown configuration: {config}")

class ModelSwitcher:
    """
    Handles single-phase vs two-phase model switching.
    This is separate from configuration switching.
    """

    def __init__(self, fluid):
        self.fluid = fluid
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

        return self.models[self.current_model]['ode_func'](t, y, *args)

    def solve(self, t, y, *args):
        """
        Solve the complete ODE system by selecting appropriate model and computing all derivatives.

        Args:
            t: Time
            y: State vector [m, T, Ts]
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
        else:
            selected_model = "single_phase"

        # Update the current model
        self.current_model = selected_model

        # Mass balance
        mdot_f = mdot_fuel_func(t)
        mdot_d = mdot_disch_func(t)
        mdot_v = mdot_vent_func(t)
        dm_dt = mdot_f - mdot_d - mdot_v

        # Temperature balance using selected model
        dT_dt = self.compute_dT_dt(t, y, *args)

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

# ----------------- Simulation Setup and User Parameters -----------------
fluid = "Hydrogen"      # Working fluid
V_t = 0.5          # Vessel volume [m^3]

# Heat transfer constants
A_in = 4.0         # m2
A_out = 4.1        # m2
k_amb = 0.025        # W/K
T_amb = 298.15     # K
m_liner = 100.0      # kg
m_wall = 150.0      # kg

# Debug printing parameters
PRINT_EVERY_N_STEPS = 50  # Print state every N integration steps (set to 0 to disable)
step_counter = 0  # Global counter for tracking steps

# Geometric parameters for alpha_s calculation
D_inner = 0.4      # Inner diameter [m] - tank liner
D_outer = 0.45     # Outer diameter [m] - includes insulation/wall
annular_fluid = "Hydrogen"  # Fluid in the gap (could be vacuum, air, etc.)

# Configuration pressure thresholds
p_min = 15e5        # Pa - minimum pressure threshold for configuration B
p_vent = 450e5      # Pa - venting pressure threshold for configuration C

# Initialize framework managers
scenario_manager = ScenarioManager()
config_manager = ConfigurationManager(fluid, p_min, p_vent)
model_switcher = ModelSwitcher(fluid)

# Set scenario - CHANGE THIS to switch scenarios
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

# Get scenario-specific mass flow functions
mdot_fuel_func = scenario_config['mass_flow_functions']['mdot_fuel']
mdot_disch_func = scenario_config['mass_flow_functions']['mdot_disch']
mdot_vent_func = scenario_config['mass_flow_functions']['mdot_vent']
Qdot_disch_func = scenario_config['Qdot_disch']

# Calculate initial density from given pressure and temperature
rho0 = PropsSI("Dmass", "P", p0, "T", T0, fluid)
m0 = rho0 * V_t

# Time span - from scenario configuration
t_span = (0.0, max_time)

# Stopping condition parameters
# Define stopping density in kg/m³ (convert from g/L)
rho_stop = 78.0  # kg/m³ (0.020 g/L)

# Mass flow rates (customize as needed)
def mdot_fuel_func(t):
    return 0.07  # kg/s constant for now
def mdot_disch_func(t):
    return 0.0
def mdot_vent_func(t):
    return 0.0
Qdot_disch_func = lambda t: 0.0

# Initialize NIST materials
liner_material = NISTMetal.aluminum_6061T6_nist()  # Aluminum 6061-T6 for liner
wall_material = NISTComposite.g10_nist(winding_angle=0.0)  # G10 for wall

# Specific heat functions using NIST data
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

    return h_i

# ----------------- Test Function -----------------
def test_alpha_s_function():
    """
    Test the get_alpha_s function with sample conditions
    """
    print("\n--- Testing get_alpha_s function ---")
    print("Function test starting...")

    # Test conditions
    T_inner = 60.0  # K (cold hydrogen)
    T_outer = 300.0  # K (ambient temperature)

    try:
        alpha_s_test = get_alpha_s(T_inner, T_outer, D_inner, D_outer, annular_fluid)
        print(f"Sample calculation:")
        print(f"  Inner temperature: {T_inner:.1f} K")
        print(f"  Outer temperature: {T_outer:.1f} K")
        print(f"  Inner diameter: {D_inner:.3f} m")
        print(f"  Outer diameter: {D_outer:.3f} m")
        print(f"  Fluid: {annular_fluid}")
        print(f"  Calculated alpha_s: {alpha_s_test:.2f} W/(m2·K)")

        # Calculate Rayleigh number for reference
        L = (D_outer - D_inner) / 2.0
        T_mean = 0.5 * (T_inner + T_outer)
        try:
            mu = PropsSI('V', 'T', T_mean, 'P', 101325, annular_fluid)
            rho = PropsSI('D', 'T', T_mean, 'P', 101325, annular_fluid)
            cp = PropsSI('C', 'T', T_mean, 'P', 101325, annular_fluid)
            k = PropsSI('L', 'T', T_mean, 'P', 101325, annular_fluid)
            nu = mu / rho
            alpha = k / (rho * cp)
            beta = 1.0 / T_mean
            Ra = 9.81 * beta * abs(T_inner - T_outer) * L**3 / (nu * alpha)
            print(f"  Rayleigh number: {Ra:.2e}")
        except:
            print("  Could not calculate Rayleigh number")

    except Exception as e:
        print(f"Error testing alpha_s function: {e}")

    print("Function test completed.")

# ----------------- Model Implementations -----------------
def single_phase_dT_dt(t, y, model_switcher, config_manager, current_config):
    """
    Single-phase energy balance.
    Uses cv for temperature derivative calculation.
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    T = max(T, 1.0)

    rho = m / V_t

    # Get configuration-dependent algebraic equations
    config_eqs = config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=False)
    p = config_eqs['pressure']

    h = PropsSI("Hmass", "T", T, "Dmass", rho, fluid)
    c_v = PropsSI("Cvmass", "T", T, "Dmass", rho, fluid)
    nu = 1.0 / rho

    # Get mass flow rates (may be config-dependent)
    mdot_f = mdot_fuel_func(t)
    mdot_d = mdot_disch_func(t)
    mdot_v = config_eqs['mdot_vent']  # Configuration-dependent

    # Use pump outlet enthalpy for refueling
    h_fuel = compute_pump_outlet_hydrogen(p, T)
    h_dich = h
    h_vent = h
    Qdot_disch = config_eqs['qdot_disch']  # Configuration-dependent

    # Calculate alpha_s based on current temperature conditions
    alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)

    numerator_T = (mdot_f * (h_fuel - h)
                   - mdot_d * (h_dich - h)
                   - mdot_v * (h_vent - h)
                   + p * nu * (mdot_f - mdot_d - mdot_v)
                   + alpha_s * A_in * (Ts - T)
                   + Qdot_disch)

    return numerator_T / (m * c_v)

def two_phase_dT_dt(t, y, model_switcher, config_manager, current_config):
    """
    Two-phase energy balance implementation.
    Uses c_v2P (two-phase specific heat capacity).
    """
    m, T, Ts = y
    m = max(m, 1e-12)
    T = max(T, 1.0)

    rho = m / V_t

    # Get configuration-dependent algebraic equations for two-phase
    config_eqs = config_manager.get_algebraic_equations(current_config, T, rho, is_two_phase=True)
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
    dp_sat_dT = p / (t + 1e-6)

    nu = 1.0 / rho

    # Get mass flow rates (may be config-dependent)
    mdot_f = mdot_fuel_func(t)
    mdot_d = mdot_disch_func(t)
    mdot_v = config_eqs['mdot_vent']  # Configuration-dependent

    # Use pump outlet enthalpy for refueling
    h_fuel = compute_pump_outlet_hydrogen(p, T)
    h_dich = h
    h_vent = h
    Qdot_disch = config_eqs['qdot_disch']  # Configuration-dependent

    # Calculate alpha_s based on current temperature conditions
    alpha_s = get_alpha_s(T, Ts, D_inner, D_outer, annular_fluid, p)

    # Two-phase energy balance numerator
    numerator_T = (mdot_f * (h_fuel - h)
                   - mdot_d * (h_dich - h)
                   - mdot_v * (h_vent - h)
                   + (T / rho) * dp_sat_dT * (mdot_f - mdot_d - mdot_v)
                   + alpha_s * A_in * (Ts - T)
                   + Qdot_disch)

    return numerator_T / (m * c_v2P)

# Initialize model switcher
model_switcher = ModelSwitcher(fluid)

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
def single_phase_wrapper(t, y, model_switcher):
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

    return single_phase_dT_dt(t, y, model_switcher, config_manager, current_config)

def two_phase_wrapper(t, y, model_switcher):
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

    return two_phase_dT_dt(t, y, model_switcher, config_manager, current_config)

# Register the models with wrappers
model_switcher.register_model("single_phase", is_single_phase_condition, single_phase_wrapper)
model_switcher.register_model("two_phase", is_two_phase_condition, two_phase_wrapper)

def compute_pump_outlet_hydrogen(tank_pressure: float, tank_temperature: float):
    """
    Calculate the hydrogen properties and pressure at pump outlet.

    This function computes the enthalpy of hydrogen after being compressed
    by a cryogenic pump during refueling, accounting for pump efficiency
    and resulting temperature rise.

    Parameters
    ----------
    tank_pressure : float
        Target tank pressure [Pa]
    tank_temperature : float
        Current tank temperature [K] (used for reference)

    Returns
    -------
    h2 : float
        Enthalpy at pump outlet [J/kg]
    """
    fluid_local = "Hydrogen"  # Local fluid name to avoid conflicts
    P1 = 3e5       # Pa (3 bar) - dewar pressure
    P2 = tank_pressure  # Target pressure (Pa)
    eta_p = 0.78   # Pump isentropic efficiency (78%)

    # 1. Inlet state: saturated liquid at P1
    h1 = PropsSI("H", "P", P1, "Q", 0, fluid_local)  # Enthalpy (J/kg)
    s1 = PropsSI("S", "P", P1, "Q", 0, fluid_local)  # Entropy (J/kg/K)

    # 2. Ideal isentropic outlet at P2
    h2s = PropsSI("H", "P", P2, "S", s1, fluid_local)

    # 3. Actual outlet enthalpy with efficiency
    h2 = h1 + (h2s - h1)/eta_p

    # 4. Outlet temperature from (h2,P2) - for reference
    T2 = PropsSI("T", "P", P2, "H", h2, fluid_local)

    # Return the enthalpy at the pump outlet
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

    return model_switcher.solve(t, y, model_switcher)

def density_event(t, y):
    """
    Event function to detect when density reaches stopping threshold.

    This function returns zero when the density equals rho_stop.
    The solver will detect this zero crossing and stop the integration.

    Parameters
    ----------
    t : float
        Time [s]
    y : array
        State vector [m, T, Ts]

    Returns
    -------
    float
        Difference between current density and stopping density.
        Zero crossing indicates stopping condition is met.
    """
    m, T, Ts = y
    m = max(m, 1e-12)  # Avoid division by zero

    current_density = m / V_t  # Current density [kg/m³]

    # Return the difference: negative when rho < rho_stop, positive when rho > rho_stop
    # Zero crossing occurs exactly when current_density = rho_stop
    return current_density - rho_stop

# Configure the event to stop integration when density threshold is reached
density_event.terminal = True    # Stop integration when event occurs
density_event.direction = 1      # Detect only increasing density (positive crossing)

# Initial state
y0 = [m0, T0, Ts0]

print(f"Starting simulation...")
print(f"GIVEN initial conditions:")
print(f"  Temperature: {T0:.2f} K")
print(f"  Pressure: {p0/1e5:.2f} bar")
print(f"  Solid Temperature: {Ts0:.2f} K")
print(f"CALCULATED from CoolProp:")
print(f"  Density: {rho0:.2f} kg/m³)")
print(f"  Mass: {m0:.2f} kg")
print(f"Density stopping condition: {rho_stop:.1f} kg/m³")
print(f"Will stop when density reaches {rho_stop:.1f} kg/m³")

# Check initial model selection
initial_p = PropsSI('P', 'T', T0, 'Dmass', rho0, fluid)
if is_near_saturation(T0, initial_p, fluid):
    initial_model = "two_phase"
else:
    initial_model = "single_phase"
print(f"Starting with {initial_model} model")

try:
    p_sat_initial = PropsSI("P", "T", T0, "Q", 0, fluid)
    print(f"Saturation pressure at {T0:.2f} K: {p_sat_initial/1e5:.2f} bar")
except:
    print(f"Could not calculate saturation pressure at {T0:.2f} K (possibly above critical point)")


print("\nSolving ODEs...")
print(f"Time span: {t_span[0]:.1f} to {t_span[1]:.1f} seconds")
print("Integration method: Radau with enhanced stability tolerances")
print(f"Density stopping condition: {rho_stop:.1f} kg/m³")
if PRINT_EVERY_N_STEPS > 0:
    print(f"Progress printing: Every {PRINT_EVERY_N_STEPS} integration steps")
else:
    print("Progress printing: Disabled")
print("Starting integration...")

# Reset step counter before integration
step_counter = 0

# Use scenario-specified time span directly
t_span_limited = t_span

sol = solve_ivp(odes, t_span_limited, y0, method="Radau", atol=1e-10, rtol=1e-8,
                dense_output=True, events=density_event, max_step=0.05)

print(f"\nIntegration completed!")
print(f"Solution completed. Success: {sol.success}")
if not sol.success:
    print(f"Solution message: {sol.message}")
else:
    print(f"Number of function evaluations: {sol.nfev}")
    print(f"Final time reached: {sol.t[-1]:.3f} seconds")
    print(f"Integration took {len(sol.t)} internal timesteps")

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
print("\nPostprocessing results...")

# Determine the actual time span for evaluation based on whether simulation stopped early
if hasattr(sol, 't_events') and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
    # Simulation stopped due to density event
    actual_t_final = sol.t_events[0][0]
    print(f"Using actual simulation time span: 0.0 to {actual_t_final:.2f} seconds")
else:
    # Simulation completed full time span
    actual_t_final = t_span[1]
    print(f"Using full time span: {t_span[0]:.1f} to {t_span[1]:.1f} seconds")

t_eval = np.linspace(t_span[0], actual_t_final, 400)
print(f"Evaluating solution at {len(t_eval)} time points...")
y_eval = sol.sol(t_eval)
m_sol, T_sol, Ts_sol = y_eval
rho_sol = m_sol / V_t

print("Computing pressure and model selection for each time point...")
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

# ----------------- Model Switching Test Function -----------------
def test_model_switching(T_test=25.0, p_range=(1e5, 50e5)):
    """
    Test function to demonstrate model switching at different pressures.

    Args:
        T_test: Temperature to test at [K]
        p_range: Pressure range to test (min, max) [Pa]
    """
    print(f"\n--- Model Switching Test at T = {T_test} K ---")

    pressures = np.linspace(p_range[0], p_range[1], 20)

    try:
        p_sat = PropsSI("P", "T", T_test + 273.15, "Q", 0, fluid)  # Convert to K
        print(f"Saturation pressure at {T_test + 273.15:.2f} K: {p_sat/1e5:.2f} bar")

        for p in pressures:
            try:
                rho_test = PropsSI("Dmass", "P", p, "T", T_test + 273.15, fluid)
                near_sat = is_near_saturation(T_test + 273.15, p, fluid)
                selected_model = "two_phase" if near_sat else "single_phase"

                print(f"P = {p/1e5:6.2f} bar: {selected_model:12s} model (near sat: {near_sat})")

            except Exception as e:
                print(f"P = {p/1e5:6.2f} bar: Error calculating properties - {str(e)[:50]}")

    except Exception as e:
        print(f"Could not calculate saturation pressure: {e}")

# Uncomment the line below to run the test
# test_model_switching()

# ----------------- Future Model Extensions -----------------
"""
To add new models to the switcher, follow this pattern:

1. Define the condition function:
   def is_new_condition(T, p, rho):
       # Your condition logic here
       return True/False

2. Define the ODE function:
   def new_model_dT_dt(t, y, model_switcher):
       # Your ODE implementation here
       return dT_dt_value

3. Register the model:
   model_switcher.register_model("new_model_name", is_new_condition, new_model_dT_dt)

Example additional conditions you might want to add:
- High pressure supercritical region
- Low temperature specific models
- Pressure-dependent heat transfer models
- Different fluid phase models
- Safety valve activation models

The model switcher will automatically select the first model whose condition
returns True, so order of registration matters if multiple conditions could be true.
"""

