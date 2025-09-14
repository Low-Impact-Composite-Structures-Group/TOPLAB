import numpy as np
from scipy.integrate import solve_ivp
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
            'DISCHARGE': {
                'initial_conditions': {
                    'p0': 400e5,      # Initial pressure [Pa]
                    'T0': 53.25,      # Initial temperature [K]
                    'Ts0': 'thermal_equilibrium',    # Initial solid temperature [K]
                },
                'rho_stop': 5.8,     # Stopping density [kg/m³]
                'max_time': 40000.0, # Maximum simulation time [s]
                'solver_settings': {
                    'method': 'RK45',
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
            'REFUEL': {
                'initial_conditions': {
                    'p0': 15.3e5,      # Initial pressure [Pa]
                    'T0': 65.5,        # Initial temperature [K]
                    'Ts0': "thermal_equilibrium",   # Initial solid temperature [K]
                },
                'rho_stop': 78.0,    # Stopping density [kg/m³]
                'max_time': 700.0,   # Maximum simulation time [s]
                'solver_settings': {
                    'method': 'RK45',
                    'atol': 1e-10,
                    'rtol': 1e-8,
                    'max_step': 0.05,
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
                'description': 'p = p(T,rho), qdot_disch= 0, mdot_vent = 0'
            },
            'B': {
                'name': 'Minimum Pressure Mode',
                'description': 'p = p_min, qdot_disch= config_B_value, mdot_vent = 0'
            },
            'C': {
                'name': 'Maximum Pressure Mode',
                'description': 'p = p_vent, qdot_disch= 0, mdot_vent = config_C_value'
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
        if p <= self.p_min and global_scenario_manager.current_scenario != 'REFUEL':
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
            # Normal operation: p = p(T,rho), qdot_disch= 0, mdot_vent = 0
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
            # Minimum pressure mode: p = p_min, qdot_disch= config_B_value, mdot_vent = 0
            # Calculate Configuration B discharge heat
            qdot_disch = self._calculate_config_B_qdot_disch(T, rho, is_two_phase, t, mdot_disch_func, Ts)

            return {
                'pressure': self.p_min,
                'qdot_disch': qdot_disch,
                'mdot_vent': 0.0
            }

        elif config == 'C':
            # Maximum pressure mode: p = p_vent, qdot_disch= 0, mdot_vent = config_C_value
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
        alpha_s = get_alpha_s(T, Ts, diameter, convective_medium, p)
        Qdot_s = alpha_s * A_in * (Ts - T)  # Environmental heat leak [W]

        # Term 1: T/ρ · (∂p/∂T)_ρ using real gas relationship
        dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, fluid)
        term1 = (T / rho) * dp_dT_rho

        # Term 2: ρ·cv·(∂T/∂ρ)_p using real CoolProp derivatives for accuracy
        dT_drho_p = cp.PropsSI('d(T)/d(D)|P', 'P', p, 'T', T, self.fluid)

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
        alpha_s = get_alpha_s(T, Ts, diameter, convective_medium, p)

        # compute heat transfered from solid to fluid
        Qdot_s = alpha_s * A_in * (Ts - T)  # Environmental heat leak [W]

        # Term 1: T/ρ · (∂p/∂T)_ρ using real gas relationship
        dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, fluid)
        term1 = (T / rho) * dp_dT_rho

        # Term 2: ρ·cv·(∂T/∂ρ)_p using real CoolProp derivatives for accuracy
        dT_drho_p = cp.PropsSI('d(T)/d(D)|P', 'P', p, 'T', T, self.fluid)

        term2 = rho * cv * dT_drho_p

        # Calculate denominator
        denominator = term1 - term2 + h_vent - h

        # compute mdot_vent
        mdot_vent = Qdot_s / denominator

        # kill simulation if mdot_vent is negative or NaN
        if mdot_vent < 0 or np.isnan(mdot_vent):
            raise ValueError("Negative or NaN venting mass flow calculated, stopping simulation.")

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

        # try:
        # Algebraic equation (7): h = h(T, ρ)
        # h = PropsSI("Hmass", "T", T, "Dmass", rho, self.fluid)
        p = PropsSI("P", "T", T, "Dmass", rho, self.fluid)

        # Pre-calculate thermodynamic derivatives at current state
        is_two_phase = is_near_saturation(T, p, self.fluid)

        # Select configuration for algebraic equations (8,9,10)
        current_config = self.config_manager.select_configuration(p, is_two_phase)
        config_eqs = self.config_manager.get_algebraic_equations(
            current_config, T, rho, is_two_phase=is_two_phase,
            t=t, mdot_disch_func=mdot_disch_func, Ts=Ts, scenario_manager=scenario_manager
        )

        p_config = config_eqs['pressure']      # May override p for configs B,C
        Qdot_disch = config_eqs['qdot_disch']  # Config-dependent discharge heat
        mdot_vent = config_eqs['mdot_vent']    # Config-dependent venting

        # Mass flow rates
        mdot_f = mdot_fuel_func(t)
        mdot_d = mdot_disch_func(t)

        # Collect heat flow data for plotting (global storage)
        global heat_flow_data
        heat_flow_data['t'].append(t)
        heat_flow_data['qdot_disch'].append(Qdot_disch)
        heat_flow_data['qdot_ohex'].append(0.0)  # Will be calculated in post-processing
        heat_flow_data['mdot_disch'].append(mdot_d)
        heat_flow_data['T'].append(T)
        heat_flow_data['rho'].append(rho)

        # Algebraic equations (4,5): Heat transfer (coupled with temperatures)
        alpha_s = get_alpha_s(T, Ts, diameter, convective_medium, p_config)
        self.alpha_s_last = alpha_s  # Store for debugging
        Qdot_s = alpha_s * A_in * (Ts - T)     # Equation (4)
        Qdot_amb = k_amb * A_out * (T_amb - Ts) # Equation (5)

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

# oHEX (Outer Heat Exchanger) target conditions for heat requirement calculation
OHEX_TARGET_TEMPERATURE = 200.0  # Target temperature [K]
OHEX_TARGET_PRESSURE = 20e5      # Target pressure [Pa] (20 bar)

# Heat flow data collection for plotting
heat_flow_data = {
    't': [],           # Time points [s]
    'qdot_disch': [],  # iHEX heat flow requirement [W]
    'qdot_ohex': [],   # oHEX heat flow requirement [W] (calculated in post-processing)
    'mdot_disch': [],  # Discharge mass flow rate [kg/s] (needed for oHEX calculation)
    'T': [],           # Fluid temperature [K] (needed for discharge enthalpy)
    'rho': []          # Fluid density [kg/m³] (needed for discharge enthalpy)
}

def reset_heat_flow_data():
    """Reset heat flow data collection before new simulation."""
    global heat_flow_data
    heat_flow_data['t'].clear()
    heat_flow_data['qdot_disch'].clear()
    heat_flow_data['qdot_ohex'].clear()
    heat_flow_data['mdot_disch'].clear()
    heat_flow_data['T'].clear()
    heat_flow_data['rho'].clear()

def get_heat_flow_data():
    """Get current heat flow data for plotting."""
    global heat_flow_data
    return {
        't': heat_flow_data['t'].copy(),
        'qdot_disch': heat_flow_data['qdot_disch'].copy(),
        'qdot_ohex': heat_flow_data['qdot_ohex'].copy(),
        'mdot_disch': heat_flow_data['mdot_disch'].copy(),
        'T': heat_flow_data['T'].copy(),
        'rho': heat_flow_data['rho'].copy()
    }

def calculate_ohex_heat_requirements(heat_flow_data,
                                   target_temperature=OHEX_TARGET_TEMPERATURE,
                                   target_pressure=OHEX_TARGET_PRESSURE):
    """
    Calculate oHEX (Outer Heat Exchanger) heat requirements by post-processing simulation data.

    The oHEX heat requirement is calculated as:
    Q_oHEX = mdot_disch * (h_disch - h_target)

    Where:
    - mdot_disch: discharge mass flow rate [kg/s]
    - h_disch: enthalpy of discharge stream at current T, rho [J/kg]
    - h_target: enthalpy at target temperature and pressure [J/kg]

    Parameters
    ----------
    heat_flow_data : dict
        Dictionary containing simulation data with keys: 't', 'mdot_disch', 'T', 'rho'
    target_temperature : float, optional
        Target temperature for oHEX outlet [K] (default: OHEX_TARGET_TEMPERATURE)
    target_pressure : float, optional
        Target pressure for oHEX outlet [Pa] (default: OHEX_TARGET_PRESSURE)

    Returns
    -------
    dict
        Updated heat_flow_data dictionary with calculated 'qdot_ohex' values
    """
    if not heat_flow_data or len(heat_flow_data.get('T', [])) == 0:
        return heat_flow_data

    # Calculate target enthalpy (constant for all time points)
    h_target = PropsSI("Hmass", "T", target_temperature, "P", target_pressure, fluid)

    # Calculate oHEX heat requirement for each time point
    qdot_ohex_calculated = []

    for i, (T, rho, mdot_d) in enumerate(zip(heat_flow_data['T'],
                                           heat_flow_data['rho'],
                                           heat_flow_data['mdot_disch'])):
        try:
            # Calculate discharge enthalpy at current conditions
            h_disch = PropsSI("Hmass", "T", T, "Dmass", rho, fluid)

            # Calculate oHEX heat requirement: Q = mdot * (h_disch - h_target)
            qdot_ohex = mdot_d * (h_target - h_disch)
            qdot_ohex_calculated.append(qdot_ohex)

        except Exception as e:
            # Handle CoolProp errors gracefully
            print(f"Warning: Could not calculate oHEX heat requirement at point {i}: {e}")
            qdot_ohex_calculated.append(0.0)

    # Update the heat flow data with calculated oHEX values
    updated_data = heat_flow_data.copy()
    updated_data['qdot_ohex'] = qdot_ohex_calculated

    return updated_data

# Geometric parameters for alpha_s calculation
diameter = 1.0      # Inner diameter [m]  # Outer diameter [m]
convective_medium = "Hydrogen"  # Fluid in the gap

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
    """Get specific heat of aluminum 5083 liner using NIST data"""
    return liner_material.determine_specific_heat(Ts)

def c_wall_func(Ts):
    """Get specific heat of G10 wall using NIST data"""
    return wall_material.determine_specific_heat(Ts)

# ----------------- Heat Transfer Coefficient Function -----------------
def get_alpha_s(T, Ts, D, fluid="Air", p=101325):
    """
    Compute equivalent convective heat transfer coefficient (h_i)
    for a horizontal cylinder annulus using the Churchill & Chu correlation.

    Parameters
    ----------
    T : float
        Inner cylinder surface temperature [K]
    D : float
        Inner cylinder diameter [m]
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
    Churchill & Chu (1975) Correlating equations for laminar and turbulent free convection from a horizontal cylinder
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

    Nu_D = (0.60 + (0.387 * Ra_D**(1/6)) /
           ( (1 + (0.559/Pr)**(9/16))**(8/27) ))**2


    # den_1 = ((Ra_D**0.25)**(1/5)+(0.12*Ra_D**(1/3))**15)**(1/15)
    # den_2 = np.log(1 - 2/den_1)

    # Nu_Do = -2/den_2


     # Heat transfer coefficient
    h = Nu_D * k / D
    return h


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

    h_disch = h
    h_vent = h

    # Energy balance terms
    h_term = mdot_f * (h_fuel - h) - mdot_d * (h_disch - h) - mdot_v * (h_vent - h)

    # compute work term
    net_mass_flow = mdot_f - mdot_d - mdot_v
    dp_dT_rho = PropsSI('d(P)/d(T)|D', 'T', T, 'Dmass', rho, fluid)

    # Calculate (T/ρ) * (∂p/∂T)_ρ
    work_term = (T / rho) * dp_dT_rho * net_mass_flow

    numerator_T = h_term + work_term + Qdot_s + Qdot_disch
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

    h_disch = h
    h_vent = h

    # Energy balance terms
    h_term = mdot_f * (h_fuel - h) - mdot_d * (h_disch - h) - mdot_v * (h_vent - h)

    # PV work term - EXPERIMENTAL: Disable during refueling to avoid double-counting compression work
    net_mass_flow = mdot_f - mdot_d - mdot_v
    work_term = (T / rho) * dp_sat_dT * net_mass_flow

    # (Qdot_s and Qdot_disch are passed as parameters)
    numerator_T = h_term + work_term + Qdot_s + Qdot_disch

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
    m, T, _ = y
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
    _, T, _ = y

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

    # Reset heat flow data collection before simulation
    reset_heat_flow_data()

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
        'metadata': metadata,
        'heat_flow_data': get_heat_flow_data()  # Include heat flow data for plotting
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

    # Get heat flow data and calculate oHEX requirements
    raw_heat_flow_data = result.get('heat_flow_data', {'t': [], 'qdot_disch': [], 'qdot_ohex': []})
    processed_heat_flow_data = calculate_ohex_heat_requirements(raw_heat_flow_data)

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
        't_offset': t_offset,
        'heat_flow_data': processed_heat_flow_data
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
    Plot results from multiple scenarios with different colors using SeabornPlotter.

    Parameters
    ----------
    results : list
        List of result dictionaries from run_hydrogen_tank_simulation()
    postprocessed_data : list, optional
        List of postprocessed data dictionaries. If None, will postprocess automatically.
    """
    if postprocessed_data is None:
        postprocessed_data = [postprocess_simulation_result(result) for result in results]

    # Create SeabornPlotter instance
    plotter = SeabornPlotter(font="Cambria", palette="delft")

    # Create the plot using the plotter method
    fig = plotter.plot_chained_scenarios(results, postprocessed_data)

    # Print detailed statistics using the plotter method
    plotter.print_detailed_simulation_statistics(results, postprocessed_data)

    return fig

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

    # Run chained scenarios
    print("\n" + "="*80)
    print("Running full operational cycle (3 scenarios: DISCHARGE → REFUEL → DORMANCY)")
    chained_results = run_chained_scenarios(['DISCHARGE', 'REFUEL', 'DORMANCY'], verbose=True)

    # Postprocess all results
    chained_data = [postprocess_simulation_result(result) for result in chained_results]

    # Plot all scenarios together
    plot_chained_scenarios(chained_results, chained_data)

    # Create the combined density-temperature plot using SeabornPlotter
    plot_combined_density_temperature(chained_results, chained_data)

    # Demonstrate the new heat exchanger requirements plotting
    plotter = SeabornPlotter()

    print(f"\nHeat Exchanger Analysis:")
    print(f"oHEX Target Conditions: {OHEX_TARGET_TEMPERATURE}K, {OHEX_TARGET_PRESSURE/1e5:.1f} bar")

    # Plot heat exchanger requirements only for discharge scenario
    for i, (result, data) in enumerate(zip(chained_results, chained_data)):
        scenario_name = result['scenario']

        # Only create heat exchanger plot for discharge scenario
        if scenario_name.upper() == 'DISCHARGE':
            heat_flow_data = data.get('heat_flow_data', {})

            if heat_flow_data and 'qdot_disch' in heat_flow_data:
                fig_hex = plotter.plot_heat_exchanger_requirements(
                    heat_flow_data,
                    scenario_name=scenario_name,
                    plot_total=True  # Enable total heat flow curve
                )
                print(f"Created heat exchanger plot for {scenario_name} scenario")

                # Check if oHEX data was calculated
                if any(q != 0.0 for q in heat_flow_data.get('qdot_ohex', [])):
                    print(f"  - iHEX, oHEX, and total heat requirements plotted")
                else:
                    print(f"  - Only iHEX requirements plotted (oHEX data all zeros, no total curve)")
            else:
                print(f"No heat flow data available for {scenario_name} scenario")

    plt.show()

