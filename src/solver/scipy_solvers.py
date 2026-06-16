"""
SciPy-based ODE solvers for multi-tank systems.

This module provides SciPy wrapper classes for various ODE integration methods
optimized for hydrogen tank system dynamics. All solvers support both step-by-step
and full integration modes.

Available Solvers:
- RK45Solver: Explicit Runge-Kutta (good for non-stiff problems)
- RadauSolver: Implicit Runge-Kutta (excellent for stiff problems)
- DOP853Solver: High-order explicit method (high precision)
- BDFSolver: Backward Differentiation Formula (stiff systems)
- LSODASolver: Adaptive Adams/BDF with automatic stiffness detection

Example Usage:
    solver = RK45Solver(timestep=1.0, rtol=1e-6, atol=1e-9)
    solver.set_ode_function(my_ode_function)
    result = solver.integrate_full(t_span=(0, 1000), y0=initial_state)
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SciPySolver:
    """
    Base class for SciPy-based ODE solvers.

    This class provides common configuration parameters and functionality
    for scipy.integrate.solve_ivp-based solvers. Child classes implement
    specific integration methods (e.g., RK45, Radau, DOP853).

    Unlike MultistepMethod classes that work with derivatives and current values,
    SciPy solvers require an ODE function that takes (t, y) and returns dy/dt.

    Common Configuration Parameters:
    - timestep: Default timestep for integration
    - rtol: Relative tolerance
    - atol: Absolute tolerance
    - max_step: Maximum step size
    - min_step: Minimum step size
    - first_step: First step size
    - dense_output: Whether to compute dense output
    """
    timestep: float
    rtol: float = 1e-6    # relative tolerance
    atol: float = 1e-9    # absolute tolerance
    max_step: float = None  # maximum step size
    min_step: float = None  # minimum step size
    first_step: float = None  # first step size
    dense_output: bool = False  # whether to compute dense output

    # Storage for ODE function and current state
    _ode_function: Callable = None
    _current_time: float = 0.0
    _current_state: np.ndarray = None

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the scipy integration method name"""
        pass

    def set_ode_function(self, ode_func: Callable):
        """Set the ODE function for integration.

        Args:
            ode_func: Function that takes (t, y) and returns dy/dt
        """
        self._ode_function = ode_func

    def set_current_state(self, time: float, state: np.ndarray):
        """Set the current time and state for integration.

        Args:
            time: Current time value
            state: Current state vector (e.g., [m, T, Ts])
        """
        self._current_time = time
        self._current_state = np.array(state)

    def _get_solver_kwargs(self, **override_kwargs) -> dict:
        """
        Get solver keyword arguments, allowing overrides.

        Args:
            **override_kwargs: Keyword arguments to override defaults

        Returns:
            dict: Solver configuration dictionary
        """
        solver_kwargs = {
            'method': self.method_name,
            'rtol': override_kwargs.get('rtol', self.rtol),
            'atol': override_kwargs.get('atol', self.atol),
            'dense_output': override_kwargs.get('dense_output', self.dense_output)
        }

        # Add optional parameters if specified
        if 'max_step' in override_kwargs or self.max_step is not None:
            solver_kwargs['max_step'] = override_kwargs.get('max_step', self.max_step)
        if 'min_step' in override_kwargs or self.min_step is not None:
            solver_kwargs['min_step'] = override_kwargs.get('min_step', self.min_step)
        if 'first_step' in override_kwargs or self.first_step is not None:
            solver_kwargs['first_step'] = override_kwargs.get('first_step', self.first_step)

        # Add any other kwargs that aren't already handled
        for key, value in override_kwargs.items():
            if key not in solver_kwargs:
                solver_kwargs[key] = value

        return solver_kwargs

    def integrate_step(self, **kwargs) -> tuple[np.ndarray, bool]:
        """
        Integrate one timestep using scipy solve_ivp.

        Args:
            **kwargs: Override parameters for this integration step

        Returns:
            tuple: (new_state, success) where new_state is the integrated state
                   and success indicates if integration was successful
        """
        if self._ode_function is None:
            raise ValueError("ODE function not set. Call set_ode_function() first.")

        if self._current_state is None:
            raise ValueError("Current state not set. Call set_current_state() first.")

        # Define time span for single step
        timestep = kwargs.get('timestep', self.timestep)
        t_span = (self._current_time, self._current_time + timestep)

        # Get solver configuration
        solver_kwargs = self._get_solver_kwargs(**kwargs)

        try:
            # Integrate one step
            sol = solve_ivp(self._ode_function, t_span, self._current_state, **solver_kwargs)

            if sol.success and len(sol.y) > 0:
                # Get final state
                new_state = sol.y[:, -1]

                # Update internal state for next step
                self._current_time += timestep
                self._current_state = new_state

                return new_state, True
            else:
                print(f"SciPy integration failed ({self.method_name}): {sol.message}")
                return self._current_state, False

        except Exception as e:
            print(f"SciPy integration error ({self.method_name}): {e}")
            return self._current_state, False

    def integrate_full(self, t_span: tuple, y0: np.ndarray, t_eval: np.ndarray = None, **kwargs) -> object:
        """
        Integrate over full time span using scipy solve_ivp.

        This method performs complete integration over the specified time span,
        which is more efficient than step-by-step integration for many use cases.

        Args:
            t_span: (t_start, t_end) time span for integration
            y0: Initial state vector
            t_eval: Specific times at which to store the computed solution
            **kwargs: Additional arguments for solve_ivp

        Returns:
            scipy solve_ivp solution object
        """
        if self._ode_function is None:
            raise ValueError("ODE function not set. Call set_ode_function() first.")

        # Get solver configuration
        solver_kwargs = self._get_solver_kwargs(**kwargs)

        # Add t_eval if provided
        if t_eval is not None:
            solver_kwargs['t_eval'] = t_eval

        return solve_ivp(self._ode_function, t_span, y0, **solver_kwargs)


@dataclass
class RK45Solver(SciPySolver):
    """
    Explicit Runge-Kutta method of order 5(4).

    This is the default scipy method and works well for non-stiff problems.
    Good balance of accuracy and computational efficiency.

    Characteristics:
    - Explicit method (good for non-stiff problems)
    - Adaptive step size
    - 5th order accuracy with 4th order error control
    - Efficient for smooth problems
    """

    @property
    def method_name(self) -> str:
        return 'RK45'


@dataclass
class RadauSolver(SciPySolver):
    """
    Implicit Runge-Kutta method of the Radau IIA family of order 5.

    Excellent for stiff problems and DAEs. Uses implicit integration
    which is more stable for stiff systems.

    Characteristics:
    - Implicit method (excellent for stiff problems)
    - L-stable (very good stability properties)
    - 5th order accuracy
    - More computationally expensive per step but can take larger steps
    """

    @property
    def method_name(self) -> str:
        return 'Radau'


@dataclass
class DOP853Solver(SciPySolver):
    """
    Explicit Runge-Kutta method of order 8.

    High-order method for problems requiring high accuracy.
    More expensive per step but can achieve very high precision.

    Characteristics:
    - Explicit method (for non-stiff problems)
    - 8th order accuracy
    - Dense output available
    - Best for high-precision requirements
    """

    @property
    def method_name(self) -> str:
        return 'DOP853'


@dataclass
class BDFSolver(SciPySolver):
    """
    Implicit multi-step variable-order (1 to 5) method based on
    Backward Differentiation Formulas.

    Specifically designed for stiff systems. Good for problems
    with widely separated time scales.

    Characteristics:
    - Implicit method (excellent for stiff problems)
    - Variable order (1-5)
    - Good for stiff ODEs and some DAEs
    - Efficient for stiff systems
    """

    @property
    def method_name(self) -> str:
        return 'BDF'


@dataclass
class LSODASolver(SciPySolver):
    """
    Adams/BDF method with automatic stiffness detection and switching.

    Automatically switches between non-stiff (Adams) and stiff (BDF) methods
    based on problem characteristics.

    Characteristics:
    - Adaptive method selection (Adams for non-stiff, BDF for stiff)
    - Automatic stiffness detection
    - Good general-purpose solver
    - Variable order
    """

    @property
    def method_name(self) -> str:
        return 'LSODA'


# Backward compatibility alias (maintains existing interface)
ScipyMethod = RK45Solver