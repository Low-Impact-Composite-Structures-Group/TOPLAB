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
from scipy.optimize import OptimizeResult


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
            state: Current state vector (e.g., [m, T, Tstruct, Tinsulation, Tshell])
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

@dataclass
class RK4FixedSolver(SciPySolver):
    """
    Fixed-step 4th order Runge-Kutta method.

    Characteristics:
    - Explicit method (for non-stiff problems)
    - Fixed timestep
    - 4th order accuracy
    - For debbugging when adaptive stepping is not desired or when a fixed timestep is required
    """

    @property
    def method_name(self) -> str:
        return 'RK4_FIXED'

    def integrate_step(self, **kwargs) -> tuple[np.ndarray, bool]:
        """Advance the current state by one fixed fourth-order Runge-Kutta step."""
        if self._ode_function is None:
            raise ValueError("ODE function not set. Call set_ode_function() first.")
        if self._current_state is None:
            raise ValueError("Current state not set. Call set_current_state() first.")

        timestep = float(kwargs.get('timestep', self.timestep))
        try:
            new_state = self._rk4_step(self._current_time, self._current_state, timestep)
        except Exception as exc:
            print(f"RK4 fixed-step integration error: {exc}")
            return self._current_state, False

        self._current_time += timestep
        self._current_state = new_state
        return new_state, True

    def integrate_full(self, t_span: tuple, y0: np.ndarray, t_eval: np.ndarray = None, **kwargs) -> OptimizeResult:
        """Integrate with fixed RK4 steps and terminal-event detection at step boundaries."""
        if self._ode_function is None:
            raise ValueError("ODE function not set. Call set_ode_function() first.")

        start_time, end_time = map(float, t_span)
        timestep = float(kwargs.get('timestep', self.timestep))
        if timestep <= 0.0:
            raise ValueError("RK4 fixed-step solver requires a positive timestep.")

        events = kwargs.get('events', [])
        if callable(events):
            events = [events]
        event_times = [[] for _ in events]
        time_points = [start_time]
        state_points = [np.asarray(y0, dtype=float).copy()]
        current_time = start_time
        current_state = state_points[0]
        event_values = [event(current_time, current_state) for event in events]
        function_evaluations = 0

        while current_time < end_time:
            step = min(timestep, end_time - current_time)
            next_state = self._rk4_step(current_time, current_state, step)
            function_evaluations += 4
            next_time = current_time + step
            next_event_values = [event(next_time, next_state) for event in events]

            terminal_event = False
            for index, event in enumerate(events):
                direction = getattr(event, 'direction', 0.0)
                crossed_up = event_values[index] < 0.0 <= next_event_values[index]
                crossed_down = event_values[index] > 0.0 >= next_event_values[index]
                crossed = crossed_up if direction > 0.0 else crossed_down if direction < 0.0 else crossed_up or crossed_down
                if crossed:
                    event_times[index].append(next_time)
                    terminal_event = terminal_event or bool(getattr(event, 'terminal', False))

            time_points.append(next_time)
            state_points.append(next_state.copy())
            current_time, current_state, event_values = next_time, next_state, next_event_values
            if terminal_event:
                break

        times = np.asarray(time_points)
        states = np.asarray(state_points).T
        if t_eval is not None:
            requested_times = np.asarray(t_eval, dtype=float)
            keep = np.isin(times, requested_times)
            if not keep[-1]:
                keep[-1] = True
            times = times[keep]
            states = states[:, keep]

        return OptimizeResult(
            t=times,
            y=states,
            t_events=[np.asarray(times) for times in event_times],
            success=True,
            message='The solver successfully reached the end of the integration interval.',
            nfev=function_evaluations,
        )

    def _rk4_step(self, time: float, state: np.ndarray, timestep: float) -> np.ndarray:
        k1 = np.asarray(self._ode_function(time, state), dtype=float)
        k2 = np.asarray(self._ode_function(time + 0.5 * timestep, state + 0.5 * timestep * k1), dtype=float)
        k3 = np.asarray(self._ode_function(time + 0.5 * timestep, state + 0.5 * timestep * k2), dtype=float)
        k4 = np.asarray(self._ode_function(time + timestep, state + timestep * k3), dtype=float)
        return state + timestep * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# Backward compatibility alias (maintains existing interface)
ScipyMethod = RK45Solver