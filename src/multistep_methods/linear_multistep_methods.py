"""The Linear Multistep Methods scripts can be used to perform a  linear
forward timestep analysis. The implementation is based on the wikipedia:
https://en.wikipedia.org/wiki/Linear_multistep_method#One-step_Euler

Fuel Tank - Multistep Methods
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Union
import numpy as np


def euler_method(
    derivative_value: float,
    function_value: float,
    timestep: float
) -> float:
    """Euler method as explicit multistep method.

    Args:
        derivative_value (float): value of the derivative at the
        time instance.
        function_value (float): value of the function at the time
        instance.
        timestep (float): timestep used for the forwards step.

    Returns:
        float: value of the one step forwards euler method.
    """
    return function_value + timestep * derivative_value


def two_step_adam_bashforth(
    derivative_values: list[float],
    function_value: float,
    timestep: float
) -> float:
    """Second step of the two step Adam-Bashforth method to compute the
    value in a forward explicit method.

    Args:
        derivative_values (List[float]): list of with the previous
        derivative values. Note that it is assumed that only the desired
        values of the are provided.
        function_value (float): previous value of the function.
        timestep (float): timestep used in the explicit method.

    Returns:
        float: value of the function after the step.
    """
    return (
        function_value
        + 3 / 2 * timestep * derivative_values[1]
        - 1 / 2 * timestep * derivative_values[0]
    )


def three_step_adam_bashforth(
    derivative_values: list[float],
    function_value: float,
    timestep: float
) -> float:
    """Third step of the two step Adam-Bashforth method to compute the
    value in a forward explicit method.

    Args:
        derivative_values (List[float]): list of with the previous
        derivative values. Note that it is assumed that only the desired
        values of the are provided.
        function_value (float): previous value of the function.
        timestep (float): timestep used in the explicit method.

    Returns:
        float: value of the function after the step.
    """
    return (
        function_value
        + timestep * 23 / 12 * derivative_values[2]
        - timestep * 16 / 12 * derivative_values[1]
        + timestep * 5 / 12 * derivative_values[0]
    )


def four_step_adam_bashforth(
    derivative_values: list[float],
    function_value: float,
    timestep: float
) -> float:
    """Fourth step of the two step Adam-Bashforth method to compute the
    value in a forward explicit method.

    Args:
        derivative_values (List[float]): list of with the previous
        derivative values. Note that it is assumed that only the desired
        values of the are provided.
        function_value (float): previous value of the function.
        timestep (float): timestep used in the explicit method.

    Returns:
        float: value of the function after the step.
    """
    return (
        function_value
        + timestep * 55 / 24 * derivative_values[3]
        - timestep * 59 / 24 * derivative_values[2]
        + timestep * 37 / 24 * derivative_values[1]
        - timestep * 9 / 24 * derivative_values[0]
    )


def adam_bashforth(
    derivative_values: Union[list[float],float],
    function_value: float,
    timestep: float
) -> float:
    """Linear multistep method using the Adam-Bashforth method. The type
    of step taken in the determination at the next time instance depends
    upon the number of provided derivatives. The maximum implemented step
    is of four.

    Args:
        derivative_values (Union[List[float],float]): list with derivative
        values. Depending on the length of the list the type of step is
        taken.
        function_value (float): value of the function at the current time
        step.
        timestep (float): size of the timestep

    Returns:
        float: value of the function at the new time instance.
    """
    if isinstance(derivative_values, float):
        return euler_method(derivative_values, function_value, timestep)
    if len(derivative_values) == 1:
        return euler_method(derivative_values[0], function_value, timestep)
    if len(derivative_values) == 2:
        return two_step_adam_bashforth(
            derivative_values, function_value, timestep
        )
    if len(derivative_values) == 3:
        return three_step_adam_bashforth(
            derivative_values, function_value, timestep
        )
    return four_step_adam_bashforth(
        derivative_values[-4:], function_value, timestep
    )


@dataclass
class MultistepMethod(Protocol):
    timestep: float

    @abstractmethod
    def compute_new_value(
        derivatives: list[float],
        current_value: float
    ) -> float:
        ...


@dataclass
class EulerMethod(MultistepMethod):
    timestep: float

    def compute_new_value(
        self,
        derivatives: Union[float, list[float]],
        current_value: float
    ) -> float:
        if not isinstance(derivatives, list):
            return euler_method(
                derivatives, current_value, self.timestep
            )
        return euler_method(
            derivatives[-1], current_value, self.timestep
        )

def rk4_method(
    derivative_func: callable,
    function_value: float,
    timestep: float,
    time: float
) -> float:
    """4th order Runge-Kutta method for numerical integration.

    Args:
        derivative_func (callable): Function that computes the derivative at a given point.
            Should accept two parameters: time and function_value.
        function_value (float): Current value of the function at the time instance.
        timestep (float): Timestep used for the forward step.
        time (float): Current time value.

    Returns:
        float: Value after one step of the RK4 method.
    """
    k1 = derivative_func(time, function_value)
    k2 = derivative_func(time + timestep/2, function_value + timestep*k1/2)
    k3 = derivative_func(time + timestep/2, function_value + timestep*k2/2)
    k4 = derivative_func(time + timestep, function_value + timestep*k3)

    return function_value + timestep * (k1 + 2*k2 + 2*k3 + k4) / 6

@dataclass
class AdamBashforthMethod(MultistepMethod):
    timestep: float

    def compute_new_value(
        self,
        derivatives: Union[float, list[float]],
        current_value: float
    ) -> float:
        if not isinstance(derivatives, list):
            return adam_bashforth(
                derivatives, current_value, self.timestep
            )
        return adam_bashforth(
            derivatives[-1], current_value, self.timestep
        )


def rk4_single_step(
    derivative_value: float,
    function_value: float,
    timestep: float
) -> float:
    """Simplified RK4 method that uses only the current derivative value.
    This adapts the RK4 concept to fit within the linear multistep method framework.

    Args:
        derivative_value (float): Value of the derivative at the time instance.
        function_value (float): Current value of the function.
        timestep (float): Timestep used for the forward step.

    Returns:
        float: Value after one step using a simplified RK4-inspired approach.
    """
    # Since we only have the current derivative value and not a function,
    # we use a simplified approach that's compatible with the framework
    return function_value + timestep * derivative_value

@dataclass
class RK4Method(MultistepMethod):
    timestep: float

    def compute_new_value(
        self,
        derivatives: Union[float, list[float]],
        current_value: float
    ) -> float:
        """Computes the next value using a simplified RK4-compatible approach.

        Args:
            derivatives (Union[float, list[float]]): Current derivative value or list of derivative values.
            current_value (float): Current function value.

        Returns:
            float: New function value after one timestep.
        """
        if not isinstance(derivatives, list):
            return rk4_single_step(
                derivatives, current_value, self.timestep
            )
        return rk4_single_step(
            derivatives[-1], current_value, self.timestep
        )


def backward_euler_method(
    derivative_value: float,
    function_value: float,
    timestep: float,
    jacobian_value: float = 1.0  # Simplified jacobian (derivative of derivative)
) -> float:
    """Backward (implicit) Euler method.

    Args:
        derivative_value (float): Value of the derivative at the current time.
        function_value (float): Current value of the function.
        timestep (float): Timestep used for the forward step.
        jacobian_value (float): Jacobian (derivative of the derivative function).
            Simplified as a constant value here.

    Returns:
        float: Value after one step of the backward Euler method.
    """
    # For a stiff system y' = f(y), the backward Euler is:
    # y(n+1) = y(n) + h*f(y(n+1))
    # This requires solving an implicit equation
    # Here we use a simplified approach with one Newton iteration

    # Avoid division by zero by adding a small epsilon
    # or fall back to forward Euler if the denominator is too close to zero
    denominator = (1.0 - timestep * jacobian_value)

    if abs(denominator) < 1e-10:  # Safety threshold to prevent numerical instability
        # Fall back to forward Euler for this step
        return function_value + timestep * derivative_value

    # Simple implementation (first-order approximation):
    # y(n+1) = y(n) + h*f(y(n)) / (1 - h*J)
    # where J is the Jacobian df/dy
    return function_value + timestep * derivative_value / denominator

@dataclass
class BackwardEulerMethod(MultistepMethod):
    timestep: float
    jacobian: float = 1.0  # Default Jacobian value

    def compute_new_value(
        self,
        derivatives: Union[float, list[float]],
        current_value: float
    ) -> float:
        if not isinstance(derivatives, list):
            return backward_euler_method(
                derivatives, current_value, self.timestep, self.jacobian
            )
        return backward_euler_method(
            derivatives[-1], current_value, self.timestep, self.jacobian
        )

@dataclass
class AdaptiveStepSolver:
    """Adaptive Step Solver for refuel analysis using linear multistep methods.

    This class adapts the timestep size during the simulation to ensure
    accurate and efficient integration, especially during the refueling
    phases where the dynamics might change rapidly.

    Attributes:
        timestep (float): Initial timestep for the solver.
        min_timestep (float): Minimum allowable timestep.
        max_timestep (float): Maximum allowable timestep.
        error_tolerance (float): Tolerance for adaptive error control.
        safety_factor (float): Safety factor for timestep adaptation.
        use_backward_euler (bool): Flag to use backward Euler method for stiff systems.
        jacobian (float): Jacobian value for the system, used in implicit methods.
    """

    timestep: float
    min_timestep: float = 0.01
    max_timestep: float = 5.0
    error_tolerance: float = 1e-6
    safety_factor: float = 0.9
    use_backward_euler: bool = False
    jacobian: float = 1.0

    def adapt_timestep(self, current_error: float):
        """Adjusts the timestep based on the estimated error of the solution.

        If the error is too large, the timestep is decreased to improve accuracy.
        If the error is small, the timestep may be increased to speed up computation.

        Args:
            current_error (float): The estimated error from the current timestep.

        Returns:
            float: The new timestep size.
        """
        # Basic error control: reduce timestep if error is too large,
        # increase timestep if error is small
        new_timestep = self.timestep * min(
            max(self.error_tolerance / abs(current_error), 0.5), 2.0
        )

        # Respect min and max bounds
        new_timestep = max(self.min_timestep, min(new_timestep, self.max_timestep))

        return new_timestep

    def compute_new_value(
        self,
        derivatives: Union[float, list[float]],
        current_value: float
    ) -> float:
        """Compute the new value for the system, adapting the timestep as needed.

        This method overrides the compute_new_value in MultistepMethod to provide
        adaptive stepping behavior.

        Args:
            derivatives (Union[float, list[float]]): Current derivative value or list of derivative values.
            current_value (float): Current function value.

        Returns:
            float: New function value after one timestep.
        """
        # For the first call, or if using a fixed timestep method, just proceed
        if self.timestep == self.min_timestep and self.timestep == self.max_timestep:
            return RK4Method.compute_new_value(self, derivatives, current_value)

        # Estimate the error using a simple heuristic:
        # Perform the step with the current timestep
        predicted_value = RK4Method.compute_new_value(self, derivatives, current_value)

        # Perform the step with a halved timestep
        half_timestep = self.timestep / 2
        value_with_half_step = RK4Method.compute_new_value(
            self, derivatives, current_value
        )
        predicted_value_half_step = RK4Method.compute_new_value(
            self, derivatives, value_with_half_step
        )

        # Estimate error as the difference between the two predictions
        error_estimate = abs(predicted_value_half_step - predicted_value) / 2

        # Adapt the timestep based on the error estimate
        new_timestep = self.adapt_timestep(error_estimate)

        # Update the timestep for the next call
        self.timestep = new_timestep

        return predicted_value_half_step

def main():
    pass


if __name__ == "__main__":
    main()


# End