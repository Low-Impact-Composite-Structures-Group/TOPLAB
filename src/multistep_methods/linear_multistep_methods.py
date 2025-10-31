"""The Linear Multistep Methods scripts can be used to perform a  linear
forward timestep analysis. The implementation is based on the wikipedia:
https://en.wikipedia.org/wiki/Linear_multistep_method#One-step_Euler

Fuel Tank - Multistep Methods
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""


from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Union


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
        return euler_method(
            derivatives[-1], current_value, self.timestep
        )


@dataclass
class AdamBashforthMethod(MultistepMethod):
    timestep: float

    def compute_new_value(
        self,
        derivatives: Union[float, list[float]],
        current_value: float
    ) -> float:
        return adam_bashforth(
            derivatives, current_value, self.timestep
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End