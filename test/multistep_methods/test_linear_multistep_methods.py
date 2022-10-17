import math
import unittest

from src.multistep_methods.linear_multistep_methods import (
    EulerMethod, adam_bashforth, euler_method, four_step_adam_bashforth,
    three_step_adam_bashforth, two_step_adam_bashforth)


class TestMultistepMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.function = (lambda x: math.exp(x))
        self.derivative = (lambda x: math.exp(x))
        self.t0 = 0
        self.timestep = 1 / 2
        self.f0 = self.function(self.t0)
        self.places = 4

    def test_euler_method(self):
        
        f1 = euler_method(
                self.derivative(self.t0),
                self.function(self.t0),
                self.timestep
            )
        self.assertEqual(f1, 1.5)
    
    def test_two_step_adam_bashforth(self):
        # Take note in this example that the derivate of the function is
        # assumed to be equal to the value of the function, thus the
        # new derivative is computed with the previous estimate of the 
        # function

        function_values = [self.f0]

        # First step is done with the euler method
        function_values.append(
            euler_method(
                function_values[-1],
                function_values[-1],
                self.timestep
            )
        )

        expected_values = [2.375, 3.7812, 6.0234]
        for value in expected_values:
            # One step
            function_values.append(
                two_step_adam_bashforth(
                    function_values[-2:],
                    function_values[-1],
                    self.timestep
            ))
            self.assertAlmostEqual(
                function_values[-1], value, self.places
            )
    
    def test_three_step_adam_bashforth(self):

        function_values = [self.f0]

        # First step is done with the euler method
        function_values.append(
            euler_method(
                function_values[-1],
                function_values[-1],
                self.timestep
            )
        )

        # Second step is taken with the two step method
        function_values.append(
            two_step_adam_bashforth(
                function_values[-2:],
                function_values[-1],
                self.timestep
            )
        )

        # Third step is taken for the test case
        function_values.append(
            three_step_adam_bashforth(
                function_values[-3:],
                function_values[-1],
                self.timestep
            )
        )
        self.assertAlmostEqual(
            function_values[-1], 3.8594, self.places
        )

    def test_four_step_adam_bashforth(self):

        function_values = [self.f0]

        # First step is done with the euler method
        function_values.append(
            euler_method(
                function_values[-1],
                function_values[-1],
                self.timestep
            )
        )

        # Second step is taken with the two step method
        function_values.append(
            two_step_adam_bashforth(
                function_values[-2:],
                function_values[-1],
                self.timestep
            )
        )

        # Third step is taken with the three step method
        function_values.append(
            three_step_adam_bashforth(
                function_values[-3:],
                function_values[-1],
                self.timestep
            )
        )

        # Fourth step is taken for the test case
        function_values.append(
            four_step_adam_bashforth(
                function_values[-4:],
                function_values[-1],
                self.timestep
            )
        )
        
        self.assertAlmostEqual(
            function_values[-1], 6.3311, self.places
        )

    def test_adam_bashforth(self):

        self.assertAlmostEqual(
            adam_bashforth(self.f0, self.f0, self.timestep), 1.5
        )

        expected_values = [
            1.5, 2.375, 3.8594, 6.3311
        ]
        function_values = [self.f0]
        for expected_value in expected_values:
            function_values.append(adam_bashforth(
                function_values, function_values[-1], self.timestep
            ))
            self.assertAlmostEqual(
                function_values[-1], expected_value, places=4
            )


class TestEulerMethod(unittest.TestCase):

    def setUp(self) -> None:
        self.function = (lambda x: math.exp(x))
        self.derivative = (lambda x: math.exp(x))
        self.t0 = 0
        self.timestep = 1 / 2
        self.f0 = self.function(self.t0)
        self.places = 4
        self.method = EulerMethod(self.timestep)

    def test_compute_new_value(self):

        expected_value = 1.5
        actual_value = self.method.compute_new_value(
            self.derivative(self.t0), self.f0
        )
        self.assertEqual(expected_value, actual_value)

        expected_value = 1.5
        actual_value = self.method.compute_new_value(
            [self.derivative(self.t0)], self.f0
        )
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
