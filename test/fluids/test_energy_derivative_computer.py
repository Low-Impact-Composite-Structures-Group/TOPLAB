import unittest
from src.fluids.energy_derivative_computer import EnergyDerivativeComputer

from src.fluids.hydrogen_retrievers import TwoPhaseRequester


class TestEnergyDerivativeComputer(unittest.TestCase):

    def setUp(self) -> None:
        self.pressure = 130e3
        self.requester = TwoPhaseRequester()
        self.hydrogen = self.requester.get_hydrogen_properties(
            self.pressure, None
        )
        self.fill = 0.95
        self.computer = EnergyDerivativeComputer()

    def test_compute_quality(self):
        expected_value = (
            1 + (
                self.hydrogen.liquid.density / self.hydrogen.gas.density
            ) * (
                self.fill / (1 - self.fill)
            )
        ) ** (-1)
        actual_value = self.computer.compute_quality(
            self.hydrogen, self.fill
        )
        self.assertEqual(expected_value, actual_value)

    def test_compute_internal_energy(self):
        x = (
            1 + (
                self.hydrogen.liquid.density / self.hydrogen.gas.density
            ) * (
                self.fill / (1 - self.fill)
            )
        ) ** (-1)
        expected_value = (
            x * self.hydrogen.gas.internal_energy
            + (1 - x) * self.hydrogen.liquid.internal_energy
        )
        actual_value = self.computer.compute_internal_energy(
            self.hydrogen, self.fill
        )
        self.assertEqual(expected_value, actual_value)

    def test_get_reference_hydrogen(self):
        pressure = 1.01 * self.hydrogen.pressure
        expected_value = self.requester.get_hydrogen_properties(
            pressure, None
        )
        actual_value = self.computer.get_reference_hydrogen(
            self.hydrogen
        )
        self.assertEqual(expected_value, actual_value)

    def test_compute_dU_dP(self):
        ref_hydrogen = self.computer.get_reference_hydrogen(
            self.hydrogen
        )
        p1 = self.hydrogen.pressure
        u1 = self.computer.compute_internal_energy(
            self.hydrogen, self.fill
        )
        p2 = ref_hydrogen.pressure
        u2 = self.computer.compute_internal_energy(
            ref_hydrogen, self.fill
        )
        expected_value = (u2 - u1) / (p2 - p1)
        actual_value = self.computer.compute_dU_dP(
            self.hydrogen, self.fill
        )
        self.assertEqual(expected_value, actual_value)

    def test_compute_density(self):
        x = self.computer.compute_quality(
            self.hydrogen, self.fill
        )
        expected_value = (
            x / self.hydrogen.gas.density
            + (1 - x) / self.hydrogen.liquid.density
        ) ** (-1)
        actual_value = self.computer.compute_density(
            self.hydrogen, self.fill
        )
        self.assertEqual(expected_value, actual_value)

    def test_compute_energy_derivative(self):

        dU_dP = self.computer.compute_dU_dP(self.hydrogen, self.fill)
        density = self.computer.compute_density(
            self.hydrogen, self.fill
        )
        expected_value = 1 / (dU_dP * density)
        actual_value = self.computer.compute_energy_derivative(
            self.hydrogen, self.fill
        )
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
