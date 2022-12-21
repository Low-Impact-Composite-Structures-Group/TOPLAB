from dataclasses import dataclass
import unittest

from src.dynamics.stopping_criteria import LowerPressureReached, MaxPressure, MaxPressureReached, NoFuelMass, TankIsEmpty, TankIsFull, TargetFillReached, TargetMassReached


@dataclass
class TargetState:
    max_pressure: float
    min_pressure: float
    min_temperature: float
    fill: float
    mass: float

    @classmethod
    def test_state(cls):
        max_pressure = 300e5
        min_pressure = 1.2e5
        min_temperature = 20
        fill = 0.95
        mass = 0

        return cls(
            max_pressure, min_pressure, min_temperature, fill, mass
        )

@dataclass
class FuelTankState:
    pressure: float
    fill: float
    fuel_mass: float
    phase: str

    @classmethod
    def test_state(cls):
        pressure = 1.5e5
        fill = 0.95
        fuel_mass = 100
        phase = "liquid"

        return cls(pressure, fill, fuel_mass, phase)


class BaseTestStoppingCriterion:

    def setUp(self):
        self.target_state = TargetState.test_state()
        self.state = FuelTankState.test_state()


class TestMaxPressure(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = MaxPressure()
        return self

    def test_is_met(self):

        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestIsEmpty(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = TankIsEmpty()
        return self

    def test_is_met(self):

        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.state.fill = 0
        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.state.phase = "twophase"
        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestNoFuelMass(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = NoFuelMass()
        return self

    def test_is_met(self):

        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.state.fuel_mass = 0
        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestTankIsFull(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = TankIsFull()
        return self

    def test_is_met(self):

        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.state.fill = 1.0001
        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestTargetFillReached(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = TargetFillReached()
        return self

    def test_is_met(self):

        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.target_state.fill = 1.0
        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestTargetMassReached(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = TargetMassReached()
        return self  

    def test_is_met(self):

        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.target_state.mass = 1e3
        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestLowerPressureReached(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = LowerPressureReached()
        return self  

    def test_is_met(self):

        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.target_state.min_pressure = 1.5e5
        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )


class TestMaxPressureReached(
    BaseTestStoppingCriterion, unittest.TestCase
):

    def setUp(self):
        super().setUp()
        self.criterion = MaxPressureReached()
        return self  

    def test_is_met(self):

        self.assertFalse(
            self.criterion.is_met(self.state, self.target_state)
        )

        self.target_state.max_pressure = 1.2e5
        self.assertTrue(
            self.criterion.is_met(self.state, self.target_state)
        )




if __name__ == "__main__":
    unittest.main()


# End