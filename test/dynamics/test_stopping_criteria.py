from dataclasses import dataclass
import unittest

from src.dynamics.stopping_criteria import MaxPressure



class TestMaxPressure(unittest.TestCase):

    def test_is_met(self):

        criterion = MaxPressure()

        @dataclass
        class TankState:
            pressure: float
        
        current_state = TankState(300e5)
        lower_state = TankState(200e5)
        higher_state = TankState(400e5)
        equal_state = TankState(300e5)

        self.assertFalse(
            criterion.is_met(current_state, higher_state)
        )
        self.assertTrue(
            criterion.is_met(current_state, equal_state)
        )
        self.assertTrue(
            criterion.is_met(current_state, lower_state)
        )



if __name__ == "__main__":
    unittest.main()


# End