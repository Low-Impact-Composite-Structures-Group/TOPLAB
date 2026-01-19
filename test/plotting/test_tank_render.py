import sys
import os
import unittest
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add the parent directory of 'plotting' to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from plotting.tank_render import plot_tank

class TestTankRender(unittest.TestCase):

    def test_plot_tank(self):
        radius = 1
        b = radius/2
        length = 3
        fig = plot_tank(radius, b, length)
        self.assertIsInstance(fig, Figure)
        plt.show()

if __name__ == "__main__":
    unittest.main()