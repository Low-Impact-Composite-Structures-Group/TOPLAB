"""
Multi-tank plotting module for hydrogen storage analysis.

This module provides clean, modular plotting functionality for multi-tank systems
using seaborn styling. Each plotting function is designed to be general and versatile.

Key Features:
- DelftColourPlotter base class with seaborn styling
- Tank evolution plots (pressure, temperature, mass, density vs time)
- Optional reference lines (e.g., P_min, P_vent, stopping density)
- Configurable titles based on analysis_name from config
- Support for both single and multi-tank systems

Author: Dante Raso
"""

from toplab.plotting.figures.core import PlotterBase
from toplab.plotting.figures.phase_colour_map import PhaseColourMapFigures
from toplab.plotting.figures.evolution import EvolutionFigures
from toplab.plotting.figures.density_temperature import DensityTemperatureFigures
from toplab.plotting.figures.flows import FlowFigures
from toplab.plotting.figures.thermal import ThermalFigures
from toplab.plotting.figures.pressure import PressureFigures
from toplab.plotting.figures.components import ComponentFigures


class DelftColourPlotter(
    PlotterBase,
    PhaseColourMapFigures,
    EvolutionFigures,
    DensityTemperatureFigures,
    FlowFigures,
    ThermalFigures,
    PressureFigures,
    ComponentFigures,
):
    """Multi-tank plotter with Delft styling; each figure family lives in ``toplab.plotting.figures``."""


def main():
    """Test the DelftColourPlotter with sample data."""
    print("Testing DelftColourPlotter...")

    # This would normally come from actual simulation results
    # For now, just demonstrate the plotter initialization
    plotter = DelftColourPlotter("Test Analysis")
    print("DelftColourPlotter test successful")


if __name__ == "__main__":
    main()