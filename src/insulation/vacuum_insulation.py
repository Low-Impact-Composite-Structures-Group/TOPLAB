from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Protocol

from src.thermodynamics.thermal_resistances import ThermalResistance


class FuelTank(Protocol):
    radius: float
    surface_area: float


@dataclass
class VacuumInsulation:
    """Vacuum insulation with constant thermal conductivity of 0.025 W/mK."""

    def compute_thermal_conductivity(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        # Fixed thermal conductivity regardless of temperature
        return 0.025  # W/mK

    def compute_heat_transfer_coefficient(
        self,
        hot_temperature: float,
        cold_temperature: float,
        tank: FuelTank
    ) -> float:
        """Method to determine the heat transfer coefficient of the insulator."""
        # For a vacuum insulation, we use the fixed thermal conductivity
        thermal_conductivity = self.compute_thermal_conductivity(hot_temperature, cold_temperature)
        # Using a simple heat transfer coefficient calculation
        return thermal_conductivity / tank.radius

    def compute_thermal_resistances(
        self,
        temperatures: list[float],
        tank: FuelTank
    ) -> list[float]:
        """Compute thermal resistances for each layer based on temperatures."""
        # Calculate heat transfer coefficients between adjacent temperature nodes
        heat_transfer_coefficients = [
            self.compute_heat_transfer_coefficient(
                temperatures[i], temperatures[i+1], tank
            )
            for i in range(len(temperatures)-1)
        ]

        # Convert to thermal resistances
        resistances = [
            ThermalResistance(coefficient, tank.surface_area).value
            for coefficient in heat_transfer_coefficients
        ]

        return resistances