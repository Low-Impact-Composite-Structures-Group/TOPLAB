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
    """Vacuum insulation with configurable heat transfer coefficient.

    For vacuum insulation, thermal resistance is dominated by radiation heat transfer
    and minimal residual gas conduction. The heat transfer coefficient represents the
    overall heat transfer coefficient (U-value) for the complete vacuum insulation
    system, not a material thermal conductivity.
    """
    surface_area: float = None  # Surface area in m²
    k_amb: float = 0.025        # Heat transfer coefficient in W/(m²·K)

    def __post_init__(self):
        """Initialize with default values if not provided."""
        if self.k_amb is None:
            self.k_amb = 0.025  # Default value W/(m²·K)

    def compute_heat_transfer_coefficient_value(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        """Compute the overall heat transfer coefficient for vacuum insulation.

        Args:
            hot_temperature: Temperature at hot side (K)
            cold_temperature: Temperature at cold side (K)

        Returns:
            float: Heat transfer coefficient in W/m²K
        """
        # Return the configured heat transfer coefficient
        return self.k_amb

    def compute_thermal_conductivity(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        """Legacy method for backward compatibility.

        Note: This method is deprecated. For vacuum insulation, the proper
        value is a heat transfer coefficient (0.025 W/m²K), not thermal conductivity.

        Args:
            hot_temperature: Temperature at hot side (K)
            cold_temperature: Temperature at cold side (K)

        Returns:
            float: Heat transfer coefficient value (W/m²K) for compatibility
        """
        return self.compute_heat_transfer_coefficient_value(hot_temperature, cold_temperature)

    def compute_heat_transfer_coefficient(
        self,
        hot_temperature: float,
        cold_temperature: float,
        tank: FuelTank
    ) -> float:
        """Compute the heat transfer coefficient for vacuum insulation.

        For vacuum insulation, the heat transfer coefficient is a system property
        that already accounts for the complete thermal resistance mechanism
        (radiation + minimal conduction), so no geometric corrections are needed.

        Args:
            hot_temperature: Temperature at hot side (K)
            cold_temperature: Temperature at cold side (K)
            tank: Tank object (not used for vacuum insulation)

        Returns:
            float: Heat transfer coefficient in W/m²K
        """
        # Return the overall heat transfer coefficient directly
        return self.compute_heat_transfer_coefficient_value(hot_temperature, cold_temperature)

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
        # Use the surface area from the insulation object if available, otherwise use tank surface area
        area = self.surface_area if self.surface_area is not None else tank.surface_area
        resistances = [
            ThermalResistance(coefficient, area).value
            for coefficient in heat_transfer_coefficients
        ]
        # print(f"Computed thermal resistances: {resistances}")

        return resistances