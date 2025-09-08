"""This module defines the liner for hydrogen fuel tanks.
The liner is an optional component that adds an additional thermal
resistance between the tank contents and the tank wall.

Fuel Tank - Liner
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from src.thermodynamics.thermal_resistances import ThermalResistance
from src.materials.materials import Metal


class TankShape(Protocol):
    """Protocol defining the necessary methods and attributes that a
    tank shape should have to be compatible with a liner.
    """
    radius: float
    surface_area: float


@dataclass
class Liner:
    """Defines a liner that provides an additional thermal resistance
    between the tank contents and the tank wall.

    The liner can be initialized in two ways:
    1. With a thickness and material - the mass is calculated
    2. With a mass and material - the thickness is calculated

    Args:
        material: The material of the liner (currently only Metal.aluminum())
        thickness: Optional thickness of the liner in meters
        mass: Optional mass of the liner in kg
        tank: Reference to the tank shape for calculations
    """
    material: Metal
    thickness: Optional[float] = None
    mass: Optional[float] = None
    tank: Optional[TankShape] = None

    def __post_init__(self):
        """Validates and completes the initialization of the liner.

        Either thickness or mass must be provided, but not both.
        If mass is provided, thickness will be calculated from the tank shape.
        """
        # Check that exactly one of thickness or mass is provided
        if (self.thickness is None and self.mass is None) or \
           (self.thickness is not None and self.mass is not None):
            raise ValueError("Either thickness or mass must be provided, but not both")

        # If tank is provided and mass is provided but not thickness,
        # calculate the thickness
        if self.tank is not None and self.mass is not None and self.thickness is None:
            self.calculate_thickness_from_mass()

    def calculate_thickness_from_mass(self) -> float:
        """Calculates the liner thickness based on the provided mass,
        tank surface area, and material density.

        Returns:
            float: The calculated liner thickness in meters
        """
        if self.tank is None:
            raise ValueError("Tank reference is required to calculate thickness from mass")

        # Get surface area - handle both Tank objects and TankDimensions objects
        if hasattr(self.tank, 'surface_area'):
            surface_area = self.tank.surface_area
        elif hasattr(self.tank, 'radius'):
            # For a sphere, surface area = 4πr²
            import math
            surface_area = 4 * math.pi * self.tank.radius**2
            print(f"Calculated surface area for sphere: {surface_area:.3f} m²")
        else:
            raise ValueError("Tank object must have either surface_area or radius attribute")

        self.thickness = self.mass / (self.material.density * surface_area)
        return self.thickness

    def calculate_mass(self) -> float:
        """Calculates the liner mass based on thickness, surface area, and density.

        Returns:
            float: The calculated liner mass in kg
        """
        if self.tank is None:
            raise ValueError("Tank reference is required to calculate mass")

        if self.thickness is None:
            raise ValueError("Thickness is required to calculate mass")

        # Get surface area - handle both Tank objects and TankDimensions objects
        if hasattr(self.tank, 'surface_area'):
            surface_area = self.tank.surface_area
        elif hasattr(self.tank, 'radius'):
            # For a sphere, surface area = 4πr²
            import math
            surface_area = 4 * math.pi * self.tank.radius**2
        else:
            raise ValueError("Tank object must have either surface_area or radius attribute")

        self.mass = self.material.density * surface_area * self.thickness
        return self.mass

    def compute_thermal_conductivity(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        """Compute the thermal conductivity of the liner material.

        Uses temperature-dependent material properties from the liner material.

        Args:
            hot_temperature (float): Temperature at the hot side of the liner
            cold_temperature (float): Temperature at the cold side of the liner

        Returns:
            float: Thermal conductivity of the liner material in W/(m·K)
        """
        # Use average temperature for thermal conductivity calculation
        avg_temperature = (hot_temperature + cold_temperature) / 2

        # Use material's temperature-dependent thermal conductivity if available
        if hasattr(self.material, 'determine_thermal_conductivity'):
            try:
                return self.material.determine_thermal_conductivity(avg_temperature)
            except Exception as e:
                print(f"Warning: Could not compute liner thermal conductivity at {avg_temperature}K: {e}")
                # Fallback to hardcoded values

        # Fallback: Use realistic thermal conductivity values based on material type
        # Check if it's an aluminum instance (by checking the density ~2700 kg/m³)
        if isinstance(self.material, Metal) and abs(self.material.density - 2700) < 100:
            # Aluminum has high thermal conductivity ~200-240 W/(m·K) at room temperature
            # Thermal conductivity decreases at lower temperatures
            if avg_temperature < 100:  # For cryogenic temperatures
                return 180.0  # W/(m·K)
            else:
                return 220.0  # W/(m·K)
        else:
            # Default value for other metals
            return 50.0  # W/(m·K)

    def compute_heat_transfer_coefficient(
        self,
        hot_temperature: float,
        cold_temperature: float
    ) -> float:
        """Calculate the heat transfer coefficient of the liner.

        Args:
            hot_temperature (float): Temperature at the hot side of the liner
            cold_temperature (float): Temperature at the cold side of the liner

        Returns:
            float: Heat transfer coefficient of the liner
        """
        if self.thickness is None or self.thickness == 0:
            # If thickness is zero, return infinite conductivity
            return float('inf')

        # Calculate thermal conductivity
        thermal_conductivity = self.compute_thermal_conductivity(
            hot_temperature, cold_temperature
        )

        # For a thin liner, we can use the simplified formula k/L
        # where k is thermal conductivity and L is thickness
        return thermal_conductivity / self.thickness

    def compute_thermal_resistance(
        self,
        hot_temperature: float,
        cold_temperature: float
    ) -> float:
        """Compute the thermal resistance value of the liner.

        Args:
            hot_temperature (float): Temperature at the hot side of the liner
            cold_temperature (float): Temperature at the cold side of the liner

        Returns:
            float: Thermal resistance value
        """
        if self.tank is None:
            raise ValueError("Tank reference is required to compute thermal resistance")

        heat_transfer_coeff = self.compute_heat_transfer_coefficient(
            hot_temperature, cold_temperature
        )

        # Get surface area - handle both Tank objects and TankDimensions objects
        if hasattr(self.tank, 'surface_area'):
            surface_area = self.tank.surface_area
        elif hasattr(self.tank, 'radius'):
            # For a sphere, surface area = 4πr²
            import math
            surface_area = 4 * math.pi * self.tank.radius**2
        else:
            raise ValueError("Tank object must have either surface_area or radius attribute")

        return ThermalResistance(
            heat_transfer_coeff, surface_area
        ).value

    def compute_thermal_resistances(
        self,
        temperatures: list[float],
        num_layers: int = 1
    ) -> list[float]:
        """Compute multiple thermal resistances for discretized liner layers.

        This method allows the liner to be divided into multiple thermal layers
        for improved accuracy, similar to how insulation is discretized.

        Args:
            temperatures: List of temperatures at layer interfaces
            num_layers: Number of layers to discretize the liner into

        Returns:
            list[float]: List of thermal resistance values for each layer
        """
        if len(temperatures) < 2:
            raise ValueError("At least 2 temperatures required for thermal resistance calculation")

        if len(temperatures) - 1 != num_layers:
            raise ValueError(f"Temperature list length ({len(temperatures)}) should be num_layers + 1 ({num_layers + 1})")

        thermal_resistances = []

        # Calculate thermal resistance for each layer
        for i in range(num_layers):
            resistance = self.compute_thermal_resistance(
                temperatures[i], temperatures[i + 1]
            )
            thermal_resistances.append(resistance)

        return thermal_resistances

    @classmethod
    def from_thickness(
        cls, thickness: float, tank: TankShape, material: Optional[Metal] = None
    ) -> Liner:
        """Create a liner with specified thickness.

        Args:
            thickness (float): Thickness of the liner in meters
            tank (TankShape): Reference to the tank shape
            material (Optional[Metal]): Material of the liner, defaults to aluminum

        Returns:
            Liner: A new Liner instance
        """
        if material is None:
            material = Metal.aluminum()

        liner = cls(material=material, thickness=thickness, tank=tank)
        liner.calculate_mass()
        return liner

    @classmethod
    def from_mass(
        cls, mass: float, tank: TankShape, material: Optional[Metal] = None
    ) -> Liner:
        """Create a liner with specified mass.

        Args:
            mass (float): Mass of the liner in kg
            tank (TankShape): Reference to the tank shape
            material (Optional[Metal]): Material of the liner, defaults to aluminum

        Returns:
            Liner: A new Liner instance
        """
        if material is None:
            material = Metal.aluminum()

        return cls(material=material, mass=mass, tank=tank)


# End
