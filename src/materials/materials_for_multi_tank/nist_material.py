"""
NIST Material Class for Multi-Tank Framework

Barebones material class that provides:
1. Basic material properties (density, failure_stress)
2. Temperature-dependent specific heat via NIST data
3. Simple interface compatible with existing thermal/structural models

This replaces the complex Material class hierarchy with a simplified
approach focused on NIST temperature-dependent properties.

Author: Multi-Tank Framework
Date: September 2025
"""

from dataclasses import dataclass
from typing import Callable

# Import NIST property functions
from .nist_properties.aluminum_6061T6_nist import specific_heat as aluminum_6061T6_specific_heat
from .nist_properties.g10_nist import specific_heat as g10_specific_heat


@dataclass
class NISTMaterial:
    """
    Simplified material class using NIST temperature-dependent properties.

    Attributes:
        density (float): Material density in kg/m³
        failure_stress (float): Material failure stress in Pa
        nist_path (str): Path identifier for NIST properties
        specific_heat_func (Callable): Function for temperature-dependent specific heat
        name (str): Human-readable material name
    """

    density: float                    # kg/m³
    failure_stress: float            # Pa
    nist_path: str                   # NIST path identifier
    specific_heat_func: Callable     # Function: T -> Cp
    name: str                        # Material name
    type: str                        # Material type: "metal" or "composite"
    winding_angle: float = None      # Winding angle for composites (radians)

    def get_specific_heat(self, temperature: float) -> float:
        """
        Get temperature-dependent specific heat from NIST data.

        Args:
            temperature (float): Temperature in Kelvin (10-400K)

        Returns:
            float: Specific heat in J/(kg·K)
        """
        return self.specific_heat_func(temperature)

    def determine_specific_heat(self, temperature: float) -> float:
        """
        Compatibility method with existing thermal models.
        Same as get_specific_heat but maintains existing interface.

        Args:
            temperature (float): Temperature in Kelvin (10-400K)

        Returns:
            float: Specific heat in J/(kg·K)
        """
        return self.get_specific_heat(temperature)

    def determine_thermal_capacity(self, temperature: float, mass: float) -> float:
        """
        Compute thermal capacity for given mass and temperature.
        Compatibility method with existing thermal models.

        Args:
            temperature (float): Temperature in Kelvin (10-400K)
            mass (float): Mass in kg

        Returns:
            float: Thermal capacity in J/K
        """
        return self.get_specific_heat(temperature) * mass

    @classmethod
    def aluminum_6061T6_nist(cls):
        """
        Create aluminum 6061-T6 material with NIST properties.

        Properties:
        - Density: 2700 kg/m³ (2.7 g/cm³)
        - Failure stress: 276 MPa
        - Temperature range: 10-400K

        Returns:
            NISTMaterial: Configured aluminum 6061-T6 material
        """
        return cls(
            density=2700.0,                              # kg/m³
            failure_stress=276e6,                        # Pa (276 MPa)
            nist_path="aluminum_6061T6_nist",
            specific_heat_func=aluminum_6061T6_specific_heat,
            name="Aluminum 6061-T6 (NIST)",
            type="metal"
        )

    @classmethod
    def g10_nist(cls):
        """
        Create G10 composite material with NIST properties.

        Properties:
        - Density: 1800 kg/m³ (1.8 g/cm³)
        - Failure stress: 310 MPa
        - Temperature range: 10-400K

        Returns:
            NISTMaterial: Configured G10 composite material
        """
        return cls(
            density=1800.0,                              # kg/m³
            failure_stress=310e6,                        # Pa (310 MPa)
            nist_path="g10_nist",
            specific_heat_func=g10_specific_heat,
            name="G10 Composite (NIST)",
            type="composite",
            winding_angle=54.7 * 3.14159/180            # Default optimal angle (54.7°) in radians
        )

    def __str__(self):
        """String representation of the material."""
        return (f"{self.name}: ρ={self.density:.0f} kg/m³, "
                f"σ_fail={self.failure_stress/1e6:.0f} MPa, "
                f"path={self.nist_path}")

    def __repr__(self):
        """Detailed representation of the material."""
        return (f"NISTMaterial(name='{self.name}', "
                f"density={self.density}, "
                f"failure_stress={self.failure_stress}, "
                f"nist_path='{self.nist_path}')")


def get_material_by_nist_path(nist_path: str) -> NISTMaterial:
    """
    Get material by NIST path identifier.

    Args:
        nist_path (str): NIST path identifier from configuration

    Returns:
        NISTMaterial: Configured material

    Raises:
        ValueError: If nist_path is not recognized
    """
    material_registry = {
        "aluminum_6061T6_nist": NISTMaterial.aluminum_6061T6_nist,
        "g10_nist": NISTMaterial.g10_nist,
    }

    if nist_path not in material_registry:
        available = ", ".join(material_registry.keys())
        raise ValueError(f"Unknown NIST path '{nist_path}'. Available: {available}")

    return material_registry[nist_path]()


if __name__ == "__main__":
    # Demonstration
    print("NIST Materials Framework Demonstration")
    print("=" * 50)

    # Create materials
    aluminum = NISTMaterial.aluminum_6061T6_nist()
    g10 = NISTMaterial.g10_nist()

    print(f"Aluminum: {aluminum}")
    print(f"G10: {g10}")

    # Test temperature-dependent properties
    test_temps = [20, 77, 150, 300]  # K

    print(f"\nTemperature-Dependent Properties:")
    print(f"{'Material':<20} {'Temp (K)':<10} {'Cp (J/kg·K)':<12}")
    print("-" * 50)

    for material in [aluminum, g10]:
        for T in test_temps:
            cp = material.get_specific_heat(T)
            print(f"{material.name:<20} {T:<10.0f} {cp:<12.1f}")

    # Test material registry
    print(f"\nMaterial Registry Test:")
    al_from_path = get_material_by_nist_path("aluminum_6061T6_nist")
    g10_from_path = get_material_by_nist_path("g10_nist")

    print(f"From path: {al_from_path.name}")
    print(f"From path: {g10_from_path.name}")

    print(f"\n✅ NIST Materials Framework Ready!")