"""
Temperature-dependent material classes using NIST data.

This module provides enhanced material classes that use NIST polynomial
fits for temperature-dependent specific heat calculations instead of the
Debye model approach used in the base Material class.
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.materials.materials import Material, Metal, Composite
from src.materials.aluminum5083_properties import specific_heat as aluminum_cp_nist
from src.materials.g10_properties import specific_heat as g10_cp_nist


@dataclass
class NISTMaterial(Material):
    """
    Base class for materials using NIST temperature-dependent property data.

    This class maintains compatibility with the existing Material interface
    while providing more accurate temperature-dependent specific heat calculations.
    """
    _nist_cp_func: Optional[callable] = None
    _temperature_range: tuple = (4, 300)

    def __post_init__(self):
        """Initialize NIST-specific attributes after dataclass initialization."""
        if not hasattr(self, '_nist_cp_func'):
            self._nist_cp_func = None
        if not hasattr(self, '_temperature_range'):
            self._temperature_range = (4, 300)

    def determine_specific_heat(self, temperature: float) -> float:
        """
        Compute the specific heat using NIST data if available, fallback to Debye model.

        Args:
            temperature: Temperature in Kelvin

        Returns:
            Specific heat in J/(kg·K)
        """
        if self._nist_cp_func is not None:
            try:
                # Check if temperature is within NIST data range
                min_temp, max_temp = self._temperature_range
                if min_temp <= temperature <= max_temp:
                    return float(self._nist_cp_func(temperature))
                else:
                    print(f"Warning: Temperature {temperature}K outside NIST range {min_temp}-{max_temp}K, using Debye model")
            except Exception as e:
                print(f"Warning: NIST calculation failed at {temperature}K: {e}, using Debye model")

        # Fallback to original Debye model
        return super().determine_specific_heat(temperature)

    def set_temperature_range(self, min_temp: float, max_temp: float):
        """Set the valid temperature range for NIST data."""
        self._temperature_range = (min_temp, max_temp)

    def set_nist_function(self, nist_func: callable):
        """Set the NIST specific heat function."""
        self._nist_cp_func = nist_func


@dataclass
class NISTMetal(NISTMaterial, Metal):
    """Metal material class using NIST temperature-dependent data."""

    def __post_init__(self):
        super().__post_init__()
        self.type = "metal"  # Keep original type for compatibility

    @classmethod
    def aluminum_5083_nist(cls):
        """
        Create aluminum 5083 material using NIST temperature-dependent specific heat.

        This uses the polynomial fit data from aluminum5083_properties.py
        """
        failure_stress = 185e6  # Pa - aluminum 5083 equivalent
        density = 2650  # kg/m³ - from NIST data for Al 5083
        characteristic_temperature = 1500  # K - kept for fallback
        molecular_weight = 26.981539  # g/mol - aluminum

        material = cls(
            failure_stress=failure_stress,
            density=density,
            characteristic_temperature=characteristic_temperature,
            molecular_weight=molecular_weight
        )
        material.set_nist_function(aluminum_cp_nist)
        material.set_temperature_range(4, 300)  # NIST data range
        return material




@dataclass
class NISTComposite(NISTMaterial, Composite):
    """Composite material class using NIST temperature-dependent data."""

    winding_angle: float

    def __post_init__(self):
        super().__post_init__()
        self.type = "composite"  # Keep original type for compatibility

    def determine_specific_heat(self, temperature: float) -> float:
        """
        Compute the specific heat using NIST data for composites.

        For composite materials like G10, the Debye model is not physically meaningful
        since composites don't have uniform crystalline lattice structures.
        We use only NIST experimental data or extrapolation.

        Args:
            temperature: Temperature in Kelvin

        Returns:
            Specific heat in J/(kg·K)
        """
        if self._nist_cp_func is not None:
            try:
                # For composites, always try to use NIST data even outside range
                min_temp, max_temp = self._temperature_range
                if min_temp <= temperature <= max_temp:
                    return float(self._nist_cp_func(temperature))
                else:
                    # For composites, extrapolate using boundary values rather than Debye model
                    if temperature < min_temp:
                        print(f"Warning: Temperature {temperature}K below NIST range, using {min_temp}K value")
                        return float(self._nist_cp_func(min_temp))
                    else:  # temperature > max_temp
                        print(f"Warning: Temperature {temperature}K above NIST range, using {max_temp}K value")
                        return float(self._nist_cp_func(max_temp))
            except Exception as e:
                print(f"Error: NIST calculation failed at {temperature}K: {e}")
                # For composites, use a reasonable constant value rather than Debye model
                return 1000.0  # J/(kg·K) - typical composite specific heat

        # If no NIST function available, use a reasonable composite estimate
        print(f"Warning: No NIST data available for composite at {temperature}K, using estimate")
        return 1000.0  # J/(kg·K) - typical composite specific heat

    @classmethod
    def g10_nist(cls, winding_angle: float):
        """
        Create G10 composite material using NIST temperature-dependent specific heat.

        Args:
            winding_angle: Fiber winding angle in radians
        """
        failure_stress = 3.10264e+8  # Pa - original value
        density = 1800  # kg/m³ - original value
        characteristic_temperature = 1500.0  # K - original value for fallback
        molecular_weight = 12.01  # g/mol - original value

        material = cls(
            failure_stress=failure_stress,
            density=density,
            characteristic_temperature=characteristic_temperature,
            molecular_weight=molecular_weight,
            winding_angle=winding_angle
        )
        material.set_nist_function(g10_cp_nist)
        material.set_temperature_range(4, 300)  # NIST data range
        return material


def compare_material_models(temperature_range=(20, 300), num_points=50):
    """
    Compare original and NIST material models over temperature range.

    Args:
        temperature_range: (min_temp, max_temp) in Kelvin
        num_points: Number of temperature points to evaluate

    Returns:
        Dictionary with comparison data
    """
    import numpy as np

    temperatures = np.linspace(temperature_range[0], temperature_range[1], num_points)

    # Create material instances
    al_original = Metal.aluminum()
    al_nist = NISTMetal.aluminum_5083_nist()

    g10_original = Composite.g10(math.radians(55))
    g10_nist = NISTComposite.g10_nist(math.radians(55))

    # Compute specific heats
    al_orig_cp = [al_original.determine_specific_heat(T) for T in temperatures]
    al_nist_cp = [al_nist.determine_specific_heat(T) for T in temperatures]

    g10_orig_cp = [g10_original.determine_specific_heat(T) for T in temperatures]
    g10_nist_cp = [g10_nist.determine_specific_heat(T) for T in temperatures]

    return {
        'temperatures': temperatures,
        'aluminum_original': np.array(al_orig_cp),
        'aluminum_nist': np.array(al_nist_cp),
        'g10_original': np.array(g10_orig_cp),
        'g10_nist': np.array(g10_nist_cp)
    }


def demonstrate_thermal_capacity_impact():
    """
    Demonstrate the impact of using NIST data on thermal capacity calculations.
    """
    print("=== NIST MATERIAL THERMAL CAPACITY IMPACT ===\n")

    # Create materials
    al_original = Metal.aluminum()
    al_nist = NISTMetal.aluminum_5083_nist()

    g10_original = Composite.g10(math.radians(55))
    g10_nist = NISTComposite.g10_nist(math.radians(55))

    # Test conditions
    test_temps = [50, 77, 150, 200, 300]  # K
    liner_mass = 100  # kg
    tank_mass = 150  # kg

    print(f"{'Temp (K)':<8} {'Liner Impact (%)':<15} {'Tank Impact (%)':<15} {'Total Impact':<15}")
    print("-" * 65)

    for T in test_temps:
        # Thermal capacities
        liner_orig = al_original.determine_thermal_capacity(T, liner_mass)
        liner_nist = al_nist.determine_thermal_capacity(T, liner_mass)
        liner_impact = (liner_nist / liner_orig - 1) * 100

        tank_orig = g10_original.determine_thermal_capacity(T, tank_mass)
        tank_nist = g10_nist.determine_thermal_capacity(T, tank_mass)
        tank_impact = (tank_nist / tank_orig - 1) * 100

        total_orig = liner_orig + tank_orig
        total_nist = liner_nist + tank_nist
        total_impact = (total_nist / total_orig - 1) * 100

        impact_level = "Major" if abs(total_impact) > 20 else "Moderate" if abs(total_impact) > 5 else "Minor"

        print(f"{T:<8.0f} {liner_impact:<15.1f} {tank_impact:<15.1f} {total_impact:<15.1f}")

    print(f"\nConclusion: NIST data provides significantly different thermal capacity")
    print(f"calculations, especially at low temperatures where hydrogen tanks operate.")


if __name__ == "__main__":
    # Demonstrate the new material classes
    demonstrate_thermal_capacity_impact()

    print("\n=== EXAMPLE USAGE ===")
    print("# Replace current materials with NIST versions:")
    print("# Instead of: Metal.aluminum()")
    print("# Use: NISTMetal.aluminum_5083_nist()")
    print("# Instead of: Composite.g10(angle)")
    print("# Use: NISTComposite.g10_nist(angle)")

    print("\n=== INTEGRATION STEPS ===")
    print("1. Update verification_cch2.py to use NIST materials")
    print("2. Compare results with current model")
    print("3. Validate against experimental data if available")
    print("4. Gradually migrate other analysis scripts")
