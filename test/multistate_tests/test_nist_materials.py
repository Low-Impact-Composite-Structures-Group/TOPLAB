"""
Test Suite for NIST Materials Framework

Tests the NISTMaterial class and associated functionality for compatibility
with existing thermal and structural model interfaces.

Test Coverage:
- Basic material creation and properties
- Temperature-dependent specific heat calculations
- Thermal capacity calculations
- Material registry functionality
- Compatibility with existing model interfaces
- Temperature range validation
- Comparison with original NIST data

Usage:
    pytest test/multi_tank_tests/test_nist_materials.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the new NIST materials framework
from src.materials.materials_for_multi_tank.nist_material import NISTMaterial, get_material_by_nist_path

# Import multi_tank NIST properties for comparison
from src.multi_tank.materials.nist_material_properties.aluminum6061t6_properties import specific_heat as al_original_cp
from src.multi_tank.materials.nist_material_properties.g10_properties import specific_heat as g10_original_cp


class TestNISTMaterialBasics:
    """Test basic material creation and properties."""

    def test_aluminum_properties(self):
        """Test aluminum 6061-T6 material properties."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        assert aluminum.density == 2700.0
        assert aluminum.failure_stress == 276e6
        assert aluminum.nist_path == "aluminum_6061T6_nist"
        assert aluminum.name == "Aluminum 6061-T6 (NIST)"

    def test_g10_properties(self):
        """Test G10 composite material properties."""
        g10 = NISTMaterial.g10_nist()

        assert g10.density == 1800.0
        assert g10.failure_stress == 310e6
        assert g10.nist_path == "g10_nist"
        assert g10.name == "G10 Composite (NIST)"

    def test_string_representations(self):
        """Test string and repr methods."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        str_repr = str(aluminum)
        assert "Aluminum 6061-T6 (NIST)" in str_repr
        assert "2700 kg/m³" in str_repr
        assert "276 MPa" in str_repr

        repr_str = repr(aluminum)
        assert "NISTMaterial" in repr_str
        assert "aluminum_6061T6_nist" in repr_str


class TestTemperatureDependentProperties:
    """Test temperature-dependent specific heat calculations."""

    @pytest.fixture
    def materials(self):
        """Create test materials."""
        return {
            'aluminum': NISTMaterial.aluminum_6061T6_nist(),
            'g10': NISTMaterial.g10_nist()
        }

    @pytest.mark.parametrize("temperature", [20, 77, 150, 300])
    def test_specific_heat_values(self, materials, temperature):
        """Test specific heat values at various temperatures."""
        for material in materials.values():
            cp = material.get_specific_heat(temperature)

            # Validate reasonable values (specific heat can be very low at cryogenic temperatures)
            assert 1 < cp < 5000, f"Unreasonable Cp value: {cp} at {temperature}K for {material.name}"

    def test_interface_consistency(self, materials):
        """Test that both interface methods return the same values."""
        temperature = 150.0

        for material in materials.values():
            cp1 = material.get_specific_heat(temperature)
            cp2 = material.determine_specific_heat(temperature)

            assert abs(cp1 - cp2) < 1e-6, "Interface methods don't match"

    def test_cryogenic_behavior(self, materials):
        """Test that materials behave correctly at cryogenic temperatures."""
        aluminum = materials['aluminum']

        # At very low temperatures, specific heat should be very low
        cp_20k = aluminum.get_specific_heat(20.0)
        cp_300k = aluminum.get_specific_heat(300.0)

        assert cp_20k < cp_300k, "Specific heat should increase with temperature"
        assert cp_20k < 50, "Cryogenic specific heat should be low"


class TestThermalCapacityCalculations:
    """Test thermal capacity calculations."""

    def test_thermal_capacity_calculation(self):
        """Test thermal capacity calculations for typical masses."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()
        g10 = NISTMaterial.g10_nist()

        # Test masses
        liner_mass = 100.0  # kg
        composite_mass = 150.0  # kg
        test_temp = 77.0  # K

        # Calculate thermal capacities
        liner_capacity = aluminum.determine_thermal_capacity(test_temp, liner_mass)
        composite_capacity = g10.determine_thermal_capacity(test_temp, composite_mass)

        # Validate calculations
        expected_liner = aluminum.get_specific_heat(test_temp) * liner_mass
        expected_composite = g10.get_specific_heat(test_temp) * composite_mass

        assert abs(liner_capacity - expected_liner) < 1e-6
        assert abs(composite_capacity - expected_composite) < 1e-6

        # Validate reasonable magnitudes
        assert 10000 < liner_capacity < 100000, f"Liner capacity unreasonable: {liner_capacity}"
        assert 10000 < composite_capacity < 100000, f"Composite capacity unreasonable: {composite_capacity}"


class TestMaterialRegistry:
    """Test material registry functionality."""

    def test_valid_material_paths(self):
        """Test material creation from valid NIST paths."""
        aluminum = get_material_by_nist_path("aluminum_6061T6_nist")
        g10 = get_material_by_nist_path("g10_nist")

        assert aluminum.nist_path == "aluminum_6061T6_nist"
        assert g10.nist_path == "g10_nist"

        assert isinstance(aluminum, NISTMaterial)
        assert isinstance(g10, NISTMaterial)

    def test_invalid_material_path(self):
        """Test that invalid paths raise appropriate errors."""
        with pytest.raises(ValueError, match="Unknown NIST path"):
            get_material_by_nist_path("nonexistent_material")

    def test_registry_completeness(self):
        """Test that all class methods are available in registry."""
        # These should not raise exceptions
        aluminum = get_material_by_nist_path("aluminum_6061T6_nist")
        g10 = get_material_by_nist_path("g10_nist")

        # Should be equivalent to class methods
        aluminum_direct = NISTMaterial.aluminum_6061T6_nist()
        g10_direct = NISTMaterial.g10_nist()

        assert aluminum.density == aluminum_direct.density
        assert g10.density == g10_direct.density


class TestModelCompatibility:
    """Test compatibility with existing thermal model interfaces."""

    def test_required_methods_exist(self):
        """Test that all required methods exist for thermal model compatibility."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        # These methods must exist for compatibility with thermal models
        assert hasattr(aluminum, 'determine_specific_heat')
        assert hasattr(aluminum, 'determine_thermal_capacity')
        assert hasattr(aluminum, 'get_specific_heat')

    def test_method_functionality(self):
        """Test that interface methods work correctly."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        temp = 150.0  # K
        mass = 50.0   # kg

        cp = aluminum.determine_specific_heat(temp)
        thermal_cap = aluminum.determine_thermal_capacity(temp, mass)

        assert cp > 0, "Specific heat should be positive"
        assert thermal_cap > 0, "Thermal capacity should be positive"
        assert abs(thermal_cap - cp * mass) < 1e-6, "Thermal capacity calculation inconsistent"


class TestTemperatureRangeValidation:
    """Test temperature range handling."""

    @pytest.mark.parametrize("temperature,expected_behavior", [
        (5.0, "clamped"),      # Below minimum
        (10.0, "valid"),       # At minimum
        (77.0, "valid"),       # Typical cryogenic
        (300.0, "valid"),      # At maximum
        (500.0, "clamped")     # Above maximum
    ])
    def test_temperature_clamping(self, temperature, expected_behavior):
        """Test temperature range clamping behavior."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        # Should not raise exceptions
        cp = aluminum.get_specific_heat(temperature)

        # Should return valid values
        assert cp > 0, f"Invalid Cp for {temperature}K: {cp}"

        # Extreme temperatures should be clamped
        if expected_behavior == "clamped":
            # Test that clamping occurred by comparing with edge values
            if temperature < 10:
                cp_10k = aluminum.get_specific_heat(10.0)
                assert abs(cp - cp_10k) < 1e-6, "Low temperature not clamped properly"
            elif temperature > 400:
                cp_400k = aluminum.get_specific_heat(400.0)
                assert abs(cp - cp_400k) < 1e-6, "High temperature not clamped properly"


class TestOriginalNISTComparison:
    """Compare with original NIST implementation."""

    @pytest.mark.parametrize("temperature", [50, 77, 150, 300])
    def test_aluminum_comparison(self, temperature):
        """Compare aluminum specific heat with original implementation."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        new_cp = aluminum.get_specific_heat(temperature)
        orig_cp = al_original_cp(temperature)

        # Should be very close (within 1%)
        diff_percent = abs(new_cp - orig_cp) / orig_cp * 100
        assert diff_percent < 1.0, f"Aluminum Cp difference too large: {diff_percent}% at {temperature}K"

    @pytest.mark.parametrize("temperature", [50, 77, 150, 300])
    def test_g10_comparison(self, temperature):
        """Compare G10 specific heat with original implementation."""
        g10 = NISTMaterial.g10_nist()

        new_cp = g10.get_specific_heat(temperature)
        orig_cp = g10_original_cp(temperature)

        # Should be very close (within 1%)
        diff_percent = abs(new_cp - orig_cp) / orig_cp * 100
        assert diff_percent < 1.0, f"G10 Cp difference too large: {diff_percent}% at {temperature}K"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_typical_tank_configuration(self):
        """Test a typical tank configuration scenario."""
        # Create materials as they would be used in the orchestrator
        liner_material = get_material_by_nist_path("aluminum_6061T6_nist")
        composite_material = get_material_by_nist_path("g10_nist")

        # Typical tank parameters
        liner_mass = 100.0      # kg
        composite_mass = 150.0  # kg
        operating_temp = 77.0   # K (liquid nitrogen temperature)

        # Calculate thermal capacities
        liner_capacity = liner_material.determine_thermal_capacity(operating_temp, liner_mass)
        composite_capacity = composite_material.determine_thermal_capacity(operating_temp, composite_mass)
        total_structural_capacity = liner_capacity + composite_capacity

        # Validate realistic values for cryogenic tank
        assert 30000 < liner_capacity < 40000, f"Liner capacity unrealistic: {liner_capacity} J/K"
        assert 30000 < composite_capacity < 40000, f"Composite capacity unrealistic: {composite_capacity} J/K"
        assert 60000 < total_structural_capacity < 80000, f"Total capacity unrealistic: {total_structural_capacity} J/K"

    def test_temperature_sweep(self):
        """Test behavior across full operating temperature range."""
        aluminum = NISTMaterial.aluminum_6061T6_nist()

        temperatures = [20, 50, 77, 100, 150, 200, 250, 300]
        specific_heats = [aluminum.get_specific_heat(T) for T in temperatures]

        # Specific heat should generally increase with temperature
        for i in range(1, len(specific_heats)):
            assert specific_heats[i] >= specific_heats[i-1], \
                f"Specific heat decreased from {temperatures[i-1]}K to {temperatures[i]}K"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])