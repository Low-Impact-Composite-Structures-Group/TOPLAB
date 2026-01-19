"""
Test cases for the cryopump model module.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.dynamics.cryopump_model import (
    CryoPumpModel,
    CryopumpParameters
)


class TestCryopumpParameters(unittest.TestCase):
    """Test the cryopump parameters dataclass."""

    def test_default_parameters(self):
        """Test the default parameters."""
        params = CryopumpParameters()
        self.assertEqual(params.reservoir_pressure, 3.0e5)
        self.assertEqual(params.efficiency, 0.78)

    def test_custom_parameters(self):
        """Test custom parameters."""
        params = CryopumpParameters(reservoir_pressure=5.0e5, efficiency=0.85)
        self.assertEqual(params.reservoir_pressure, 5.0e5)
        self.assertEqual(params.efficiency, 0.85)


class TestCryopumpModel(unittest.TestCase):
    """Test the cryopump model."""

    def setUp(self):
        """Set up for each test."""
        # Create test parameters
        self.test_params = CryopumpParameters(reservoir_pressure=4.0e5, efficiency=0.80)
        # Mock HydrogenRetriever and SinglePhaseRequester to avoid actual calls to CoolProp
        self.hydrogen_mock = MagicMock()
        self.patcher1 = patch('src.dynamics.cryopump_model.HydrogenRetriever')
        self.retriever_mock = self.patcher1.start()
        self.retriever_instance = MagicMock()
        self.retriever_mock.return_value = self.retriever_instance
        self.retriever_instance.get_hydrogen_properties.return_value = self.hydrogen_mock

        # Also mock PropsSI
        self.patcher2 = patch('src.dynamics.cryopump_model.PropsSI')
        self.props_mock = self.patcher2.start()

        # Setup PropsSI mock to return reasonable values
        def mock_props_si(prop, *args):
            if prop == "H":
                return 1e6  # Enthalpy in J/kg
            elif prop == "S":
                return 1e3  # Entropy in J/kg-K
            elif prop == "D":
                return 70.0  # Density in kg/m^3
            elif prop == "T":
                return 20.0  # Temperature in K
            return 0.0

        self.props_mock.side_effect = mock_props_si

    def tearDown(self):
        """Tear down after each test."""
        self.patcher1.stop()
        self.patcher2.stop()

    def test_init(self):
        """Test initialization of the cryopump model."""
        # With default parameters
        model = CryoPumpModel()
        self.assertEqual(model.parameters.reservoir_pressure, 3.0e5)
        self.assertEqual(model.parameters.efficiency, 0.78)
        self.assertTrue(model.enable_cache)

        # With custom parameters
        model = CryoPumpModel(self.test_params, enable_cache=False)
        self.assertEqual(model.parameters.reservoir_pressure, 4.0e5)
        self.assertEqual(model.parameters.efficiency, 0.80)
        self.assertFalse(model.enable_cache)

    def test_compute_pump_outlet_hydrogen(self):
        """Test computation of pump outlet hydrogen."""
        model = CryoPumpModel(self.test_params)

        # First call should compute
        result = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result, self.hydrogen_mock)

        # Verify the retriever was called with the right parameters
        self.retriever_instance.get_hydrogen_properties.assert_called_once()

        # Reset the mock to check caching
        self.retriever_instance.get_hydrogen_properties.reset_mock()

        # Second call with the same pressure should use cache
        result = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result, self.hydrogen_mock)

        # Verify the retriever was NOT called again (using cache)
        self.retriever_instance.get_hydrogen_properties.assert_not_called()

    def test_compute_without_cache(self):
        """Test computation without caching."""
        model = CryoPumpModel(self.test_params, enable_cache=False)

        # First call should compute
        result = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result, self.hydrogen_mock)

        # Reset the mock to check that caching is disabled
        self.retriever_instance.get_hydrogen_properties.reset_mock()

        # Second call should compute again (no cache)
        result = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result, self.hydrogen_mock)

        # Verify the retriever was called again (no caching)
        self.retriever_instance.get_hydrogen_properties.assert_called_once()

    def test_clear_cache(self):
        """Test clearing the cache."""
        model = CryoPumpModel(self.test_params)

        # First call should compute
        result = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result, self.hydrogen_mock)

        # Clear the cache
        model.clear_cache()

        # Reset the mock to check that cache was cleared
        self.retriever_instance.get_hydrogen_properties.reset_mock()

        # Next call should compute again (cache cleared)
        result = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result, self.hydrogen_mock)

        # Verify the retriever was called again (cache was cleared)
        self.retriever_instance.get_hydrogen_properties.assert_called_once()

    def test_cache_different_pressures(self):
        """Test that different pressures create different cache entries."""
        model = CryoPumpModel(self.test_params)

        # Call with first pressure
        result1 = model.compute_pump_outlet_hydrogen(100e5)
        self.assertEqual(result1, self.hydrogen_mock)

        # Reset the mock
        self.retriever_instance.get_hydrogen_properties.reset_mock()

        # Call with different pressure - should not use cache
        result2 = model.compute_pump_outlet_hydrogen(200e5)
        self.assertEqual(result2, self.hydrogen_mock)

        # Verify the retriever was called again (different pressure = cache miss)
        self.retriever_instance.get_hydrogen_properties.assert_called_once()

    def test_get_cache_info(self):
        """Test getting cache information."""
        model = CryoPumpModel(self.test_params)

        # Initially cache should be empty
        cache_info = model.get_cache_info()
        self.assertTrue(cache_info["enabled"])
        self.assertEqual(cache_info["hits"], 0)
        self.assertEqual(cache_info["misses"], 0)

        # Make a call to populate cache
        model.compute_pump_outlet_hydrogen(100e5)

        # Check cache info after first call (should have 1 miss)
        cache_info = model.get_cache_info()
        self.assertTrue(cache_info["enabled"])
        self.assertEqual(cache_info["hits"], 0)
        self.assertEqual(cache_info["misses"], 1)

        # Make the same call again (should be a hit)
        model.compute_pump_outlet_hydrogen(100e5)

        # Check cache info after second call (should have 1 hit, 1 miss)
        cache_info = model.get_cache_info()
        self.assertTrue(cache_info["enabled"])
        self.assertEqual(cache_info["hits"], 1)
        self.assertEqual(cache_info["misses"], 1)

        # Test with disabled cache
        no_cache_model = CryoPumpModel(enable_cache=False)
        cache_info = no_cache_model.get_cache_info()
        self.assertFalse(cache_info["enabled"])


class TestModuleFunctions(unittest.TestCase):
    """Test for module-level functions."""

    def test_module_compute_function_exists(self):
        """Simple test to verify the module-level compute function exists."""
        from src.dynamics.cryopump_model import compute_pump_outlet_hydrogen
        self.assertTrue(callable(compute_pump_outlet_hydrogen))

    def test_module_level_caching(self):
        """Test that caching works with the module-level function."""
        # Import directly to ensure we're using the actual function and instance
        from src.dynamics.cryopump_model import (
            compute_pump_outlet_hydrogen,
            default_cryopump
        )

        # Clear cache to start fresh
        default_cryopump.clear_cache()

        # Get initial cache info
        initial_info = default_cryopump.get_cache_info()
        initial_hits = initial_info.get("hits", 0)
        initial_misses = initial_info.get("misses", 0)

        # Make first call to the same pressure - should be a cache miss
        result1 = compute_pump_outlet_hydrogen(123.45e5)  # Use a unique pressure

        # Get cache info after first call
        after_first_call = default_cryopump.get_cache_info()
        self.assertEqual(after_first_call["misses"], initial_misses + 1,
                         "Cache misses should increase after first call")

        # Make second call to the same pressure - should be a cache hit
        result2 = compute_pump_outlet_hydrogen(123.45e5)

        # Get cache info after second call
        after_second_call = default_cryopump.get_cache_info()
        self.assertEqual(after_second_call["hits"], initial_hits + 1,
                        "Cache hits should increase after second call")

        # Verify we got the same result object both times
        self.assertIs(result1, result2,
                     "Same result object should be returned from cache")
if __name__ == '__main__':
    unittest.main()
