"""
Cryopump model for hydrogen refueling simulations.

This module contains models for simulating cryopumps used in hydrogen refueling
operations. It provides functions to compute hydrogen properties at pump outlet
based on isentropic pump model with efficiency.

Fuel Tank - Cryopump Models
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022-2025
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
from functools import lru_cache

from CoolProp.CoolProp import PropsSI
from src.fluids.hydrogen_retrievers import HydrogenRetriever, SinglePhaseRequester

# Default parameters for the cryopump model
DEFAULT_RESERVOIR_PRESSURE = 2.0e5  # 2 bar in Pa
DEFAULT_PUMP_EFFICIENCY = 0.78     # 78% isentropic efficiency


@dataclass
class CryopumpParameters:
    """Parameters for cryopump operation."""
    reservoir_pressure: float = DEFAULT_RESERVOIR_PRESSURE
    efficiency: float = DEFAULT_PUMP_EFFICIENCY


class CryopumpModel:
    """
    Model for a cryogenic pump used in hydrogen refueling simulations.

    The model calculates the thermodynamic properties of hydrogen at the pump outlet
    based on an isentropic efficiency model.
    """

    def __init__(self, parameters: Optional[CryopumpParameters] = None, enable_cache: bool = True):
        """
        Initialize the cryopump model with the given parameters.

        Args:
            parameters: Parameters for the cryopump operation. If None, default parameters are used.
            enable_cache: Whether to enable caching of computation results. Default is True.
        """
        self.parameters = parameters or CryopumpParameters()
        self.enable_cache = enable_cache

        # Create a cached version of the computation function
        self._cached_compute = lru_cache(maxsize=128)(self._compute_pump_outlet_hydrogen)

    def compute_pump_outlet_hydrogen(self, tank_pressure: float):
        """
        Computes the hydrogen properties at pump outlet based on isentropic pump model with efficiency.
        Uses caching if enabled.

        Args:
            tank_pressure: The current pressure in the tank (Pa)

        Returns:
            Union[Hydrogen, TwoPhaseHydrogen]: Hydrogen property object at pump outlet conditions
        """
        if not self.enable_cache:
            return self._compute_pump_outlet_hydrogen(tank_pressure)

        # Use the cached version if caching is enabled
        return self._cached_compute(tank_pressure)

    def _compute_pump_outlet_hydrogen(self, tank_pressure: float):
        """
        Internal computation function for hydrogen properties at pump outlet.

        Args:
            tank_pressure: The current pressure in the tank (Pa)

        Returns:
            Union[Hydrogen, TwoPhaseHydrogen]: Hydrogen property object at pump outlet conditions
        """
        reservoir_pressure = self.parameters.reservoir_pressure
        eta = self.parameters.efficiency

        # Get saturated liquid properties at reservoir_pressure (always use saturated liquid from reservoir)
        try:
            h_liq_in = PropsSI("H", "P", reservoir_pressure, "Q", 0, "hydrogen")
            s_liq_in = PropsSI("S", "P", reservoir_pressure, "Q", 0, "hydrogen")
        except ValueError as e:
            # If we can't get saturated properties at the reservoir pressure
            # Use properties near the critical point
            print(f"Using fallback reservoir properties due to error: {str(e)}")
            # Use properties at 3 bar (default)
            h_liq_in = PropsSI("H", "P", 3.0e5, "Q", 0, "hydrogen")
            s_liq_in = PropsSI("S", "P", 3.0e5, "Q", 0, "hydrogen")

        # Isentropic enthalpy at tank_pressure (ideal pump work)
        # Try to calculate isentropic enthalpy - this is where the error occurs
        try:
            h_liq_isentropic = PropsSI("H", "P", tank_pressure, "S", s_liq_in, "hydrogen")
        except ValueError as e:
            # If we get a flash error, use a simpler model to approximate the enthalpy change
            # This is a simplified model for when CoolProp can't solve the flash calculation
            print(f"Using simplified pump model due to flash calculation error at P={tank_pressure/1e5:.2f}bar")
            # Simple work calculation: v * (P2 - P1) - approximate isentropic pump work
            v_liq = 1.0 / PropsSI("D", "P", reservoir_pressure, "Q", 0, "hydrogen")
            h_liq_isentropic = h_liq_in + v_liq * (tank_pressure - reservoir_pressure)

        # Real enthalpy after pump (using efficiency)
        # For a pump, the real work is greater than isentropic work, so divide by efficiency
        h_liq_out = h_liq_in + (h_liq_isentropic - h_liq_in) / eta

        # Find temperature at tank_pressure and h_liq_out
        try:
            t_liq_out = PropsSI("T", "P", tank_pressure, "H", h_liq_out, "hydrogen")
        except ValueError as e:
            # If we can't get temperature from enthalpy, estimate it
            print(f"Using estimated temperature due to calculation error")
            # Estimate temperature change based on enthalpy change and a typical Cp value
            t_liq_in = PropsSI("T", "P", reservoir_pressure, "Q", 0, "hydrogen")
            cp_avg = 12000.0  # J/kg-K - approximate for liquid hydrogen
            t_liq_out = t_liq_in + (h_liq_out - h_liq_in) / cp_avg

        # Print debug information
        print(f"Pump outlet conditions: P={tank_pressure/1e5:.2f}bar, T={t_liq_out:.2f}K")

        # Use HydrogenRetriever which will automatically determine the correct phase
        try:
            hydrogen = HydrogenRetriever().get_hydrogen_properties(tank_pressure, t_liq_out)
        except ValueError as e:
            # If HydrogenRetriever fails, fallback to SinglePhaseRequester
            print(f"Using fallback SinglePhaseRequester due to error: {str(e)}")
            try:
                hydrogen = SinglePhaseRequester().get_hydrogen_properties(tank_pressure, t_liq_out)
            except ValueError:
                # Last resort: use properties at a safe condition near the actual point
                print(f"Using last-resort property calculation")
                # Try slightly different temperature to avoid edge cases
                t_safe = t_liq_out * 1.05
                hydrogen = SinglePhaseRequester().get_hydrogen_properties(tank_pressure, t_safe)

        # Print debug information about the hydrogen phase
        if hasattr(hydrogen, 'phase'):
            print(f"Hydrogen phase: {hydrogen.phase}")
        elif hasattr(hydrogen, 'liquid') and hasattr(hydrogen, 'gas'):
            print(f"Two-phase hydrogen")
        else:
            print(f"Unknown hydrogen type: {type(hydrogen)}")

        return hydrogen

    def clear_cache(self):
        """Clear the computation cache."""
        if hasattr(self._cached_compute, 'cache_clear'):
            self._cached_compute.cache_clear()

    def get_cache_info(self):
        """
        Get information about the cache usage.

        Returns:
            Dict: Information about cache hits, misses and size
        """
        if not self.enable_cache:
            return {"enabled": False}

        if hasattr(self._cached_compute, 'cache_info'):
            info = self._cached_compute.cache_info()
            return {
                "enabled": True,
                "hits": info.hits,
                "misses": info.misses,
                "maxsize": info.maxsize,
                "currsize": info.currsize,
                "hit_ratio": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
            }
        return {"enabled": True, "details": "Cache info not available"}


# Create a default instance for easy access to the compute function
default_cryopump = CryopumpModel()
compute_pump_outlet_hydrogen = default_cryopump.compute_pump_outlet_hydrogen


def main():
    """Test function for the cryopump model."""
    # Test the cryopump model with different pressures
    test_pressures = [10e5, 100e5, 200e5, 300e5, 400e5]

    print("Testing default cryopump with caching:")
    for pressure in test_pressures:
        hydrogen = compute_pump_outlet_hydrogen(pressure)
        print(f"Tested at P={pressure/1e5:.1f} bar")
        print("-" * 30)

    # Test cache hits by requesting the same pressures again
    print("\nTesting cache hits (should be faster and show fewer debug messages):")
    for pressure in test_pressures:
        hydrogen = compute_pump_outlet_hydrogen(pressure)
        print(f"Tested at P={pressure/1e5:.1f} bar (from cache)")

    print(f"\nCache info: {default_cryopump.get_cache_info()}")

    # Test with custom parameters and disabled cache
    custom_params = CryopumpParameters(
        reservoir_pressure=5.0e5,  # 5 bar
        efficiency=0.85            # 85% efficiency
    )
    custom_cryopump = CryopumpModel(custom_params, enable_cache=True)

    print("\nTesting with custom parameters:")
    hydrogen = custom_cryopump.compute_pump_outlet_hydrogen(200e5)
    print(f"Custom cryopump tested at 200 bar")

    # Test cache clearing
    print("\nTesting cache clearing:")
    default_cryopump.clear_cache()
    print(f"Cache after clearing: {default_cryopump.get_cache_info()}")

    # Test without caching
    print("\nTesting without cache:")
    no_cache_cryopump = CryopumpModel(enable_cache=False)
    hydrogen = no_cache_cryopump.compute_pump_outlet_hydrogen(300e5)
    print(f"No-cache cryopump tested at 300 bar")

    # Show cache is disabled
    print(f"No-cache info: {no_cache_cryopump.get_cache_info()}")


if __name__ == "__main__":
    main()
