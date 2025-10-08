"""
Carbon-Epoxy Composite NIST Properties for Multi-Tank Framework

Simplified interface to temperature-dependent properties using CSV data.
Uses linear interpolation for CSV data (10-88K) and linear extrapolation beyond 90K.

Source: Experimental data provided via CSV file
"""

import os
import csv
from typing import List, Tuple


class CarbonEpoxyProperties:
    """
    Carbon-epoxy composite properties using CSV data and interpolation.

    Data is loaded once and cached for efficient repeated lookups.
    """

    def __init__(self):
        self._data_loaded = False
        self._temperatures = []
        self._specific_heats = []
        self._load_data()

    def _load_data(self):
        """Load CSV data from carbon_epoxy_cps.csv file."""
        if self._data_loaded:
            return

        # Get path to CSV file (one directory up from this file)
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, '..', 'carbon_epoxy_cps.csv')

        try:
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row

                for row in reader:
                    temp = float(row[0])  # Temperature in K
                    cp = float(row[1])    # Specific heat in J/(kg·K)
                    self._temperatures.append(temp)
                    self._specific_heats.append(cp)

            self._data_loaded = True
            print(f"✅ Loaded {len(self._temperatures)} data points for carbon-epoxy specific heat")
            print(f"   Temperature range: {min(self._temperatures):.1f} - {max(self._temperatures):.1f} K")

        except FileNotFoundError:
            raise FileNotFoundError(f"Carbon-epoxy CSV data file not found at: {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading carbon-epoxy CSV data: {e}")

    def _linear_interpolate(self, temperature: float) -> float:
        """
        Linear interpolation between data points.

        Args:
            temperature: Temperature in K

        Returns:
            Interpolated specific heat in J/(kg·K)
        """
        # Handle edge cases
        if temperature <= self._temperatures[0]:
            return self._specific_heats[0]
        if temperature >= self._temperatures[-1]:
            # For temperatures beyond CSV data (>88K), use a more conservative approach
            # The specific heat trend becomes roughly linear at higher temperatures
            # Use a reduced slope to avoid unrealistically high extrapolated values

            # Get the last data point
            T_max = self._temperatures[-1]  # 88K
            cp_max = self._specific_heats[-1]  # ~8190 J/(kg·K)

            # For typical carbon-epoxy, Cp approaches ~1000-1500 J/(kg·K) at room temperature
            # Use a much more conservative linear extrapolation
            if temperature <= 300:  # Up to room temperature
                # Linear interpolation to realistic room temperature value
                cp_300k = 1200  # Typical carbon-epoxy Cp at 300K [J/(kg·K)]
                slope = (cp_300k - cp_max) / (300 - T_max)
                return cp_max + slope * (temperature - T_max)
            else:
                # Beyond room temperature, use constant value
                return 1200  # Constant Cp for high temperatures

        # Find bracketing indices
        for i in range(len(self._temperatures) - 1):
            if self._temperatures[i] <= temperature <= self._temperatures[i + 1]:
                T1, T2 = self._temperatures[i], self._temperatures[i + 1]
                cp1, cp2 = self._specific_heats[i], self._specific_heats[i + 1]

                # Linear interpolation
                fraction = (temperature - T1) / (T2 - T1)
                return cp1 + fraction * (cp2 - cp1)

        # Should never reach here
        raise ValueError(f"Temperature {temperature} K could not be interpolated")


# Create global instance for efficient data sharing
_carbon_epoxy_props = CarbonEpoxyProperties()


def specific_heat(temperature: float) -> float:
    """
    Specific heat of carbon-epoxy composite using CSV data interpolation.

    The data in the CSV file was computed using a rule of mixtures approach based on the reported data for graphite fibres [1] and epoxy resin [2].

    [1] DeSorbo, W.; Tyler, W. W. (1953): The Specific Heat of Graphite from 13° to 300°K. In The Journal of Chemical Physics 21 (10), pp. 1660–1663. DOI: 10.1063/1.1698640.
    [2] Hartwig, Gunther (1994): Polymer properties at room and cryogenic temperatures. New York, London: Springer (International cryogenics monograph series).

    Args:
        temperature (float): Temperature in Kelvin

    Returns:
        float: Specific heat in J/(kg·K)

    Valid range: 10-88K (CSV data), linear extrapolation beyond 90K
    """
    # Clamp temperature to reasonable range
    T = max(5.0, min(500.0, temperature))

    return _carbon_epoxy_props._linear_interpolate(T)


def thermal_conductivity(temperature: float) -> float:
    """
    Thermal conductivity of carbon-epoxy composite.

    Note: This is a placeholder implementation.
    TODO: Add proper thermal conductivity data/model.

    Args:
        temperature (float): Temperature in Kelvin

    Returns:
        float: Thermal conductivity in W/(m·K)
    """
    # Placeholder - typical carbon-epoxy values range from 0.5-20 W/(m·K)
    # depending on fiber orientation and volume fraction
    return 2.0  # Conservative estimate for structural composite


if __name__ == "__main__":
    # Test the functions
    test_temps = [10, 20, 30, 50, 77, 88, 100, 150, 300]  # K
    print("Carbon-Epoxy Composite NIST Properties:")
    print(f"{'Temp (K)':<10} {'Cp (J/kg·K)':<12} {'k (W/m·K)':<12}")
    print("-" * 40)

    for T in test_temps:
        cp = specific_heat(T)
        k = thermal_conductivity(T)
        print(f"{T:<10.0f} {cp:<12.1f} {k:<12.1f}")

    print(f"\n✅ Carbon-epoxy properties test complete!")