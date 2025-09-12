"""
Aluminum 3003F Temperature-Dependent Properties from NIST Data

This module provides polynomial functions for thermal conductivity and specific heat
of aluminum 3003F based on NIST curve fits.

Data range: 4-300K
Equation range: 1-300K (thermal conductivity), 4-300K (specific heat)
Curve fit error: 2% (thermal conductivity), 5% (specific heat) relative to data

The polynomial form used is:
log10(property) = a + b*log10(T) + c*log10(T)^2 + ... + i*log10(T)^8

where T is temperature in Kelvin.
"""

import math


def thermal_conductivity(T):
    """
    Thermal conductivity of aluminum 3003F using NIST polynomial fit.

    Args:
        T: Temperature in Kelvin

    Returns:
        Thermal conductivity in W/(m·K)

    Valid range: 1-300K (equation range), 4-300K (data range)
    Curve fit error: 2% relative to data
    """
    # Coefficients from NIST table
    a = 0.63736
    b = -1.1437
    c = 7.4624
    d = -12.6905
    e = 11.9165
    f = -6.18721
    g = 1.63939
    h = -0.172667
    i = 0

    log_T = math.log10(T)
    log_k = (a + b*log_T + c*log_T**2 + d*log_T**3 + e*log_T**4 +
            f*log_T**5 + g*log_T**6 + h*log_T**7 + i*log_T**8)

    return 10**log_k


def specific_heat(T):
    """
    Specific heat of aluminum 3003F using NIST polynomial fit.

    Args:
        T: Temperature in Kelvin

    Returns:
        Specific heat in J/(kg·K)

    Valid range: 4-300K (both data and equation range)
    Curve fit error: 5% relative to data
    """
    # Coefficients from NIST table
    a = 46.6467
    b = -314.292
    c = 866.662
    d = -1298.3
    e = 1162.27
    f = -637.795
    g = 210.351
    h = -38.3094
    i = 2.96344

    log_T = math.log10(T)
    log_cp = (a + b*log_T + c*log_T**2 + d*log_T**3 + e*log_T**4 +
             f*log_T**5 + g*log_T**6 + h*log_T**7 + i*log_T**8)

    return 10**log_cp


if __name__ == "__main__":
    # Test the functions at various temperatures
    test_temps = [4, 20, 77, 150, 300]  # K

    print("Aluminum 3003F Properties Test:")
    print("=" * 50)
    print(f"{'Temp (K)':<10} {'k (W/m·K)':<12} {'Cp (J/kg·K)':<12}")
    print("-" * 50)

    for T in test_temps:
        k = thermal_conductivity(T)
        cp = specific_heat(T)
        print(f"{T:<10.1f} {k:<12.3f} {cp:<12.1f}")

    print("\nValid ranges:")
    print("Thermal conductivity: 1-300K (equation), 4-300K (data)")
    print("Specific heat: 4-300K")
    print("Curve fit errors: 2% (k), 5% (Cp) relative to data")
