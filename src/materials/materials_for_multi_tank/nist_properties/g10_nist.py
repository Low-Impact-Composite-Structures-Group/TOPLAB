"""
G10 NIST Properties for Multi-Tank Framework

Simplified interface to NIST temperature-dependent properties.
Based on NIST polynomial fits for temperature range 10-400K.

Source: NIST Cryogenic Material Properties Database
"""

import math


def specific_heat(temperature):
    """
    Specific heat of G10 using NIST polynomial fit.

    Args:
        temperature (float): Temperature in Kelvin (10-400K range)

    Returns:
        float: Specific heat in J/(kg·K)

    Valid range: 4-300K (data), extended to 400K for hydrogen applications
    """
    # Clamp temperature to valid range
    T = max(10.0, min(400.0, temperature))

    # NIST coefficients for G10 specific heat
    a = -2.4083
    b = 7.6006
    c = -8.2982
    d = 7.3301
    e = -4.2386
    f = 1.4294
    g = -0.24396
    h = 0.015236
    i = 0

    log_T = math.log10(T)
    log_cp = (a + b*log_T + c*log_T**2 + d*log_T**3 + e*log_T**4 +
             f*log_T**5 + g*log_T**6 + h*log_T**7 + i*log_T**8)

    return 10**log_cp


def thermal_conductivity_normal(temperature):
    """
    Thermal conductivity of G10 in normal direction using NIST polynomial fit.

    Args:
        temperature (float): Temperature in Kelvin (10-400K range)

    Returns:
        float: Thermal conductivity in W/(m·K)

    Valid range: 10-300K (equation), extended to 400K for hydrogen applications
    """
    # Clamp temperature to valid range
    T = max(10.0, min(400.0, temperature))

    # NIST coefficients for G10 thermal conductivity (normal direction)
    a = -4.1236
    b = 13.788
    c = -26.068
    d = 26.272
    e = -14.663
    f = 4.4954
    g = -0.6905
    h = 0.0397
    i = 0

    log_T = math.log10(T)
    log_k = (a + b*log_T + c*log_T**2 + d*log_T**3 + e*log_T**4 +
            f*log_T**5 + g*log_T**6 + h*log_T**7 + i*log_T**8)

    return 10**log_k


def thermal_conductivity_warp(temperature):
    """
    Thermal conductivity of G10 in warp direction using NIST polynomial fit.

    Args:
        temperature (float): Temperature in Kelvin (10-400K range)

    Returns:
        float: Thermal conductivity in W/(m·K)

    Valid range: 12-300K (equation), extended to 400K for hydrogen applications
    """
    # Clamp temperature to valid range
    T = max(12.0, min(400.0, temperature))

    # NIST coefficients for G10 thermal conductivity (warp direction)
    a = -2.64827
    b = 8.80228
    c = -24.8998
    d = 41.1625
    e = -39.8754
    f = 23.1778
    g = -7.95635
    h = 1.48806
    i = -0.11701

    log_T = math.log10(T)
    log_k = (a + b*log_T + c*log_T**2 + d*log_T**3 + e*log_T**4 +
            f*log_T**5 + g*log_T**6 + h*log_T**7 + i*log_T**8)

    return 10**log_k


# Default to normal direction for structural applications
def thermal_conductivity(temperature):
    """Default thermal conductivity (normal direction) for structural applications."""
    return thermal_conductivity_normal(temperature)


if __name__ == "__main__":
    # Test the functions
    test_temps = [10, 50, 77, 150, 300, 400]  # K
    print("G10 NIST Properties:")
    print(f"{'Temp (K)':<10} {'Cp (J/kg·K)':<12} {'k_normal (W/m·K)':<17} {'k_warp (W/m·K)':<16}")
    print("-" * 60)

    for T in test_temps:
        cp = specific_heat(T)
        k_n = thermal_conductivity_normal(T)
        k_w = thermal_conductivity_warp(T)
        print(f"{T:<10.0f} {cp:<12.1f} {k_n:<17.6f} {k_w:<16.6f}")