"""
Aluminum 6061-T6 NIST Properties for Multi-Tank Framework

Simplified interface to NIST temperature-dependent properties.
Based on NIST polynomial fits for temperature range 10-400K.

Source: NIST Cryogenic Material Properties Database
"""

import math


def specific_heat(temperature):
    """
    Specific heat of aluminum 6061-T6 using NIST polynomial fit.

    Args:
        temperature (float): Temperature in Kelvin (10-400K range)

    Returns:
        float: Specific heat in J/(kg·K)

    Valid range: 4-300K (data), extended to 400K for hydrogen applications
    Curve fit error: 5% relative to data
    """
    # Clamp temperature to valid range
    T = max(10.0, min(400.0, temperature))

    # NIST coefficients for aluminum 6061-T6 specific heat
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


def thermal_conductivity(temperature):
    """
    Thermal conductivity of aluminum 6061-T6 using NIST polynomial fit.

    Args:
        temperature (float): Temperature in Kelvin (10-400K range)

    Returns:
        float: Thermal conductivity in W/(m·K)

    Valid range: 1-300K (equation), extended to 400K for hydrogen applications
    Curve fit error: 0.5% relative to data
    """
    # Clamp temperature to valid range
    T = max(10.0, min(400.0, temperature))

    # NIST coefficients for aluminum 6061-T6 thermal conductivity
    a = 0.07918
    b = 1.0957
    c = -0.07277
    d = 0.08084
    e = 0.02803
    f = -0.09464
    g = 0.04179
    h = -0.00571
    i = 0

    log_T = math.log10(T)
    log_k = (a + b*log_T + c*log_T**2 + d*log_T**3 + e*log_T**4 +
            f*log_T**5 + g*log_T**6 + h*log_T**7 + i*log_T**8)

    return 10**log_k


if __name__ == "__main__":
    # Test the functions
    test_temps = [10, 50, 77, 150, 300, 400]  # K
    print("Aluminum 6061-T6 NIST Properties:")
    print(f"{'Temp (K)':<10} {'Cp (J/kg·K)':<12} {'k (W/m·K)':<12}")
    print("-" * 40)

    for T in test_temps:
        cp = specific_heat(T)
        k = thermal_conductivity(T)
        print(f"{T:<10.0f} {cp:<12.1f} {k:<12.3f}")