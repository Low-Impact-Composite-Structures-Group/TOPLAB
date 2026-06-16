"""Approximate carbon-epoxy cryogenic property correlations used by multistate."""


def specific_heat(temperature: float) -> float:
    temp = max(5.0, min(500.0, float(temperature)))
    if temp <= 50.0:
        return 50.0 + 8.0 * temp
    if temp <= 150.0:
        return 450.0 + 5.0 * (temp - 50.0)
    return min(1200.0, 950.0 + 2.5 * (temp - 150.0))


def thermal_conductivity(temperature: float) -> float:
    _ = temperature
    return 2.0