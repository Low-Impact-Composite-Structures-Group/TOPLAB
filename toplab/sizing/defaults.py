"""Fallback values used when a tank spec omits an optional field (single source of truth)."""

INITIAL_TEMPERATURE_K = 53.25
FILL_FRACTION = 0.90

# Wall-design pressure when ``venting_pressure`` is not given, per sizing route.
MISSION_SIZED_OPERATING_PRESSURE_PA = 450e5

# Simulation limits derived from the initial pressure when not given.
VENTING_PRESSURE_FACTOR = 1.125
MINIMUM_PRESSURE_FACTOR = 0.1
MINIMUM_PRESSURE_CAP_PA = 15e5
