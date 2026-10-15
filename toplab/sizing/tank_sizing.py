"""
Pure tank sizing.

``size_tank`` turns a :class:`~toplab.config.TankSpec` and a fuel requirement into a
:class:`TankDesign`: the capsule geometry plus the resolved initial state and
pressure limits the simulation starts from. It does not print, read global state
or mutate its inputs; anything worth reporting is returned in ``TankDesign.notes``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from CoolProp.CoolProp import PropsSI

from toplab.config import ConfigError, TankSpec
from toplab.sizing import defaults


@dataclass(frozen=True)
class TankDesign:
    geometry: Any                       # CapsuleTank
    method: str                         # 'radius' | 'mission'
    initial_pressure: float             # [Pa]
    initial_temperature: float          # [K]
    initial_density: Optional[float]    # [kg/m^3]; None if not configured or derived
    venting_pressure: float             # [Pa]
    minimum_pressure: float             # [Pa]
    initial_mass: Optional[float] = None  # [kg], only if the spec fixes it
    required_fuel: Optional[float] = None  # [kg], mission-sized tanks only
    notes: Tuple[str, ...] = ()


def size_tank(spec: TankSpec, required_fuel: Optional[float] = None) -> TankDesign:
    """
    Size one tank.

    A tank with an explicit ``geometry.radius`` keeps it. Otherwise the radius is
    chosen so that ``required_fuel`` (or ``geometry.fuel_mass`` if given) fills
    ``fill_fraction`` of the volume at the initial state.
    """
    from toplab.materials.nist_materials import NISTMaterial
    from toplab.tank_design.tank_shapes import CapsuleTank

    data = {k: v for k, v in spec.geometry_data().items() if v is not None}
    fluid = spec.coolprop
    notes = []

    if "initial_pressure" not in data:
        raise ConfigError(f"Tank {spec.node_id!r}: initial_conditions.pressure is required")
    pressure = float(data["initial_pressure"])
    phi = float(data.get("phi", 0.0))
    venting = data.get("venting_pressure")
    material = NISTMaterial.aluminum_6061T6_nist()
    fuel = None

    if "radius" in data:
        method = "radius"
        radius = float(data["radius"])
        operating_pressure = float(venting) if venting is not None else pressure
        if "initial_temperature" in data:
            temperature = float(data["initial_temperature"])
        else:
            temperature, _ = _initial_temperature(data, pressure, fluid, notes)
        notes.append(f"radius {radius:g} m given")
        density = float(data["initial_density"]) if "initial_density" in data else None
    else:
        method = "mission"
        fuel = float(data["fuel_mass"]) if "fuel_mass" in data else required_fuel
        if fuel is None or fuel <= 0.0:
            raise ConfigError(
                f"Tank {spec.node_id!r}: cannot size from a fuel requirement of {fuel}. "
                f"Give geometry.radius or geometry.fuel_mass, or size from a discharge mission."
            )
        if venting is None:
            operating_pressure = defaults.MISSION_SIZED_OPERATING_PRESSURE_PA
            notes.append(f"venting_pressure not set; wall designed for {operating_pressure / 1e5:.0f} bar")
        else:
            operating_pressure = float(venting)

        temperature, density = _initial_temperature(data, pressure, fluid, notes, need_density=True)
        fill = float(data.get("fill_fraction", defaults.FILL_FRACTION))
        volume = fuel / (density * fill)
        # V = pi r^3 (phi + 4/3), phi = cylinder length / radius
        radius = (volume / (math.pi * (max(phi, 0.0) + 4.0 / 3.0))) ** (1.0 / 3.0)
        notes.append(f"sized for {fuel:.2f} kg at {density:.2f} kg/m^3, fill {fill:.0%}: "
                     f"V={volume:.4f} m^3, r={radius:.3f} m")

    geometry = CapsuleTank(radius=radius, material=material, operating_pressure=operating_pressure, phi=phi)

    if "minimum_pressure" in data:
        minimum_pressure = float(data["minimum_pressure"])
    else:
        minimum_pressure = min(defaults.MINIMUM_PRESSURE_CAP_PA, pressure * defaults.MINIMUM_PRESSURE_FACTOR)
        notes.append(f"minimum_pressure not set; using {minimum_pressure / 1e5:.1f} bar")

    return TankDesign(
        geometry=geometry,
        method=method,
        initial_pressure=pressure,
        initial_temperature=temperature,
        initial_density=density,
        venting_pressure=float(venting) if venting is not None else pressure * defaults.VENTING_PRESSURE_FACTOR,
        minimum_pressure=minimum_pressure,
        initial_mass=data.get("initial_mass"),
        required_fuel=fuel,
        notes=tuple(notes),
    )


def _initial_temperature(data, pressure, fluid, notes, need_density=False):
    """Resolve (temperature, density) from the configured initial conditions."""
    configured_t = data.get("initial_temperature")
    if "initial_density" in data:
        density = float(data["initial_density"])
        try:
            return float(PropsSI("T", "P", pressure, "D", density, fluid)), density
        except Exception as exc:
            notes.append(f"could not solve T from P={pressure / 1e5:.0f} bar, rho={density:g}: {exc}")
            temperature = float(configured_t) if configured_t is not None else defaults.INITIAL_TEMPERATURE_K
            return temperature, float(PropsSI("D", "P", pressure, "T", temperature, fluid))

    if configured_t is None:
        temperature = defaults.INITIAL_TEMPERATURE_K
        notes.append(f"initial temperature not set; using {temperature} K")
    else:
        temperature = float(configured_t)
    density = float(PropsSI("D", "P", pressure, "T", temperature, fluid)) if need_density else None
    return temperature, density
