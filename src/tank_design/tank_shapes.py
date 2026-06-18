from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq


FUEL_HEIGHT_TOLERANCE = 1e-8


@dataclass
class SphericalTank:
    radius: float
    material: object
    operating_pressure: float
    phi: float = 0.0

    @property
    def volume(self) -> float:
        sphere_volume = 4.0 / 3.0 * math.pi * self.radius ** 3
        cylinder_volume = math.pi * self.radius ** 2 * self.cylindrical_section_length
        return sphere_volume + cylinder_volume

    @property
    def surface_area(self) -> float:
        sphere_area = 4.0 * math.pi * self.radius ** 2
        cylinder_area = 2.0 * math.pi * self.radius * self.cylindrical_section_length
        return sphere_area + cylinder_area

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def characteristic_height(self) -> float:
        return self.diameter

    @property
    def characteristic_length(self) -> float:
        return self.cylindrical_section_length

    @property
    def exposed_surface(self) -> float:
        return 2.0 * math.pi * self.radius * self.cylindrical_section_length

    @property
    def body_length(self) -> float:
        return self.cylindrical_section_length

    @property
    def cylindrical_section_length(self) -> float:
        return max(self.phi, 0.0) * self.radius

    def compute_fuel_volume(self, fuel_height: float) -> float:
        bounded_height = min(max(fuel_height, 0.0), self.diameter)
        return math.pi * bounded_height ** 2 * (self.radius - bounded_height / 3.0)

    def compute_fuel_height(self, fuel_volume: float) -> float:
        if fuel_volume <= 0.0:
            return 0.0
        if fuel_volume >= self.volume:
            return self.diameter
        return brentq(
            lambda height: self.compute_fuel_volume(height) - fuel_volume,
            a=0.0,
            b=self.diameter,
            xtol=FUEL_HEIGHT_TOLERANCE,
            maxiter=100,
        )