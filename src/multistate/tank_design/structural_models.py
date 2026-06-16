from __future__ import annotations

import math
from typing import Protocol


class CompositeMaterial(Protocol):
    failure_stress: float
    winding_angle: float


class TankSection(Protocol):
    radius: float
    material: CompositeMaterial


class CompositeModel:
    @staticmethod
    def hoop_stress(pressure: float, radius: float) -> float:
        return pressure * radius

    @staticmethod
    def meridional_stress(pressure: float, radius: float) -> float:
        return pressure * radius / 2.0

    @classmethod
    def helical_thickness(
        cls,
        pressure: float,
        radius: float,
        material: CompositeMaterial,
    ) -> float:
        return (
            cls.meridional_stress(pressure, radius)
            / material.failure_stress
            / math.cos(material.winding_angle) ** 2
        )


class CompositeSphericalEndCap(CompositeModel):
    def compute_thickness(self, tank_section: TankSection, pressure: float) -> float:
        return self.helical_thickness(pressure, tank_section.radius, tank_section.material)


class CompositeCylinder(CompositeModel):
    def compute_thickness(self, tank_section: TankSection, pressure: float) -> float:
        return self.helical_thickness(
            pressure,
            tank_section.radius,
            tank_section.material,
        ) + self.hoop_thickness(pressure, tank_section.radius, tank_section.material)

    @classmethod
    def hoop_thickness(
        cls,
        pressure: float,
        radius: float,
        material: CompositeMaterial,
    ) -> float:
        return (
            cls.hoop_stress(pressure, radius)
            - cls.meridional_stress(pressure, radius) * math.tan(material.winding_angle) ** 2
        ) / material.failure_stress