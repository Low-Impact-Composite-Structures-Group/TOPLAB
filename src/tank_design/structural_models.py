

import math
from abc import abstractmethod
from typing import Protocol


class Material(Protocol):
    failure_stress: float
    type: float


class CompositeMaterial(Material):
    winding_angle: float


class TankSection(Protocol):
    radius: float
    type: str
    material: Material


class StructuralModel(Protocol):

    @abstractmethod
    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        ...


class MetalSphericalEndCap(StructuralModel):

    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        return (
            pressure * tank_section.radius
            / tank_section.material.failure_stress
        )


class MetalCylinder(StructuralModel):

    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        return (
            pressure * tank_section.radius
            / tank_section.material.failure_stress
        )

class MetalEllipsoidalEndCap(StructuralModel):
    # TODO : replace with correct calculation for new geometry

    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        return (
            pressure * tank_section.radius
            / tank_section.material.failure_stress
        )


class MetalEllipticCylinder(StructuralModel):
    # TODO : replace with correct calculation for new geometry
    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        return (
            pressure * tank_section.radius
            / tank_section.material.failure_stress
        )


class CompositeModel(StructuralModel):

    @staticmethod
    def hoop_stress(pressure: float, radius: float) -> float:
        return pressure * radius

    @staticmethod
    def meridional_stress(pressure: float, radius: float) -> float:
        return pressure * radius / 2
    
    @staticmethod
    def meridional_stress_ellipse(pressure: float, radius: float, b: float) -> float:
        return pressure * radius  * (1- radius**2/2*b**2)

    @classmethod
    def helical_thickness(
        cls,
        pressure: float,
        radius: float,
        material: CompositeMaterial
    ) -> float:
        return (
            cls.meridional_stress(pressure, radius)
            / material.failure_stress
            / math.cos(material.winding_angle) ** 2
        )
        
    @classmethod
    def helical_thickness_ellipse(
        cls,
        pressure: float,
        radius: float,
        b: float,
        material: CompositeMaterial
    ) -> float:
        return (
            cls.meridional_stress_ellipse(pressure, radius, b)
            / material.failure_stress
            / math.cos(material.winding_angle) ** 2
        )


class CompositeSphericalEndCap(CompositeModel):

    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        return self.helical_thickness(
            pressure, tank_section.radius, tank_section.material
        )


class CompositeCylinder(CompositeModel):

    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        helical = self.helical_thickness(
            pressure, tank_section.radius, tank_section.material
        )
        hoop = self.hoop_thickness(
            pressure, tank_section.radius, tank_section.material
        )
        return helical + hoop

    @classmethod
    def hoop_thickness(
        cls,
        pressure: float,
        radius: float,
        material: CompositeMaterial
    ) -> float:
        num = (
            cls.hoop_stress(pressure, radius)
            - cls.meridional_stress(pressure, radius)
            * math.tan(material.winding_angle) ** 2
        )
        return num / material.failure_stress

class CompositeEllipsoidalEndCap(CompositeModel):
    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        return self.helical_thickness_ellipse(
            pressure, tank_section.radius, tank_section.radius/2.0, tank_section.material
        )


class CompositeEllipticCylinder(CompositeModel):
    # TODO : replace with correct calculation for new geometry
    def compute_thickness(
        self,
        tank_section: TankSection,
        pressure: float
    ) -> float:
        helical = self.helical_thickness(
            pressure, tank_section.radius, tank_section.material
        )
        hoop = self.hoop_thickness(
            pressure, tank_section.radius, tank_section.material
        )
        return helical + hoop

    @classmethod
    def hoop_thickness(
        cls,
        pressure: float,
        radius: float,
        material: CompositeMaterial
    ) -> float:
        num = (
            cls.hoop_stress(pressure, radius)
            - cls.meridional_stress(pressure, radius)
            * math.tan(material.winding_angle) ** 2
        )
        return num / material.failure_stress


class StructuralModelFactory:

    def get_structural_model(
        self, tank_section: TankSection
    ) -> StructuralModel:
        if tank_section.material is None:
            raise ValueError(
                "Tank section has no material.."
            )
        if tank_section.material.type == "metal":
            if tank_section.type == "cylinder":
                return MetalCylinder()
            if tank_section.type == "spherical_end_cap":
                return MetalSphericalEndCap()
            if tank_section.type == "elliptic_cylinder":
                return MetalEllipticCylinder()
            if tank_section.type == "ellipsoidal_end_cap":
                return MetalEllipsoidalEndCap()
        if tank_section.material.type == "composite":
            if tank_section.type == "cylinder":
                return CompositeCylinder()
            if tank_section.type == "spherical_end_cap":
                return CompositeSphericalEndCap()
            if tank_section.type == "elliptic_cylinder":
                return CompositeEllipticCylinder()
            if tank_section.type == "ellipsoidal_end_cap":
                return CompositeEllipsoidalEndCap()
        raise ValueError(
            f"{tank_section.material.type} and {tank_section.type}" \
                "not supported in StructuralModelFactory"
        )


def main():
    pass


if __name__ == "__main__":
    pass


# End
