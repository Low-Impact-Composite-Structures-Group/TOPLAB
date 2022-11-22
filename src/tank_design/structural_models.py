

from abc import abstractmethod
import math
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




class CompositeModel(StructuralModel):

    @staticmethod
    def hoop_stress(pressure: float, radius: float) -> float:
        return pressure * radius
    
    @staticmethod
    def meridional_stress(pressure: float, radius: float) -> float:
        return pressure * radius / 2

    @classmethod
    def helical_thickness(
        cls,
        pressure: float,
        radius: float,
        material: CompositeMaterial
    ) -> float:
        return (
            cls.hoop_stress(pressure, radius)
            / material.failure_stress
            / math.cos(math.radians(material.winding_angle)) ** 2
        )



class CompositeEndCap(CompositeModel):

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
            * math.tan(math.radians(material.winding_angle)) ** 2
        )
        return num / material.failure_stress


class StructuralModelFactory:

    def get_structural_model(
        self, tank_section: TankSection
    ) -> StructuralModel:
        if tank_section.material is None:
            return None
        if tank_section.material.type == "metal":
            if tank_section.type == "cylinder":
                return MetalCylinder()
            if tank_section.type == "spherical_end_cap":
                return MetalSphericalEndCap()
        if tank_section.material.type == "composite":
            if tank_section.type == "cylinder":
                return CompositeCylinder()
            if tank_section.type == "spherical_end_cap":
                return CompositeEndCap()
        raise ValueError(
            f"{tank_section.material.type} and {tank_section.type}" \
                "not supported in StructuralModelFactory"
        )


def main():
    pass


if __name__ == "__main__":
    pass


# End
