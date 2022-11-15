

from abc import abstractmethod
from typing import Protocol



class Material(Protocol):
    failure_stress: float
    type: float


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


class StructuralModelFactory:

    def get_structural_model(
        self, tank_section: TankSection
    ) -> StructuralModel:
        if tank_section.material.type == "metal":
            if tank_section.type == "cylinder":
                return MetalCylinder()
            if tank_section.type == "spherical_end_cap":
                return MetalSphericalEndCap()
        if tank_section.material.type == "composite":
            ...
        raise ValueError(
            f"{tank_section.material.type} and {tank_section.type}" \
                "not supported in StructuralModelFactory"
        )


def main():
    pass


if __name__ == "__main__":
    pass


# End
