

from abc import abstractmethod
from typing import Protocol


class TankSection(Protocol):
    radius: float
    type: str


class Material(Protocol):
    failure_stress: float
    type: float


class StructuralModel(Protocol):

    @abstractmethod
    def compute_thickness(
        self,
        tank_section: TankSection,
        material: Material,
        pressure: float
    ) -> float:
        ...


class MetalSphericalEndCap(StructuralModel):

    def compute_thickness(
        self,
        tank_section: TankSection,
        material: Material,
        pressure: float
    ) -> float:
        return pressure * tank_section.radius / material.failure_stress


class MetalCylinder(StructuralModel):

    def compute_thickness(
        self,
        tank_section: TankSection,
        material: Material,
        pressure: float
    ) -> float:
        return pressure * tank_section.radius / material.failure_stress


class StructuralModelFactory:

    def get_structural_model(
        self, tank_section: TankSection, material: Material
    ) -> StructuralModel:
        if material.type == "metal":
            if tank_section.type == "cylinder":
                return MetalCylinder()
            if tank_section.type == "spherical_end_cap":
                return MetalSphericalEndCap()
        if material.type == "composite":
            ...
        raise ValueError(
            f"{material.type} not supported material type for " \
                "structural models"
        )


def main():
    pass


if __name__ == "__main__":
    pass


# End
