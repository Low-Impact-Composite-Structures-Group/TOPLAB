"""Self-contained NIST-backed materials used by the multistate solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from toplab.materials.nist_material_properties.aluminum3003f_properties import specific_heat as aluminum3003f_cp_nist, thermal_conductivity as aluminum3003f_k_nist
from toplab.materials.nist_material_properties.aluminum5083_properties import specific_heat as aluminum5083_cp_nist, thermal_conductivity as aluminum5083_k_nist
from toplab.materials.nist_material_properties.aluminum6061t6_properties import specific_heat as aluminum6061t6_cp_nist, thermal_conductivity as aluminum6061t6_k_nist
from toplab.materials.nist_material_properties.carbon_epoxy_properties import specific_heat as carbon_epoxy_cp_nist, thermal_conductivity as carbon_epoxy_k_nist
from toplab.materials.nist_material_properties.g10_properties import specific_heat as g10_cp_nist, thermal_conductivity_normal as g10_k_normal_nist


def _clamp_temperature(temperature: float, minimum: float = 10.0, maximum: float = 400.0) -> float:
    return min(max(float(temperature), minimum), maximum)


@dataclass
class Material:
    failure_stress: float
    density: float
    type: str


@dataclass
class NISTMaterial(Material):
    nist_path: str
    specific_heat_func: Callable[[float], float]
    name: str
    thermal_conductivity_func: Callable[[float], float] | None = None
    winding_angle: float | None = None

    def get_specific_heat(self, temperature: float) -> float:
        return float(self.specific_heat_func(_clamp_temperature(temperature)))

    def determine_specific_heat(self, temperature: float) -> float:
        return self.get_specific_heat(temperature)

    def determine_thermal_capacity(self, temperature: float, mass: float) -> float:
        return self.get_specific_heat(temperature) * mass

    def determine_thermal_conductivity(self, temperature: float) -> float:
        if self.thermal_conductivity_func is None:
            raise RuntimeError(
                f"No NIST thermal conductivity data available for {self.nist_path}"
            )
        return float(
            self.thermal_conductivity_func(
                _clamp_temperature(temperature, minimum=4.0)
            )
        )

    @classmethod
    def aluminum_6061T6_nist(cls) -> "NISTMaterial":
        return NISTMetal.aluminum_6061T6_nist()

    @classmethod
    def g10_nist(cls, winding_angle: float | None = None) -> "NISTMaterial":
        return NISTComposite.g10_nist(winding_angle=winding_angle)

    @classmethod
    def carbon_epoxy_nist(cls, winding_angle: float | None = None) -> "NISTMaterial":
        return NISTComposite.carbon_epoxy_nist(winding_angle=winding_angle)

    def __str__(self) -> str:
        return (
            f"{self.name}: ρ={self.density:.0f} kg/m³, "
            f"σ_fail={self.failure_stress / 1e6:.0f} MPa, "
            f"path={self.nist_path}"
        )

    def __repr__(self) -> str:
        return (
            f"NISTMaterial(name='{self.name}', "
            f"density={self.density}, "
            f"failure_stress={self.failure_stress}, "
            f"nist_path='{self.nist_path}')"
        )


@dataclass(repr=False)
class NISTMetal(NISTMaterial):
    @classmethod
    def aluminum_5083_nist(cls) -> "NISTMetal":
        return cls(
            failure_stress=185e6,
            density=2650.0,
            type="metal",
            nist_path="aluminum_5083_nist",
            specific_heat_func=aluminum5083_cp_nist,
            thermal_conductivity_func=aluminum5083_k_nist,
            name="Aluminum 5083 (NIST)",
        )

    @classmethod
    def aluminum_3003F_nist(cls) -> "NISTMetal":
        return cls(
            failure_stress=110e6,
            density=2730.0,
            type="metal",
            nist_path="aluminum_3003F_nist",
            specific_heat_func=aluminum3003f_cp_nist,
            thermal_conductivity_func=aluminum3003f_k_nist,
            name="Aluminum 3003-F (NIST)",
        )

    @classmethod
    def aluminum_6061T6_nist(cls) -> "NISTMetal":
        return cls(
            failure_stress=276e6,
            density=2700.0,
            type="metal",
            nist_path="aluminum_6061T6_nist",
            specific_heat_func=aluminum6061t6_cp_nist,
            thermal_conductivity_func=aluminum6061t6_k_nist,
            name="Aluminum 6061-T6 (NIST)",
        )


@dataclass(repr=False)
class NISTComposite(NISTMaterial):
    @classmethod
    def g10_nist(cls, winding_angle: float | None = None) -> "NISTComposite":
        return cls(
            failure_stress=310e6,
            density=1800.0,
            type="composite",
            nist_path="g10_nist",
            specific_heat_func=g10_cp_nist,
            thermal_conductivity_func=g10_k_normal_nist,
            name="G10 Composite (NIST)",
            winding_angle=math.radians(54.7) if winding_angle is None else winding_angle,
        )

    @classmethod
    def carbon_epoxy_nist(cls, winding_angle: float | None = None) -> "NISTComposite":
        return cls(
            failure_stress=5000e6,
            density=1500.0,
            type="composite",
            nist_path="carbon_epoxy_nist",
            specific_heat_func=carbon_epoxy_cp_nist,
            thermal_conductivity_func=carbon_epoxy_k_nist,
            name="Carbon-Epoxy Composite (NIST)",
            winding_angle=0.0 if winding_angle is None else winding_angle,
        )


def get_material_by_nist_path(nist_path: str) -> NISTMaterial:
    material_registry = {
        "aluminum_5083_nist": NISTMetal.aluminum_5083_nist,
        "aluminum_3003F_nist": NISTMetal.aluminum_3003F_nist,
        "aluminum_6061T6_nist": NISTMetal.aluminum_6061T6_nist,
        "g10_nist": NISTComposite.g10_nist,
        "carbon_epoxy_nist": NISTComposite.carbon_epoxy_nist,
    }

    if nist_path not in material_registry:
        available = ", ".join(material_registry.keys())
        raise ValueError(f"Unknown NIST path '{nist_path}'. Available: {available}")

    return material_registry[nist_path]()


__all__ = [
    "Material",
    "NISTComposite",
    "NISTMaterial",
    "NISTMetal",
    "get_material_by_nist_path",
]
