"""
NIST-backed materials colocated under multi_tank.

Copied from `src/materials/nist_materials.py` and updated to import
property polynomials from `src.multi_tank.materials.nist_material_properties`.
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.materials.materials import Material, Metal, Composite
from src.multi_tank.materials.nist_material_properties.aluminum5083_properties import specific_heat as aluminum_cp_nist, thermal_conductivity as aluminum_k_nist
from src.multi_tank.materials.nist_material_properties.g10_properties import specific_heat as g10_cp_nist, thermal_conductivity_normal as g10_k_normal_nist
from src.multi_tank.materials.nist_material_properties.aluminum3003f_properties import specific_heat as aluminum3003f_cp_nist, thermal_conductivity as aluminum3003f_k_nist
from src.multi_tank.materials.nist_material_properties.aluminum6061t6_properties import specific_heat as aluminum6061t6_cp_nist, thermal_conductivity as aluminum6061t6_k_nist


@dataclass
class NISTMaterial(Material):
    _nist_cp_func: Optional[callable] = None
    _nist_k_func: Optional[callable] = None
    _temperature_range: tuple = (4, 300)

    def __post_init__(self):
        if not hasattr(self, '_nist_cp_func'):
            self._nist_cp_func = None
        if not hasattr(self, '_nist_k_func'):
            self._nist_k_func = None
        if not hasattr(self, '_temperature_range'):
            self._temperature_range = (4, 300)

    def determine_specific_heat(self, temperature: float) -> float:
        if self._nist_cp_func is not None:
            min_temp, max_temp = self._temperature_range
            if min_temp <= temperature <= max_temp:
                return float(self._nist_cp_func(temperature))
            elif temperature < min_temp:
                raise ValueError(f"Temperature {temperature}K below NIST range {min_temp}-{max_temp}K")
            elif temperature > max_temp:
                raise ValueError(f"Temperature {temperature}K above NIST range {min_temp}-{max_temp}K")

    def determine_thermal_conductivity(self, temperature: float) -> float:
        if self._nist_k_func is not None:
            min_temp, max_temp = self._temperature_range
            if min_temp <= temperature <= max_temp:
                return float(self._nist_k_func(temperature))
            else:
                raise ValueError(f"Temperature {temperature}K outside NIST range {min_temp}-{max_temp}K for thermal conductivity")
        raise RuntimeError(f"No NIST thermal conductivity data available at {temperature}K - simulation cannot continue")

    def set_temperature_range(self, min_temp: float, max_temp: float):
        self._temperature_range = (min_temp, max_temp)

    def set_nist_function(self, nist_func: callable):
        self._nist_cp_func = nist_func

    def set_nist_thermal_conductivity_function(self, nist_k_func: callable):
        self._nist_k_func = nist_k_func


@dataclass
class NISTMetal(NISTMaterial, Metal):
    def __post_init__(self):
        super().__post_init__()
        self.type = "metal"

    @classmethod
    def aluminum_5083_nist(cls):
        failure_stress = 185e6
        density = 2650
        characteristic_temperature = 1500
        molecular_weight = 26.981539
        material = cls(failure_stress=failure_stress, density=density, characteristic_temperature=characteristic_temperature, molecular_weight=molecular_weight)
        material.set_nist_function(aluminum_cp_nist)
        material.set_nist_thermal_conductivity_function(aluminum_k_nist)
        material.set_temperature_range(4, 300)
        return material

    @classmethod
    def aluminum_3003F_nist(cls):
        failure_stress = 110e6
        density = 2730
        characteristic_temperature = 1500
        molecular_weight = 26.981539
        material = cls(failure_stress=failure_stress, density=density, characteristic_temperature=characteristic_temperature, molecular_weight=molecular_weight)
        material.set_nist_function(aluminum3003f_cp_nist)
        material.set_nist_thermal_conductivity_function(aluminum3003f_k_nist)
        material.set_temperature_range(4, 300)
        return material

    @classmethod
    def aluminum_6061T6_nist(cls):
        failure_stress = 310e6
        density = 2700
        characteristic_temperature = 1500
        molecular_weight = 26.981539
        material = cls(failure_stress=failure_stress, density=density, characteristic_temperature=characteristic_temperature, molecular_weight=molecular_weight)
        material.set_nist_function(aluminum6061t6_cp_nist)
        material.set_nist_thermal_conductivity_function(aluminum6061t6_k_nist)
        material.set_temperature_range(4, 300)
        return material


@dataclass
class NISTComposite(NISTMaterial, Composite):
    winding_angle: float

    def __post_init__(self):
        super().__post_init__()
        self.type = "composite"

    def determine_specific_heat(self, temperature: float) -> float:
        if self._nist_cp_func is not None:
            min_temp, max_temp = self._temperature_range
            if min_temp <= temperature <= max_temp:
                return float(self._nist_cp_func(temperature))
            else:
                if temperature < min_temp:
                    raise ValueError(f"Temperature {temperature}K below NIST range {min_temp}-{max_temp}K for composite material")
                else:
                    raise ValueError(f"Temperature {temperature}K above NIST range {min_temp}-{max_temp}K for composite material")
        raise RuntimeError(f"No NIST data available for composite at {temperature}K - simulation cannot continue")

    @classmethod
    def g10_nist(cls, winding_angle: float):
        failure_stress = 2000e6
        density = 1800
        characteristic_temperature = 1500.0
        molecular_weight = 12.01
        material = cls(failure_stress=failure_stress, density=density, characteristic_temperature=characteristic_temperature, molecular_weight=molecular_weight, winding_angle=winding_angle)
        material.set_nist_function(g10_cp_nist)
        material.set_nist_thermal_conductivity_function(g10_k_normal_nist)
        material.set_temperature_range(4, 300)
        return material
