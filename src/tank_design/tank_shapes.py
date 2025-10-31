"""Tanks is used to define the shapes of the fuel tank. This is achieved
by defining different sections, such as the body and the end caps. This
can be used to compute the tank volume and surface area, as well as fuel
height and other variables required for convective computations.

Fuel Tank - Tanks
Hydrogen Storage in Civil Aviation PhD
Victor Kees Poorte, 2022
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from scipy.optimize import brentq

from src.tank_design.structural_models import StructuralModelFactory


STRUCTURAL_MODEL_FACTORY = StructuralModelFactory()
FUEL_HEIGHT_TOLERANCE = 1e-3


class Material(Protocol):
    density: float

    @abstractmethod
    def determine_specific_heat(self, temperature: float) -> float:
        ...


def radius_from_volume_sphere(volume: float) -> float:
    return (3 * volume / (4 * math.pi)) ** (1/3)


class TankSection(ABC):
    material: Material
    operating_pressure: float

    def __post_init__(self):
        self.define_structural_model()

    def define_structural_model(self):
        self.structural_model = (
            STRUCTURAL_MODEL_FACTORY.get_structural_model(self)
        )
        return self.structural_model

    @property
    def thickness(self):
        return self.structural_model.compute_thickness(
            self, self.operating_pressure
        )

    @property
    def structural_volume(self):
        return self.surface_area * self.thickness

    @property
    def structural_mass(self):
        return self.structural_volume * self.material.density

    def compute_thermal_capacity(
        self, temperature: float
    ) -> float:
        return (
            self.structural_mass 
            * self.material.determine_specific_heat(temperature)
        )

    def set_operating_pressure(self, pressure: float) -> float:
        self.operating_pressure = pressure
        return self.operating_pressure

    @property
    @abstractmethod
    def volume(self) -> float:
        """Method to compute the volume of the tank section"""
        ...

    @property
    @abstractmethod
    def surface_area(self) -> float:
        """Method to compute the surface area of the tank section"""
        ...

    @abstractmethod
    def compute_volume_section(self, fuel_height: float) -> float:
        """Method to compute the volume section of the fuel, for the 
        given height of the fuel.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: Volume of the fuel in the tank for the given height.
        """
        ...

    @abstractmethod
    def compute_wetted_surface(self, fuel_height: float) -> float:
        """Method to compute the wetted surface by the fuel for the 
        given fuel height.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: Wetted area by the fuel for the provided height.
        """
        ...
    

@dataclass
class CylindricalBody(TankSection):
    radius: float
    length: float
    material: Material
    operating_pressure: float

    def __post_init__(self):
        self.type = "cylinder"
        super().__post_init__()

    @property
    def surface_area(self) -> float:
        return 2 * np.pi * self.radius * self.length
    
    @property
    def volume(self) -> float:
        return np.pi * self.radius ** 2 * self.length

    def compute_volume_section(self, fuel_height: float) -> float:
        if fuel_height < 0:
            raise ValueError("Negative fuel height...")
        if fuel_height > 2 * self.radius:
            raise ValueError("Fuel height higher than tank...")
        self.volume_section = (
            self.length * (
                (self.radius ** 2 * np.arccos(
                    (self.radius - fuel_height) / self.radius)
                )
                - (self.radius - fuel_height) * (
                    2 * self.radius *  fuel_height - fuel_height ** 2
                ) ** (1 / 2)
            )
        )
        return self.volume_section

    def compute_fuel_amplitude_angle(self, fuel_height: float) -> float:
        """Method to compute the fuel amplitude angle. This is used for
        the computation of the wetted area by the fuel.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Raises:
            ValueError: Raises ValueError when the provided fuel height 
            is larger than the diameter of the fuel tank.

        Returns:
            float: Amplitude angle spanned by the fuel in radians.
        """
        if fuel_height > 2 * self.radius:
            raise ValueError("Fuel height larger than diameter...")
        if fuel_height > self.radius:
            self.theta = (
                np.pi / 2
                + np.arcsin((fuel_height - self.radius) / self.radius)
            ) * 2
            return self.theta
        if fuel_height <= self.radius:
            self.theta = (
                np.arccos((self.radius - fuel_height) / self.radius)
            ) * 2
            return self.theta

    def compute_wetted_surface(self, fuel_height: float) -> float:
        self.surface_area_section = (
            self.compute_fuel_amplitude_angle(fuel_height) * self.radius
            * self.length
        )
        return self.surface_area_section


@dataclass
class SphericalEndCap(TankSection):
    radius: float
    material: Material
    operating_pressure: float

    def __post_init__(self):
        self.type = "spherical_end_cap"
        self.define_structural_model()

    @property
    def surface_area(self) -> float:
        return  2 * np.pi * self.radius ** 2

    @property
    def volume(self) -> float:
        return 2 * np.pi * self.radius ** 3 / 3
    
    def compute_volume_section(self, fuel_height: float) -> float:
        if fuel_height < 0:
            raise ValueError("Negative fuel height...")
        if fuel_height > self.radius * 2:
            raise ValueError("Fuel height higher than diameter...")
        self.volume_section = (
            np.pi * fuel_height ** 2 * (3 * self.radius - fuel_height)
            / 6
        )
        return self.volume_section

    def compute_wetted_surface(self, fuel_height: float) -> float:
        self.surface_area_section = (
            self.radius * np.pi * fuel_height
        )
        return self.surface_area_section


class Tank:
    sections: list[TankSection]
    material: Material
    operating_pressure: float

    def set_sections(
        self, sections: list[TankSection]
    ) -> list[TankSection]:
        """Method to assign the sections to the tank.

        Args:
            sections (List[TankSection]): List with tank sections.

        Returns:
            list[TankSection]: List with tank sections.
        """
        self.sections = sections
        return self.sections
        
    def compute_thermal_capacity(
        self, temperature: float
    ) -> float:
        return sum([
            section.compute_thermal_capacity(temperature)
            for section in self.sections
        ])

    def set_operating_pressure(self, pressure: float):
        self.operating_pressure = pressure
        for section in self.sections:
            section.set_operating_pressure(pressure)
        return self.operating_pressure

    @property
    @abstractmethod
    def characteristic_height(self):
        ...

    @property
    @abstractmethod
    def characteristic_length(self):
        ...

    @property
    def volume(self) -> float:
        return sum([
            section.volume
            for section in self.sections
        ])

    @property
    def structural_volume(self):
        return sum([
            section.structural_volume for section in self.sections
        ])

    @property
    def structural_mass(self):
        return sum([
            section.structural_mass for section in self.sections
        ])
    
    @property
    def surface_area(self) -> float:
        return sum([
            section.surface_area
            for section in self.sections
        ])

    def compute_fuel_volume(self, fuel_height: float) -> float:
        return sum([
            section.compute_volume_section(fuel_height)
            for section in self.sections
        ])

    @abstractmethod
    def compute_fuel_height(self, fuel_volume: float) -> float:
        """Method to compute at which height the fuel reaches in the 
        tank. This is required in the computation of the heat transfer
        modes.

        Args:
            fuel_volume (float): Volume of the fuel in the tank.

        Returns:
            float: Height of the fuel in the tank.
        """
        ...

    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        """Method to compute the surface of the whetter area by the
        liquid part of the fuel.
        This is required for the heat transfer mode computations.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: wetted area by the fuel.
        """
        return sum([
            section.compute_wetted_surface(fuel_height)
            for section in self.sections
        ])

    def compute_gas_wetted_surface(self, fuel_height: float) -> float:
        """Method to compute the surface of the whetter area by the
        gas part of the fuel.
        This is required for the heat transfer mode computations.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: wetted area by the fuel.
        """
        fuel_surface = self.compute_fuel_wetted_surface(fuel_height)
        return self.surface_area - fuel_surface

    
@dataclass
class CylindricalTankSphericalCaps(Tank):
    radius: float
    total_length: float
    material: Material
    operating_pressure: float

    def __post_init__(self):
        self.create_body()
        self.create_end_cap()
        self.create_sections()

    def __str__(self):
        return f"Cylindrical Tank Spherical End Caps (Radius: {self.radius}, Body: {self.body_length})"

    @property
    def thickness(self):
        return self.body.thickness

    @property
    def body_length(self):
        return self.total_length - 2 * self.radius

    @property
    def volume(self) -> float:
        return sum([
            section.volume
            for section in self.sections
        ])
    
    @property
    def diameter(self):
        return self.radius * 2

    @property
    def characteristic_height(self):
        return self.diameter

    @property
    def characteristic_length(self):
        return self.body_length

    def create_body(self) -> CylindricalBody:
        """Method to create the body of the fuel tank.

        Returns:
            CylindricalBody: Body of the fuel tank.
        """
        self.body = CylindricalBody(
            self.radius,
            self.body_length,
            self.material,
            self.operating_pressure
        )
        return self.body

    def create_end_cap(self) -> SphericalEndCap:
        """Method to create the end cap of the fuel tank.

        Returns:
            SphericalEndCap: End cap of the fuel tank.
        """
        self.end_cap = SphericalEndCap(
            self.radius, self.material, self.operating_pressure
        )
        return self.end_cap
    
    @property
    def exposed_surface(self) -> float:
        return self.body.surface_area

    def create_sections(self):
        """Method te create the list with fuel tank sections.

        Returns:
            _type_: _description_
        """
        self.sections: list[TankSection] = [
            self.end_cap, self.body, self.end_cap
        ]
        return self.sections

    def compute_fuel_height(self, fuel_volume: float) -> float:
        if abs(fuel_volume - self.volume) / self.volume <= FUEL_HEIGHT_TOLERANCE:
            return 2 * self.radius
        return brentq(
            lambda h: self.compute_fuel_volume(h) - fuel_volume,
            a=0,
            b=2*self.radius,
            xtol=FUEL_HEIGHT_TOLERANCE,
            maxiter=100
        )
    
    def compute_fuel_area_zones(self, fuel_height: float) -> list[float]:
        """Method to compute the areas of the zones as used in Rompokos
        (2018). This is required to compute the heat transfer
        coefficients of the fuel tank. See Figure 9 of Rompokos 2018.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            list[float]: List with surface area values, in the order as
            the names of the zones.
        """
        gas_surface = (
            self.surface_area - self.compute_fuel_wetted_surface(
                fuel_height
            )
        )
        return [
            gas_surface,
            self.compute_zone_1_area(fuel_height),
            self.compute_zone_2_area(fuel_height),
            self.compute_zone_3_area(fuel_height)
        ]

    def compute_zone_1_area(self, fuel_height: float) -> float:
        """Method to compute area section 1 of the fuel tank. This is 
        the upper part of the fuel tank. Dive into Figure 9 of 
        Winnefeld 2018 for better understanding.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: surface are of the area wetted in zone 1.
        """
        if fuel_height <= self.diameter - self.outer_segment:
            return 0
        total_zone_1_area = self.compute_fuel_wetted_surface(
            self.outer_segment
        )
        # This is the upper part not filled by the fuel which needs to 
        # be removed from the total zone 1 area
        remove_upper_area = self.compute_fuel_wetted_surface(
            self.diameter - fuel_height
        )
        return total_zone_1_area - remove_upper_area

    def compute_zone_2_area(self, fuel_height: float) -> float:
        """Method to compute area section 2 of the fuel tank. This is 
        the lower part of the fuel tank. Dive into Figure 9 of 
        Winnefeld 2018 for better understanding.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: surface are of the area wetted in zone 2.
        """
        if fuel_height > self.outer_segment:
            return sum([
                section.compute_wetted_surface(
                    self.outer_segment
                )
                for section in self.sections
            ])
        return sum([
            section.compute_wetted_surface(fuel_height)
            for section in self.sections
        ])

    def compute_zone_3_area(self, fuel_height: float) -> float:
        """Method to compute area section 3 of the fuel tank. This is 
        the vertical part of the fuel tank. Dive into Figure 9 of 
        Winnefeld 2018 for better understanding.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: surface are of the area wetted in zone 3.
        """
        if fuel_height <= self.outer_segment:
            return 0
        return (
            self.compute_fuel_wetted_surface(fuel_height)
            - self.compute_zone_2_area(fuel_height)
            - self.compute_zone_1_area(fuel_height)
        )

    def compute_zone_1_length(self, fuel_height: float) -> float:
        """Method to compute the characteristic length of Zone 1 type of
        convection, which is the upper part of the fuel tank.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: Characteristic convection length.
        """
        # Case where the fuel height is not in zone 1 anymore
        if fuel_height <= self.diameter - self.outer_segment:
            return 0
        return (
            self.outer_segment - (self.diameter - fuel_height)
        )

    def compute_zone_2_length(self, fuel_height: float) -> float:
        """Method to compute the characteristic length of Zone 2 type of
        convection, which is the lower part of the fuel tank.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: Characteristic convection length.
        """
        if fuel_height > self.outer_segment:
            return self.outer_segment
        return fuel_height

    def compute_zone_3_length(self, fuel_height: float) -> float:
        """Method to compute the characteristic length of Zone 3 type of
        convection, which is the vertical part of the fuel tank.

        Args:
            fuel_height (float): Height of the fuel in the tank.

        Returns:
            float: Characteristic convection length.
        """
        # Case where the fuel height is not in zone 3 anymore
        if fuel_height <= self.outer_segment:
            return 0
        # Case where the fuel height is still in the upper zone
        if fuel_height >= 2 * self.radius - self.outer_segment:
            return self.radius * np.cos(np.pi / 4) * 2
        # Case where the fuel is somewhere in zone 3
        return fuel_height - self.outer_segment

    def compute_convective_lengths(
        self, fuel_height: float
    ) -> list[float]:
        """Method to compute the convective lengths for the given fuel
        height.

        Args:
            fuel_height (float): heigh of the fuel in the tank.

        Returns:
            list[float]: List with characteristic convective lengths.
        """
        return [
            self.diameter - fuel_height,
            self.compute_zone_1_length(fuel_height),
            self.compute_zone_2_length(fuel_height),
            self.compute_zone_3_length(fuel_height)
        ]

    @property
    def outer_segment(self) -> float:
        """Method to compute the small segment of that is included 
        between the square enclosed in a circle and the circle itself.
        This is required for the fuel zone computations. For a better 
        understanding dive into the paper of Winnefeld published in
        2018.

        Returns:
            float: Length of the segment between the enclosed square and
            the circle.
        """
        return self.radius * (1 - np.cos(np.pi / 4))

    @classmethod
    def rompokos(
        cls, material: Material, operating_pressure: float
    ) -> CylindricalTankSphericalCaps:
        """Method to create the tank as used by Rompokos.

        Returns:
            CylindricalTankSphericalCaps: Instance of the Cylindrical
            Tank with Spherical End Caps as used by ROmpokos (2021)
        """
        # Define tank properties
        tank_outer_diameter = 2.5       # [m]
        insulation_thickness = 8e-2     # [m]
        tank_length = 19.1              # [m]
        tank_outer_radius = tank_outer_diameter / 2
        tank_inner_radius = tank_outer_radius - insulation_thickness
        tank  = cls(
            tank_inner_radius, tank_length, material, operating_pressure
        )
        return tank

    @classmethod
    def ahluwalia(
        cls, material: Material, operating_pressure: float
    ) -> CylindricalTankSphericalCaps:
        # Define tank properties
        tank_radius = 0.25                      # [m]
        tank_body_length = 0.43570335168670493  # [m]
        tank_length = tank_body_length + 2 * tank_radius
        tank = cls(
            tank_radius, tank_length, material, operating_pressure
        )
        return tank
        
    @classmethod
    def example(
        cls, material: Material, operating_pressure: float
    ) -> CylindricalTankSphericalCaps:
        # Define tank properties
        tank_body_length = 4.0      # [m]
        tank_radius = 1.25          # [m]
        tank_length = tank_body_length + 2 * tank_radius
        tank = cls(
            tank_radius, tank_length, material, operating_pressure
        )
        return tank

    @staticmethod
    def length_from_radius_and_volume(
        radius: float, volume: float
    ) -> float:
        num = volume - 4 / 3 * math.pi * radius ** 3
        if num < 0: return None
        den = math.pi * radius ** 2
        return num / den
        

@dataclass
class SphericalTank(Tank):
    radius: float
    material: Material
    operating_pressure: float

    @property
    def diameter(self):
        return 2 * self.radius

    @property
    def characteristic_height(self):
        return self.diameter

    @property
    def characteristic_length(self):
        return 0

    @property
    def exposed_surface(self):
        # It should be null; however to avoid zero division it is set
        # really small
        return 0

    @property
    def body_length(self):
        return 0

    def __post_init__(self) -> None:
        self.create_sections()

    def create_sections(self) -> list[TankSection]:
        self.sections = [
            SphericalEndCap(
                self.radius, self.material, self.operating_pressure
            ),
            SphericalEndCap(
                self.radius, self.material, self.operating_pressure
            )
        ]
        return self.sections

    def compute_fuel_height(self, fuel_volume: float) -> float:
        return brentq(
            lambda h: self.compute_fuel_volume(h) - fuel_volume,
            a=0,
            b=2*self.radius,
            xtol=FUEL_HEIGHT_TOLERANCE,
            maxiter=100
        )

    @classmethod
    def lin(
        cls, 
        material: Material, 
        operating_pressure: float
    ) -> SphericalTank:
        tank_volume = 52
        def radius_from_volume(volume: float) -> float:
            return (volume * 3 / 4 / math.pi) ** (1 / 3)
        return cls(
            radius_from_volume(tank_volume),
            material,
            operating_pressure
        )


@dataclass
class TankDimensions:
    radius: float = None
    body_length: float = None

    @property
    def total_tank_length(self):
        return self.body_length + 2 * self.radius


class TankFactory():

    @staticmethod
    def create_tank(
        type: str,
        tank_dimensions: TankDimensions,
        material: Material,
        operating_pressure: float
    ) -> Tank:
        if type != "cylindrical_spherical_end_caps":
            raise ValueError(f"'{type}' not supported tank type")
        if tank_dimensions.body_length is None:
            return SphericalTank(tank_dimensions.radius, material, operating_pressure)
        total_length = tank_dimensions.body_length + 2 * tank_dimensions.radius
        return CylindricalTankSphericalCaps(
            tank_dimensions.radius, total_length, material, operating_pressure
        )


def main():
    
    pass


if __name__ == "__main__":
    main()


#  End
