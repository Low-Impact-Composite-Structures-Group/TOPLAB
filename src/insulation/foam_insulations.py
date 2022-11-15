from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
import math
import json
from pathlib import Path
from typing import Protocol

from src.thermodynamics.thermal_resistances import ThermalResistance


class FuelTank(Protocol):
    radius: float
    surface_area: float


class Insulation(Protocol):

    @abstractmethod
    def compute_thermal_conductivity(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        """Method to compute the thermal conductivity of the insulator.
        the temperatures are required for variable conductivity foam
        insulators and for MLI types of insulation.

        Args:
            hot_temperature (float): Temperature at the hot side of the
            insulator.
            cold_temperature (float): Temperature at the cold side of
            the insulator.

        Returns:
            float: Thermal conductivity of the insulator.
        """
        ...

    @abstractmethod
    def compute_heat_transfer_coefficient(
        self,
        hot_temperature: float,
        cold_temperature: float,
        tank: FuelTank
    ) -> float:
        """Method to determine the heat transfer coefficient of the
        insulator.

        Args:
            hot_temperature (float): Temperature at the hot side of the
            insulator.
            cold_temperature (float): Temperature at the cold side of
            the insulator.

        Returns:
            float: Heat transfer coefficient of the insulator.
        """
        ...

    @abstractmethod
    def compute_thermal_resistances(
        self,
        temperatures: list[float],
        tank: FuelTank
    ) -> list[float]:
        ...


@dataclass
class FoamInsulation(Insulation):
    thickness: float
    density: float

    def compute_heat_transfer_coefficient(
        self,
        thermal_conductivity: float,
        outer_radius: float,
        inner_radius: float
    ) -> float:
        return (
            thermal_conductivity / math.log(outer_radius / inner_radius)
        )
    
    def compute_thermal_resistances(
        self,
        temperatures: list[float],
        tank: FuelTank
    ) -> list[float]:
        layer_thickness = self.thickness / (len(temperatures) - 1)
        def compute_inner_radius(layer: int) -> float:
            return tank.radius + (layer - 1) * layer_thickness
        def compute_outer_radius(layer: int) -> float:
            return tank.radius + layer * layer_thickness
        heat_transfer_coefficients = [
            self.compute_heat_transfer_coefficient(
                self.compute_thermal_conductivity(
                    temperature, temperatures[layer]
                ),
                compute_outer_radius(layer),
                compute_inner_radius(layer)
            )
            for layer, temperature in enumerate(temperatures[:-1], 1)
        ]
        resistances = [
            ThermalResistance(coefficient, tank.surface_area).value
            for coefficient in heat_transfer_coefficients
        ]
        return resistances


@dataclass
class ConstantFoamInsulation(FoamInsulation):
    thermal_conductivity: float

    def compute_thermal_conductivity(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        return self.thermal_conductivity

    @classmethod
    def polyvinylchloride(
        cls, thickness: float
    ) -> ConstantFoamInsulation:
        """Polyvinylchloride closed-cell foam.

        Args:
            inner_radius (float): Inner radius of the insulator.
            outer_radius (float): Outer radius of the insulator.

        Returns:
            _type_: Foam insulation
        """
        density = None
        thermal_conductivity = 0.0046
        return cls(thickness, density, thermal_conductivity)

    @classmethod
    def rohacell(
        cls, thickness: float
    ) -> ConstantFoamInsulation:
        """Rohacell closed-cell foam.

        Args:
            inner_radius (float): Inner radius of the insulator.
            outer_radius (float): Outer radius of the insulator.

        Returns:
            _type_: Foam insulation
        """
        thermal_conductivity = 0.015
        density = 51.1
        return cls(thickness, density, thermal_conductivity)


@dataclass
class VariableFoamInsulation(FoamInsulation):
    name: str

    def __post_init__(self):
        self.create_path()
        self.load_foam_data()

    def create_path(self) -> Path:
        self.path = (
            Path("src")
            / "insulation"
            / "foam_data"
            / f"{self.name}.json"
        )

    def load_foam_data(self):
        with self.path.open("r") as file:
            self.thermal_conductivity_data = json.load(file)
        return self.thermal_conductivity_data

    def compute_average_temperature(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        return (hot_temperature + cold_temperature) / 2

    def compute_thermal_conductivity(
        self, hot_temperature: float, cold_temperature: float
    ) -> float:
        target_temperature = self.compute_average_temperature(
            hot_temperature, cold_temperature
        )
        first = True
        for temperature, conductivity in zip(
            self.thermal_conductivity_data["temperature"],
            self.thermal_conductivity_data["thermal_conductivity"]
        ):
            if first:
                if target_temperature < temperature:
                    raise ValueError(
                        "Temperature too cold for foam data..."
                    )
                first = False
            if temperature >= target_temperature:
                return conductivity
        raise ValueError("Temperature too hot for foam data...")

    @classmethod
    def rohacell(cls, thickness: float) -> VariableFoamInsulation:
        return cls(thickness, "rohacell")

    @classmethod
    def polyurethane(cls, thickness: float) -> VariableFoamInsulation:
        return cls(thickness, "polyurethane")


def main():
    pass


if __name__ == "__main__":
    main()


# End
