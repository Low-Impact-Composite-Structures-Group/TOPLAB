from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from src.thermodynamics.heat_transfer_modes import (GasPhaseConvection,
                                                    LiquidPhaseConvection,
                                                    NaturalConvection,
                                                    RohsenowNaturalConvection)
from src.thermodynamics.thermal_resistances import (ParallelResistances,
                                                    ThermalResistance)


class FuelTank(Protocol):
    characteristic_length: float
    characteristic_height: float
    surface_area: float

    @abstractmethod
    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_gas_wetted_surface(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_zone_1_length(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_zone_2_length(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_zone_3_length(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_zone_1_area(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_zone_2_area(self, fuel_height: float) -> float:
        ...

    @abstractmethod
    def compute_zone_3_area(self, fuel_height: float) -> float:
        ...


class Hydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen


class TankState(Protocol):
    fuel_height: float
    hydrogen: Hydrogen
    is_full: bool
    is_empty: bool


class InternalModel(Protocol):

    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> float:
        return ParallelResistances().compute_equivalent_resistance(
            [
                resistance.value
                for resistance in self.get_thermal_resistances(
                    tank, tank_state, surface_temperature
                )
            ]
        )

    def get_thermal_resistances(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        # Only gas in the tank
        if tank_state.is_empty:
            return [self.create_gas_resistance(
                tank, tank_state, surface_temperature
            )]
        # Full liquid tank
        if tank_state.is_full:
            return self.create_liquid_resistance(
                tank, tank_state, surface_temperature
            )
        # Partial gas partial liquid tank
        return self.create_two_phase_thermal_resistances(
            tank, tank_state, surface_temperature
        )

    def create_two_phase_thermal_resistances(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        liquid_resistance = self.create_liquid_resistance(
            tank, tank_state, surface_temperature
        )
        gas_resistance = self.create_gas_resistance(
            tank, tank_state, surface_temperature
        )
        return [*liquid_resistance, gas_resistance]

    def create_gas_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> ThermalResistance:
        gas_convection = GasPhaseConvection(
            tank_state.hydrogen.gas,
            tank.characteristic_height - tank_state.fuel_height,
            surface_temperature
        )
        gas_resistance = ThermalResistance(
            gas_convection.heat_transfer_coefficient,
            tank.compute_gas_wetted_surface(
                tank_state.fuel_height
            )
        )
        return gas_resistance

    @abstractmethod
    def create_liquid_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        ...


class SingleZoneModel(InternalModel):

    def create_liquid_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        liquid_convection = LiquidPhaseConvection(
            tank_state.hydrogen.liquid,
            tank_state.fuel_height,
            surface_temperature
        )
        liquid_resistance = ThermalResistance(
            liquid_convection.heat_transfer_coefficient,
            tank.compute_fuel_wetted_surface(
                tank_state.fuel_height
            )
        )
        return [liquid_resistance]


class ThreeZoneModel(InternalModel):

    def create_liquid_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        convective_motions = self.create_convective_motions(
            tank, tank_state, surface_temperature
        )
        surfaces = self.create_surfaces(tank, tank_state)
        resistances = [
            ThermalResistance(
                convection.heat_transfer_coefficient, surface
            )
            for convection, surface in zip(convective_motions, surfaces)
        ]
        return resistances

    def create_surfaces(
        self,
        tank: FuelTank,
        tank_state: TankState
    ) -> list[float]:
        surfaces = [
            tank.compute_zone_1_area(tank_state.fuel_height),
            tank.compute_zone_2_area(tank_state.fuel_height),
            tank.compute_zone_3_area(tank_state.fuel_height)
        ]

        return surfaces

    def create_convective_motions(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[NaturalConvection]:
        convective_motions = [
            RohsenowNaturalConvection(
                tank_state.hydrogen.liquid,
                tank.compute_zone_1_length(tank_state.fuel_height),
                surface_temperature
            ),
            RohsenowNaturalConvection(
                tank_state.hydrogen.liquid,
                tank.compute_zone_2_length(tank_state.fuel_height),
                surface_temperature
            ),
            RohsenowNaturalConvection(
                tank_state.hydrogen.liquid,
                tank.compute_zone_3_length(tank_state.fuel_height),
                surface_temperature
            )
        ]

        return convective_motions


class InternalModelFactory:

    _models = {
        "single_zone_model": SingleZoneModel(),
        "three_zone_model": ThreeZoneModel(),
    }

    @property
    def _available(self):
        return ", ".join(self._models.keys())

    def create_model(self, type: str):
        model = self._models.get(type)

        if model is not None: return model

        raise ValueError(
            f"'{type}' is an invalid internal thermal model.\n"
            f"Available models are: {self._available}"
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
