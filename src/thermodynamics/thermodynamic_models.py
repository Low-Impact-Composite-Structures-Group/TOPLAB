
from abc import abstractmethod
from typing import Protocol

from src.thermodynamics.heat_transfer_modes import (ForcedConvection,
                                                    GasPhaseConvection,
                                                    LiquidPhaseConvection)
from src.thermodynamics.thermal_resistances import ThermalResistance


class FuelTank(Protocol):
    characteristic_height: float
    characteristic_length: float
    surface_area: float
    exposed_surface: float
    covered_surface: float

    @abstractmethod
    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        ...


class ConvectiveMedium(Protocol):
    ...


class TwoPhaseHydrogen(Protocol):
    liquid: ConvectiveMedium
    gas: ConvectiveMedium


class InternalThermodynamicFactory(Protocol):

    @abstractmethod
    def get_thermal_resistances(
        self,
        fuel_tank: FuelTank,
        hydrogen: ConvectiveMedium,
        fuel_height: float,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        ...


class SingleZoneFactory(InternalThermodynamicFactory):
	
    def get_thermal_resistances(
        self,
        fuel_tank: FuelTank,
        hydrogen: ConvectiveMedium,
        fuel_height: float,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        # Only gas in the tank
        if fuel_height == 0: 
            return self.create_gas_thermal_resistance(
                fuel_tank, hydrogen, surface_temperature
            )
        # Full liquid tank
        if fuel_height == fuel_tank.characteristic_height:
            return self.create_liquid_thermal_resistance(
                fuel_tank, hydrogen, surface_temperature
            )
        # Partial gas partial liquid tank 
        return self.create_two_phase_thermal_resistances(
            fuel_tank, hydrogen, surface_temperature, fuel_height
        )

    def create_gas_thermal_resistance(
        self,
        fuel_tank: FuelTank,
        hydrogen: ConvectiveMedium,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        convection = GasPhaseConvection(
            hydrogen,
            fuel_tank.characteristic_height,
            surface_temperature
        )
        resistance = ThermalResistance(
            convection.heat_transfer_coefficient,
            fuel_tank.surface_area
        )
        return [resistance]

    def create_liquid_thermal_resistance(
        self,
        fuel_tank: FuelTank,
        hydrogen: ConvectiveMedium,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        convection = LiquidPhaseConvection(
            hydrogen,
            fuel_tank,
            surface_temperature
        )
        resistance = ThermalResistance(
            convection.heat_transfer_coefficient,
            fuel_tank.surface_area,
        )
        return [resistance]

    def create_two_phase_thermal_resistances(
        self,
        fuel_tank: FuelTank,
        hydrogen: TwoPhaseHydrogen,
        surface_temperature: float,
        fuel_height: float
    ) -> list[ThermalResistance]:
        liquid_convection = LiquidPhaseConvection(
            hydrogen.liquid,
            fuel_height,
            surface_temperature
        )
        liquid_surface = fuel_tank.compute_fuel_wetted_surface(
            fuel_height
        )
        liquid_resistance = ThermalResistance(
            liquid_convection.heat_transfer_coefficient,
            liquid_surface
        )
        gas_convection = GasPhaseConvection(
            hydrogen.gas,
            fuel_tank.characteristic_height - fuel_height,
            surface_temperature
        )
        gas_surface = fuel_tank.surface_area - liquid_surface
        gas_resistance = ThermalResistance(
            gas_convection.heat_transfer_coefficient,
            gas_surface
        )
        return [liquid_resistance, gas_resistance]


class ExternalThermodynamicFactory(Protocol):
    
    @abstractmethod
    def get_thermal_resistances(
        self,
        fuel_tank: FuelTank,
        ambient: ConvectiveMedium,
        flight_speed: float,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        ...


class ForcedConvectionFactory(ExternalThermodynamicFactory):

    def get_thermal_resistances(
        self,
        fuel_tank: FuelTank,
        ambient: ConvectiveMedium,
        flight_speed: float,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        exposed_convection = ForcedConvection(
            ambient,
            fuel_tank.characteristic_length,
            surface_temperature,
            flight_speed
        )
        exposed_resistance = ThermalResistance(
            exposed_convection.heat_transfer_coefficient,
            fuel_tank.exposed_surface
        )
        return [exposed_resistance]


def main():
    pass


if __name__ == "__main__":
    main()


# End
