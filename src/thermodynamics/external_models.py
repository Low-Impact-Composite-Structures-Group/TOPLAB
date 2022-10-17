


from abc import abstractmethod
from typing import Protocol

from src.thermodynamics.heat_transfer_modes import ForcedConvection
from src.thermodynamics.thermal_resistances import (SeriesResistances,
                                                    ThermalResistance)


class FuelTank(Protocol):
    characteristic_length: float
    exposed_surface: float


class Ambient(Protocol):
    ...


class MissionSection(Protocol):
    ambient: Ambient
    flight_speed: float


class ExternalModel(Protocol):

    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> float:
        return SeriesResistances().compute_equivalent_resistance(
            [
                resistance.value
                for resistance in self.get_thermal_resistances(
                    tank, mission_section, surface_temperature
                )
            ]
        )
    
    @abstractmethod
    def get_thermal_resistances(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        ...


class ForcedConvectionModel(ExternalModel):

    def get_thermal_resistances(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        exposed_convection = ForcedConvection(
            mission_section.ambient,
            tank.characteristic_length,
            surface_temperature,
            mission_section.flight_speed
        )
        exposed_resistance = ThermalResistance(
            exposed_convection.heat_transfer_coefficient,
            tank.exposed_surface
        )
        return [exposed_resistance]


def main():
    pass


if __name__ == "__main__":
    main()


# End
