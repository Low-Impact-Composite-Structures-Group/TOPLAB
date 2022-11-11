

from abc import abstractmethod
from typing import Protocol

from src.thermodynamics.heat_transfer_modes import (ForcedConvection,
                                                    NaturalCylinderConvection,
                                                    NaturalSphereConvection,
                                                    Radiation)
from src.thermodynamics.thermal_resistances import (ParallelResistances,
                                                    SeriesResistances,
                                                    ThermalResistance)


class FuelTank(Protocol):
    characteristic_length: float
    characteristic_height: float
    exposed_surface: float
    surface_area: float


class Ambient(Protocol):
    temperature: float


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
        convection_resistance = self.equivalent_convection_resistance(
            tank, mission_section, surface_temperature
        )
        radiation_resistance = self.define_radiation_resistance(
            tank, mission_section, surface_temperature
        )
        return SeriesResistances().compute_equivalent_resistance(
            [convection_resistance, radiation_resistance]
        )
    
    def equivalent_convection_resistance(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> float:
        return ParallelResistances().compute_equivalent_resistance(
            [
                resistance.value
                for resistance in self.get_convective_motions(
                    tank, mission_section, surface_temperature
                )
            ]
        )

    def define_radiation_resistance(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> float:
        radiation = Radiation(
            surface_temperature,
            mission_section.ambient.temperature
        )
        return ThermalResistance(
            radiation.heat_transfer_coefficient,
            tank.surface_area
        ).value
  
    @abstractmethod
    def get_convective_motions(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        ...


class ForcedConvectionModel(ExternalModel):

    def get_convective_motions(
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


class NaturalConvectionModel(ExternalModel):

    def get_convective_motions(
        self, 
        tank: FuelTank, 
        mission_section: MissionSection, 
        surface_temperature: float
    ) -> list[ThermalResistance]:
        cylinder_convection = NaturalCylinderConvection(
            mission_section.ambient,
            tank.characteristic_height,
            surface_temperature
        )
        cylinder_convection = ThermalResistance(
            cylinder_convection.heat_transfer_coefficient,
            tank.exposed_surface
        )
        spheres_convection = NaturalSphereConvection(
            mission_section.ambient,
            tank.characteristic_height,
            surface_temperature
        )
        spheres_convection = ThermalResistance(
            spheres_convection.heat_transfer_coefficient,
            tank.surface_area - tank.exposed_surface
        )
        return [cylinder_convection, spheres_convection]


def main():
    pass


if __name__ == "__main__":
    main()


# End
