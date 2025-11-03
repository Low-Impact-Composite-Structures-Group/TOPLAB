
from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from src.thermodynamics.thermal_resistances import SeriesResistances


class MissionSection(Protocol):
    temperature: float


class TankState(Protocol):
    temperature: float


class FuelTank(Protocol):
    ...

    @abstractmethod
    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        ...


class Insulation(Protocol):
    
    @abstractmethod
    def compute_thermal_resistances(
        self,
        temperatures: list[float],
        tank: FuelTank
    ) -> list[float]:
        ...


class InternalModel(Protocol):

    @abstractmethod
    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> float:
        ...


class ExternalModel(Protocol):

    @abstractmethod
    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        mission_section: MissionSection,
        surface_temperature: float
    ) -> float:
        ...


class NoInsulation:
    def compute_thermal_resistances(self, temperatures, tank) -> list[float]:
        return []  # No insulation layers → no additional resistances


@dataclass
class ThermodynamicModel:
    internal_model: InternalModel
    external_model: ExternalModel
    insulation: Insulation

    insulation_layers: int = 12
    max_iterations: int = int(1e3)

    def __post_init__(self):
        if self.insulation is None:
            self.insulation = NoInsulation()
            self.insulation_layers = 0

    def compute_heat_flux(
        self,
        tank: FuelTank,
        tank_state: TankState,
        mission_section: MissionSection
    ) -> tuple[float, list]:
        
        temperatures = self.define_initial_temperatures(
            tank_state.temperature, mission_section.temperature
        )

        for _ in range(self.max_iterations):
            
            # Compute the thermal resistances that are in series in the 
            # fuel tank
            series_resistances = self.compute_thermal_resistances(
                tank, tank_state, mission_section, temperatures
            )

            # Compute the total heat flux
            heat_flux = self.compute_total_tank_heat_flux(
                mission_section.temperature,
                tank_state.temperature,
                SeriesResistances().compute_equivalent_resistance(
                    series_resistances
                )
            )

            # If there is no insulation, the heat flux convergence iterations are not required
            if isinstance(self.insulation, NoInsulation):
                return heat_flux, temperatures

            # Compute new temperatures
            new_temperatures = self.compute_new_temperatures(
                self.construct_a_matrix(
                    self.insulation_layers
                ),
                self.construct_y_vector(
                    series_resistances,
                    tank_state.temperature,
                    mission_section.temperature,
                    heat_flux
                )
            )
            
            # Test convergence
            if self.temperatures_have_converged(
                temperatures, new_temperatures
            ):
                return heat_flux, temperatures
            temperatures = new_temperatures
        raise StopIteration("Reached maximum number of iterations.")

    def compute_thermal_resistances(
        self,
        tank: FuelTank,
        tank_state: TankState,
        mission_section: MissionSection,
        temperatures
    ) -> list[float]:
        
        internal_resistance = self.internal_model.compute_equivalent_resistance(
            tank, tank_state, temperatures[0]
        )
        insulation_resistances = self.insulation.compute_thermal_resistances(
            temperatures, tank
        )
        external_resistance = self.external_model.compute_equivalent_resistance(
            tank, mission_section, temperatures[-1]
        )

        return [
            internal_resistance,
            *insulation_resistances,
            external_resistance,
        ]
            
    def define_initial_temperatures(
        self, fuel_temperature: float, ambient_temperature: float
    ) -> list[float]:
        if self.insulation_layers == 0:
            return [fuel_temperature, ambient_temperature]
        
        temp_difference = ambient_temperature - fuel_temperature
        temperature_step = temp_difference / self.insulation_layers
        return [
            fuel_temperature + i * temperature_step
            for i in range(self.insulation_layers + 1)
        ]

    @staticmethod
    def construct_a_matrix(insulation_layers: int) -> npt.ArrayLike:
        """Method to construct the matrix required to compute the new
        temperatures, for the tank inner and outer wall, and when
        implemented, the different layers in the insulation.

        Args:
            insulation_layers (int): number of insulation layers.
        """
        a = np.zeros((insulation_layers+2, insulation_layers+1))
        for i in range(insulation_layers+1):
            a[i,i] = 1
            a[i+1,i] = -1
        return a

    @staticmethod
    def construct_y_vector(
        resistances: list[float],
        fuel_temperature: float,
        ambient_temperature: float,
        heat_flux: float
    ) -> npt.ArrayLike:
        """Method to construct the y vector, of the thermal resistances,
        heat flux and boundary temperatures.

        Args:
            resistances (List[float]): List of all the resistances, note
            that the first value is the fuel resistance, then goes 
            through the insulation to the outer wall and the ambient 
            resistance.
            fuel_temperature (float): Temperature of the fuel.
            ambient_temperature (float): Ambient temperature.
            heat_flux (float): Total heat flux through the tank wall.

        Returns:
            npt.ArrayLike: Target vector.
        """
        y = np.ones((len(resistances), 1))
        for i, resistance in enumerate(resistances):
            y[i][0] = heat_flux * resistance
        y[0][0] += fuel_temperature
        y[-1][0] -= ambient_temperature
        return y

    @staticmethod
    def compute_total_tank_heat_flux(
        ambient_temperature: float,
        fuel_temperature: float,
        total_resistance: float
    ) -> float:
        """Static method to compute the total heat flux going into the 
        fuel tank.

        Args:
            ambient_temperature (float): Outside temperature.
            fuel_temperature (float): Fuel temperature.
            total_resistance (float): Total heat resistance of the tank.

        Returns:
            float: Heat flux into the fuel tank.
        """
        return (
            (ambient_temperature - fuel_temperature) / total_resistance
        )

    @staticmethod
    def temperatures_have_converged(
        old_temperatures: list, new_temperatures: list, threshold=1.0
    ) -> bool:
        """Verify if the temperatures have converged in the iterative
        process.

        Args:
            old_temperatures (list): Temperatures of the previous 
            iteration.
            new_temperatures (list): Temperatures of the new 
            iteration.
            threshold (float, optional): Threshold to define if the 
            temperatures have converged. Defaults to 0.1.

        Returns:
            bool: Bool defining if the temperatures have converged.
        """
        for old, new in zip(old_temperatures, new_temperatures):
            if abs(old - new) > threshold:
                return False
        return True

    @staticmethod
    def compute_new_temperatures(
        a_matrix: npt.ArrayLike, y_vector: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Compute the new temperatures through the thickness of the 
        fuel tank. This is computed with the normal equation used in 
        leased square methods.

        Args:
            a_matrix (npt.ArrayLike): A matrix with the zeros and ones.
            y_vector (npt.ArrayLike): Thermal resistance, heat flux, and
            temperatures vector.

        Returns:
            npt.ArrayLike: Array with the new suggested temperatures.
        """
        step1 = np.linalg.matrix_power(
            np.dot(np.transpose(a_matrix), a_matrix), -1
        )
        step2 = np.dot(step1, np.transpose(a_matrix))
        temperatures = [
            temp[0]
            for temp in np.dot(step2, y_vector)
        ]
        return temperatures


def main():
    pass


if __name__ == "__main__":
    main()


# End
