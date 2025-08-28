from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from src.thermodynamics.thermal_resistances import SeriesResistances, ParallelResistances


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


@dataclass
class ThermodynamicModel:
    internal_model: InternalModel
    external_model: ExternalModel
    insulation: Insulation
    insulation_layers: int = 12
    max_iterations: int = int(1e3)
    constant_heat_flux: float = None

    def compute_heat_flux(
        self,
        tank: FuelTank,
        tank_state: TankState,
        mission_section: MissionSection
    ) -> tuple[float, list]:

        if self.constant_heat_flux is not None:
            # Use the constant heat flux value
            heat_flux = self.constant_heat_flux

            # Define the number of temperature interfaces based on insulation_layers
            num_interfaces = self.insulation_layers

            # Define initial temperatures with the fixed number of interfaces
            temperatures = self.define_initial_temperatures(
                tank_state.temperature, mission_section.temperature, num_interfaces
            )
            return heat_flux, temperatures

        # Define the number of temperature interfaces based on insulation_layers
        num_interfaces = self.insulation_layers

        # Define initial temperatures with the fixed number of interfaces
        temperatures = self.define_initial_temperatures(
            tank_state.temperature, mission_section.temperature, num_interfaces
        )

        for iteration in range(self.max_iterations):

            # Compute the thermal resistances that are in series in the fuel tank
            thermal_resistances = self.compute_thermal_resistances(
                tank, tank_state, mission_section, temperatures
            )

            # Compute the total heat flux
            total_resistance = SeriesResistances().compute_equivalent_resistance(thermal_resistances)

            heat_flux = self.compute_total_tank_heat_flux(
                mission_section.temperature,
                tank_state.temperature,
                total_resistance
            )            # Compute new temperatures
            # The number of resistances determines both matrix dimensions
            num_resistances = len(thermal_resistances)

            # Create y vector first to get its dimensions
            y_vector = self.construct_y_vector(
                thermal_resistances,
                tank_state.temperature,
                mission_section.temperature,
                heat_flux
            )

            # Now create A matrix with dimensions that match y_vector
            # A matrix should have dimensions (num_resistances, num_resistances-1)
            a_matrix = np.zeros((num_resistances, num_resistances-1))
            for i in range(num_resistances-1):
                a_matrix[i,i] = 1
                a_matrix[i+1,i] = -1

            try:
                new_temperatures = self.compute_new_temperatures(
                    a_matrix,
                    y_vector
                )

                # Test convergence
                has_converged = self.temperatures_have_converged(
                    temperatures, new_temperatures
                )

                if has_converged:
                    return heat_flux, temperatures

                temperatures = new_temperatures
            except Exception as e:
                raise
        raise StopIteration("Reached maximum number of iterations.")

    def compute_thermal_resistances(
        self,
        tank: FuelTank,
        tank_state: TankState,
        mission_section: MissionSection,
        temperatures
    ) -> list[float]:

        # Start with internal model resistance
        internal_resistance = self.internal_model.compute_equivalent_resistance(
            tank, tank_state, temperatures[0]
        )
        thermal_resistances = [internal_resistance]

        # Add liner resistance if present
        has_liner = hasattr(tank, 'liner') and tank.liner is not None

        if has_liner and hasattr(tank.liner, 'compute_thermal_resistance'):
            # Calculate liner thermal resistance between internal temperature and first layer
            liner_resistance = tank.liner.compute_thermal_resistance(
                tank_state.temperature, temperatures[0]
            )
            thermal_resistances.append(liner_resistance)

        # We need to select temperatures for the insulation layers
        # Ensure we use only the needed number of temperature interfaces
        # Skip the first temperature if we have a liner
        start_idx = 1 if has_liner else 0
        insulation_temps = temperatures[start_idx:start_idx+self.insulation_layers+1]

        # Add insulation resistances using only the needed temperatures
        insulation_resistances = self.insulation.compute_thermal_resistances(
            insulation_temps, tank
        )
        thermal_resistances.extend(insulation_resistances)

        # Add external model resistance
        external_resistance = self.external_model.compute_equivalent_resistance(
            tank, mission_section, temperatures[-1]
        )
        thermal_resistances.append(external_resistance)

        return thermal_resistances

    def define_initial_temperatures(
        self, fuel_temperature: float, ambient_temperature: float, num_layers=None
    ) -> list[float]:
        """Define initial temperatures for each interface between resistances.

        Args:
            fuel_temperature: Temperature of the fuel
            ambient_temperature: Ambient temperature
            num_layers: Optional number of layers to use (defaults to self.insulation_layers)

        Returns:
            list[float]: List of temperatures at each interface
        """
        if num_layers is None:
            num_layers = self.insulation_layers

        temp_difference = ambient_temperature - fuel_temperature
        temperature_step = temp_difference / num_layers
        return [
            fuel_temperature + i * temperature_step
            for i in range(num_layers + 1)
        ]

    @staticmethod
    def construct_a_matrix(num_resistances: int) -> npt.ArrayLike:
        """Method to construct the matrix required to compute the new
        temperatures, for the tank inner and outer wall, and when
        implemented, the different layers in the insulation.

        Args:
            num_resistances (int): Number of resistances in the thermal model.
                                  This can include insulation layers plus other components
                                  like liner, internal and external resistances.
        """
        a = np.zeros((num_resistances, num_resistances-1))
        for i in range(num_resistances-1):
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
        least square methods.

        Args:
            a_matrix (npt.ArrayLike): A matrix with the zeros and ones.
            y_vector (npt.ArrayLike): Thermal resistance, heat flux, and
            temperatures vector.

        Returns:
            npt.ArrayLike: Array with the new suggested temperatures.
        """
        # Ensure dimensions are compatible
        if a_matrix.shape[0] != y_vector.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: a_matrix shape {a_matrix.shape}, y_vector shape {y_vector.shape}")

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
