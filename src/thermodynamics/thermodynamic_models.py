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
    liner_layers: int = 1
    max_iterations: int = int(1e3)
    constant_heat_flux: float = None


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
    liner_layers: int = 5
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

            # Define the number of temperature interfaces based on total layers
            num_interfaces = self._compute_total_temperature_interfaces(tank)

            # Define initial temperatures with the calculated number of interfaces
            temperatures = self.define_initial_temperatures(
                tank_state.temperature, mission_section.temperature, num_interfaces
            )
            return heat_flux, temperatures

        # Define the number of temperature interfaces based on total layers
        num_interfaces = self._compute_total_temperature_interfaces(tank)

        # Define initial temperatures with the calculated number of interfaces
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

    def _compute_total_temperature_interfaces(self, tank: FuelTank) -> int:
        """Compute the total number of temperature interfaces needed.

        This accounts for liner layers and insulation layers.

        Args:
            tank: Tank object to check for liner presence

        Returns:
            int: Total number of temperature interfaces needed
        """
        # Check if tank has a liner
        has_liner = hasattr(tank, 'liner') and tank.liner is not None

        if has_liner:
            # Total interfaces = liner_layers + insulation_layers
            return self.liner_layers + self.insulation_layers
        else:
            # Only insulation layers
            return self.insulation_layers

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

        # Detailed breakdown of internal resistance
        if hasattr(self.internal_model, 'get_thermal_resistances'):
            internal_resistances = self.internal_model.get_thermal_resistances(
                tank, tank_state, temperatures[0]
            )
            for i, resistance in enumerate(internal_resistances):
                # Try to identify the heat transfer mode
                if hasattr(resistance, 'heat_transfer_coefficient'):
                    htc = resistance.heat_transfer_coefficient
                    area = resistance.surface_area

        # Add liner resistances if present (potentially multiple layers)
        has_liner = hasattr(tank, 'liner') and tank.liner is not None
        temp_idx = 0  # Track which temperature index we're at

        if has_liner:

            if self.liner_layers == 1:
                # Use single resistance method for backward compatibility
                if hasattr(tank.liner, 'compute_thermal_resistance'):
                    liner_resistance = tank.liner.compute_thermal_resistance(
                        temperatures[temp_idx], temperatures[temp_idx + 1]
                    )
                    thermal_resistances.append(liner_resistance)
                    temp_idx += 1

                    # Get liner details
                    if hasattr(tank.liner, 'thickness') and tank.liner.thickness:
                        liner_thickness = tank.liner.thickness
                    if hasattr(tank.liner, 'material'):
                        liner_material = tank.liner.material

                    # Calculate thermal conductivity and HTC
                    if hasattr(tank.liner, 'compute_thermal_conductivity'):
                        k = tank.liner.compute_thermal_conductivity(
                            temperatures[temp_idx-1], temperatures[temp_idx]
                        )

                    if hasattr(tank.liner, 'compute_heat_transfer_coefficient'):
                        htc = tank.liner.compute_heat_transfer_coefficient(
                            temperatures[temp_idx-1], temperatures[temp_idx]
                        )
            else:
                # Use multiple resistance method for discretized liner
                if hasattr(tank.liner, 'compute_thermal_resistances'):
                    liner_temps = temperatures[temp_idx:temp_idx + self.liner_layers + 1]
                    liner_resistances = tank.liner.compute_thermal_resistances(
                        liner_temps, self.liner_layers
                    )
                    thermal_resistances.extend(liner_resistances)
                    temp_idx += self.liner_layers
                else:
                    # Fallback to single resistance repeated for each layer
                    total_liner_resistance = 0
                    for i in range(self.liner_layers):
                        liner_resistance = tank.liner.compute_thermal_resistance(
                            temperatures[temp_idx], temperatures[temp_idx + 1]
                        )
                        thermal_resistances.append(liner_resistance)
                        total_liner_resistance += liner_resistance
                        temp_idx += 1
        # Select temperatures for the insulation layers
        # Start from the current temperature index
        insulation_temps = temperatures[temp_idx:temp_idx+self.insulation_layers+1]

        # Add insulation resistances using the remaining temperatures
        insulation_resistances = self.insulation.compute_thermal_resistances(
            insulation_temps, tank
        )
        thermal_resistances.extend(insulation_resistances)

        total_insulation_resistance = 0
        for i, resistance in enumerate(insulation_resistances):
            total_insulation_resistance += resistance

        # Get insulation properties if available
        if hasattr(self.insulation, 'compute_thermal_conductivity'):
            k_insulation = self.insulation.compute_thermal_conductivity(
                insulation_temps[0], insulation_temps[-1]
            )

        # Add external model resistance
        external_resistance = self.external_model.compute_equivalent_resistance(
            tank, mission_section, temperatures[-1]
        )
        thermal_resistances.append(external_resistance)

        # Detailed breakdown of external resistance
        if hasattr(self.external_model, 'get_convective_motions'):
            convective_resistances = self.external_model.get_convective_motions(
                tank, mission_section, temperatures[-1]
            )
            for i, resistance in enumerate(convective_resistances):
                if hasattr(resistance, 'heat_transfer_coefficient'):
                    htc = resistance.heat_transfer_coefficient
                    area = resistance.surface_area

        # Add radiation resistance details
        if hasattr(self.external_model, 'define_radiation_resistance'):
            radiation_resistance = self.external_model.define_radiation_resistance(
                tank, mission_section, temperatures[-1]
            )

        # Summary
        total_resistance = sum(thermal_resistances)

        if has_liner:
            liner_total = sum(thermal_resistances[1:1+self.liner_layers])
            insulation_start_idx = 1 + self.liner_layers
        else:
            insulation_start_idx = 1

        insulation_total = sum(thermal_resistances[insulation_start_idx:insulation_start_idx+self.insulation_layers])

        # Calculate overall heat flux
        total_temp_diff = mission_section.temperature - tank_state.temperature
        heat_flux = total_temp_diff / total_resistance

        return thermal_resistances

    def define_initial_temperatures(
        self, fuel_temperature: float, ambient_temperature: float, num_layers=None
    ) -> list[float]:
        """Define initial temperatures for each interface between resistances.

        Args:
            fuel_temperature: Temperature of the fuel
            ambient_temperature: Ambient temperature
            num_layers: Optional number of layers to use (defaults to total layers)

        Returns:
            list[float]: List of temperatures at each interface
        """
        if num_layers is None:
            # Use the provided value (calculated by _compute_total_temperature_interfaces)
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
