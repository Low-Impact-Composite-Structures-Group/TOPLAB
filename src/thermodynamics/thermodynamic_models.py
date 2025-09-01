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

        print(f"\n🔥 THERMAL RESISTANCE ANALYSIS")
        print(f"{'='*50}")
        print(f"Tank surface temperature: {temperatures[0]:.2f} K")
        print(f"Ambient temperature: {mission_section.temperature:.2f} K")
        print(f"Temperature difference: {mission_section.temperature - temperatures[0]:.2f} K")

        # Start with internal model resistance
        internal_resistance = self.internal_model.compute_equivalent_resistance(
            tank, tank_state, temperatures[0]
        )
        thermal_resistances = [internal_resistance]

        print(f"\n🔄 1. INTERNAL HEAT TRANSFER MODES:")
        print(f"   Internal resistance: {internal_resistance:.6e} K/W")

        # Detailed breakdown of internal resistance
        if hasattr(self.internal_model, 'get_thermal_resistances'):
            internal_resistances = self.internal_model.get_thermal_resistances(
                tank, tank_state, temperatures[0]
            )
            print(f"   Internal resistance breakdown:")
            for i, resistance in enumerate(internal_resistances):
                print(f"     Mode {i+1}: {resistance.value:.6e} K/W")
                # Try to identify the heat transfer mode
                if hasattr(resistance, 'heat_transfer_coefficient'):
                    htc = resistance.heat_transfer_coefficient
                    area = resistance.surface_area
                    print(f"       HTC: {htc:.2e} W/(m²·K), Area: {area:.3f} m²")

        # Add liner resistances if present (potentially multiple layers)
        has_liner = hasattr(tank, 'liner') and tank.liner is not None
        temp_idx = 0  # Track which temperature index we're at

        if has_liner:
            print(f"\n🛡️  2. LINER THERMAL RESISTANCE:")
            print(f"   Number of liner layers: {self.liner_layers}")

            if self.liner_layers == 1:
                # Use single resistance method for backward compatibility
                if hasattr(tank.liner, 'compute_thermal_resistance'):
                    liner_resistance = tank.liner.compute_thermal_resistance(
                        temperatures[temp_idx], temperatures[temp_idx + 1]
                    )
                    thermal_resistances.append(liner_resistance)
                    temp_idx += 1

                    print(f"   Single liner resistance: {liner_resistance:.6e} K/W")

                    # Get liner details
                    if hasattr(tank.liner, 'thickness') and tank.liner.thickness:
                        print(f"   Liner thickness: {tank.liner.thickness*1000:.2f} mm")
                    if hasattr(tank.liner, 'material'):
                        print(f"   Liner material: {tank.liner.material.__class__.__name__}")

                    # Calculate thermal conductivity and HTC
                    if hasattr(tank.liner, 'compute_thermal_conductivity'):
                        k = tank.liner.compute_thermal_conductivity(
                            temperatures[temp_idx-1], temperatures[temp_idx]
                        )
                        print(f"   Thermal conductivity: {k:.2f} W/(m·K)")

                    if hasattr(tank.liner, 'compute_heat_transfer_coefficient'):
                        htc = tank.liner.compute_heat_transfer_coefficient(
                            temperatures[temp_idx-1], temperatures[temp_idx]
                        )
                        print(f"   Heat transfer coefficient: {htc:.2e} W/(m²·K)")
                        print(f"   Surface area: {tank.surface_area:.3f} m²")
            else:
                # Use multiple resistance method for discretized liner
                if hasattr(tank.liner, 'compute_thermal_resistances'):
                    liner_temps = temperatures[temp_idx:temp_idx + self.liner_layers + 1]
                    liner_resistances = tank.liner.compute_thermal_resistances(
                        liner_temps, self.liner_layers
                    )
                    thermal_resistances.extend(liner_resistances)
                    temp_idx += self.liner_layers

                    print(f"   Multiple liner resistances:")
                    for i, resistance in enumerate(liner_resistances):
                        print(f"     Layer {i+1}: {resistance:.6e} K/W")
                    print(f"   Total liner resistance: {sum(liner_resistances):.6e} K/W")
                else:
                    # Fallback to single resistance repeated for each layer
                    print(f"   Fallback: Using single resistance per layer")
                    total_liner_resistance = 0
                    for i in range(self.liner_layers):
                        liner_resistance = tank.liner.compute_thermal_resistance(
                            temperatures[temp_idx], temperatures[temp_idx + 1]
                        )
                        thermal_resistances.append(liner_resistance)
                        total_liner_resistance += liner_resistance
                        temp_idx += 1
                        print(f"     Layer {i+1}: {liner_resistance:.6e} K/W")
                    print(f"   Total liner resistance: {total_liner_resistance:.6e} K/W")
        else:
            print(f"\n🛡️  2. LINER: None present")

        # Select temperatures for the insulation layers
        # Start from the current temperature index
        insulation_temps = temperatures[temp_idx:temp_idx+self.insulation_layers+1]

        print(f"\n🧊 3. INSULATION THERMAL RESISTANCE:")
        print(f"   Number of insulation layers: {self.insulation_layers}")
        print(f"   Temperature range: {insulation_temps[0]:.2f} K → {insulation_temps[-1]:.2f} K")

        # Add insulation resistances using the remaining temperatures
        insulation_resistances = self.insulation.compute_thermal_resistances(
            insulation_temps, tank
        )
        thermal_resistances.extend(insulation_resistances)

        print(f"   Insulation resistances:")
        total_insulation_resistance = 0
        for i, resistance in enumerate(insulation_resistances):
            total_insulation_resistance += resistance
            print(f"     Layer {i+1}: {resistance:.6e} K/W")
        print(f"   Total insulation resistance: {total_insulation_resistance:.6e} K/W")

        # Get insulation properties if available
        if hasattr(self.insulation, 'compute_thermal_conductivity'):
            k_insulation = self.insulation.compute_thermal_conductivity(
                insulation_temps[0], insulation_temps[-1]
            )
            print(f"   Insulation thermal conductivity: {k_insulation:.4f} W/(m·K)")

        # Add external model resistance
        external_resistance = self.external_model.compute_equivalent_resistance(
            tank, mission_section, temperatures[-1]
        )
        thermal_resistances.append(external_resistance)

        print(f"\n🌬️  4. EXTERNAL HEAT TRANSFER MODES:")
        print(f"   External resistance: {external_resistance:.6e} K/W")

        # Detailed breakdown of external resistance
        if hasattr(self.external_model, 'get_convective_motions'):
            convective_resistances = self.external_model.get_convective_motions(
                tank, mission_section, temperatures[-1]
            )
            print(f"   External convection breakdown:")
            for i, resistance in enumerate(convective_resistances):
                print(f"     Mode {i+1}: {resistance.value:.6e} K/W")
                if hasattr(resistance, 'heat_transfer_coefficient'):
                    htc = resistance.heat_transfer_coefficient
                    area = resistance.surface_area
                    print(f"       HTC: {htc:.2e} W/(m²·K), Area: {area:.3f} m²")

        # Add radiation resistance details
        if hasattr(self.external_model, 'define_radiation_resistance'):
            radiation_resistance = self.external_model.define_radiation_resistance(
                tank, mission_section, temperatures[-1]
            )
            print(f"   Radiation resistance: {radiation_resistance:.6e} K/W")

        # Summary
        total_resistance = sum(thermal_resistances)
        print(f"\n📊 THERMAL RESISTANCE SUMMARY:")
        print(f"{'='*50}")
        print(f"   Internal:     {internal_resistance:.6e} K/W ({internal_resistance/total_resistance*100:.1f}%)")

        if has_liner:
            liner_total = sum(thermal_resistances[1:1+self.liner_layers])
            print(f"   Liner:        {liner_total:.6e} K/W ({liner_total/total_resistance*100:.1f}%)")
            insulation_start_idx = 1 + self.liner_layers
        else:
            insulation_start_idx = 1

        insulation_total = sum(thermal_resistances[insulation_start_idx:insulation_start_idx+self.insulation_layers])
        print(f"   Insulation:   {insulation_total:.6e} K/W ({insulation_total/total_resistance*100:.1f}%)")
        print(f"   External:     {external_resistance:.6e} K/W ({external_resistance/total_resistance*100:.1f}%)")
        print(f"   TOTAL:        {total_resistance:.6e} K/W")

        # Calculate overall heat flux
        total_temp_diff = mission_section.temperature - tank_state.temperature
        heat_flux = total_temp_diff / total_resistance
        print(f"   Heat flux:    {heat_flux:.2f} W")
        print(f"   Heat flux density: {heat_flux/tank.surface_area:.2f} W/m²")

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
