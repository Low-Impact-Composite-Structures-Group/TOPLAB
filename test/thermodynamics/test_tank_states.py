import unittest
from dataclasses import dataclass
from plotting.plot_tank_states import PASCAL_TO_BAR

from src.fluids.hydrogen_retrievers import HydrogenRetriever
from src.thermodynamics.tank_states import SECONDS_TO_HOURS, InitialState, TankState, TankStates


@dataclass
class FuelTank:
    volume: float

    def compute_fuel_height(self, fuel_volume: float) -> float:
        return 13.11

    @classmethod
    def test_tank(cls):
        volume = 88.5

        return cls(volume)


class FuelFlow:
    ...


@dataclass
class StateDerivatives:
    pressure: float
    temperature: float
    heat_flux: float

    @classmethod
    def test_derivatives(cls, number: int):
        pressure = f"pres_{number}_der"
        temperature = f"temp_{number}_der"
        heat_flux = f"flux_{number}"

        return cls(pressure, temperature, heat_flux)


class DynamicModel:

    def compute_state_derivatives(
        self, tank_sate: TankState, fuel_flows: list[FuelFlow]
    ) -> StateDerivatives:
        return "test"


class TestInitialState(unittest.TestCase):

    def setUp(self) -> None:
        self.pressure = 150e3
        self.temperature = None
        self.fill = 0.85
        self.state = InitialState(
            self.pressure, self.temperature, self.fill
        )

    def test_get_hydrogen_properties(self):
        expected_value = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        actual_value = self.state.get_hydrogen_properties()
        self.assertEqual(expected_value, actual_value)

    def test_compute_fuel_mass(self):
        tank_volume = 33.3
        expected_value = (
            self.fill * self.state.hydrogen.liquid.density
            + (1 - self.fill) * self.state.hydrogen.gas.density
        ) * tank_volume
        actual_value = self.state.compute_fuel_mass(tank_volume)
        self.assertEqual(expected_value, actual_value)

        # Only gas phase so no fill
        pressure = 300e5
        temperature = 70
        fill = 0.0
        hydrogen = HydrogenRetriever().get_hydrogen_properties(
            pressure, temperature
        )
        expected_value = tank_volume * hydrogen.density
        state = InitialState(
            pressure, temperature, fill
        )
        actual_value = state.compute_fuel_mass(tank_volume)
        self.assertEqual(expected_value, actual_value)


class TestTankState(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = FuelTank.test_tank()
        self.temperature = None
        self.pressure = 1.4e5
        self.fuel_mass = 600
        self.fill = 0.07374514524884335
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        self.state = TankState(
            self.tank, self.temperature, self.pressure, self.fuel_mass
        )

    def test_fill(self):
        actual_value = self.state.fill
        expected_value = self.fill
        self.assertEqual(expected_value, actual_value)

        # Gas state
        temperature = 70
        pressure = 300e5
        fuel_mass = 500
        state = TankState(
            self.tank, temperature, pressure, fuel_mass
        )
        expected_value = 0
        actual_value = state.fill
        self.assertEqual(expected_value, actual_value)

        # Liquid state
        temperature = 25
        pressure = 30e5
        fuel_mass = 500
        state = TankState(
            self.tank, temperature, pressure, fuel_mass
        )
        expected_value = 1.0
        actual_value = state.fill
        self.assertEqual(expected_value, actual_value)

    def test_volume(self):
        expected_value = self.tank.volume
        actual_value = self.state.volume
        self.assertEqual(expected_value, actual_value)

    def test_liquid_mass(self):
        expected_value = (
            self.fill * self.tank.volume * self.hydrogen.liquid.density
        )
        actual_value = self.state.liquid_mass
        self.assertEqual(expected_value, actual_value)

        # Gas state
        temperature = 70
        pressure = 300e5
        fuel_mass = 500
        state = TankState(
            self.tank, temperature, pressure, fuel_mass
        )
        expected_value = 0
        actual_value = state.liquid_mass
        self.assertEqual(expected_value, actual_value)

    def test_gas_mass(self):
        expected_value = (
            (1 - self.fill)
            * self.tank.volume * self.hydrogen.gas.density
        )
        actual_value = self.state.gas_mass
        self.assertEqual(expected_value, actual_value)

        # Liquid state
        temperature = 25
        pressure = 30e5
        fuel_mass = 500
        state = TankState(
            self.tank, temperature, pressure, fuel_mass
        )
        expected_value = 0.0
        actual_value = state.gas_mass
        self.assertEqual(expected_value, actual_value)

    def test_fuel_volume(self):
        expected_value = (
            self.tank.volume * self.fill
        )
        actual_value = self.state.fuel_volume
        self.assertEqual(expected_value, actual_value)

    def test_fuel_height(self):
        expected_value = self.tank.compute_fuel_height(None)
        actual_value = self.state.fuel_height
        self.assertEqual(expected_value, actual_value)

    def test_is_full(self):
        self.assertFalse(self.state.is_full)
        fuel_mass = 1e4
        state = TankState(
            self.tank, self.temperature, self.pressure, fuel_mass
        )
        self.assertTrue(state.is_full)

    def test_is_empty(self):
        self.assertFalse(self.state.is_empty)

    def test_complete_state_properties(self):
        self.tank = FuelTank.test_tank()
        self.temperature = 23
        self.pressure = None
        self.fuel_mass = 600
        self.fill = 0.07374514524884335
        self.hydrogen = HydrogenRetriever().get_hydrogen_properties(
            self.pressure, self.temperature
        )
        self.state = TankState(
            self.tank, self.temperature, self.pressure, self.fuel_mass
        )
        self.assertIsNotNone(self.state.pressure)

    def test_compute_state_derivative(self):

        dynamic_model = DynamicModel()
        fuel_flows = None
        heat_flux = None
        tank_thermal_capacity = None
        actual_value = self.state.compute_state_derivatives(
            dynamic_model, fuel_flows, heat_flux, tank_thermal_capacity
        )
        expected_value = dynamic_model.compute_state_derivatives(
            self.state, fuel_flows
        )
        self.assertEqual(expected_value, actual_value)
    

class TestTankStates(unittest.TestCase):

    def setUp(self) -> None:
        self.tank = FuelTank.test_tank()
        self.temperature = None
        self.pressure_1 = 1.4e5
        self.pressure_2 = 3e5
        self.fuel_mass = 600
        self.timestep = 60

        self.state_1 = TankState(
            self.tank, self.temperature, self.pressure_1, self.fuel_mass
        )
        self.state_1.derivatives = StateDerivatives.test_derivatives(1)
        self.state_2 = TankState(
            self.tank, self.temperature, self.pressure_2, self.fuel_mass
        )
        self.state_2.derivatives = StateDerivatives.test_derivatives(2)
        self.states = TankStates(
            [self.state_1, self.state_2], self.timestep
        )

    def test_last_state(self):

        expected_value = self.state_2
        actual_value = self.states.last_state
        self.assertEqual(expected_value, actual_value)

    def test_first_state(self):

        expected_value = self.state_1
        actual_value = self.states.first_state
        self.assertEqual(expected_value, actual_value)

    def test_pressures(self):

        expected_value = [self.pressure_1, self.pressure_2]
        actual_value = self.states.pressures
        self.assertEqual(expected_value, actual_value)

    def test_pressures_in_bars(self):

        expected_value = [
            p * PASCAL_TO_BAR
            for p in [self.pressure_1, self.pressure_2]
        ]
        actual_value = self.states.pressures_in_bar
        self.assertEqual(expected_value, actual_value)

    def test_temperatures(self):

        expected_value = [21.51574187925867, 24.682982971786814]
        actual_value = self.states.temperatures
        self.assertEqual(expected_value, actual_value)

    def test_pressure_derivatives(self):

        expected_value = ["pres_1_der", "pres_2_der"]
        actual_value = self.states.pressure_derivatives
        self.assertEqual(expected_value, actual_value)

    def test_temperature_derivatives(self):

        expected_value = ["temp_1_der", "temp_2_der"]
        actual_value = self.states.temperature_derivatives
        self.assertEqual(expected_value, actual_value)

    def test_initial_temperature(self):

        expected_value = self.state_1.temperature
        actual_value = self.states.initial_temperature
        self.assertEqual(expected_value, actual_value)

    def test_last_temperature(self):

        expected_value = self.state_2.temperature
        actual_value = self.states.last_temperature
        self.assertEqual(expected_value, actual_value)

    def test_last_pressure(self):

        expected_value = self.state_2.pressure
        actual_value = self.states.last_pressure
        self.assertEqual(expected_value, actual_value)

    def test_last_fill(self):

        expected_value = self.state_2.fill
        actual_value = self.states.last_fill
        self.assertEqual(expected_value, actual_value)
        
    def test_average_temperature(self):

        temperatures = [21.51574187925867, 24.682982971786814]
        expected_value = sum(temperatures) / len(temperatures)
        actual_value = self.states.average_temperature
        self.assertEqual(expected_value, actual_value)

    def test_max_pressure(self):

        expected_value = self.state_2.pressure
        actual_value = self.states.max_pressure
        self.assertEqual(expected_value, actual_value)

    def test_min_temperature(self):

        expected_value = self.state_1.temperature
        actual_value = self.states.min_temperature
        self.assertEqual(expected_value, actual_value)

    def test_hydrogens(self):

        expected_value = [
            state.hydrogen
            for state in [self.state_1, self.state_2]
        ]
        actual_value = self.states.hydrogens
        self.assertEqual(expected_value, actual_value)

    def test_fills(self):

        expected_value = [self.state_1.fill, self.state_2.fill]
        actual_value = self.states.fills
        self.assertEqual(expected_value, actual_value)

    def test_volumes(self):

        expected_value = [self.state_1.volume, self.state_2.volume]
        actual_value = self.states.volumes
        self.assertEqual(expected_value, actual_value)

    def test_gas_masses(self):

        expected_value = [self.state_1.gas_mass, self.state_2.gas_mass]
        actual_value = self.states.gas_masses
        self.assertEqual(expected_value, actual_value)

    def test_liquid_masses(self):

        expected_value = [
            self.state_1.liquid_mass, 
            self.state_2.liquid_mass
        ]
        actual_value = self.states.liquid_masses
        self.assertEqual(expected_value, actual_value)

    def test_total_masses(self):

        expected_value = [
            self.state_1.fuel_mass, self.state_2.fuel_mass
        ]
        actual_value = self.states.total_masses
        self.assertEqual(expected_value, actual_value)

    def test_state_derivatives(self):

        expected_value = [
            self.state_1.derivatives
        ]
        actual_value = self.states.state_derivatives
        self.assertEqual(expected_value, actual_value)

    def test_required_fluxes(self):

        expected_value = [
            self.state_1.derivatives.heat_flux
        ]
        actual_value = self.states.required_fluxes
        self.assertEqual(expected_value, actual_value)

    def test__add__(self):

        # Test initial empty state
        states = TankStates(list(), self.timestep)
        states += self.states
        expected_value = [self.state_1, self.state_2]
        actual_value = states.states
        self.assertEqual(expected_value, actual_value)

        # Test state with last state corresponding
        new_states = TankStates([self.state_2], self.timestep)
        states += new_states
        expected_value = [self.state_1, self.state_2]
        actual_value = states.states
        self.assertEqual(expected_value, actual_value)

        # Normal sum
        new_states = TankStates(
            [self.state_1, self.state_2], self.timestep
        )
        states += new_states
        expected_value = [
            self.state_1, self.state_2, self.state_1, self.state_2
        ]
        actual_value = states.states
        self.assertEqual(expected_value, actual_value)

    def test_add_tank_state(self):

        self.states.add_tank_state(self.state_1)
        expected_value = [
            self.state_1, self.state_2, self.state_1
        ]
        actual_value = self.states.states
        self.assertEqual(expected_value, actual_value)

    def test_timestep_in_hours(self):

        expected_value = [
            i * SECONDS_TO_HOURS * self.timestep
            for i in range(2)
        ]
        actual_value = self.states.timesteps_in_hours
        self.assertEqual(expected_value, actual_value)





if __name__ == "__main__":
    unittest.main()


# End
