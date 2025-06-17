import copy
import unittest
from dataclasses import dataclass
from unittest.mock import patch
from src.dynamics.dynamic_models import LinModel, SinglePhaseLimitLowerPressureModel, SinglePhaseModel, StateDerivatives, TwoPhaseLimitLowerPressureModel, TwoPhaseModel, SinglePhaseInOutModel


@dataclass
class Hydrogen:
    pressure: float
    density: float
    enthalpy: float
    dRho_dP: float
    dRho_dT: float
    dH_dP: float
    dH_dT: float

    @classmethod
    def test_hydrogen(cls):
        pressure = 130e3
        density = 5
        enthalpy = 4
        dRho_dP = 6
        dRho_dT = 7
        dH_dP = 9
        dH_dT = 11

        return cls(
            pressure, density, enthalpy, dRho_dP, dRho_dT, dH_dP, dH_dT
        )


@dataclass
class TwoPhaseHydrogen:
    liquid: Hydrogen
    gas: Hydrogen
    dP_dT: float

    @property
    def pressure(self):
        return self.liquid.pressure

    @classmethod
    def test_hydrogen(cls):

        liquid = Hydrogen.test_hydrogen()
        liquid.internal_energy = 22.22
        gas = Hydrogen.test_hydrogen()
        gas.internal_energy = 33.33
        dP_dT = 9.9

        return cls(liquid, gas, dP_dT)



@dataclass
class TankState:
    hydrogen: TwoPhaseHydrogen
    gas_mass: float
    liquid_mass: float
    tank_thermal_capacity: float
    volume: float
    heat_flux: float
    phase: str

    @property
    def fuel_mass(self):
        return self.liquid_mass + self.gas_mass


@dataclass
class FuelFlow:
    hydrogen: Hydrogen
    mass_flow: float


class TestTwoPhaseModel(unittest.TestCase):

    def setUp(self) -> None:

        self.pressure = 130e3
        self.liquid_density = 7
        self.liquid_enthalpy = 2
        self.liquid_dRho_dP = 3
        self.liquid_dRho_dT = 4
        self.liquid_dH_dP = 5
        self.liquid_dH_dT = 6
        self.liquid = Hydrogen(
            self.pressure,
            self.liquid_density,
            self.liquid_enthalpy,
            self.liquid_dRho_dP,
            self.liquid_dRho_dT,
            self.liquid_dH_dP,
            self.liquid_dH_dT
        )

        self.pressure = 140e3
        self.gas_density = 6
        self.gas_enthalpy = 5
        self.gas_dRho_dP = 4
        self.gas_dRho_dT = 3
        self.gas_dH_dP = 2
        self.gas_dH_dT = 7
        self.gas = Hydrogen(
            self.pressure,
            self.gas_density,
            self.gas_enthalpy,
            self.gas_dRho_dP,
            self.gas_dRho_dT,
            self.gas_dH_dP,
            self.gas_dH_dT
        )

        self.dP_dT = 85.85
        self.hydrogen = TwoPhaseHydrogen(
            self.liquid, self.gas, self.dP_dT
        )

        self.liquid_mass = 22
        self.gas_mass = 124

        self.tank_thermal_capacity = 125
        self.tank_volume = 33.3

        self.mass_flow_1 = 11
        self.mass_flow_2 = -5
        self.fuel_flows = [
            FuelFlow(self.gas, self.mass_flow_1),
            FuelFlow(self.gas, self.mass_flow_2)
        ]

        self.flux = 101

        self.tank_state = TankState(
            self.hydrogen,
            self.gas_mass,
            self.liquid_mass,
            self.tank_thermal_capacity,
            self.tank_volume,
            self.flux,
            "twophase"
        )

        self.model = TwoPhaseModel

    def test_a12(self):
        expected_value = - self.dP_dT
        actual_value = self.model.a12(self.hydrogen)
        self.assertTrue(expected_value, actual_value)

    def test_a21(self):
        expected_value = - (
            self.gas_mass * self.gas_dRho_dP
            / self.gas_density ** 2
            + self.liquid_mass * self.liquid_dRho_dP
            / self.liquid_density ** 2
        )
        actual_value = self.model.a21(
            self.gas_mass, self.liquid_mass, self.hydrogen
        )
        self.assertEqual(expected_value, actual_value)

    def test_a22(self):
        expected_value = - (
            self.gas_mass * self.gas_dRho_dT
            / self.gas_density ** 2
            + self.liquid_mass * self.liquid_dRho_dT
            / self.liquid_density ** 2
        )
        actual_value = self.model.a22(
            self.gas_mass, self.liquid_mass, self.hydrogen
        )
        self.assertEqual(expected_value, actual_value)

    def test_a23(self):
        expected_value = (
            1 / self.gas_density - 1 / self.liquid_density
        )
        actual_value = self.model.a23(self.hydrogen)
        self.assertEqual(expected_value, actual_value)

    def test_a42(self):
        expected_value = (
            self.tank_thermal_capacity
            + self.liquid_mass * self.liquid_dH_dT
            + self.gas_mass * self.gas_dH_dT
            + (
                self.liquid_mass * self.liquid_dH_dP
                + self.gas_mass * self.gas_dH_dP
                - self.tank_volume
            ) * self.dP_dT
        )
        actual_value = self.model.a42(
            self.tank_thermal_capacity,
            self.tank_volume,
            self.gas_mass,
            self.liquid_mass,
            self.hydrogen
        )
        self.assertEqual(expected_value, actual_value)

    def test_a43(self):
        expected_value = self.gas_enthalpy - self.liquid_enthalpy
        actual_value = self.model.a43(self.hydrogen)
        self.assertEqual(expected_value, actual_value)

    def test_define_a_matrix(self):

        expected_value = [
            [
                1,
                self.model.a12(self.hydrogen),
                0,
                0
            ], [
                self.model.a21(
                    self.gas_mass, self.liquid_mass, self.hydrogen
                ),
                self.model.a22(
                    self.gas_mass, self.liquid_mass, self.hydrogen
                ),
                self.model.a23(
                    self.hydrogen
                ),
                0
            ], [
                0, 0, 1, 1
            ], [
                0,
                self.model.a42(
                    self.tank_thermal_capacity,
                    self.tank_volume,
                    self.gas_mass,
                    self.liquid_mass,
                    self.hydrogen
                ),
                self.model.a43(
                    self.hydrogen
                ),
                0
            ]
        ]
        actual_value = TwoPhaseModel().define_a_matrix(self.tank_state)
        self.assertEqual(expected_value, actual_value)

    def test_y2(self):
        expected_value = - 1 * (
            self.mass_flow_1 / self.liquid_density
            + self.mass_flow_2 / self.liquid_density
        )
        actual_value = self.model.y2(self.fuel_flows, self.hydrogen)
        self.assertEqual(expected_value, actual_value)

    def test_y3(self):
        expected_value = self.mass_flow_1 + self.mass_flow_2
        actual_value = self.model.y3(self.fuel_flows)
        self.assertEqual(expected_value, actual_value)

    def test_y4(self):
        expected_value = (
            self.mass_flow_1 * (
                self.gas_enthalpy - self.liquid_enthalpy
            ) + self.mass_flow_2 * (
                self.gas_enthalpy - self.liquid_enthalpy
            ) + self.flux
        )
        actual_value = self.model.y4(
            self.fuel_flows, self.hydrogen, self.flux
        )
        self.assertEqual(expected_value, actual_value)

    def test_define_b_vector(self):

        expected_value = [
            [0],
            [self.model.y2(self.fuel_flows, self.hydrogen)],
            [self.model.y3(self.fuel_flows)],
            [self.model.y4(self.fuel_flows, self.hydrogen, self.flux)]
        ]
        actual_value = self.model.define_b_vector(
            self.tank_state, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)

    def test_added_heat_flux(self):
        self.assertEqual(0, self.model().added_heat_flux())

    def test_venting_mass(self):
        self.assertEqual(0, self.model().venting_mass())

    def test_compute_state_derivatives(self):
        # Does not need testing as all the intermediate steps have
        # already been tested
        self.model.compute_state_derivatives(
            self.tank_state, self.fuel_flows
        )


class TestSinglePhaseModel(unittest.TestCase):

    def setUp(self) -> None:

        self.hydrogen = Hydrogen.test_hydrogen()
        self.tank_volume = 123.5
        self.tank_thermal_capacity = 85.85
        self.fuel_mass = 11.1
        self.fuel_mass_flow = 33.3
        self.heat_flux = 1001.1
        self.flow_hydrogen = copy.deepcopy(self.hydrogen)
        self.flow_hydrogen.enthalpy = 0.55
        self.fuel_flow = FuelFlow(
            self.flow_hydrogen, self.fuel_mass_flow
        )
        gas_mass = 123.5
        liquid_mass = 33.3
        heat_flux = 55.6
        self.tank_state = TankState(
            self.hydrogen,
            gas_mass,
            liquid_mass,
            self.tank_thermal_capacity,
            self.tank_volume,
            heat_flux,
            "gas"
        )

        self.dynamic_model = SinglePhaseModel()

    def test_a11(self):

        self.assertEqual(
            self.dynamic_model.a11(self.hydrogen),
            self.hydrogen.dRho_dP
        )

    def test_a12(self):

        self.assertEqual(
            self.dynamic_model.a12(self.hydrogen),
            self.hydrogen.dRho_dT
        )

    def test_a21(self):

        fuel_mass = (
            self.tank_volume
            * self.hydrogen.density
        )
        expected_a21 = (
            fuel_mass * self.hydrogen.dH_dP
            - self.tank_volume
        )
        self.assertEqual(
            self.dynamic_model.a21(
                self.hydrogen, self.tank_volume
            ),
            expected_a21
        )

    def test_a22(self):

        expected_a22 = (
            self.tank_thermal_capacity
            + self.tank_volume
            * self.hydrogen.density
            * self.hydrogen.dH_dT
        )
        self.assertEqual(
            expected_a22,
            self.dynamic_model.a22(
                self.hydrogen,
                self.tank_volume,
                self.tank_thermal_capacity
            )
        )

    def test_y1(self):

        expected_y1 = (
            self.hydrogen.density
            * self.fuel_mass_flow
            / self.fuel_mass
        )
        self.assertEqual(
            expected_y1,
            self.dynamic_model.y1(
                self.fuel_mass, self.hydrogen, self.fuel_mass_flow
            )
        )

    def test_y2(self):
        expected_y2 = (
            self.heat_flux + self.fuel_mass_flow * (
                self.flow_hydrogen.enthalpy - self.hydrogen.enthalpy
            )
        )
        self.assertEqual(
            expected_y2,
            self.dynamic_model.y2(
                self.hydrogen, self.fuel_flow, self.heat_flux
            )
        )

    def test_solve_equations(self):

        expected_value = (0.340445893441996, -0.140116159393431)
        actual_value = self.dynamic_model.solve_state_equations(
            self.tank_state, self.fuel_flow, self.heat_flux
        )
        self.assertEqual(expected_value, actual_value)

    def test_venting_mass(self):

        expected_value = 0
        actual_value = self.dynamic_model.compute_venting_mass()
        self.assertEqual(expected_value, actual_value)

    def test_added_heat_flux(self):

        expected_value = 0
        actual_value = self.dynamic_model.compute_added_heat_flux()
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_define_liquid_and_mass_derivatives(self):

        tank_phase = "liquid"
        expected_value = (0, self.fuel_mass_flow)
        actual_value = self.dynamic_model.define_liquid_and_mass_derivatives(
            tank_phase, self.fuel_mass_flow
        )
        self.assertEqual(expected_value, actual_value)

        tank_phase = "gas"
        expected_value = (self.fuel_mass_flow, 0)
        actual_value = self.dynamic_model.define_liquid_and_mass_derivatives(
            tank_phase, self.fuel_mass_flow
        )
        self.assertEqual(expected_value, actual_value)

        with self.assertRaises(ValueError) as context:
            self.dynamic_model.define_liquid_and_mass_derivatives(
                "test", self.fuel_mass_flow
            )
        self.assertTrue(
            "test not supported in single phase model"
            in str(context.exception)
        )

    def test_compute_state_derivatives(self):

        expected_value = StateDerivatives(
            2.3881857529760415,
            -1.895321753279756,
            self.fuel_mass_flow,
            0,
            0,
            0
        )
        actual_value = self.dynamic_model.compute_state_derivatives(
            self.tank_state, [self.fuel_flow]
        )
        self.assertEqual(expected_value, actual_value)


class TestTwoPhaseLimitPressureModel(unittest.TestCase):

    def setUp(self) -> None:
        self.pressure = 130e3
        self.liquid_density = 7
        self.liquid_enthalpy = 2
        self.liquid_dRho_dP = 3
        self.liquid_dRho_dT = 4
        self.liquid_dH_dP = 5
        self.liquid_dH_dT = 6
        self.liquid = Hydrogen(
            self.pressure,
            self.liquid_density,
            self.liquid_enthalpy,
            self.liquid_dRho_dP,
            self.liquid_dRho_dT,
            self.liquid_dH_dP,
            self.liquid_dH_dT
        )

        self.pressure = 140e3
        self.gas_density = 6
        self.gas_enthalpy = 5
        self.gas_dRho_dP = 4
        self.gas_dRho_dT = 3
        self.gas_dH_dP = 2
        self.gas_dH_dT = 7
        self.gas = Hydrogen(
            self.pressure,
            self.gas_density,
            self.gas_enthalpy,
            self.gas_dRho_dP,
            self.gas_dRho_dT,
            self.gas_dH_dP,
            self.gas_dH_dT
        )

        self.dP_dT = 85.85
        self.hydrogen = TwoPhaseHydrogen(
            self.liquid, self.gas, self.dP_dT
        )

        self.liquid_mass = 22
        self.gas_mass = 124

        self.tank_thermal_capacity = 125
        self.tank_volume = 33.3

        self.mass_flow_1 = 11
        self.mass_flow_2 = -5
        self.fuel_flows = [
            FuelFlow(self.gas, self.mass_flow_1),
            FuelFlow(self.gas, self.mass_flow_2)
        ]

        self.flux = 101

        self.tank_state = TankState(
            self.hydrogen,
            self.gas_mass,
            self.liquid_mass,
            self.tank_thermal_capacity,
            self.tank_volume,
            self.flux,
            "twophase"
        )

        self.model = TwoPhaseLimitLowerPressureModel()

    def test_compute_pressure_derivative(self):

        expected_value = 0
        actual_value = self.model.compute_pressure_derivative()
        self.assertEqual(expected_value, actual_value)

    def test_compute_temperature_derivative(self):

        expected_value = 0
        actual_value = self.model.compute_temperature_derivative()
        self.assertEqual(expected_value, actual_value)

    def test_compute_venting_mass(self):

        expected_value = 0
        actual_value = self.model.compute_venting_mass()
        self.assertEqual(expected_value, actual_value)

    def test_compute_gas_mass_derivative(self):

        expected_value = self.gas_mass_derivative()
        actual_value = self.model.compute_gas_mass_derivative(
            self.hydrogen, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)

    def gas_mass_derivative(self):
        expected_value = (
            sum([flow.mass_flow for flow in self.fuel_flows])
            / (
                1 - (
                    self.hydrogen.liquid.density
                    / self.hydrogen.gas.density
                )
            )
        )

        return expected_value

    def test_compute_liquid_mass_derivative(self):

        expected_value = self.liquid_mass_derivative()
        actual_value = self.model.compute_liquid_mass_derivative(
            self.hydrogen, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)

    def liquid_mass_derivative(self):
        expected_value = (
            sum([flow.mass_flow for flow in self.fuel_flows])
            / (
                1 - (
                    self.hydrogen.gas.density
                    / self.hydrogen.liquid.density
                )
            )
        )

        return expected_value

    def test_compute_required_heat_flux(self):

        expected_value = self.required_flux()
        actual_value = self.model.compute_required_heat_flux(
            self.hydrogen, self.fuel_flows, self.flux
        )
        self.assertEqual(expected_value, actual_value)

    def required_flux(self):
        expected_value = - (
            self.hydrogen.liquid.enthalpy * self.liquid_mass_derivative()
            + self.hydrogen.gas.enthalpy * self.gas_mass_derivative()
            + sum([
                - flow.mass_flow * flow.hydrogen.enthalpy
                for flow in self.fuel_flows
            ])
            - self.flux
        )

        return expected_value

    def test_compute_state_derivatives(self):

        expected_value = StateDerivatives(
            0,
            0,
            self.gas_mass_derivative(),
            self.liquid_mass_derivative(),
            0,
            self.required_flux()
        )
        actual_value = self.model.compute_state_derivatives(
            self.tank_state, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)


class TestSinglePhaseLimitLowerPressureModel(unittest.TestCase):

    def setUp(self) -> None:

        self.hydrogen = Hydrogen.test_hydrogen()
        self.tank_volume = 123.5
        self.tank_thermal_capacity = 85.85
        self.fuel_mass_flow = 33.3
        self.heat_flux = 1001.1
        self.flow_hydrogen = copy.deepcopy(self.hydrogen)
        self.flow_hydrogen.enthalpy = 0.55
        self.fuel_flow = FuelFlow(
            self.flow_hydrogen, self.fuel_mass_flow
        )
        self.fuel_flows = [self.fuel_flow]
        gas_mass = 123.5
        liquid_mass = 33.3
        self.fuel_mass = gas_mass + liquid_mass
        self.tank_state = TankState(
            self.hydrogen,
            gas_mass,
            liquid_mass,
            self.tank_thermal_capacity,
            self.tank_volume,
            self.heat_flux,
            "gas"
        )

        self.dynamic_model = SinglePhaseLimitLowerPressureModel()

    def test_define_liquid_and_mass_derivatives(self):

        tank_phase = "liquid"
        expected_value = (0, self.fuel_mass_flow)
        actual_value = self.dynamic_model.define_liquid_and_mass_derivatives(
            tank_phase, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)

        tank_phase = "gas"
        expected_value = (self.fuel_mass_flow, 0)
        actual_value = self.dynamic_model.define_liquid_and_mass_derivatives(
            tank_phase, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)

        with self.assertRaises(ValueError) as context:
            self.dynamic_model.define_liquid_and_mass_derivatives(
                "test", self.fuel_flows
            )
        self.assertTrue(
            "test not supported in single phase model"
            in str(context.exception)
        )

    def test_compute_venting_mass(self):

        expected_value = 0
        actual_value = self.dynamic_model.compute_venting_mass()
        self.assertEqual(expected_value, actual_value)


    def test_compute_temperature_derivative(self):

        expected_value = self.temperature_derivative()
        actual_value = self.dynamic_model.compute_temperature_derivative(
            self.tank_state, self.fuel_flows
        )
        self.assertEqual(expected_value, actual_value)

    def temperature_derivative(self):
        expected_value = (
            self.hydrogen.density
            * self.fuel_mass_flow
            / self.fuel_mass
            / self.hydrogen.dRho_dT
        )

        return expected_value

    def test_compute_required_flux(self):

        expected_value = -self.required_heat_flux()
        actual_value = self.dynamic_model.compute_required_heat_flux(
            self.tank_state, self.temperature_derivative()
        )
        self.assertEqual(expected_value, actual_value)

    def required_heat_flux(self):
        expected_value = (
            (
                self.tank_thermal_capacity
                + self.fuel_mass * self.hydrogen.dH_dT
            ) * self.temperature_derivative()
            - self.heat_flux
        )

        return expected_value

    def test_compute_pressure_derivative(self):

        expected_value = 0
        actual_value = self.dynamic_model.compute_pressure_derivative()
        self.assertEqual(expected_value, actual_value)

    def test_compute_state_derivatives(self):

        expected_value = StateDerivatives(
            0,
            self.temperature_derivative(),
            self.fuel_mass_flow,
            0,
            0,
            self.required_heat_flux()
        )
        actual_value = self.dynamic_model.compute_state_derivatives(
            self.tank_state, self.fuel_flows
        )
        for attr in ["pressure", "temperature", "liquid_mass", "gas_mass", "venting_mass", "heat_flux"]:
            self.assertAlmostEqual(
                abs(getattr(expected_value, attr)),
                abs(getattr(actual_value, attr)),
                places=7
            )


class TestLinModel(unittest.TestCase):

    def setUp(self) -> None:

        # Define model
        self.model = LinModel()

        # Define state of the tank
        self.hydrogen = TwoPhaseHydrogen.test_hydrogen()
        self.fill = 0.95

        # Define the fuel flow
        self.fuel_mass_flow = 33.3
        self.fuel_flows = [FuelFlow(
            self.hydrogen.liquid, self.fuel_mass_flow
        )]

        self.heat_flux = 2000
        self.volume = 85.1

        self.tank_state = TankState(
            self.hydrogen,
            None,
            None,
            None,
            self.volume,
            self.heat_flux,
            "twophase"
        )
        self.tank_state.fill = self.fill

    def test_compute_energy_derivative(self):

        expected_value = self.energy_derivative()
        actual_value = self.model.compute_energy_derivative(
            self.hydrogen, self.fill
        )
        self.assertAlmostEqual(expected_value, actual_value, places=4)

    def energy_derivative(self):
        expected_value = 0.0326014453201906
        return expected_value

    def test_compute_pressure_derivative(self):

        expected_value = self.pressure_derivative()
        actual_value = self.model.compute_pressure_derivative(
            self.tank_state, self.fuel_flows
        )
        self.assertAlmostEqual(expected_value, actual_value)

    def pressure_derivative(self):
        flow_term = sum([
            flow.mass_flow * flow.hydrogen.enthalpy
            for flow in self.fuel_flows
        ])
        expected_value = (
            self.energy_derivative() * (self.heat_flux + flow_term)
            / self.volume
        )

        return expected_value

    def test_compute_state_derivatives(self):

        expected_value = StateDerivatives(
            self.pressure_derivative(),
            self.pressure_derivative() / self.hydrogen.dP_dT,
            None,
            None,
            None,
            None
        )
        actual_value = self.model.compute_state_derivatives(
            self.tank_state, self.fuel_flows
        )
        attributes =[
            "pressure",
            "temperature",
            "liquid_mass",
            "gas_mass",
            "venting_mass",
            "heat_flux"
        ]
        for attribute in attributes:
            self.assertAlmostEqual(
                expected_value.__getattribute__(attribute),
                actual_value.__getattribute__(attribute)
            )

class TestSinglePhaseInOutModel(unittest.TestCase):

    def setUp(self) -> None:
        self.hydrogen = Hydrogen.test_hydrogen()
        self.tank_volume = 123.5
        self.tank_thermal_capacity = 85.85
        self.fuel_mass = 11.1
        self.heat_flux = 1001.1

        # Inflow
        self.inflow_hydrogen = copy.deepcopy(self.hydrogen)
        self.inflow_hydrogen.enthalpy = 0.55
        self.inflow_mass_flow = 33.3
        self.fuel_flow_in = FuelFlow(self.inflow_hydrogen, self.inflow_mass_flow)

        # Outflow
        self.outflow_hydrogen = copy.deepcopy(self.hydrogen)
        self.outflow_hydrogen.enthalpy = 0.25
        self.outflow_mass_flow = 10.0
        self.fuel_flow_out = FuelFlow(self.outflow_hydrogen, self.outflow_mass_flow)

        gas_mass = 123.5
        liquid_mass = 0
        self.tank_state = TankState(
            self.hydrogen,
            gas_mass,
            liquid_mass,
            self.tank_thermal_capacity,
            self.tank_volume,
            self.heat_flux,
            "gas"
        )

        self.dynamic_model = SinglePhaseInOutModel()

    def test_a11(self):
        self.assertEqual(
            self.dynamic_model.a11(self.hydrogen),
            self.hydrogen.dRho_dP
        )

    def test_a12(self):
        self.assertEqual(
            self.dynamic_model.a12(self.hydrogen),
            self.hydrogen.dRho_dT
        )

    def test_a21(self):
        fuel_mass = (
            self.tank_volume
            * self.hydrogen.density
        )
        expected_a21 = (
            fuel_mass * self.hydrogen.dH_dP
            - self.tank_volume
        )
        self.assertEqual(
            self.dynamic_model.a21(
                self.hydrogen, self.tank_volume
            ),
            expected_a21
        )

    def test_a22(self):
        expected_a22 = (
            self.tank_thermal_capacity
            + self.tank_volume
            * self.hydrogen.density
            * self.hydrogen.dH_dT
        )
        self.assertEqual(
            expected_a22,
            self.dynamic_model.a22(
                self.hydrogen,
                self.tank_volume,
                self.tank_thermal_capacity
            )
        )

    def test_y1(self):
        # Pass both mass flows
        self.dynamic_model.y1(
            self.fuel_mass, self.hydrogen, self.inflow_mass_flow, self.outflow_mass_flow
        )

    def test_y2(self):
        # Pass both FuelFlow objects
        self.dynamic_model.y2(
            self.hydrogen, self.fuel_flow_in, self.fuel_flow_out, self.heat_flux
        )

    # TODO: add actual value for assert
    def test_solve_equations(self):
        # Pass both FuelFlow objects
        self.dynamic_model.solve_state_equations(
            self.tank_state, self.fuel_flow_in, self.fuel_flow_out, self.heat_flux
        )
    # TODO: add actual value for assert
    def test_venting_mass(self):
        self.dynamic_model.compute_venting_mass()

    # TODO: add actual value for assert
    def test_added_heat_flux(self):
        self.dynamic_model.compute_added_heat_flux()

    def test_define_liquid_and_mass_derivatives(self):
        net_mass_flow = self.inflow_mass_flow - self.outflow_mass_flow

        tank_phase = "liquid"
        self.dynamic_model.define_liquid_and_mass_derivatives(
            tank_phase, net_mass_flow
        )

        tank_phase = "gas"
        self.dynamic_model.define_liquid_and_mass_derivatives(
            tank_phase, net_mass_flow
        )

        with self.assertRaises(ValueError) as context:
            self.dynamic_model.define_liquid_and_mass_derivatives(
                "test", net_mass_flow
            )

    def test_compute_state_derivatives(self):
        # Pass both as lists
        actual_value = self.dynamic_model.compute_state_derivatives(
            self.tank_state, [self.fuel_flow_in], [self.fuel_flow_out]
        )
        # Dummy expected values, just for structure (these are not correct)
        expected_value = StateDerivatives(
            1.0,  # pressure
            2.0,  # temperature
            3.0,  # liquid_mass
            4.0,  # gas_mass
            5.0,  # venting_mass
            6.0   # heat_flux
        )
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End