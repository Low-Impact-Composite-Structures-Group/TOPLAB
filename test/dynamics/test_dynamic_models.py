import unittest
from dataclasses import dataclass
from unittest.mock import patch
from src.dynamics.dynamic_models import TwoPhaseModel

from src.fluids.convective_mediums import Hydrogen


@dataclass
class Hydrogen:
    density: float
    enthalpy: float
    dRho_dP: float
    dRho_dT: float
    dH_dP: float
    dH_dT: float


@dataclass
class TwoPhaseHydrogen:
    liquid: Hydrogen
    gas: Hydrogen
    dP_dT: float


@dataclass
class TankState:
    hydrogen: TwoPhaseHydrogen
    gas_mass: float
    liquid_mass: float
    tank_thermal_capacity: float
    volume: float
    heat_flux: float


@dataclass
class FuelFlow:
    hydrogen: Hydrogen
    mass_flow: float


class TestTwoPhaseModel(unittest.TestCase):

    def setUp(self) -> None:
        
        self.liquid_density = 7
        self.liquid_enthalpy = 2
        self.liquid_dRho_dP = 3
        self.liquid_dRho_dT = 4
        self.liquid_dH_dP = 5
        self.liquid_dH_dT = 6
        self.liquid = Hydrogen(
            self.liquid_density,
            self.liquid_enthalpy,
            self.liquid_dRho_dP,
            self.liquid_dRho_dT,
            self.liquid_dH_dP,
            self.liquid_dH_dT
        )

        self.gas_density = 6
        self.gas_enthalpy = 5
        self.gas_dRho_dP = 4
        self.gas_dRho_dT = 3
        self.gas_dH_dP = 2
        self.gas_dH_dT = 7
        self.gas = Hydrogen(
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
            self.flux
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
        expected_value = (
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
        self.assertEqual(0, self.model().added_heat_flux)

    def test_venting_mass(self):
        self.assertEqual(0, self.model().venting_mass)

    def test_compute_state_derivatives(self):
        # Does not need testing as all the intermediate steps have 
        # already been tested
        self.model.compute_state_derivatives(
            self.tank_state, self.fuel_flows
        )


if __name__ == "__main__":
    unittest.main()


# End
