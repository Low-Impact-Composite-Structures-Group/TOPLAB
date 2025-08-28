from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from src.thermodynamics.heat_transfer_modes import (GasPhaseConvection,
                                                    LiquidPhaseConvection,
                                                    NaturalConvection,
                                                    RohsenowNaturalConvection)
from src.thermodynamics.thermal_resistances import (ParallelResistances,
                                                    ThermalResistance)


class FuelTank(Protocol):
    characteristic_length: float
    characteristic_height: float
    surface_area: float
    
    @abstractmethod
    def compute_fuel_wetted_surface(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_gas_wetted_surface(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_zone_1_length(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_zone_2_length(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_zone_3_length(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_zone_1_area(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_zone_2_area(self, fuel_height: float) -> float:
        ...
    
    @abstractmethod
    def compute_zone_3_area(self, fuel_height: float) -> float:
        ...


class Hydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen


class TankState(Protocol):
    fuel_height: float
    hydrogen: Hydrogen
    is_full: bool
    is_empty: bool


class InternalModel(Protocol):

    def compute_equivalent_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> float:
        return ParallelResistances().compute_equivalent_resistance(
            [
                resistance.value
                for resistance in self.get_thermal_resistances(
                    tank, tank_state, surface_temperature
                )
            ]
        )

    def get_thermal_resistances(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        # Check if we're in a transition state
        is_transition = False
        if hasattr(tank_state, '_forced_phase'):
            # Check if we're transitioning between phases
            if (tank_state._forced_phase == "twophase" and 
                (not hasattr(tank_state.hydrogen, "liquid") or not hasattr(tank_state.hydrogen, "gas"))):
                print(f"WARNING: In transition state for thermal model. Using gas-only model temporarily.")
                is_transition = True
        
        # In transition state or empty tank
        if is_transition or tank_state.is_empty:
            return [self.create_gas_resistance(
                tank, tank_state, surface_temperature
            )]
        
        # Check phase to determine thermal model
        phase = tank_state.phase
        
        # Single-phase liquid tank
        if phase == "liquid" or tank_state.is_full:
            try:
                return self.create_liquid_resistance(
                    tank, tank_state, surface_temperature
                )
            except ValueError as e:
                print(f"WARNING: Error creating liquid resistances: {e}. Trying gas model.")
                # Only try gas model if liquid fails and gas properties are available
                try:
                    return [self.create_gas_resistance(
                        tank, tank_state, surface_temperature
                    )]
                except ValueError as e2:
                    raise ValueError(f"Cannot create thermal resistance - both liquid and gas models failed: {e}, {e2}")
        
        # Single-phase gas tank  
        if phase == "gas":
            try:
                return [self.create_gas_resistance(
                    tank, tank_state, surface_temperature
                )]
            except ValueError as e:
                print(f"WARNING: Error creating gas resistances: {e}. Trying liquid model.")
                # Fall back to liquid model if gas fails
                return self.create_liquid_resistance(
                    tank, tank_state, surface_temperature
                )
                
        # Two-phase tank (default case)
        try:
            return self.create_two_phase_thermal_resistances(
                tank, tank_state, surface_temperature
            )
        except ValueError as e:
            print(f"WARNING: Error creating two-phase resistances: {e}. Using gas model.")
            # Fall back to gas model if two-phase properties unavailable
            return [self.create_gas_resistance(
                tank, tank_state, surface_temperature
            )]

    def create_two_phase_thermal_resistances(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        try:
            # First ensure we have liquid phase available
            if not hasattr(tank_state.hydrogen, "liquid"):
                print(f"WARNING: Liquid phase unavailable in two-phase model. Using gas-only resistance.")
                gas_resistance = self.create_gas_resistance(
                    tank, tank_state, surface_temperature
                )
                return [gas_resistance]
                
            # Normal case - both liquid and gas available
            liquid_resistance = self.create_liquid_resistance(
                tank, tank_state, surface_temperature
            )
            gas_resistance = self.create_gas_resistance(
                tank, tank_state, surface_temperature
            )
            return [*liquid_resistance, gas_resistance]
        except Exception as e:
            print(f"WARNING: Error in create_two_phase_thermal_resistances: {e}. Using gas-only model.")
            gas_resistance = self.create_gas_resistance(
                tank, tank_state, surface_temperature
            )
            return [gas_resistance]

    def create_gas_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> ThermalResistance:
        gas_convection = GasPhaseConvection(
            tank_state.hydrogen.gas,
            tank.characteristic_height - tank_state.fuel_height,
            surface_temperature
        )
        gas_resistance = ThermalResistance(
            gas_convection.heat_transfer_coefficient,
            tank.compute_gas_wetted_surface(
                tank_state.fuel_height
            )
        )
        return gas_resistance

    @abstractmethod
    def create_liquid_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        ...


class SingleZoneModel(InternalModel):

    def create_liquid_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        liquid_convection = LiquidPhaseConvection(
            tank_state.hydrogen.liquid,
            tank_state.fuel_height,
            surface_temperature
        )
        liquid_resistance = ThermalResistance(
            liquid_convection.heat_transfer_coefficient,
            tank.compute_fuel_wetted_surface(
                tank_state.fuel_height
            )
        )
        return [liquid_resistance]


class ThreeZoneModel(InternalModel):

    def create_liquid_resistance(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[ThermalResistance]:
        convective_motions = self.create_convective_motions(
            tank, tank_state, surface_temperature
        )
        surfaces = self.create_surfaces(tank, tank_state)
        resistances = [
            ThermalResistance(
                convection.heat_transfer_coefficient, surface
            )
            for convection, surface in zip(convective_motions, surfaces)
        ]
        return resistances

    def create_surfaces(
        self,
        tank: FuelTank,
        tank_state: TankState
    ) -> list[float]:
        surfaces = [
            tank.compute_zone_1_area(tank_state.fuel_height),
            tank.compute_zone_2_area(tank_state.fuel_height),
            tank.compute_zone_3_area(tank_state.fuel_height)
        ]

        return surfaces

    def create_convective_motions(
        self,
        tank: FuelTank,
        tank_state: TankState,
        surface_temperature: float
    ) -> list[NaturalConvection]:
        convective_motions = [
            RohsenowNaturalConvection(
                tank_state.hydrogen.liquid,
                tank.compute_zone_1_length(tank_state.fuel_height),
                surface_temperature
            ),
            RohsenowNaturalConvection(
                tank_state.hydrogen.liquid,
                tank.compute_zone_2_length(tank_state.fuel_height),
                surface_temperature
            ),
            RohsenowNaturalConvection(
                tank_state.hydrogen.liquid,
                tank.compute_zone_3_length(tank_state.fuel_height),
                surface_temperature
            )
        ]
        
        return convective_motions


def main():
    pass


if __name__ == "__main__":
    main()


# End
