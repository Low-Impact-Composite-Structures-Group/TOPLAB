"""
Tank geometry utility functions for multi-tank systems.

This module provides functions for creating tank geometries based on
mission requirements or fuel mass specifications.
"""

import math
import CoolProp.CoolProp as CP
from typing import Tuple

from toplab.tank_design.tank_shapes import CapsuleTank
from toplab.materials.nist_materials import NISTMetal, NISTComposite
from toplab.missions.isochoric_missions import DischargeMission


def create_tank_from_mission(
    mission: DischargeMission,
    initial_pressure: float,
    initial_temperature: float,
    operating_pressure: float,
    phi: float = 0.0,
    safety_margin: float = 1.2,
    liner_thickness: float = 0.005,
    insulation_thickness: float = 0.05,
) -> tuple[CapsuleTank, float]:
    """
    Create tank geometry based on mission fuel requirements.

    Args:
        mission: Mission defining fuel requirements
        initial_pressure: Initial tank pressure [Pa]
        initial_temperature: Initial tank temperature [K]
        operating_pressure: Maximum operating pressure for wall design [Pa]
        safety_margin: Safety factor for fuel mass
        liner_thickness: Liner thickness [m]
        insulation_thickness: Insulation thickness [m]

    Returns:
        tuple: (SphericalTank, required_fuel_volume [m³])
    """

    # Calculate total fuel mass required from mission
    total_fuel_mass = 0.0
    for section in mission.sections:
        # Get outflows (fuel consumption)
        outflows = section.get_outflows()
        section_fuel_consumption = 0.0

        for flow in outflows:
            if hasattr(flow, 'mass_flow'):
                if isinstance(flow.mass_flow, list):
                    # Time-varying flow: integrate using trapezoidal rule
                    # For linear interpolation between start and end values
                    start_rate = abs(flow.mass_flow[0])
                    end_rate = abs(flow.mass_flow[-1])
                    avg_rate = (start_rate + end_rate) / 2.0  # Trapezoidal integration
                    section_fuel_consumption += avg_rate * section.duration
                else:
                    # Constant flow
                    consumption_rate = abs(flow.mass_flow)  # kg/s
                    section_fuel_consumption += consumption_rate * section.duration  # kg

        total_fuel_mass += section_fuel_consumption

    # Apply safety margin
    required_fuel_mass = total_fuel_mass * safety_margin

    # Calculate hydrogen density at initial state
    try:
        rho_h2 = CP.PropsSI('D', 'P', initial_pressure, 'T', initial_temperature, 'Hydrogen')  # kg/m³
    except Exception as e:
        raise RuntimeError(
            f"CoolProp failed to calculate hydrogen density at P={initial_pressure/1e5:.1f} bar, T={initial_temperature:.1f} K. "
            f"CoolProp is required for all fluid property calculations. Error: {e}"
        ) from e

    # Calculate required fuel volume
    fuel_volume_required = required_fuel_mass / rho_h2  # m³

    # Add some vapor space (typically 10% ullage)
    ullage_factor = 1.1
    internal_volume_required = fuel_volume_required * ullage_factor

    # Calculate internal radius for the configured capsule geometry.
    # V = πr³(φ + 4/3), where φ = L/r and L is cylindrical section length.
    volume_factor = math.pi * (max(phi, 0.0) + 4.0 / 3.0)
    internal_radius = (internal_volume_required / volume_factor) ** (1/3)

    # Account for liner thickness - external radius includes liner
    external_radius = internal_radius + liner_thickness

    # Create materials
    liner_material = NISTMetal.aluminum_6061T6_nist()
    # For SphericalTank, we use the external radius and the liner material
    # (the wall/insulation will be handled by the thermal model)

    tank = CapsuleTank(
        radius=external_radius,
        material=liner_material,
        operating_pressure=operating_pressure,
        phi=phi,
    )

    print(f"Tank sizing from mission requirements:")
    print(f"  Total fuel mass required: {total_fuel_mass:.3f} kg")
    print(f"  With {safety_margin}x safety margin: {required_fuel_mass:.3f} kg")
    print(f"  Density at {initial_pressure/1e5:.1f} bar, {initial_temperature:.1f} K: {rho_h2:.2f} kg/m³")
    print(f"  Fuel volume required: {fuel_volume_required:.4f} m³")
    print(f"  Internal volume (with ullage): {internal_volume_required:.4f} m³")
    print(f"  Internal radius: {internal_radius:.3f} m")
    print(f"  External radius (with liner): {external_radius:.3f} m")
    print(f"  Tank phi (L/R): {phi:.3f}")
    print(f"  Tank volume: {tank.volume:.4f} m³")
    print(f"  Operating pressure: {operating_pressure/1e5:.1f} bar")

    return tank, fuel_volume_required


def create_tank_from_fuel_mass(
    fuel_mass: float,
    initial_pressure: float,
    initial_temperature: float,
    operating_pressure: float,
    phi: float = 0.0,
    safety_margin: float = 1.1,
    liner_thickness: float = 0.005,
    insulation_thickness: float = 0.05,
) -> CapsuleTank:
    """
    Create tank geometry based on required fuel mass.

    Args:
        fuel_mass: Required fuel mass [kg]
        initial_pressure: Initial tank pressure [Pa]
        initial_temperature: Initial tank temperature [K]
        operating_pressure: Maximum operating pressure for wall design [Pa]
        safety_margin: Safety factor for tank volume (ullage space)
        liner_thickness: Liner thickness [m]
        insulation_thickness: Insulation thickness [m]

    Returns:
        SphericalTank: Tank designed for operating pressure
    """

        # Calculate hydrogen density at initial state using CoolProp
    try:
        # Use CoolProp to get hydrogen density at initial P & T
        rho_h2 = CP.PropsSI('D', 'P', initial_pressure, 'T', initial_temperature, 'Hydrogen')  # kg/m³
    except Exception as e:
        raise RuntimeError(
            f"CoolProp failed to calculate hydrogen density at P={initial_pressure/1e5:.1f} bar, T={initial_temperature:.1f} K. "
            f"CoolProp is required for all fluid property calculations. Error: {e}"
        ) from e

    # Calculate required fuel volume
    fuel_volume_required = fuel_mass / rho_h2  # m³

    # Add ullage space
    internal_volume_required = fuel_volume_required * safety_margin

    # Calculate internal radius for the configured capsule geometry.
    # V = πr³(φ + 4/3), where φ = L/r and L is cylindrical section length.
    volume_factor = math.pi * (max(phi, 0.0) + 4.0 / 3.0)
    internal_radius = (internal_volume_required / volume_factor) ** (1/3)

    # Account for liner thickness - external radius includes liner
    external_radius = internal_radius + liner_thickness

    # Create materials
    liner_material = NISTMetal.aluminum_6061T6_nist()
    # Create SphericalTank with operating pressure (P_VENT) for wall design
    tank = CapsuleTank(
        radius=external_radius,
        material=liner_material,
        operating_pressure=operating_pressure,  # Use P_VENT for structural design
        phi=phi,
    )

    print(f"Tank sizing from fuel mass:")
    print(f"  Required fuel mass: {fuel_mass:.3f} kg")
    print(f"  Density at {initial_pressure/1e5:.1f} bar, {initial_temperature:.1f} K: {rho_h2:.2f} kg/m³")
    print(f"  Fuel volume required: {fuel_volume_required:.4f} m³")
    print(f"  Internal volume (with ullage): {internal_volume_required:.4f} m³")
    print(f"  Internal radius: {internal_radius:.3f} m")
    print(f"  External radius (with liner): {external_radius:.3f} m")
    print(f"  Tank phi (L/R): {phi:.3f}")
    print(f"  Tank volume: {tank.volume:.4f} m³")
    print(f"  Operating pressure: {operating_pressure/1e5:.1f} bar")

    return tank