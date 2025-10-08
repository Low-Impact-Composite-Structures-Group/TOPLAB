#!/usr/bin/env python3
"""
Test script to validate pressure calculations for discharge piping
"""

import math

def test_pressure_calculation():
    """Test the pressure calculation with realistic parameters."""

    # Parameters from config
    pipe_diameter = 0.01  # 10mm
    pipe_length = 2.0     # 2m
    pipe_roughness = 1.5e-6  # stainless steel
    loss_coefficient = 2.5   # K-factor for bends/valves
    lh2_density = 70.8       # kg/m³

    print("🔍 DISCHARGE PRESSURE CALCULATION VALIDATION")
    print("="*60)
    print(f"Pipe: {pipe_diameter*1000:.0f}mm diameter × {pipe_length:.1f}m length")
    print(f"Roughness: {pipe_roughness*1e6:.1f} µm (stainless steel)")
    print(f"Loss coefficient (K): {loss_coefficient}")
    print(f"LH2 density: {lh2_density:.1f} kg/m³")
    print("="*60)

    # Test different flow rates
    flow_rates = [0.004, 0.020, 0.049, 0.098]  # kg/s (4, 20, 49, 98 g/s)

    for flow_rate in flow_rates:
        print(f"\n📊 Flow Rate: {flow_rate*1000:.0f} g/s ({flow_rate:.3f} kg/s)")

        # Convert mass flow to volumetric flow
        volumetric_flow = flow_rate / lh2_density  # m³/s
        print(f"   Volumetric flow: {volumetric_flow*1000:.2f} L/s")

        # Calculate flow velocity in pipe
        pipe_area = math.pi * (pipe_diameter / 2) ** 2
        velocity = volumetric_flow / pipe_area  # m/s
        print(f"   Pipe velocity: {velocity:.1f} m/s")

        # Calculate Reynolds number
        kinematic_viscosity = 1e-7  # m²/s for LH2 (assumed)
        reynolds = velocity * pipe_diameter / kinematic_viscosity
        print(f"   Reynolds number: {reynolds:.0f}")

        # Calculate friction factor
        if reynolds > 2300:  # Turbulent flow
            friction_factor = 0.316 / (reynolds ** 0.25)
            flow_regime = "Turbulent"
        else:  # Laminar flow
            friction_factor = 64 / reynolds
            flow_regime = "Laminar"
        print(f"   Flow regime: {flow_regime}")
        print(f"   Friction factor: {friction_factor:.6f}")

        # Calculate pressure losses
        # Frictional losses (Darcy-Weisbach equation)
        friction_loss = friction_factor * (pipe_length / pipe_diameter) * (lh2_density * velocity**2 / 2)

        # Minor losses (K-factors)
        minor_loss = loss_coefficient * (lh2_density * velocity**2 / 2)

        # Total pressure drop
        total_pressure_drop = friction_loss + minor_loss

        print(f"   Friction loss: {friction_loss/1000:.2f} kPa")
        print(f"   Minor loss: {minor_loss/1000:.2f} kPa")
        print(f"   Total pressure drop: {total_pressure_drop/1000:.2f} kPa")

        # Check for choked flow
        sonic_velocity = 1000  # m/s for LH2 (assumed)
        if velocity > 0.5 * sonic_velocity:
            print(f"   ⚠️  Approaching choked flow! (v = {velocity:.1f} m/s)")
            choked_flow_factor = 2.0
            total_pressure_drop *= choked_flow_factor
            print(f"   Choked flow factor applied: {choked_flow_factor}x")
            print(f"   Adjusted pressure drop: {total_pressure_drop/1000:.2f} kPa")

        # Minimum tank pressure = atmospheric + pressure drops
        atmospheric_pressure = 1.01325e5  # Pa
        min_tank_pressure = atmospheric_pressure + total_pressure_drop
        min_tank_pressure_bar = min_tank_pressure / 1e5

        print(f"   Required tank pressure: {min_tank_pressure_bar:.2f} bar")
        print(f"   Pressure above atmospheric: {total_pressure_drop/1000:.2f} kPa")


if __name__ == "__main__":
    test_pressure_calculation()