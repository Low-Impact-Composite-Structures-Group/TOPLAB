import matplotlib.pyplot as plt
from plotting.plot_tank_states import plot_single_tank_fill, plot_tank_loads, plot_single_tank_temperatures, plot_single_required_flux, plot_single_tank_loads, plot_density_vs_temperature
from facades.analysis_facades import MissionAnalysisFacade, OperatingEnvelope, TankDimensions, FillingAnalysisFacade, TargetConditions, InitialConditions, InOutTankAnalysisFacade
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, OutFlow, MissionSection
from src.thermodynamics.tank_states import InitialState
from src.tank_design.tank_shapes import SphericalTank
from src.fluids.hydrogen_retrievers import SinglePhaseRequester, TwoPhaseRequester
import numpy as np
import os
import yaml

def perform_analysis():

    # Define tank parameters
    p_init = 101325  # Pa (use atmospheric pressure)
    t_init = 20  # K (well below critical temperature of 33K)
    fill = 0.5  # Start with half liquid, half gas
    p_max = 500000 # Pa
    p_min = 150000 # Pa
    t_min = None # K
    ambient_heat_load = 5.0 # W/m^2
    stopping_mass = 10.0 # kg
    duration_in_hours = 1.0 # hours

    # Create multi-flow mission
    inflow_rate = 0.03  # kg/s
    outflow_rate = 0.05  # kg/s

    # Add debugging to see what's happening
    print(f"Requesting two-phase hydrogen at T={t_init}K")
    hydrogen_requester = TwoPhaseRequester()
    hydrogen_props = hydrogen_requester.get_hydrogen_properties(p_init, t_init)

    # Print debug information about the hydrogen object
    print(f"Hydrogen properties obtained - checking phases")
    has_liquid = hasattr(hydrogen_props, 'liquid')
    has_gas = hasattr(hydrogen_props, 'gas')
    print(f"  Has liquid attribute: {has_liquid}")
    print(f"  Has gas attribute: {has_gas}")

    # Create inflow using liquid properties if available
    if has_liquid:
        try:
            print(f"  Liquid density: {hydrogen_props.liquid.density} kg/m³")
            inflow = InFlow(inflow_rate, hydrogen_props.liquid)
        except ValueError as e:
            print(f"  Error accessing liquid properties: {e}")
            # Fallback to using the base hydrogen object
            inflow = InFlow(inflow_rate, hydrogen_props)
    else:
        inflow = InFlow(inflow_rate, hydrogen_props)

    # Create outflow specifically targeting gas phase
    outflow = OutFlow(outflow_rate, "gas")

    # Create mission with both flows
    mission = Mission([
        MissionSection(
            duration = 3600*duration_in_hours, # Convert hours to seconds
            fuel_flows=[inflow, outflow],  # Pass as a list
            altitude=0,
            mach_number=0.0
        )
    ])

    print(f"Mission required fuel mass: {mission.required_fuel} kg")

    # Calculate appropriate radius based on required mass
    VOLUME_MARGIN = 1.1  # make the tank 10% larger than the required volume

    try:
        # Try using average density of two-phase mixture
        avg_density = fill * hydrogen_props.liquid.density
        if hasattr(hydrogen_props, 'gas') and hasattr(hydrogen_props.gas, 'density'):
            avg_density += (1 - fill) * hydrogen_props.gas.density
        else:
            # If gas phase not available, just use liquid density
            print("Using only liquid density for volume calculation")
            avg_density = hydrogen_props.liquid.density

        required_volume = VOLUME_MARGIN * (mission.required_fuel / avg_density)
    except Exception as e:
        print(f"Error calculating volume: {e}")
        # Fallback: use a reasonable hydrogen density (~70 kg/m³)
        required_volume = VOLUME_MARGIN * (mission.required_fuel / 70.0)

    radius = (3 * required_volume / (4 * np.pi))**(1/3)

    # instantiate the Tank and TankDimensions object
    tank_material = Composite.carbon(np.radians(55))
    tank_dimensions = TankDimensions(radius, 0.0) # this will instantiate a spherical tank

    # Tank material and insulation
    tank_material = Composite.carbon(np.radians(55))

    # TODO: remove dummy insulation; not used since we are applying constant heat load
    insulation_thickness = 0.001  # [m]
    insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

    initial_conditions = InitialConditions(p_init, t_init, fill)
    operating_window = OperatingEnvelope(p_max, p_min, t_min)

    tank_performance = InOutTankAnalysisFacade.analyse(
        tank_dimensions,
        tank_material,
        insulation,
        mission,
        ambient_heat_load,
        initial_conditions,
        operating_window,
        TargetConditions(stopping_mass, 0.0),
    )
    tank_performance = tank_performance.tank_states

    # Generate figures as before, but don't call plt.show() yet
    fig_tank_fill = plot_single_tank_fill(tank_performance)
    fig_tank_temperatures = plot_single_tank_temperatures(tank_performance)
    fig_tank_pressures = plot_single_tank_loads(tank_performance)

    # Collect all axes from the generated figures
    axes_list = [
        fig_tank_fill.ax[0],         # Row 1, Col 1: Tank 1 Fill Level
        fig_tank_temperatures.ax[0], # Row 1, Col 2: Tank 1 Temperature
        fig_tank_pressures.ax[0],    # Row 1, Col 3: Tank 1 Pressure

    ]
    titles = [
        "Tank Fill Level",
        "Tank Internal Temperature",
        "Tank Internal Pressure",
    ]

    # Create a new figure with a 1x3 grid
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    axs = axs.flatten()

    for ax_target, ax_source, title in zip(axs, axes_list, titles):
        # Copy lines and labels from the original axes to the new axes
        for line in ax_source.get_lines():
            ax_target.plot(
                line.get_xdata(), line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker()
            )
        ax_target.set_title(title)
        ax_target.set_xlabel(ax_source.get_xlabel())
        ax_target.set_ylabel(ax_source.get_ylabel())
        if ax_source.get_legend():
            ax_target.legend()

    plt.close(fig_tank_fill.fig)
    plt.close(fig_tank_temperatures.fig)
    plt.close(fig_tank_pressures.fig)


    plt.tight_layout()
    plt.show()

