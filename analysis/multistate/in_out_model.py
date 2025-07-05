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
    # Get the directory of the current script
    radius = 1.0 # m
    p_init = 40000000 # Pa
    t_init = 70 # K
    fill = 0.0
    p_max = 45000000 # Pa
    p_min = 1500000 # Pa
    t_min = None # K
    ambient_heat_load = 5.0 # W/m^2

    inflow = 0.01 # kg/s
    outflow = 0.03 # kg/s

    # instantiate the Tank and TankDimensions object
    tank_material = Composite.carbon(np.radians(55))
    tank = SphericalTank(radius, tank_material, p_init)
    tank_dimensions = TankDimensions(radius, 0.0)

    fuel_mass = 200 # kg
    total_duration = 1 # hours

    # Create multi-flow mission
    inflow_rate = 0.01  # kg/s
    outflow_rate = -0.03  # kg/s

    # Get hydrogen properties - FIX: Create instance first
    hydrogen_requester = SinglePhaseRequester()
    hydrogen_props = hydrogen_requester.get_hydrogen_properties(p_init, t_init)

    # Create mission with both flows
    mission = Mission([
        Mission.multi_flow_section(
            duration=1.0,  # hours
            altitude=0,
            inflow=inflow_rate,
            outflow=outflow_rate,
            hydrogen=hydrogen_props,
            phase="gas",
            mach_number=0.0
        )
    ])

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
        TargetConditions(fuel_mass, 0.0),
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

