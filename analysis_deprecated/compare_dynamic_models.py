
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

import plotting.plot_style
from src.dynamics.dynamic_models.protocols import DynamicModel
from src.dynamics.dynamic_models.lin import LinModel
from src.dynamics.dynamic_models.ahluwalia import TwoPhaseModel
from src.fluids.hydrogen_retrievers import TwoPhaseRequester
from src.materials.materials import Metal
from src.tank_design.tank_shapes import SphericalTank

DAY_IN_HOURS = 24
VALUE_TO_KILO = 1e-3
HOURS_TO_DAYS = 1 / DAY_IN_HOURS
HOURS_IN_SECONDS = 60 ** 2
STRATIFICATION_FACTOR = 2


"""Support classes for the analysis."""

class Hydrogen(Protocol):
    density: float


class TwoPhaseHydrogen(Protocol):
    liquid: Hydrogen
    gas: Hydrogen
    pressure: float


@dataclass
class TankState:
    fill: float
    heat_flux: float
    volume: float
    hydrogen: TwoPhaseHydrogen

    @property
    def pressure(self):
        return self.hydrogen.pressure

    @property
    def gas_mass(self):
        return (1 - self.fill) * self.volume * self.hydrogen.gas.density

    @property
    def liquid_mass(self):
        return self.fill * self.volume * self.hydrogen.liquid.density

    @property
    def fuel_mass(self):
        return self.liquid_mass + self.gas_mass

    @property
    def tank_thermal_capacity(self):
        return 0

    @property
    def tank_temperature(self):
        None


@dataclass
class FuelFlow:
    hydrogen: TwoPhaseHydrogen
    mass_flow: float


"""Main function to be called for the analysis."""

def perform_analysis():

    # Tank properties, fuel flows definition, and initial conditions
    fill_percentages = [0.95, 0.3]      # [%]
    heat_fluxes = [20, 100]             # [W]
    mass_flow = 0
    initial_pressure = 101e3            # [kPa]

    # Define the maximum pressure for the tank
    max_pressure = 138e3                # [kPa]

    # Define the dynamic models to be iterated over
    dynamic_models = [
        # Dynamic model, Marker style, Linestyle
        (LinModel(), None, ""),
        (TwoPhaseModel(), "", None)
    ]

    # The timesteps need to be changed per loading and fill as the 
    # amount of markers need to be reasonable for each load case
    timesteps = [
        DAY_IN_HOURS * HOURS_IN_SECONDS,
        DAY_IN_HOURS * HOURS_IN_SECONDS * 0.2,
        DAY_IN_HOURS * HOURS_IN_SECONDS * 0.5,
        DAY_IN_HOURS * HOURS_IN_SECONDS * 0.08
    ]

    # Loop through the dynamic models and load cases to create the plot
    for dynamic_model, marker, linestyle in dynamic_models:
        plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
        i = 0
        for fill in fill_percentages:
            for heat_flux in heat_fluxes:
                tank_states: list[TankState] = compute_tank_states(
                    initial_pressure,
                    max_pressure,
                    mass_flow,
                    fill,
                    heat_flux,
                    dynamic_model,
                    timesteps[i]
                )
                plt.plot(
                    [
                        step * timesteps[i]
                        / DAY_IN_HOURS / HOURS_IN_SECONDS
                        for step in range(len(tank_states))
                    ],
                    [
                        state.pressure * VALUE_TO_KILO
                        for state in tank_states
                    ],
                    marker=marker,
                    linestyle=linestyle
                )
                i += 1

    create_legend(fill_percentages, heat_fluxes)
    prettify_plot()
    plt.show()


"""Support functions to organise the code."""

def define_tank_state(
    pressure: float,
    fill_percentage: float,
    heat_flux: float
) -> None:
    hydrogen = TwoPhaseRequester().get_hydrogen_properties(
                pressure, None
            )
    return TankState(
        fill=fill_percentage,
        heat_flux=heat_flux,
        volume=SphericalTank.lin(
            Metal.aluminum(), pressure
        ).volume,
        hydrogen=hydrogen
    )

def compute_tank_states(
    initial_pressure: float,
    max_pressure: float,
    mass_flow: float,
    fill_percentage: float,
    heat_flux: float,
    dynamic_model: DynamicModel,
    timestep: float
) -> list[TankState]:
    pressure = initial_pressure
    tank_states: list[TankState] = [
                define_tank_state(
                    pressure,
                    fill_percentage,
                    heat_flux
                )
            ]
    fuel_flows = [FuelFlow(tank_states[-1].hydrogen.liquid, mass_flow)]
    while tank_states[-1].pressure < max_pressure:
        state_derivatives = dynamic_model.compute_state_derivatives(
            tank_states[-1], fuel_flows
        )
        dP_dt = state_derivatives.pressure
        pressure += dP_dt * timestep * STRATIFICATION_FACTOR
        tank_states.append(
            define_tank_state(
                pressure,
                fill_percentage,
                heat_flux
            )
        )
    return tank_states

def create_legend(
    fill_percentages: float,
    heat_fluxes: float
) -> None:
    # Fake plot to get the desired legend
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    for fill_percentage in fill_percentages:
        for heat_flux in heat_fluxes:
            label = f"{int(fill_percentage*100)} %, {heat_flux} W"
            # Plot reference data
            plt.plot([], [], label=label)

def prettify_plot() -> None:
    x_ticks = np.arange(0, 15, 2)
    y_ticks = np.arange(100, 151, 10)
    plt.xticks(x_ticks)
    plt.xlim((x_ticks[0], x_ticks[-1]))
    plt.yticks(y_ticks)
    plt.ylim((y_ticks[0], y_ticks[-1]))
    plt.ylabel("Tank Pressure [kPa]")
    plt.xlabel("Time [days]")
    plt.grid()
    plt.legend()


"""Main function."""

def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
