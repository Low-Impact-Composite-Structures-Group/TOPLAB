

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

import plotting.plot_style
from src.dynamics.dynamic_models import LinModel
from src.fluids.hydrogen_retrievers import TwoPhaseRequester
from src.materials.materials import Metal
from src.tank_design.tank_shapes import SphericalTank


HOURS_IN_SECONDS = 60 ** 2
TIMESTEP = HOURS_IN_SECONDS
VALUE_TO_KILO = 1e-3
STRATIFICATION_FACTOR = 2


"""Supporting classes for the analysis."""

class TwoPhaseHydrogen(Protocol):
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

    # Plot the analysis
    plot_pressure_rise(
        fill_percentages,
        heat_fluxes,
        initial_pressure,
        max_pressure,
        mass_flow
    )
    plot_reference(fill_percentages, heat_fluxes)
    create_legend(fill_percentages, heat_fluxes)
    prettify_plot()
    plt.show()


""" Support function to create overview in the code."""

def plot_pressure_rise(
    fill_percentages: float,
    heat_fluxes: float,
    initial_pressure: float,
    max_pressure: float,
    mass_flow: float
) -> None:
    # Define hydrogen requester and initial conditions
    for fill_percentage in fill_percentages:
        for heat_flux in heat_fluxes:
            tank_states = compute_tank_states(
                initial_pressure,
                max_pressure,
                mass_flow,
                fill_percentage,
                heat_flux
            )

            plt.plot(
                [
                    hour / 24
                    for hour in range(len(tank_states))
                ],
                [
                    state.pressure * VALUE_TO_KILO
                    for state in tank_states
                ],
                marker=""
            )

def prettify_plot() -> None:
    x_ticks = np.arange(0, 13, 2)
    y_ticks = np.arange(100, 141, 10)
    plt.xticks(x_ticks)
    plt.xlim((x_ticks[0], x_ticks[-1]))
    plt.yticks(y_ticks)
    plt.ylim((y_ticks[0], y_ticks[-1]))
    plt.ylabel("Tank Pressure [kPa]")
    plt.xlabel("Time [days]")
    plt.grid()
    plt.legend()

def compute_tank_states(
    initial_pressure: float,
    max_pressure: float,
    mass_flow: float,
    fill_percentage: float,
    heat_flux: float
) -> list[TankState]:
    pressure = initial_pressure
    tank_states: list[TankState] = [
                define_tank_state(
                    pressure,
                    fill_percentage,
                    heat_flux
                )
            ]
    fuel_flows = [FuelFlow(tank_states[-1].hydrogen, mass_flow)]
    while tank_states[-1].pressure < max_pressure:
        state_derivatives = LinModel().compute_state_derivatives(
            tank_states[-1], fuel_flows
        )
        dP_dt = state_derivatives.pressure
        pressure += dP_dt * TIMESTEP * STRATIFICATION_FACTOR
        tank_states.append(
            define_tank_state(
                pressure,
                fill_percentage,
                heat_flux
            )
        )
        
    return tank_states

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

def plot_reference(
    fill_percentages: list[float],
    heat_fluxes: list[float]
) -> None:
    ref_data_path = (
        Path.cwd() / "data" / "reference" / "lin_pressure_rise.json"
    )
    with ref_data_path.open("r") as file:
        ref_data = json.load(file)
    # Plot the reference data
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    for fill_percentage in fill_percentages:
        for heat_flux in heat_fluxes:
            # Plot reference data
            fill_key = f"{int(fill_percentage*100)}_fill"
            flux_key = f"{heat_flux}_flux"
            row_data = ref_data[fill_key][flux_key]
            plt.plot(
                row_data["times"], row_data["pressures"], linestyle=""
            )
    return None


"""Main function."""

def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
