"""Production thermal-model sensitivity to foam thickness.

Each transient uses ``InsulatedTankThermalModel`` directly. Shell,
insulation, and structure temperatures evolve after an ambient step; hydrogen
mass and temperature are fixed boundary states to isolate foam thickness.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from toplab.materials.nist_materials import NISTComposite, NISTMetal
from toplab.materials.rohacell_properties import DENSITY as ROHACELL_DENSITY
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel
from toplab.thermodynamics.tank_states import IsochoricTankState

R_INNER = 0.500
T_LINER = 0.003
T_WALL = 0.015
R_STRUCTURE = R_INNER + T_LINER + T_WALL
L_CYL = 1.500
T_SHELL_THK = 0.002
A_IN = 4.0 * math.pi * R_INNER**2 + 2.0 * math.pi * R_INNER * L_CYL
V_TANK = math.pi * R_INNER**3 * (4.0 / 3.0 + L_CYL / R_INNER)

ALPHA_AMB = 5.0
EMISSIVITY = 0.05
T_H2 = 54.0
T_STRUCTURE_INIT = 55.0
T_SHELL_INIT = 288.15
T_AMB_STEP = 288.15
FUEL_MASS = 70.0 * V_TANK
T_END = 21_600.0
T_EVAL = np.arange(0.0, T_END, 30.0)
SWEEP_THICKNESSES_M = [0.025, 0.050, 0.075, 0.100]
_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class _ReferenceTank:
    volume = V_TANK

    def compute_fuel_height(self, fuel_volume: float) -> float:
        return 0.0


@dataclass
class Simulation:
    thickness_m: float
    time: np.ndarray
    structure_temperature: np.ndarray
    insulation_temperature: np.ndarray
    shell_temperature: np.ndarray
    q_ambient_to_shell: np.ndarray
    q_shell_to_insulation: np.ndarray
    q_insulation_to_structure: np.ndarray
    q_structure_to_h2: np.ndarray


def _layer_mass(density: float, radius_inner: float, radius_outer: float) -> float:
    cylinder_volume = math.pi * (radius_outer**2 - radius_inner**2) * L_CYL
    endcap_volume = (4.0 / 3.0) * math.pi * (radius_outer**3 - radius_inner**3)
    return density * (cylinder_volume + endcap_volume)


def build_model(thickness_m: float) -> InsulatedTankThermalModel:
    """Build the production five-state thermal model for one foam thickness."""
    liner = NISTMetal.aluminum_6061T6_nist()
    wall = NISTComposite.carbon_epoxy_nist()
    shell_radius = R_STRUCTURE + thickness_m
    return InsulatedTankThermalModel(
        tank_volume=V_TANK, inner_surface_area=A_IN, inner_diameter=2.0 * R_INNER,
        r_structure=R_STRUCTURE, r_shell=shell_radius, cylinder_length=L_CYL,
        liner_mass=_layer_mass(liner.density, R_INNER, R_INNER + T_LINER),
        wall_mass=_layer_mass(wall.density, R_INNER + T_LINER, R_STRUCTURE),
        foam_mass=_layer_mass(ROHACELL_DENSITY, R_STRUCTURE, shell_radius),
        shell_mass=_layer_mass(liner.density, shell_radius, shell_radius + T_SHELL_THK),
        ambient_temperature=T_AMB_STEP, alpha_amb=ALPHA_AMB, emissivity_shell=EMISSIVITY,
        liner_material=liner, wall_material=wall, shell_material=liner,
    )


def _state(tank, pressure, hydrogen, temperatures) -> IsochoricTankState:
    structure_temperature, insulation_temperature, shell_temperature = temperatures
    return IsochoricTankState(
        tank=tank, fuel_mass=FUEL_MASS, h2_temperature=T_H2,
        structure_temperature=structure_temperature,
        insulation_temperature=insulation_temperature, shell_temperature=shell_temperature,
        pressure=pressure, hydrogen=hydrogen,
    )


def run_transient(thickness_m: float, time: np.ndarray = T_EVAL) -> Simulation:
    """Integrate the production model's three thermal ODEs for one thickness."""
    model = build_model(thickness_m)
    tank = _ReferenceTank()
    initial_state = _state(tank, None, None, [T_STRUCTURE_INIT, T_STRUCTURE_INIT, T_SHELL_INIT])
    initial_insulation = model.determine_initial_insulation_temperature(T_STRUCTURE_INIT, T_SHELL_INIT)
    initial_temperatures = [T_STRUCTURE_INIT, initial_insulation, T_SHELL_INIT]

    def rhs(current_time: float, temperatures: np.ndarray) -> list[float]:
        state = _state(tank, initial_state.pressure, initial_state.hydrogen, temperatures)
        return [
            model.compute_structure_temperature_derivative(current_time, state),
            model.compute_insulation_temperature_derivative(current_time, state),
            model.compute_shell_temperature_derivative(current_time, state),
        ]

    solution = solve_ivp(
        rhs, (time[0], time[-1]), initial_temperatures, t_eval=time,
        method="LSODA", rtol=1e-9, atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    states = [
        _state(tank, initial_state.pressure, initial_state.hydrogen, temperatures)
        for temperatures in solution.y.T
    ]
    structure_temperature, insulation_temperature, shell_temperature = solution.y
    return Simulation(
        thickness_m=thickness_m, time=time,
        structure_temperature=structure_temperature,
        insulation_temperature=insulation_temperature,
        shell_temperature=shell_temperature,
        q_ambient_to_shell=np.array([
            model.compute_ambient_to_shell_heat_flux(state.shell_temperature) for state in states
        ]),
        q_shell_to_insulation=np.array([
            model.compute_shell_to_insulation_heat_flux(state.shell_temperature, state.insulation_temperature)
            for state in states
        ]),
        q_insulation_to_structure=np.array([
            model.compute_insulation_to_structure_heat_flux(
                state.insulation_temperature, state.structure_temperature
            ) for state in states
        ]),
        q_structure_to_h2=np.array([
            model.compute_structure_to_h2_heat_flux(current_time, state)
            for current_time, state in zip(time, states)
        ]),
    )


def _save(figure: plt.Figure, name: str) -> None:
    os.makedirs(_OUTDIR, exist_ok=True)
    path = os.path.join(_OUTDIR, name)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {path}")


def _plot_by_thickness(simulations: list[Simulation], panels, title: str, filename: str) -> None:
    figure, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(simulations)))
    for axis, (label, attribute) in zip(axes, panels):
        for simulation, color in zip(simulations, colors):
            axis.plot(
                simulation.time / 3600.0, getattr(simulation, attribute), color=color, lw=1.8,
                label=f"{simulation.thickness_m * 1_000:.0f} mm",
            )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[0].legend(title="Foam thickness", fontsize=9)
    axes[-1].set_xlabel("Time [h]")
    figure.tight_layout()
    _save(figure, filename)


def plot_node_temperatures(simulations: list[Simulation]) -> None:
    _plot_by_thickness(
        simulations,
        [("Shell temperature [K]", "shell_temperature"),
         ("Insulation temperature [K]", "insulation_temperature"),
         ("Structure temperature [K]", "structure_temperature")],
        "Production transient model: thermal-node temperatures",
        "foam_thickness_node_temperatures.png",
    )


def plot_heat_flows(simulations: list[Simulation]) -> None:
    _plot_by_thickness(
        simulations,
        [("Ambient to shell [W]", "q_ambient_to_shell"),
         ("Shell to insulation [W]", "q_shell_to_insulation"),
         ("Insulation to structure [W]", "q_insulation_to_structure"),
         ("Structure to H2 [W]", "q_structure_to_h2")],
        "Production transient model: directional heat flows",
        "foam_thickness_heat_flows.png",
    )


def print_summary(simulations: list[Simulation]) -> None:
    print("\nFoam thickness sensitivity at 6 h")
    print("thickness  T_shell  T_foam  T_structure  Q_foam->structure  Q_structure->H2")
    for simulation in simulations:
        print(
            f"{simulation.thickness_m * 1_000:>7.0f} mm  "
            f"{simulation.shell_temperature[-1]:>7.2f} K  "
            f"{simulation.insulation_temperature[-1]:>6.2f} K  "
            f"{simulation.structure_temperature[-1]:>11.2f} K  "
            f"{simulation.q_insulation_to_structure[-1]:>17.3f} W  "
            f"{simulation.q_structure_to_h2[-1]:>15.3f} W"
        )


def main() -> None:
    print("Production five-state thermal-model foam-thickness sensitivity")
    print(f"Ambient step: shell starts at {T_SHELL_INIT:.0f} K; ambient is {T_AMB_STEP:.2f} K")
    print(f"H2 boundary state: T_H2 = {T_H2:.1f} K; fuel mass = {FUEL_MASS:.2f} kg")
    simulations = [run_transient(thickness) for thickness in SWEEP_THICKNESSES_M]
    print_summary(simulations)
    plot_node_temperatures(simulations)
    plot_heat_flows(simulations)
    print(f"\nStudy complete. Figures saved to: {_OUTDIR}")


if __name__ == "__main__":
    main()