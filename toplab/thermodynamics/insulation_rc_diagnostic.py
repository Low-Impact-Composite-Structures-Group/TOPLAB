"""Diagnostic radial RC discretization for tank insulation.

This module is intended for validation studies only. It does not replace or
alter :class:`InsulatedTankThermalModel`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq

from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel


class InsulationRCModel:
    """N-node radial insulation network using the tank's existing material model."""

    def __init__(self, tank_model: InsulatedTankThermalModel, node_count: int):
        if node_count < 1:
            raise ValueError("node_count must be at least 1")
        if tank_model.insulation_model.integrated_conductivity is None:
            raise ValueError(
                "The radial RC diagnostic requires an insulation model with "
                "integrated_conductivity"
            )
        if tank_model.insulation_model.heat_flux is not None:
            raise ValueError(
                "The radial RC diagnostic does not support heat-flux-only "
                "insulation models"
            )

        self.tank_model = tank_model
        self.node_count = node_count
        self.radii = self._control_volume_faces(node_count)
        self.node_radii = self._node_center_radii()
        density = tank_model.insulation_model.density
        self.node_masses = np.asarray(
            [
                density * self._insulation_volume(outer) -
                density * self._insulation_volume(inner)
                for inner, outer in zip(self.radii[:-1], self.radii[1:])
            ],
            dtype=float,
        )[::-1]

    def _insulation_volume(self, radius: float) -> float:
        model = self.tank_model
        return (
            math.pi * model.L * (radius**2 - model.r_structure**2)
            + (4.0 * math.pi / 3.0) * (radius**3 - model.r_structure**3)
        )

    def _control_volume_faces(self, node_count: int) -> np.ndarray:
        model = self.tank_model
        inner, outer = model.r_structure, model.r_shell
        if node_count == 1:
            return np.asarray([inner, outer], dtype=float)

        total_volume = self._insulation_volume(outer)
        radii = [inner]
        for index in range(1, node_count):
            target_volume = total_volume * index / node_count
            radii.append(
                brentq(
                    lambda radius: self._insulation_volume(radius) - target_volume,
                    inner,
                    outer,
                )
            )
        radii.append(outer)
        return np.asarray(radii, dtype=float)

    def _node_center_radii(self) -> np.ndarray:
        centers = []
        for inner, outer in zip(self.radii[:-1], self.radii[1:]):
            inner_volume = self._insulation_volume(inner)
            outer_volume = self._insulation_volume(outer)
            centers.append(
                brentq(
                    lambda radius: self._insulation_volume(radius)
                    - 0.5 * (inner_volume + outer_volume),
                    inner,
                    outer,
                )
            )
        return np.asarray(centers[::-1], dtype=float)

    @property
    def total_insulation_mass(self) -> float:
        return float(np.sum(self.node_masses))

    def _integrated_conductivity(self, cold_temperature: float, hot_temperature: float) -> float:
        integrated = self.tank_model.insulation_model.integrated_conductivity
        if integrated is None:
            raise RuntimeError("Insulation model has no integrated conductivity")
        return float(integrated(cold_temperature, hot_temperature))

    def _segment_geometry_factor(self, inner_radius: float, outer_radius: float) -> float:
        model = self.tank_model
        return (
            2.0 * math.pi * model.L / math.log(outer_radius / inner_radius)
            + 4.0 * math.pi * inner_radius * outer_radius
            / (outer_radius - inner_radius)
        )

    def _segment_heat_flow(
        self,
        inner_radius: float,
        outer_radius: float,
        inner_temperature: float,
        outer_temperature: float,
    ) -> float:
        return self._segment_geometry_factor(inner_radius, outer_radius) * (
            self._integrated_conductivity(inner_temperature, outer_temperature)
        )

    def heat_flows(
        self,
        shell_temperature: float,
        node_temperatures: Iterable[float],
        structure_temperature: float,
    ) -> np.ndarray:
        """Return heat flows in the outward-to-inward direction, in watts."""
        temperatures = np.asarray(tuple(node_temperatures), dtype=float)
        if temperatures.shape != (self.node_count,):
            raise ValueError(
                f"Expected {self.node_count} insulation temperatures, "
                f"received {temperatures.size}"
            )

        if self.node_count == 1:
            return np.asarray(
                [
                    self.tank_model.compute_shell_to_insulation_heat_flux(
                        shell_temperature, float(temperatures[0])
                    ),
                    self.tank_model.compute_insulation_to_structure_heat_flux(
                        float(temperatures[0]), structure_temperature
                    ),
                ],
                dtype=float,
            )

        boundary_temperatures = np.concatenate(
            ([shell_temperature], temperatures, [structure_temperature])
        )
        boundary_radii = np.concatenate(
            ([self.tank_model.r_shell], self.node_radii, [self.tank_model.r_structure])
        )
        return np.asarray(
            [
                self._segment_heat_flow(
                    boundary_radii[index + 1],
                    boundary_radii[index],
                    float(boundary_temperatures[index + 1]),
                    float(boundary_temperatures[index]),
                )
                for index in range(self.node_count + 1)
            ],
            dtype=float,
        )

    def temperature_derivatives(
        self,
        shell_temperature: float,
        node_temperatures: Iterable[float],
        structure_temperature: float,
    ) -> np.ndarray:
        """Return dT/dt for each insulation node in K/s."""
        temperatures = np.asarray(tuple(node_temperatures), dtype=float)
        flows = self.heat_flows(
            shell_temperature, temperatures, structure_temperature
        )
        heat_capacities = np.asarray(
            [
                mass
                * float(
                    self.tank_model.insulation_model.specific_heat(
                        max(4.0, min(float(temperature), 400.0))
                    )
                )
                for mass, temperature in zip(self.node_masses, temperatures)
            ],
            dtype=float,
        )
        return (flows[:-1] - flows[1:]) / heat_capacities

    def energy(self, node_temperatures: Iterable[float]) -> float:
        """Return relative insulation internal energy in joules.

        The arbitrary reference is 4 K; the heat-capacity integration follows
        the same 4–400 K clamping used by the production insulation ODE.
        """
        temperatures = np.asarray(tuple(node_temperatures), dtype=float)
        if temperatures.shape != (self.node_count,):
            raise ValueError(
                f"Expected {self.node_count} insulation temperatures, "
                f"received {temperatures.size}"
            )
        specific_heat = self.tank_model.insulation_model.specific_heat

        def specific_energy(temperature: float) -> float:
            if temperature < 4.0:
                return float(specific_heat(4.0)) * (temperature - 4.0)
            if temperature > 400.0:
                return (
                    specific_energy(400.0)
                    + float(specific_heat(400.0)) * (temperature - 400.0)
                )
            sample_count = max(3, int(math.ceil((temperature - 4.0) / 0.1)) + 1)
            sample_temperatures = np.linspace(4.0, temperature, sample_count)
            heat_capacities = np.fromiter(
                (float(specific_heat(float(sample))) for sample in sample_temperatures),
                dtype=float,
                count=sample_count,
            )
            return float(simpson(heat_capacities, x=sample_temperatures))

        return float(
            sum(
                mass
                * specific_energy(float(node_temperature))
                for mass, node_temperature in zip(self.node_masses, temperatures)
            )
        )

    def energy_residual(
        self,
        shell_temperature: float,
        node_temperatures: Iterable[float],
        structure_temperature: float,
    ) -> float:
        """Return instantaneous inlet minus outlet minus insulation storage [W]."""
        temperatures = np.asarray(tuple(node_temperatures), dtype=float)
        flows = self.heat_flows(
            shell_temperature, temperatures, structure_temperature
        )
        derivatives = self.temperature_derivatives(
            shell_temperature, temperatures, structure_temperature
        )
        specific_heat = self.tank_model.insulation_model.specific_heat
        stored_power = sum(
            mass
            * float(
                specific_heat(max(4.0, min(float(temperature), 400.0)))
            )
            * derivative
            for mass, temperature, derivative in zip(
                self.node_masses, temperatures, derivatives
            )
        )
        return float(flows[0] - flows[-1] - stored_power)


def insulation_diffusion_times(
    tank_model: InsulatedTankThermalModel,
    temperatures: Iterable[float],
    conductivity_step: float = 0.01,
) -> list[dict[str, float]]:
    """Calculate insulation diffusion times from the selected material model."""
    insulation = tank_model.insulation_model
    if insulation.integrated_conductivity is None:
        raise ValueError(
            "Diffusion-time diagnostics require an insulation model with "
            "integrated_conductivity"
        )
    if conductivity_step <= 0.0:
        raise ValueError("conductivity_step must be positive")
    thickness = tank_model.r_shell - tank_model.r_structure
    rows = []
    for temperature in temperatures:
        if temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        low = temperature - conductivity_step
        high = temperature + conductivity_step
        conductivity = float(
            insulation.integrated_conductivity(low, high)
        ) / (2.0 * conductivity_step)
        specific_heat = float(insulation.specific_heat(temperature))
        diffusivity = conductivity / (insulation.density * specific_heat)
        diffusion_seconds = thickness**2 / diffusivity
        rows.append(
            {
                "temperature_K": float(temperature),
                "conductivity_W_mK": conductivity,
                "specific_heat_J_kgK": specific_heat,
                "density_kg_m3": float(insulation.density),
                "diffusivity_m2_s": diffusivity,
                "diffusion_time_s": diffusion_seconds,
                "diffusion_time_h": diffusion_seconds / 3600.0,
            }
        )
    return rows
