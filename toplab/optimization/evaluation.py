"""Typed candidate evaluation shared by the optimisers.

A candidate is a list of :class:`DesignVariable` values applied to the typed
``AnalysisConfig`` with ``dataclasses.replace``; no YAML is written. Evaluation
runs the discharge mission, the dormancy hold and aft-fuselage packaging.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from toplab.config import AnalysisConfig
from toplab.packaging.aft_placement import AftFuselageDimensions, maximize_packaging_compaction

PARAMETERS = ("radius", "phi", "insulation_thickness")
_ALIASES = {"r": "radius", "t_ins": "insulation_thickness"}


@dataclass(frozen=True)
class TankDesign:
    radius: float
    phi: float
    insulation_thickness: float


@dataclass(frozen=True)
class DesignVariable:
    """One scalar design variable on one tank (``tank`` is the 1-based tank index)."""
    tank: int
    parameter: str
    lower: float
    upper: float
    baseline: float

    @property
    def name(self) -> str:
        return f"{self.parameter}_{self.tank}"


def parse_design_variables(raw: Mapping[str, Mapping[str, float]]) -> list[DesignVariable]:
    """Parse ``{radius_1: {min, max, baseline}, ...}`` into an ordered list."""
    variables = []
    for name, spec in raw.items():
        match = re.fullmatch(r"(radius|phi|insulation_thickness|r|t_ins)_?(\d+)", name)
        if match is None:
            raise ValueError(
                f"Design variable '{name}' must be '<parameter>_<tank>' with parameter in {PARAMETERS}.")
        parameter = _ALIASES.get(match.group(1), match.group(1))
        variables.append(DesignVariable(int(match.group(2)), parameter, float(spec["min"]),
                                        float(spec["max"]), float(spec["baseline"])))
    if not variables:
        raise ValueError("design_variables must define at least one variable.")
    return variables


def apply_design(config: AnalysisConfig, variables: Sequence[DesignVariable],
                 values: Sequence[float]) -> AnalysisConfig:
    """Return a copy of ``config`` with the design values applied and mission sizing disabled."""
    if len(values) != len(variables):
        raise ValueError(f"Expected {len(variables)} design values, got {len(values)}.")
    geometries = []
    materials = []
    for tank in config.tanks:
        geometry = dict(tank.geometry)
        # A fixed radius replaces mission sizing for every tank, including untouched ones.
        geometry.pop("mission_based_sizing", None)
        geometries.append(geometry)
        materials.append(copy.deepcopy(tank.materials))
    for variable, value in zip(variables, values):
        if not 1 <= variable.tank <= len(geometries):
            raise ValueError(f"Design variable {variable.name} refers to a tank that does not exist.")
        index = variable.tank - 1
        if variable.parameter == "insulation_thickness":
            materials[index].setdefault("insulation", {})["thickness"] = float(value)
        else:
            geometries[index][variable.parameter] = float(value)
    tanks = [replace(tank, geometry=g, materials=m)
             for tank, g, m in zip(config.tanks, geometries, materials)]
    return replace(config, tanks=tanks, dormancy_check=None,
                   output={**config.output, "silent": True, "save_plots": False, "save_data": False})


def packaging_dimensions(config: AnalysisConfig) -> AftFuselageDimensions:
    values = (config.packaging or {}).get("aft_fuselage_dimensions")
    if not values:
        raise ValueError("Base configuration must define packaging.aft_fuselage_dimensions.")
    return AftFuselageDimensions(**{key: float(value) for key, value in values.items()})


@dataclass
class CandidateResult:
    designs: list[TankDesign] = field(default_factory=list)
    eta_g: float = 0.0
    eta_v: float = 0.0
    eta_vent: float = 0.0
    zeta_p: float = 0.0
    f_vent: float = 0.0
    vented_mass_kg: float = 0.0
    per_tank_vented_fraction: list[float] = field(default_factory=list)
    discharge_residuals: list[float] = field(default_factory=list)
    discharge_residual: float = 0.0
    dormancy_residuals: list[float] = field(default_factory=list)
    dormancy_residual: float = 0.0
    packaging_feasible: bool = False
    packages: bool = False
    z_shift_star: float = 0.0
    mission_completed: bool = False
    initial_hydrogen_masses_kg: list[float] = field(default_factory=list)
    final_hydrogen_masses_kg: list[float] = field(default_factory=list)
    structural_masses_kg: list[float] = field(default_factory=list)
    internal_volumes_m3: list[float] = field(default_factory=list)
    external_volumes_m3: list[float] = field(default_factory=list)
    placements: list[Any] = field(default_factory=list)
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return (self.error is None and self.mission_completed and
                self.discharge_residual <= 0.0 and self.dormancy_residual <= 0.0 and
                self.packaging_feasible)

    @property
    def objectives(self) -> list[float]:
        return [self.eta_g, self.eta_v, self.eta_vent, self.zeta_p]


def venting_performance(vented_fraction: float, maximum_vented_fraction: float) -> float:
    """Return the signed venting performance; values below zero are retained."""
    if maximum_vented_fraction <= 0.0:
        raise ValueError("maximum_vented_fraction must be positive.")
    return 1.0 - float(vented_fraction) / float(maximum_vented_fraction)


class CandidateEvaluator:
    """Evaluate discharge, dormancy, system metrics and aft packaging for one design."""

    def __init__(self, base_config: AnalysisConfig, variables: Sequence[DesignVariable],
                 maximum_vented_fraction: float, hold_period_s: float, progress: bool = True) -> None:
        self.base_config = base_config
        self.variables = list(variables)
        self.packaging_dimensions = packaging_dimensions(base_config)
        self.maximum_vented_fraction = float(maximum_vented_fraction)
        self.hold_period_s = float(hold_period_s)
        self.progress = bool(progress)
        self.evaluation_count = 0

    def _progress(self, message: str) -> None:
        if self.progress:
            print(f"[evaluation {self.evaluation_count}] {message}", flush=True)

    def evaluate(self, values: Sequence[float]) -> CandidateResult:
        from toplab.orchestration.run_analysis import run_dormancy_check
        from toplab.orchestration.system_orchestrator import SystemOrchestrator

        result = CandidateResult()
        self.evaluation_count += 1
        started = time.perf_counter()
        self._progress(f"start x={[float(v) for v in values]}")
        try:
            candidate = apply_design(self.base_config, self.variables, values)
            orchestrator = SystemOrchestrator(candidate, verbosity="quiet")
            discharge = orchestrator.run_simulation()
            tank_count = len(orchestrator.tank_geometries)

            target_s = float(orchestrator.tank_system.config.MISSION_DURATION)
            actual_s = float(discharge.times[-1])
            result.mission_completed = actual_s >= target_s
            result.discharge_residuals = [target_s - actual_s] * tank_count
            result.discharge_residual = max(result.discharge_residuals, default=0.0)
            result.initial_hydrogen_masses_kg = [
                float(discharge.multi_tank_states[0].get_tank_state(i).fuel_mass) for i in range(tank_count)]
            result.final_hydrogen_masses_kg = [
                float(discharge.multi_tank_states[-1].get_tank_state(i).fuel_mass) for i in range(tank_count)]
            properties = orchestrator.tank_system._cached_tank_properties
            result.structural_masses_kg = [
                float(properties[i]["liner_mass"]) + float(properties[i]["wall_mass"])
                for i in range(tank_count)]
            result.internal_volumes_m3 = [float(properties[i]["volume"]) for i in range(tank_count)]
            result.external_volumes_m3 = [float(properties[i]["outer_volume"]) for i in range(tank_count)]
            hydrogen = sum(result.initial_hydrogen_masses_kg)
            total_mass = hydrogen + sum(result.structural_masses_kg)
            result.eta_g = hydrogen / total_mass if total_mass > 0.0 else 0.0
            external = sum(result.external_volumes_m3)
            result.eta_v = sum(result.internal_volumes_m3) / external if external > 0.0 else 0.0
            self._progress(f"discharge actual_s={actual_s:.1f} target_s={target_s:.1f} eta_g={result.eta_g:.4f}")

            dormancy_case = replace(candidate, dormancy_check={
                "enabled": True, "duration_h": self.hold_period_s / 3600.0,
                "maximum_loss_percent_per_hour": 100.0})
            dormancy = run_dormancy_check(dormancy_case, silent=True)
            initial = [t["initial_mass_kg"] for t in dormancy["tanks"]]
            vented = [t["vented_mass_kg"] for t in dormancy["tanks"]]
            result.vented_mass_kg = sum(vented)
            result.per_tank_vented_fraction = [v / m if m > 0.0 else 0.0 for v, m in zip(vented, initial)]
            result.f_vent = result.vented_mass_kg / sum(initial) if sum(initial) > 0.0 else 0.0
            limit = self.maximum_vented_fraction
            result.eta_vent = venting_performance(result.f_vent, limit)
            result.dormancy_residuals = [f / limit - 1.0 for f in result.per_tank_vented_fraction]
            result.dormancy_residual = max(result.dormancy_residuals, default=0.0)
            self._progress(f"dormancy f_vent={result.f_vent:.6g} g_dorm={result.dormancy_residual:.6g}")

            radii = [float(properties[i]["outer_diameter"]) / 2.0 for i in range(tank_count)]
            half_lengths = [float(properties[i].get("cylindrical_section_length", 0.0)) / 2.0
                            for i in range(tank_count)]
            packaging = maximize_packaging_compaction(radii, half_lengths, self.packaging_dimensions)
            result.packaging_feasible = result.packages = bool(packaging.feasible)
            result.z_shift_star = float(packaging.z_shift)
            result.zeta_p = float(packaging.zeta_p)
            result.placements = packaging.placements
            result.designs = [
                TankDesign(float(g.radius), float(g.phi),
                           float(candidate.tanks[i].materials
                                 .get("insulation", {}).get("thickness", 0.0)))
                for i, g in enumerate(orchestrator.tank_geometries)]
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            self._progress(f"error={result.error}")
        self._progress(f"done elapsed={time.perf_counter() - started:.1f}s "
                       f"objectives={result.objectives} feasible={result.feasible}")
        return result
