"""NSGA-II optimization for the single-tank CcH2 workflow."""

from __future__ import annotations

import copy
import csv
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from toplab.configuration.scenario_configuration import ScenarioConfig
from toplab.orchestration.system_orchestrator import SystemOrchestrator
from toplab.packaging.aft_placement import AftFuselageDimensions, place_tanks_in_aft


@dataclass(frozen=True)
class TankDesign:
    radius: float
    phi: float
    insulation_thickness: float


@dataclass
class CandidateResult:
    design: TankDesign
    eta_g: float = 0.0
    eta_v: float = 0.0
    f_vent: float = 0.0
    discharge_residual: float = 0.0
    dormancy_residual: float = 0.0
    packages: bool = False
    feasible: bool = False
    mission_time_s: float = 0.0
    target_mission_time_s: float = 0.0
    initial_mass_kg: float = 0.0
    final_mass_kg: float = 0.0
    error: str | None = None

    @property
    def objectives(self) -> list[float]:
        return [-self.eta_g, -self.eta_v, self.f_vent]

    @property
    def constraints(self) -> list[float]:
        return [
            self.discharge_residual,
            self.dormancy_residual,
            0.0 if self.packages else 1.0,
        ]


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int
    generations: int
    seed: int | None
    eliminate_duplicates: bool
    lower_bounds: tuple[float, float, float]
    upper_bounds: tuple[float, float, float]
    maximum_vented_percentage: float
    hold_period_s: float
    base_config_path: Path
    output_directory: Path
    save_history: bool = True
    save_pareto_front: bool = True


def decode_design_vector(values: Sequence[float]) -> TankDesign:
    if len(values) != 3:
        raise ValueError("Single-tank design vectors must contain radius, phi, and insulation thickness.")
    return TankDesign(float(values[0]), float(values[1]), float(values[2]))


class SingleTankCandidateEvaluator:
    """Evaluate one independent candidate against a copied baseline scenario."""

    def __init__(self, config: NSGA2Config, packaging_dimensions: AftFuselageDimensions):
        self.config = config
        self.packaging_dimensions = packaging_dimensions
        with config.base_config_path.open("r", encoding="utf-8") as stream:
            self.base_config = yaml.safe_load(stream) or {}

    def evaluate(self, values: Sequence[float]) -> CandidateResult:
        design = decode_design_vector(values)
        result = CandidateResult(design=design)

        try:
            discharge_config = self._build_candidate_config(design)
            with self._temporary_config(discharge_config) as config_path:
                scenario = ScenarioConfig.from_yaml(config_path)
                orchestrator = SystemOrchestrator(scenario, verbosity="quiet")
                simulation = orchestrator.run_simulation()

            result.target_mission_time_s = float(orchestrator.tank_system.config.MISSION_DURATION)
            result.mission_time_s = float(simulation.times[-1])
            result.discharge_residual = (
                result.target_mission_time_s - result.mission_time_s
                if result.target_mission_time_s > 0.0
                else 0.0
            )
            result.initial_mass_kg = float(simulation.multi_tank_states[0].get_tank_state(0).fuel_mass)
            result.final_mass_kg = float(simulation.multi_tank_states[-1].get_tank_state(0).fuel_mass)

            if result.discharge_residual > 0.0:
                return result

            dormancy_config = self._build_dormancy_config(discharge_config)
            with self._temporary_config(dormancy_config) as config_path:
                scenario = ScenarioConfig.from_yaml(config_path)
                dormancy_orchestrator = SystemOrchestrator(scenario, verbosity="quiet")
                dormancy = dormancy_orchestrator.run_simulation()

            dormancy_initial = float(dormancy.multi_tank_states[0].get_tank_state(0).fuel_mass)
            dormancy_final = float(dormancy.multi_tank_states[-1].get_tank_state(0).fuel_mass)
            vented_mass = max(0.0, dormancy_initial - dormancy_final)
            result.f_vent = 100.0 * vented_mass / dormancy_initial if dormancy_initial > 0.0 else 0.0
            result.dormancy_residual = result.f_vent - self.config.maximum_vented_percentage

            if result.dormancy_residual > 0.0:
                return result

            tank_properties = orchestrator.tank_system._cached_tank_properties[0]
            packaging = place_tanks_in_aft(
                outer_radii=[float(tank_properties["outer_diameter"]) / 2.0],
                half_cyl_lengths=[float(tank_properties.get("cylindrical_section_length", 0.0)) / 2.0],
                dims=self.packaging_dimensions,
            )
            result.packages = bool(packaging.feasible)
            if not result.packages:
                return result

            structure_mass = float(tank_properties["liner_mass"]) + float(tank_properties["wall_mass"])
            tank_volume = float(tank_properties["volume"])
            outer_volume = float(tank_properties["outer_volume"])
            result.eta_g = result.initial_mass_kg / (result.initial_mass_kg + structure_mass)
            result.eta_v = tank_volume / outer_volume if outer_volume > 0.0 else 0.0
            result.feasible = True
            return result
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            result.error = str(exc)
            return result

    def _build_candidate_config(self, design: TankDesign) -> dict[str, Any]:
        config = copy.deepcopy(self.base_config)
        tank_nodes = [node for node in config.get("network", {}).get("nodes", []) if node.get("type") == "tank"]
        if len(tank_nodes) != 1:
            raise ValueError("The single-tank NSGA-II optimizer requires exactly one tank node.")
        tank = tank_nodes[0]
        geometry = tank.setdefault("geometry", {})
        geometry.pop("mission_based_sizing", None)
        geometry["radius"] = design.radius
        geometry["phi"] = design.phi
        tank.setdefault("materials", {}).setdefault("insulation", {})["thickness"] = design.insulation_thickness
        config.setdefault("output", {}).update({"silent": True, "save_plots": False, "save_data": False})
        config.pop("dormancy_check", None)
        return config

    def _build_dormancy_config(self, discharge_config: dict[str, Any]) -> dict[str, Any]:
        config = copy.deepcopy(discharge_config)
        config.pop("missions", None)
        config["mission"] = {
            "type": "dormancy",
            "profile": "constant_flow",
            "ambient_temperature": 288.15,
            "assigned_to_node": 1,
            "flow_rate": 0.0,
            "duration": self.config.hold_period_s,
        }
        return config

    def _temporary_config(self, config: dict[str, Any]):
        return _TemporaryConfig(config, self.config.base_config_path.parent)


class _TemporaryConfig:
    def __init__(self, config: dict[str, Any], directory: Path):
        self.config = config
        self.directory = directory
        self.path: Path | None = None

    def __enter__(self) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=self.directory, delete=False) as stream:
            yaml.safe_dump(self.config, stream)
            self.path = Path(stream.name)
        return str(self.path)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def build_optimizer_problem(config: NSGA2Config, packaging_dimensions: AftFuselageDimensions):
    from pymoo.core.problem import ElementwiseProblem

    class SingleTankNSGA2Problem(ElementwiseProblem):
        def __init__(self):
            self.evaluator = SingleTankCandidateEvaluator(config, packaging_dimensions)
            self.history: list[CandidateResult] = []
            super().__init__(
                n_var=3,
                n_obj=3,
                n_ieq_constr=3,
                xl=np.array(config.lower_bounds, dtype=float),
                xu=np.array(config.upper_bounds, dtype=float),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            result = self.evaluator.evaluate(x)
            self.history.append(result)
            out["F"] = result.objectives
            out["G"] = result.constraints

    return SingleTankNSGA2Problem()


def run_nsga2(config: NSGA2Config, packaging_dimensions: AftFuselageDimensions, initial_population=None):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    problem = build_optimizer_problem(config, packaging_dimensions)
    algorithm = NSGA2(
        pop_size=config.population_size,
        eliminate_duplicates=config.eliminate_duplicates,
        sampling=initial_population,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", config.generations),
        seed=config.seed,
        save_history=False,
        verbose=False,
    )
    return result, problem.history


def write_history(path: Path, history: Sequence[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) | asdict(result.design) for result in history]
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)