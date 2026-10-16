"""NSGA-II optimization for the single-tank CcH2 workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from toplab.config import load_config
from toplab.optimization.evaluation import DesignVariable, TankDesign, apply_design
from toplab.orchestration.run_analysis import run_dormancy_check
from toplab.orchestration.system_orchestrator import SystemOrchestrator
from toplab.packaging.aft_placement import AftFuselageDimensions, place_tanks_in_aft


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
    evaluation_progress: bool = True


def decode_design_vector(values: Sequence[float]) -> TankDesign:
    if len(values) != 3:
        raise ValueError("Single-tank design vectors must contain radius, phi, and insulation thickness.")
    return TankDesign(float(values[0]), float(values[1]), float(values[2]))


class SingleTankCandidateEvaluator:
    """Evaluate one independent candidate against a copied baseline scenario."""

    def __init__(self, config: NSGA2Config, packaging_dimensions: AftFuselageDimensions):
        self.config = config
        self.packaging_dimensions = packaging_dimensions
        self.base_config = load_config(config.base_config_path)
        if len(self.base_config.tanks) != 1:
            raise ValueError("The single-tank NSGA-II optimizer requires exactly one tank node.")
        self.variables = [DesignVariable(1, name, 0.0, 0.0, 0.0)
                          for name in ("radius", "phi", "insulation_thickness")]

    def evaluate(self, values: Sequence[float]) -> CandidateResult:
        design = decode_design_vector(values)
        result = CandidateResult(design=design)

        try:
            candidate = apply_design(self.base_config, self.variables, values)
            orchestrator = SystemOrchestrator(candidate, verbosity="quiet")
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

            dormancy = run_dormancy_check(replace(candidate, dormancy_check={
                "enabled": True, "duration_h": self.config.hold_period_s / 3600.0,
                "maximum_loss_percent_per_hour": 100.0}), silent=True)
            dormancy_initial = float(dormancy["tanks"][0]["initial_mass_kg"])
            vented_mass = float(dormancy["tanks"][0]["vented_mass_kg"])
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


def build_optimizer_problem(config: NSGA2Config, packaging_dimensions: AftFuselageDimensions):
    from pymoo.core.problem import ElementwiseProblem

    class SingleTankNSGA2Problem(ElementwiseProblem):
        def __init__(self):
            self.evaluator = SingleTankCandidateEvaluator(config, packaging_dimensions)
            self.history: list[CandidateResult] = []
            self.progress = config.evaluation_progress
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
            if self.progress:
                print(
                    f"[NSGA-II evaluation {len(self.history)}] x={[float(v) for v in x]} "
                    f"objectives={result.objectives} feasible={result.feasible}"
                    + (f" error={result.error}" if result.error else ""),
                    flush=True,
                )
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