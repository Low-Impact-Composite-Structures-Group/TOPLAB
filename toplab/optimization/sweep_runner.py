from __future__ import annotations

import multiprocessing as mp
import os
import time
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Sequence, TypeVar

import yaml


DEFAULT_RANKING = (
    "mission_completed",
    "gravimetric_efficiency",
    "volumetric_efficiency",
    "mission_completion_ratio",
)

DesignT = TypeVar("DesignT")


@dataclass(frozen=True)
class SweepRuntimeConfig:
    sweep_config_path: Path
    base_config_path: Path
    report_path: Path
    verbosity: str = "quiet"
    silent: bool = True
    save_plots: bool = False
    save_data: bool = False
    ranking: tuple[str, ...] = DEFAULT_RANKING
    require_mission_completion: bool = False
    case_timeout_s: float | None = None


@dataclass
class SweepResult:
    design: object
    mission_completed: bool
    mission_completion_ratio: float
    mission_duration_s: float
    target_duration_s: float
    gravimetric_efficiency: float
    volumetric_efficiency: float
    total_fuel_mass_kg: float
    structure_mass_kg: float
    total_inner_volume_m3: float
    total_outer_volume_m3: float
    error: str | None = None

    @property
    def is_successful(self) -> bool:
        return self.error is None


class BaseSweepStudy(Generic[DesignT]):
    def __init__(self, runtime_config: SweepRuntimeConfig) -> None:
        self.sweep_config_path = runtime_config.sweep_config_path
        self.base_config_path = runtime_config.base_config_path
        self.verbosity = runtime_config.verbosity
        self.report_path = runtime_config.report_path
        self.silent = runtime_config.silent
        self.save_plots = runtime_config.save_plots
        self.save_data = runtime_config.save_data
        self.ranking = runtime_config.ranking
        self.require_mission_completion = runtime_config.require_mission_completion
        self.case_timeout_s = runtime_config.case_timeout_s

    @classmethod
    def load_sweep_config(cls, sweep_config_path: str | Path) -> tuple[Path, dict]:
        config_path = Path(sweep_config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as stream:
            sweep_config = yaml.safe_load(stream) or {}
        return config_path, sweep_config

    @classmethod
    def create_runtime_config(
        cls,
        *,
        sweep_config_path: Path,
        sweep_config: dict,
        project_root: Path,
        default_base_config: Path,
        default_report_path: Path,
    ) -> SweepRuntimeConfig:
        output_config = sweep_config.get("output", {})
        constraints = sweep_config.get("constraints", {})
        sweep_section = sweep_config.get("sweep", {})

        base_config = sweep_config.get("base_config", default_base_config)
        report_path = output_config.get("report_path", default_report_path)
        timeout_constraint_s = constraints.get("max_case_wall_time_s", sweep_section.get("case_timeout_s"))
        ranking = tuple(sweep_config.get("ranking") or DEFAULT_RANKING)

        if not Path(base_config).is_absolute():
            base_config = (project_root / base_config).resolve()
        if not Path(report_path).is_absolute():
            report_path = (project_root / report_path).resolve()

        case_timeout_s = None if timeout_constraint_s is None else float(timeout_constraint_s)

        return SweepRuntimeConfig(
            sweep_config_path=sweep_config_path,
            base_config_path=Path(base_config),
            report_path=Path(report_path),
            verbosity=output_config.get("verbosity", "quiet"),
            silent=bool(output_config.get("silent", True)),
            save_plots=bool(output_config.get("save_plots", False)),
            save_data=bool(output_config.get("save_data", False)),
            ranking=ranking,
            require_mission_completion=bool(constraints.get("require_mission_completion", False)),
            case_timeout_s=case_timeout_s,
        )

    def run_sweep(self, design_points: Iterable[DesignT]) -> list[SweepResult]:
        design_list = list(design_points)
        results: list[SweepResult] = []
        for index, design in enumerate(design_list, start=1):
            print(
                f"Now running case {index}/{len(design_list)} with design vector {self.format_design_vector(design)}",
                flush=True,
            )
            case_start = time.perf_counter()
            result = self._evaluate_design_with_timeout(design)
            elapsed_s = time.perf_counter() - case_start
            status = "PASS" if result.mission_completed else ("FAIL" if result.error else "SHORT")
            print(
                f"Completed case {index}/{len(design_list)}: {status} "
                f"(mission={result.mission_duration_s/3600.0:.3f}/{result.target_duration_s/3600.0:.3f} h, "
                f"gravimetric={result.gravimetric_efficiency:.4f}, "
                f"volumetric={result.volumetric_efficiency:.4f}, "
                f"wall_time={elapsed_s:.2f} s)",
                flush=True,
            )
            results.append(result)
        return results

    def _evaluate_design_with_timeout(self, design: DesignT) -> SweepResult:
        if self.case_timeout_s is None or self.case_timeout_s <= 0.0:
            return self.evaluate_design(design)

        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        process = ctx.Process(
            target=_evaluate_design_worker,
            args=(queue, type(self), str(self.sweep_config_path), design),
        )
        process.start()
        process.join(self.case_timeout_s)

        if process.is_alive():
            process.terminate()
            process.join()
            return SweepResult(
                design=design,
                mission_completed=False,
                mission_completion_ratio=0.0,
                mission_duration_s=0.0,
                target_duration_s=0.0,
                gravimetric_efficiency=0.0,
                volumetric_efficiency=0.0,
                total_fuel_mass_kg=0.0,
                structure_mass_kg=0.0,
                total_inner_volume_m3=0.0,
                total_outer_volume_m3=0.0,
                error=f"Timed out after {self.case_timeout_s:.1f} s",
            )

        if not queue.empty():
            return queue.get()

        return SweepResult(
            design=design,
            mission_completed=False,
            mission_completion_ratio=0.0,
            mission_duration_s=0.0,
            target_duration_s=0.0,
            gravimetric_efficiency=0.0,
            volumetric_efficiency=0.0,
            total_fuel_mass_kg=0.0,
            structure_mass_kg=0.0,
            total_inner_volume_m3=0.0,
            total_outer_volume_m3=0.0,
            error="Case process exited without returning a result",
        )

    def _stdout_context(self):
        if not self.silent:
            return nullcontext()
        return open(os.devnull, "w")

    def _format_case_timeout_constraint(self) -> str:
        if self.case_timeout_s is None or self.case_timeout_s <= 0.0:
            return "disabled"
        return f"{self.case_timeout_s:.1f} s"

    def evaluate_design(self, design: DesignT) -> SweepResult:
        raise NotImplementedError

    def format_design_vector(self, design: DesignT) -> str:
        raise NotImplementedError


def _evaluate_design_worker(queue, study_cls, sweep_config_path: str, design) -> None:
    study, _ = study_cls.from_sweep_config(sweep_config_path)
    result = study.evaluate_design(design)
    queue.put(result)