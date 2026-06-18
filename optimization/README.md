# Optimization Directory

The `optimization/` directory contains runnable design studies and optimization-style entrypoints.

At the moment this includes sweep-based studies, but the directory is intended to grow into the home for broader design-space exploration and optimization workflows.

## Purpose

An optimization study defines:

- the system case being varied
- the design variables being explored
- the study-level constraints and ranking logic
- the report/output location for the study results

As with `analysis/`, the study folders should remain thin and study-specific.

## Structure

The optimization layer is split between:

- `optimization/<study>/`: study-local YAML, driver, and outputs
- `src/optimization/`: shared runtime logic for sweep / optimization execution

This means the study folder owns the definition of a design study, while `src/optimization/` owns the common execution machinery.

## Current Pattern

Each study is expected to follow a pattern like:

- `driver_*.py`: thin runnable entrypoint
- `*_sweep.py` or study module: study-specific design vector, config mutation, and metric extraction
- `*.yaml`: study definition and runtime options
- `output/`: generated reports and results

The pressure-buffer example under `optimization/presure_buffer_opt/` is the reference case for this pattern.

## Shared Runtime Responsibilities

The shared logic in `src/optimization/` is the right place for concerns such as:

- progress handling
- per-case timeout control
- silent execution behavior
- reusable sweep or optimization orchestration
- common runtime configuration parsing

Study-local files should only implement what is unique to that study.

## Adding A New Study

When adding a new optimization study:

1. Create a dedicated subdirectory under `optimization/`.
2. Add a study-local YAML definition file.
3. Add a thin driver entrypoint.
4. Implement only the study-specific design vector, config mutation, and metric extraction locally.
5. Reuse the common machinery from `src/optimization/` for execution flow.

That keeps new studies consistent with the analysis-layer structure while avoiding duplicated runtime logic.