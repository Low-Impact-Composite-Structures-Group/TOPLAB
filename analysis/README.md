# Analysis Directory

The `analysis/` directory contains runnable simulation cases built on top of the shared orchestration, mission, and solver stack in `src/`.

## Purpose

An analysis case answers a specific physics or system question by defining:

- a local driver entrypoint
- one or more YAML scenario files
- a local output location for plots and reports

The analysis folders should stay thin. They define the case being run, while the reusable machinery remains in `src/`.

## Expected Pattern

Each analysis study should generally follow this structure:

- `driver_*.py`: the local runnable entrypoint
- `*.yaml`: the scenario configuration(s)
- `output/`: generated plots, reports, and result files

Examples already following this pattern include the coupled, single-tank, pressure-fed, and DSE cases in this directory.

## How Analyses Fit The Repo

- `analysis/` defines the runnable case
- `src/orchestration/` governs case setup and execution
- `src/system/` contains the multi-tank simulation engine
- `src/coupling/`, `src/dynamics/`, `src/thermodynamics/`, and related modules contain the physical models

This split is intentional: new analyses should reuse the existing runtime rather than duplicating solver logic locally.

## Adding A New Analysis

When adding a new analysis case:

1. Create a dedicated subdirectory under `analysis/`.
2. Add a thin driver script that calls the shared orchestration path.
3. Keep case definition in YAML rather than in Python control flow.
4. Write outputs into an analysis-local `output/` directory.

The driver should primarily identify the config to run, not implement custom simulation behavior.

## Existing Case Notes

- `analysis/DSE/` already has its own case-local README for design-study-specific notes.
- `analysis/verification/` contains benchmark/reference data rather than a standard runnable analysis case.