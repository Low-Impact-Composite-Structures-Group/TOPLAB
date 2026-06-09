# Hydrogen Fuel Tank: Student Quickstart

This repository models cryogenic hydrogen tank behavior for mission-oriented studies.
Core simulation logic lives in `src/`, while study-specific entry points and YAML configurations live in `analysis/`.

## Important Disclaimer

- This project is still under construction.
- You do not have permission to share this repository, its data, or outputs outside the authorized group.
- For your own development, create a private fork and work there.

## Environment Setup

Use the provided Conda/Mamba environment file:

```bash
micromamba env create -f hython.yaml
micromamba activate hython
```

If the environment already exists:

```bash
micromamba activate hython
```

## Source Code Context (Very Brief)

- `src/`: reusable physics, thermodynamics, fluid models, and multistate orchestration.
- `analysis/`: runnable analyses (drivers + YAML scenarios).
- `analysis/multistate_systems/DSE/`: Design Study Environment (DSE) cases, including dormancy and discharge.
- `output/` (and analysis-local `output/` folders): generated results and plots.

## The example analysis you can base your work off

You can find the DSE folder here:

```bash
cd analysis/multistate_systems/DSE
```

### 1) Discharge Analysis

- Purpose: simulate tank depletion under a commanded mission outflow profile.
- Driver: `driver_discharge.py`
- Config: `discharge.yaml`

Run:

```bash
python driver_discharge.py
```

### 2) Dormancy (24h) Analysis

- Purpose: simulate tank behavior during idle storage (no commanded fuel demand), including pressure/thermal evolution and vent behavior.
- Driver: `driver_dormancy_24h.py`
- Config: `dormancy_24h.yaml`

Run:

```bash
python driver_dormancy_24h.py
```

## Outputs

Both analyses write outputs under:

- `analysis/multistate_systems/DSE/output/results/`
- `analysis/multistate_systems/DSE/output/plots/`

## Notes for Students

- Start from the existing YAML files and modify only one parameter group at a time.
- Keep your own branch/fork for experiments and documentation.
- If results look non-physical, first confirm you are using the `hython` environment.
