# Thermomechanical OPtimization LAB (TOPLAB)

TOPLAB is a modular analysis tool for the design of composite hydrogen storage tanks for aviation, capable of capturing interactions between tanks and other fuel system components.

This repository hosts the official implementation accompanying the paper:

> *"A thermomechanical model of on-board multi-state hydrogen fuel systems in aviation"*
> Proceedings of the 2nd Vienna Aviation Days 2025.

## Repository Scope

TOPLAB provides a configuration-driven workflow for running hydrogen fuel system analyses and studying thermomechanical behavior under mission-relevant conditions.

## Environment Setup

The repository includes a micromamba / conda environment definition at `hython.yaml`.

Create the environment from the repository root with:

```bash
micromamba env create -f hython.yaml
micromamba activate hython
```

If the environment already exists and you want to sync it to the current dependency set:

```bash
micromamba env update -f hython.yaml --prune
micromamba activate hython
```

Equivalent conda commands also work:

```bash
conda env create -f hython.yaml
conda activate hython
```

After activation, a quick sanity check is:

```bash
python -c "import yaml, seaborn, CoolProp, numpy, scipy, matplotlib, pytest"
```

For workflows that generate PGF figures or build the LaTeX documentation, the same environment also carries the required TeX tooling (`pdflatex` via `texlive-core` and `latexmk`). You can verify that with:

```bash
which pdflatex
which latexmk
```

The audited environment definition includes runtime, plotting, test, and documentation dependencies used across the repository, including plotting support (`seaborn`), YAML parsing (`pyyaml`), and LaTeX tooling for PGF/document builds.

## General Spirit of the Code

TOPLAB is designed around a few core ideas:

- Physics-first, architecture-second: governing equations and thermodynamic consistency are treated as primary design constraints, and software structure is built to preserve them.
- Graph-based system modelling: fuel systems are represented as directed networks of nodes and edges, so topology changes are handled in configuration rather than by rewriting solver code.
- Modular components: tanks, couplings, valves, peripheral conditioning elements, and thermal models are implemented as composable building blocks.
- Configuration-driven studies: scenarios are expressed in YAML and executed through common orchestration paths, enabling reproducible comparisons across missions and architectures.
- Extensibility with control: new mechanisms (for example, edge models or component chains) are intended to be added as explicit modules, not ad hoc special cases.

In practice, this means the codebase aims to make it easy to:

1. Encode a new multi-tank architecture from configuration.
2. Run the same numerical machinery across different mission profiles.
3. Compare physical assumptions and design choices without duplicating solver logic.

## Documentation

The repository-level technical documentation now lives in:

- `documentation/`

This folder contains the model and solution description in LaTeX form, including:

- network graph representation
- governing equations (mass/energy/solid thermal state)
- operating configurations and switching behavior
- numerical method and integration strategy
- legacy vs graph-based paradigm comparison

Start from `documentation/main.tex`.

To build the documentation PDF after activating the environment:

```bash
cd documentation
make
```

## Analyses

The runnable simulation cases live under `analysis/`.

Each analysis case is intended to follow a common pattern:

- one local driver script
- one or more YAML scenario files
- one local `output/` directory for generated plots and reports

These cases are thin entrypoints into the shared orchestration and solver stack under `src/`.

For the analysis-layer conventions and workflow, see:

- `analysis/README.md`

## Optimization Studies

The runnable design studies live under `optimization/`.

This layer mirrors the spirit of `analysis/`, but separates study-specific entrypoints from shared optimization machinery:

- `optimization/<study>/` contains the study-local driver, YAML study definition, and outputs
- `src/optimization/` contains the reusable sweep / optimization runtime

For the optimization-layer conventions and workflow, see:

- `optimization/README.md`

## Contact

For questions related to the paper or early access inquiries:

- d.raso@tudelft.nl

---

## Citation

If you use this work, please cite:

```bibtex
@article{raso2026,
	title     = {A thermomechanical model of on-board multi-state hydrogen fuel systems in aviation},
	author    = {Dante Raso and Nils Wieja and Jan Conde-Wolter and Lukas Hauser and Christoph Ebert and Bianca Giovanardi and Maik Gude and Julien van Campen},
	journal   = {Proceedings of the 2nd Vienna Aviation Days},
	publisher = {Springer Nature},
	year      = {2026}
}
```
