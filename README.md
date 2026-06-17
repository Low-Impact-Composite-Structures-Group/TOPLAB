# Thermomechanical OPtimization LAB (TOPLAB)

TOPLAB is a modular analysis tool for the design of composite hydrogen storage tanks for aviation, capable of capturing interactions between tanks and other fuel system components.

This repository hosts the official implementation accompanying the paper:

> *"A thermomechanical model of on-board multi-state hydrogen fuel systems in aviation"*
> Proceedings of the 2nd Vienna Aviation Days 2025.

## Repository Scope

TOPLAB provides a configuration-driven workflow for running hydrogen fuel system analyses and studying thermomechanical behavior under mission-relevant conditions.

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
