# hydrogen_fuel_tank
This package enables the analysis of hydrogen fuel tank, providing insight into the thermo-mechanical loading during filling and draining of the tank.

## Workstreams (and attribution)
This repository currently contains two related workstreams that share a common Python environment but solve different problems:

- **Original hydrogen fuel tank analysis workflow (Victor Kees Poorte)**: the original analysis + example modules, focused on single-tank fill/drain mission simulations and parameter studies. Some multi-tank examples exist here too but do not support multistate modelling.
- **Multistate extension (Dante Raso)**: newer functionality for orchestrating coupled systems (e.g., multiple tanks / multistate systems), with its own drivers and test suite.

Both workstreams can coexist in the same repo and environment, but you may run them in slightly different ways depending on whether you are running a single analysis module through the shared runner, or running a dedicated multistate driver.

## Copyright and licensing
Copyright (c) 2022–2026 Victor Kees Poorte and Dante Raso.

This repository includes both **source code** and **data/outputs**, so licensing is split:

- **Code license**: Apache License 2.0 — see [LICENSE/Apache-2.0.txt](LICENSE/Apache-2.0.txt)
- **Data and outputs license**: Creative Commons Attribution 4.0 (CC BY 4.0) — see [LICENSE/CC-BY-4.0.txt](LICENSE/CC-BY-4.0.txt)

If a file or subdirectory specifies a different license, that license takes precedence.

## Dependencies
Most failures to “run” come from using an interpreter that does not have the required scientific dependencies installed.

This project’s development environment requires:

- Python
- NumPy, SciPy, Pandas
- Matplotlib
- PyYAML
- CoolProp

For development/testing, you will also want:

- Pytest

## Running Analyses

Notes:

- Many configs in this repo are named `main.YAML` (uppercase). On case-sensitive file systems, `main.yaml` will not be found.
- Each analysis module has at least a `perform_analysis(config)` function. Some modules also provide optional `extract_data(...)` and `plot_results(...)` hooks that the runner will call.

### Single-tank workflow (Victor)
This is the standard “module + YAML” pattern using the shared runner (`main.py`). Most modules live under `analysis/` and `examples/`.

Analysis and example files are run using Python through the command line as follows:

```
python main.py path.to.main path/to/main.YAML
```

Where the first argument is a Python import path (module), and the second is the path to the YAML config file for the analysis.

Examples:

```
python main.py analysis.compare_dynamic_models examples/compare_dynamic_models/main.YAML
python main.py examples.compare_dynamic_models.main examples/compare_dynamic_models/main.YAML
```

Shorthand is also supported for common cases:

```
python main.py compare_dynamic_models examples/compare_dynamic_models/main.YAML
```

The runner may prompt to reuse/recompute pickles for reproducibility.

The deprecated files are run through `main_deprecated.py`. Here the desired module and associated `perform_analysis()` function can be imported and run by calling `perform_analysis()`.

### Multistate workflow (Dante)
The multistate/orchestrated framework includes dedicated scripts and tests under `analysis/multistate_systems/` and `test/multistate_tests/`.

Each analysis directory typically contains a YAML config file and a Python driver file. To run an analysis, navigate to the corresponding sub-directory and run the driver:

```
python driver_<name_of_analysis>.py
```

For example, to run the coupled CcH2-CH2 analysis, navigate to `analysis/multistate_systems/coupled_cch2_ch2/` and run:

```
python driver_coupled_cch2_ch2.py
```

To run only the multistate test suite, use the following:

```
python -m pytest test/multistate_tests -q
```
