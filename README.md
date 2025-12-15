# hydrogen_fuel_tank
This package enables the analysis of hydrogen fuel tank, providing insight into the thermo-mechanical loading during filling and draining of the tank.

## Multi-Tank Systems
- Analyses live under `analysis/multi_tank_systems/` (e.g., `coupled_ch2_lh2`, `single_tank_*`).
- Core modules are under `src/multi_tank/` (dynamics, fluids, materials, thermodynamics, plotting).
- Plot styling is unified in `src/multi_tank/plotting/plot_style_sb.py` (seaborn/matplotlib, optional PGF output).
- Verification reference data (Stops et al., Cryogenics 2024) lives in `analysis/multi_tank_systems/verification/`.

## Quick Start
Run the CH2→LH2 feedforward scenario (short ~10 min) using the micromamba environment:

```bash
# From repo root
micromamba run -n python-dev env \
	PYTHONPATH="$PWD" \
	H2_DEBUG=1 \
	python analysis/multi_tank_systems/coupled_ch2_lh2/driver_coupled_ch2_lh2.py
```

Outputs are written to:
- Plots: `analysis/multi_tank_systems/coupled_ch2_lh2/output/plots/`
- Results: `analysis/multi_tank_systems/coupled_ch2_lh2/output/results/`

Tip: In VS Code, you can also run the task "Run feedforward scenario (short)" after activating the `python-dev` environment.

## Notes
- PGF/LaTeX export is enabled; if custom fonts are not found, the plotter falls back to system fonts and still writes `.png`/`.pgf`.
- The verification folder contains reference CSVs and a README with the DOI for Stops et al. (2024).
