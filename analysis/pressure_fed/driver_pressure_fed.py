#!/usr/bin/env python3
"""Driver for the pressure-fed LH2/GH2 multi-tank simulation.

Topology
--------
Tank 1 (LH2) -- 100% discharge --> fuel cell  (mission CSV)
           `--- 5% split ---------> HEX (293 K) --> Compressor (11 bar) --> Tank 2 (GH2 buffer)
Tank 2 (GH2) -- pressure valve (open at 11 bar, close at 10 bar) --> sink

Run
---
    micromamba run -n hython python analysis/pressure_fed/driver_pressure_fed.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.orchestration.run_analysis import run_analysis

# Ensure output directories exist (paths are relative to cwd when invoked)
for _d in ["output/plots", "output/results"]:
    Path(_d).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    config_path = Path(__file__).parent / "input.yaml"

    result = run_analysis(
        config_path=config_path,
        analysis_name="Pressure-Fed LH2/GH2 System",
        show_material_props=False,
        verbose=True,
    )

    if result.get("success"):
        print("\nPressure-fed simulation completed successfully.")
        exec_time = result.get("execution_time", 0.0)
        print(f"   Execution time: {exec_time:.1f} s")
    else:
        print(f"\nSimulation failed: {result.get('error', 'unknown error')}")
        sys.exit(1)
