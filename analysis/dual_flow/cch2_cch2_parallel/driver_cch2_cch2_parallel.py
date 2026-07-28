#!/usr/bin/env python3
"""
CCH2-CCH2 Parallel Dual Flow Analysis Driver

Two independent cryo-compressed hydrogen tanks operating in parallel:
  - Tank 1 → Fuel Cell (FC) demand from triathlon_dual_flow.csv
  - Tank 2 → Gas Turbine (GT) demand from triathlon_dual_flow.csv

Both tanks discharge simultaneously (synchronous integration) with no
inter-tank coupling.  Each tank sees only its own sink's mass-flow column.
"""

import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

from src.orchestration.run_analysis import run_analysis


def main():
    """Execute CCH2-CCH2 parallel dual-flow analysis."""
    config_path = current_dir / "cch2_cch2_parallel.yaml"
    result = run_analysis(
        config_path=config_path,
        analysis_name="CCH2-CCH2 Parallel Dual Flow",
        show_material_props=False,
        verbose=False,
    )
    return result


if __name__ == "__main__":
    main()
