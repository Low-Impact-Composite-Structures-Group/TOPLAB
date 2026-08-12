#!/usr/bin/env python3
"""
Single Tank CCH2 Analysis Driver

Cryo-compressed hydrogen storage analysis using orchestrated multi-tank framework.

Author: Dante Raso
"""

from pathlib import Path

current_dir = Path(__file__).parent

# Import common driver function
from toplab.orchestration.run_analysis import run_analysis


def main():
    """Execute single tank CCH2 analysis."""
    config_path = current_dir / "single_tank_cch2_config.yaml"
    result = run_analysis(
        config_path=config_path,
        analysis_name="Single Tank CCH2 Benchmark",
        show_material_props=False,
        verbose=False
    )
    return result


if __name__ == "__main__":
    main()
