#!/usr/bin/env python3
"""
Single Tank CCH2 Analysis Driver

Cryo-compressed hydrogen storage analysis using orchestrated multi-tank framework.

Author: Dante Raso
"""

import sys
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import common driver function
from src.multi_tank.orchestration.run_analysis import run_analysis


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
