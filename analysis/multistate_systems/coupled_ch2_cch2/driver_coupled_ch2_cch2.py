#!/usr/bin/env python3
"""
Coupled CH2-CCH2 Multi-Tank Analysis Driver

Coupled gaseous and cryo-compressed hydrogen storage system with pressure
compensation coupling.

Author: Dante Raso
"""

import sys
from pathlib import Path

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent))

# Import common driver function
from src.multistate.orchestration.run_analysis import run_analysis


def main():
    """Execute coupled CH2-CCH2 multi-tank analysis."""
    config_path = current_dir / "coupled_ch2_cch2_config.yaml"
    result = run_analysis(
        config_path=config_path,
        analysis_name="Coupled CH2-CCH2 Multi-Tank System",
        show_material_props=False,
        verbose=False
    )
    return result


if __name__ == "__main__":
    # Make this module runnable directly via `python driver_coupled_ch2_cch2.py`
    try:
        main()
    except Exception:
        # Ensure failures are visible when run as a script
        raise


