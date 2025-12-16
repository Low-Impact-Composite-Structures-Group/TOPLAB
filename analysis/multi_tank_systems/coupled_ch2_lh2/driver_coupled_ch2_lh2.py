#!/usr/bin/env python3
"""
Coupled CH2-LH2 Multi-Tank Analysis Driver

Coupled gaseous and liquid hydrogen storage system with flow-controlled
pressurisation coupling.

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
    """Execute coupled CH2-LH2 multi-tank analysis."""
    config_path = current_dir / "coupled_ch2_lh2_config.yaml"
    result = run_analysis(
        config_path=config_path,
        analysis_name="Coupled CH2-LH2 Multi-Tank System",
        show_material_props=False,
        verbose=False
    )
    return result


