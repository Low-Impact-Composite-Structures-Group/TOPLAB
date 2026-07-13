#!/usr/bin/env python3
"""Pressure buffer sweep driver."""

import sys
from pathlib import Path


current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

try:
    from optimization.pressure_buffer_sweep.pressure_buffer_sweep import main
except ModuleNotFoundError:
    from pressure_buffer_sweep import main


if __name__ == "__main__":
    main()