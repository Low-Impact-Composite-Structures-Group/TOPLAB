#!/usr/bin/env python3
"""Pressure buffer sensitivity-step optimization driver."""

import sys
from pathlib import Path


current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

try:
    from optimization.pressure_buffer_opt.pressure_buffer_sensitivity import main
except ModuleNotFoundError:
    from pressure_buffer_sensitivity import main


if __name__ == "__main__":
    main()
