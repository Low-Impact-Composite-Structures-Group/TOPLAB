#!/usr/bin/env python3
"""CH2-CCH2 SQP optimisation driver — facade over src.optimization.sqp_optimizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.optimization.sqp_optimizer import SQPOptimizer


def main() -> None:
    cfg_path = Path(__file__).with_suffix(".yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Optimisation config not found: {cfg_path}")
    SQPOptimizer(cfg_path).run()


if __name__ == "__main__":
    main()
