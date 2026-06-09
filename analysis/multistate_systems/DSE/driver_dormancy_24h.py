from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_root_to_syspath() -> Path:
    """Allow running this driver from any working directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing 'src/'")


_REPO_ROOT = _add_repo_root_to_syspath()

from src.multistate.orchestration.run_analysis import run_analysis


def main() -> None:
    config_path = Path(__file__).with_name("dormancy_24h.yaml")
    run_analysis(config_path)


if __name__ == "__main__":
    main()
