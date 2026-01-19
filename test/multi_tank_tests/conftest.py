from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def _ensure_import_paths(repo_root: Path) -> None:
    # Ensure `from src...` imports work when pytest is run from repo root.
    sys.path.insert(0, str(repo_root))


@pytest.fixture(scope="session")
def main_analysis_configs(repo_root: Path) -> dict[str, Path]:
    """The five main analyses under analysis/multi_tank_systems (excluding single_tank_lh2)."""
    return {
        "single_tank_ch2": repo_root
        / "analysis"
        / "multi_tank_systems"
        / "single_tank_ch2"
        / "single_tank_ch2_config.yaml",
        "single_tank_slh2": repo_root
        / "analysis"
        / "multi_tank_systems"
        / "single_tank_slh2"
        / "single_tank_slh2_config.yaml",
        "single_tank_cch2": repo_root
        / "analysis"
        / "multi_tank_systems"
        / "single_tank_cch2"
        / "single_tank_cch2_config.yaml",
        "coupled_ch2_cch2": repo_root
        / "analysis"
        / "multi_tank_systems"
        / "coupled_ch2_cch2"
        / "coupled_ch2_cch2_config.yaml",
        "coupled_ch2_lh2": repo_root
        / "analysis"
        / "multi_tank_systems"
        / "coupled_ch2_lh2"
        / "coupled_ch2_lh2_config.yaml",
    }


@pytest.fixture(scope="session")
def coupled_configs(main_analysis_configs: dict[str, Path]) -> dict[str, Path]:
    return {k: v for k, v in main_analysis_configs.items() if k.startswith("coupled_")}
