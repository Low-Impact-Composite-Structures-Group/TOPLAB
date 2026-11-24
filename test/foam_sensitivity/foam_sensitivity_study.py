"""Compatibility entry point for the foam sensitivity study."""

from pathlib import Path

from toplab.thermodynamics import foam_sensitivity
from toplab.thermodynamics.foam_sensitivity import *  # noqa: F403


if __name__ == "__main__":
    foam_sensitivity._OUTDIR = str(Path(__file__).resolve().parent / "output")
    foam_sensitivity.main()
