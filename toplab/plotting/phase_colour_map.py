"""
Standalone script to generate a hydrogen density–temperature phase colour map
using the DelftColourPlotter utilities and CoolProp saturation data. It is not used as a module in the main project.

Configure values below (no CLI args needed). Set SAVE=True to write to disk; otherwise, the plot will be shown interactively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent


import matplotlib
matplotlib.use("pgf")  # Use PGF backend for LaTeX compatibility
import matplotlib.pyplot as plt

from toplab.plotting.multi_tank_plotting import DelftColourPlotter

# Configure matplotlib for PGF output
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{graphicx}",
})


# ====== Configurable values ======
# Appearance
GREYSCALE: bool = False
DPI: int = 300  # Lower DPI for PGF (vector format)

# Save options
SAVE: bool = True
OUTDIR: str = "output/"
FILENAME: Optional[str] = None  # e.g., "phase_map_custom.pgf" (if None, a default is used)
FORMAT: str = "pgf"  # one of: "pgf", "png", "pdf", "svg"

# Axes and resolution
TMIN: float = 15.0
TMAX: float = 300.0
RHOMIN: float = 0.0
RHOMAX: float = 90.0
NX: int = 1000  # Reduced resolution for smaller PGF file
NY: int = 1000  # Reduced resolution for smaller PGF file
# ====== End config ======

# Legend configuration
LEGEND_LOC: str = 'upper right'
LEGEND_NCOLS: int = 2
CALLOUT_FONT_SIZE: float = 17.0
ISOBAR_LABEL_FONT_SIZE: float = 12.0

# Isobar and marker size configuration
ISOBAR_PRESSURES: list[float] = [13, 100, 200, 400, 700]  # bar
CRITICAL_MARKER_SIZE: float = 70.0
DEFAULT_MARKER_SIZE: float = 150.0
# Marker definitions (labels support mathtext for H$_2$)
MARKERS = [
    {
        "T": 53.25,
        "rho": 78.0,
        "marker": "s",
        "size": DEFAULT_MARKER_SIZE,
        "callout": r"CcH$_2$ (40-60 K, 250-350 bar)",
        "callout_offset": (10, 18),
        "callout_ha": "left",
    },
    {
        "T": 28.2,
        "rho": 62.07,
        "marker": "s",
        "size": DEFAULT_MARKER_SIZE,
        "callout": r"sLH$_2$ (25-30 K, 5-10 bar)",
        "callout_offset": (12, 0),
        "callout_ha": "left",
    },
    {
        "T": 288.0,
        "rho": 40.0,
        "marker": "s",
        "size": DEFAULT_MARKER_SIZE,
        "callout": r"CH$_2$ (300 K, 350-700 bar)",
        "callout_offset": (-12, 12),
        "callout_ha": "right",
    },
    {
        "T": 20.7,
        "rho": 70.0,
        "marker": "s",
        "size": DEFAULT_MARKER_SIZE,
        "callout": r"LH$_2$ (20 K, 1-5 bar)",
        "callout_offset": (14, 20),
        "callout_ha": "left",
    },
]


def _compute_save_path(use_greyscale: bool) -> Optional[Path]:
    """Compute save path from the config variables above; create directories as needed.

    Defaults to saving in the same directory where this script lives when paths are relative.
    Absolute paths (either FILENAME or OUTDIR) are respected.
    """
    if not SAVE:
        return None

    # If FILENAME is an absolute path, use it directly
    if FILENAME:
        file_path = Path(FILENAME)
        if file_path.is_absolute():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure extension matches requested format
            if file_path.suffix.lower() not in (f".{FORMAT}", ".png", ".pdf", ".svg"):
                file_path = file_path.with_suffix(f".{FORMAT}")
            return file_path

    # Otherwise, resolve OUTDIR (absolute respected; relative -> script directory)
    outdir = Path(OUTDIR)
    if not outdir.is_absolute():
        outdir = SCRIPT_DIR / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Default filename if not provided
    default_name = f"phase_map_{'greyscale' if use_greyscale else 'colour'}.{FORMAT}"
    filename = FILENAME or default_name
    # Ensure extension matches requested format if user omitted it
    if Path(filename).suffix.lower() not in (f".{FORMAT}", ".png", ".pdf", ".svg", ".pgf"):
        filename = f"{filename}.{FORMAT}"
    return outdir / filename


def main():
    plotter = DelftColourPlotter(
        analysis_name="Hydrogen Phase Map",
        use_greyscale=GREYSCALE,
        enable_multi_tank_overlay=False,
    )

    save_path = _compute_save_path(GREYSCALE)

    fig = plotter.plot_phase_colour_map(
        temperature_range=(TMIN, TMAX),
        density_range=(RHOMIN, RHOMAX),
        resolution=(NX, NY),
        save_path=str(save_path) if save_path else None,
        dpi=DPI,
        marker_points=MARKERS,
        legend_location=LEGEND_LOC,
        legend_ncols=LEGEND_NCOLS,
        isobar_pressures_bar=ISOBAR_PRESSURES,
        critical_marker_size=CRITICAL_MARKER_SIZE,
        default_marker_size=DEFAULT_MARKER_SIZE,
        isobar_label_font_size=ISOBAR_LABEL_FONT_SIZE,
        callout_font_size=CALLOUT_FONT_SIZE,
    )

    if save_path is None:
        plt.show()
    else:
        print(f"PGF file saved to: {save_path}")
        print(f"Include in LaTeX with: \\input{{{save_path}}}")


if __name__ == "__main__":
    main()
