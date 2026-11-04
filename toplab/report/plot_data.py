"""CSV export of the data behind each generated figure.

The data is read back from the matplotlib artists of the finished figure, so a
CSV always matches exactly what is drawn, for every plot type, without a second
data-extraction path that could drift from the plotting code.

Format (long / tidy), one row per plotted point:
    axes, series, kind, x_label, y_label, x, y
``kind`` is ``line``, ``scatter``, ``point`` (single marker) or
``reference_line`` (constant-valued 2-point line such as P_min or an event marker).
"""

from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

COLUMNS = ["axes", "series", "kind", "x_label", "y_label", "x", "y"]


def plot_data_enabled(run, plot_key: str) -> bool:
    """Per-plot ``save_data`` overrides the global ``output.save_data`` switch."""
    output = run.config.output or {}
    if str(output.get("data_format", "csv")).lower() != "csv":
        return False
    plot_cfg = run.config.output_plots.get(plot_key, {}) or {}
    return bool(plot_cfg.get("save_data", output.get("save_data", False)))


def _series_name(artist, fallback: str) -> str:
    label = artist.get_label() or ""
    return fallback if (not label or label.startswith("_")) else label


def _axes_name(ax, index: int) -> str:
    return ax.get_ylabel() or ax.get_title() or f"axes_{index}"


def figure_to_dataframe(fig) -> pd.DataFrame:
    """Collect every plotted line and scatter series of ``fig`` into a tidy frame."""
    rows: List[pd.DataFrame] = []
    for ax_idx, ax in enumerate(fig.axes):
        axes_name = _axes_name(ax, ax_idx)
        x_label, y_label = ax.get_xlabel(), ax.get_ylabel()

        seen = {}

        def _add(name, kind, x, y):
            # repeated labels (e.g. one legend entry for several isobars) stay distinguishable
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name} #{seen[name]}"
            x = np.asarray(x, dtype=float).ravel()
            y = np.asarray(y, dtype=float).ravel()
            n = min(x.size, y.size)
            if n == 0:
                return
            rows.append(pd.DataFrame({
                "axes": axes_name, "series": name, "kind": kind,
                "x_label": x_label, "y_label": y_label,
                "x": x[:n], "y": y[:n],
            }))

        for i, line in enumerate(ax.get_lines()):
            x, y = line.get_xdata(orig=False), line.get_ydata(orig=False)
            try:
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
            except (TypeError, ValueError):
                continue  # categorical / non-numeric axis
            if x.size == 1:
                kind = "point"
            elif x.size == 2 and (np.ptp(x) == 0 or np.ptp(y) == 0):
                kind = "reference_line"
            else:
                kind = "line"
            _add(_series_name(line, f"line_{i}"), kind, x, y)

        for i, coll in enumerate(ax.collections):
            # pcolormesh / fills have no point offsets worth exporting
            if type(coll).__name__ != "PathCollection":
                continue
            offsets = np.asarray(coll.get_offsets(), dtype=float)
            if offsets.ndim == 2 and offsets.shape[1] == 2:
                _add(_series_name(coll, f"scatter_{i}"), "scatter",
                     offsets[:, 0], offsets[:, 1])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)


def _csv_path(run, figure_path: Optional[str], suffix: str = "") -> Optional[Path]:
    if not figure_path:
        return None
    results_dir = Path((run.config.output or {}).get("results_directory", "output/results"))
    return results_dir / f"{Path(figure_path).stem}{suffix}.csv"


def export_figures(run, plot_key: str, figures: Any, figure_path: Optional[str]) -> List[str]:
    """Write CSV(s) for ``figures`` when enabled for ``plot_key``.

    ``figures`` is a Figure or a dict ``{metric: Figure}`` (separate-figure mode),
    in which case each file gets ``_<metric>`` appended to the stem of
    ``figure_path`` (the base path without per-metric suffix).
    """
    if figures is None or not figure_path or not plot_data_enabled(run, plot_key):
        return []

    items: Iterable = (
        [(f"_{k}", f) for k, f in figures.items()]
        if isinstance(figures, dict) else [("", figures)]
    )
    written = []
    for suffix, fig in items:
        path = _csv_path(run, figure_path, suffix)
        df = figure_to_dataframe(fig)
        if df.empty:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        written.append(str(path))
        print(f"   Saved plot data to: {path}")
    return written
