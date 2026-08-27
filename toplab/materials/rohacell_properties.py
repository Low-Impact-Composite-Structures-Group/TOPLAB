"""
Rohacell foam: temperature-dependent thermal conductivity.

Conductivity data digitized from measurements (density = 51.1 kg/m³).
Temperature range: 20–325 K.
"""
import json
import numpy as np
from pathlib import Path

DENSITY = 51.1  # kg/m³

_T_arr: np.ndarray | None = None
_k_arr: np.ndarray | None = None


def _load() -> tuple[np.ndarray, np.ndarray]:
    global _T_arr, _k_arr
    if _T_arr is None:
        path = Path(__file__).parent / "rohacell.json"
        with open(path) as f:
            raw = json.load(f)
        T = np.array(raw["temperature"], dtype=float)
        k = np.array(raw["thermal_conductivity"], dtype=float)
        order = np.argsort(T)
        T, k = T[order], k[order]
        # remove exact duplicates to keep interp monotonic
        mask = np.concatenate(([True], np.diff(T) > 0))
        _T_arr, _k_arr = T[mask], k[mask]
    return _T_arr, _k_arr


def thermal_conductivity(temperature: float) -> float:
    """Return Rohacell 51A thermal conductivity [W/m·K] at *temperature* [K]."""
    T_arr, k_arr = _load()
    T_clamped = float(np.clip(temperature, T_arr[0], T_arr[-1]))
    return float(np.interp(T_clamped, T_arr, k_arr))
