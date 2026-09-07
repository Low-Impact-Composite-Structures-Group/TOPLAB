"""
Rohacell foam: temperature-dependent thermal properties.

Thermal conductivity: digitized from measurements, Rohacell 51A (density = 51.1 kg/m³),
temperature range 20–325 K.

Specific heat: Rohacell 31 data (rohacell31_specific_heat.csv), temperature range 20–330 K.
Rohacell 31 cp is used as the best available approximation for Rohacell 51A.
"""
import csv
import json
import numpy as np
from pathlib import Path

DENSITY = 51.1  # kg/m³

_T_arr: np.ndarray | None = None
_k_arr: np.ndarray | None = None

_T_cp_arr: np.ndarray | None = None
_cp_arr: np.ndarray | None = None


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


def _load_cp() -> tuple[np.ndarray, np.ndarray]:
    global _T_cp_arr, _cp_arr
    if _T_cp_arr is None:
        path = Path(__file__).parent / "rohacell31_specific_heat.csv"
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((float(row["temperature_K"]), float(row["specific_heat_J_kgK"])))
        T = np.array([r[0] for r in rows])
        cp = np.array([r[1] for r in rows])
        order = np.argsort(T)
        _T_cp_arr, _cp_arr = T[order], cp[order]
    return _T_cp_arr, _cp_arr


def thermal_conductivity(temperature: float) -> float:
    """Return Rohacell 51A thermal conductivity [W/m·K] at *temperature* [K]."""
    T_arr, k_arr = _load()
    T_clamped = float(np.clip(temperature, T_arr[0], T_arr[-1]))
    return float(np.interp(T_clamped, T_arr, k_arr))


def integrated_thermal_conductivity(temperature_low: float, temperature_high: float) -> float:
    """Integrate Rohacell 51A conductivity [W/m] over a temperature interval."""
    if temperature_low == temperature_high:
        return 0.0

    sign = 1.0 if temperature_high > temperature_low else -1.0
    T_arr, k_arr = _load()
    lower = float(np.clip(min(temperature_low, temperature_high), T_arr[0], T_arr[-1]))
    upper = float(np.clip(max(temperature_low, temperature_high), T_arr[0], T_arr[-1]))
    temperatures = np.concatenate((
        [lower],
        T_arr[(T_arr > lower) & (T_arr < upper)],
        [upper],
    ))
    conductivities = np.interp(temperatures, T_arr, k_arr)
    return sign * float(np.trapz(conductivities, temperatures))


def specific_heat(temperature: float) -> float:
    """Return Rohacell 31 specific heat [J/kg·K] at *temperature* [K] (clamped at data bounds)."""
    T_arr, cp_arr = _load_cp()
    T_clamped = float(np.clip(temperature, T_arr[0], T_arr[-1]))
    return float(np.interp(T_clamped, T_arr, cp_arr))
