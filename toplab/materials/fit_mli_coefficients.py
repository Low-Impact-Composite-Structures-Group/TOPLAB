"""
Standalone non-negative least-squares fit of the MLI surrogate coefficients A, B.

Model (bulk equivalent conductivity k_eq(T) = A*T + B*T**3):

    Q = G * [ A*(T_hot**2 - T_cold**2)/2 + B*(T_hot**4 - T_cold**4)/4 ]

with G the full-layer geometry factor (same expression as the thermal model):

    G = 2*pi*L / ln(r_out/r_in) + 4*pi*r_in*r_out / (r_out - r_in)

A, B >= 0 keep k_eq positive and the integral monotonic in T_hot and T_cold.
Run once, then copy the printed values into mli_properties.py (DEFAULT_A/B)
or the YAML insulation block. Not imported by the simulation.

Usage:
    python fit_mli_coefficients.py                 # built-in data points
    python fit_mli_coefficients.py data.csv        # CSV with the columns in COLUMNS
    python fit_mli_coefficients.py data.csv --relative
"""
import argparse
import csv
import math
import sys

import numpy as np
from scipy.optimize import nnls

COLUMNS = ("description", "Q_W", "T_hot_K", "T_cold_K", "L_m", "r_in_m", "r_out_m")

# (description, Q [W], T_hot [K], T_cold [K], L [m], r_in [m], r_out [m])
DATA = [
    ("CcH2 medium", 25.0, 300.0, 55.0, 1.0, 0.50, 0.52),
    ("LH2 medium", 5.0, 300.0, 25.0, 1.0, 0.50, 0.53),
    ("row 3", 25.0, 300.0, 55.0, 2.0, 0.75, 0.77),
]


def geometry_factor(L: float, r_in: float, r_out: float) -> float:
    return 2.0 * math.pi * L / math.log(r_out / r_in) + 4.0 * math.pi * r_in * r_out / (r_out - r_in)


def load_csv(path: str) -> list[tuple]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((
                row.get("description", ""),
                *(float(row[c]) for c in COLUMNS[1:]),
            ))
    return rows


def fit(data: list[tuple], relative: bool = False) -> dict:
    Q = np.array([d[1] for d in data])
    T_hot = np.array([d[2] for d in data])
    T_cold = np.array([d[3] for d in data])
    G = np.array([geometry_factor(*d[4:7]) for d in data])

    X = G[:, None] * np.column_stack((
        (T_hot ** 2 - T_cold ** 2) / 2.0,
        (T_hot ** 4 - T_cold ** 4) / 4.0,
    ))
    # Relative weighting stops the largest-Q cases dominating the fit.
    w = 1.0 / Q if relative else np.ones_like(Q)
    coeffs, _ = nnls(X * w[:, None], Q * w)
    pred = X @ coeffs
    return {
        "A": coeffs[0],
        "B": coeffs[1],
        "Q_pred": pred,
        "Q": Q,
        "G": G,
        "sse": float(np.sum((pred - Q) ** 2)),
        "n_T_hot": len(set(T_hot)),
        "n_T_cold": len(set(T_cold)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", help="CSV with columns: " + ", ".join(COLUMNS))
    parser.add_argument("--relative", action="store_true", help="minimise relative instead of absolute residuals")
    args = parser.parse_args()

    data = load_csv(args.csv) if args.csv else DATA
    res = fit(data, relative=args.relative)

    print(f"{'case':<16}{'T_hot':>8}{'T_cold':>8}{'G':>9}{'Q_data':>9}{'Q_fit':>9}{'err %':>8}")
    for d, g, q, qp in zip(data, res["G"], res["Q"], res["Q_pred"]):
        print(f"{d[0]:<16}{d[2]:>8.1f}{d[3]:>8.1f}{g:>9.2f}{q:>9.3f}{qp:>9.3f}{100 * (qp - q) / q:>8.1f}")

    print(f"\nA = {res['A']:.6e}  [W/m/K^2]")
    print(f"B = {res['B']:.6e}  [W/m/K^4]")
    print(f"sum of squared residuals = {res['sse']:.4g}")
    if res["A"] == 0.0 or res["B"] == 0.0:
        print("NOTE: a coefficient hit the non-negativity bound (0); the data do not support that term.")
    if res["n_T_hot"] < 2:
        print("WARNING: all points share one T_hot; the fit cannot be trusted at other ambient temperatures.")
    if len(data) < 4:
        print("WARNING: very few data points for two coefficients.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
