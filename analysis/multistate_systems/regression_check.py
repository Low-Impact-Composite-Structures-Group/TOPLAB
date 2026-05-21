#!/usr/bin/env python3
"""
Regression check for the five multistate_systems analyses.

Usage
-----
  # First run: save results as the reference baseline
  python regression_check.py --save-baseline

  # Subsequent runs: compare against the saved baseline
  python regression_check.py

The script reports pass/fail for each metric using a relative tolerance of 1e-4
(0.01 %) — tight enough to catch meaningful changes, loose enough to ignore
floating-point non-determinism from adaptive step solvers.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# ── import paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# ── Analysis registry ─────────────────────────────────────────────────────────
ANALYSES = [
    {
        "name": "single_tank_ch2",
        "config": Path(__file__).parent / "single_tank_ch2" / "single_tank_ch2_config.yaml",
    },
    {
        "name": "single_tank_cch2",
        "config": Path(__file__).parent / "single_tank_cch2" / "single_tank_cch2_config.yaml",
    },
    {
        "name": "single_tank_slh2",
        "config": Path(__file__).parent / "single_tank_slh2" / "single_tank_slh2_config.yaml",
    },
    {
        "name": "coupled_ch2_cch2",
        "config": Path(__file__).parent / "coupled_ch2_cch2" / "coupled_ch2_cch2_config.yaml",
    },
    {
        "name": "coupled_ch2_lh2",
        "config": Path(__file__).parent / "coupled_ch2_lh2" / "coupled_ch2_lh2_config.yaml",
    },
]

BASELINE_CSV = Path(__file__).parent / "regression_baseline.csv"
REL_TOL = 1e-4   # 0.01 % relative tolerance for pass/fail
# Metrics excluded from regression comparison (environment-dependent, not physics):
SKIP_KEYS = {"execution_time_s"}


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract_metrics(result: dict) -> dict:
    """Pull scalar metrics from a run_analysis result dict."""
    metrics = {}

    if not result.get("success"):
        metrics["success"] = 0.0
        return metrics

    metrics["success"] = 1.0
    metrics["execution_time_s"] = float(result.get("execution_time", 0.0))

    sim_results = result.get("results")
    if sim_results is None:
        return metrics

    try:
        data = sim_results.get_combined_data()
    except Exception:
        return metrics

    times = data["times"]
    metrics["final_time_s"] = float(times[-1])

    n_tanks = data["masses"].shape[0]
    for i in range(n_tanks):
        prefix = f"tank{i+1}"
        masses = data["masses"][i]
        temps  = data["temperatures"][i]
        press  = data["pressures"][i]

        metrics[f"{prefix}_final_mass_kg"]     = float(masses[-1])
        metrics[f"{prefix}_final_temp_K"]      = float(temps[-1])
        metrics[f"{prefix}_final_pressure_Pa"] = float(press[-1])
        metrics[f"{prefix}_min_pressure_Pa"]   = float(np.min(press))
        metrics[f"{prefix}_max_pressure_Pa"]   = float(np.max(press))

    return metrics


# ── Run one analysis ──────────────────────────────────────────────────────────

def run_one(entry: dict) -> dict:
    from src.multi_tank.orchestration.run_analysis import run_analysis

    print(f"\n{'─'*60}")
    print(f"Running: {entry['name']}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = run_analysis(
            config_path=entry["config"],
            analysis_name=entry["name"],
            show_material_props=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        result = {"success": False, "error": str(exc)}
    result["execution_time"] = time.time() - t0
    return result


# ── Baseline I/O ──────────────────────────────────────────────────────────────

def save_baseline(rows: list[dict]):
    """Write all metric rows to the baseline CSV."""
    all_keys = ["analysis"] + sorted({k for r in rows for k in r if k != "analysis"})
    with open(BASELINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"\nBaseline saved → {BASELINE_CSV}")


def load_baseline() -> dict[str, dict]:
    """Load baseline CSV; returns {analysis_name: {metric: float}}."""
    if not BASELINE_CSV.exists():
        return {}
    baseline = {}
    with open(BASELINE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            name = row.pop("analysis")
            baseline[name] = {k: float(v) for k, v in row.items() if v != ""}
    return baseline


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(name: str, current: dict, reference: dict) -> bool:
    """Return True if all metrics pass within REL_TOL."""
    passed = True
    for key, ref_val in reference.items():
        if key in SKIP_KEYS:
            continue
        cur_val = current.get(key)
        if cur_val is None:
            print(f"  MISSING  {key}")
            passed = False
            continue
        if ref_val == 0.0:
            ok = abs(cur_val) < 1e-9
        else:
            ok = abs(cur_val - ref_val) / abs(ref_val) <= REL_TOL
        status = "OK   " if ok else "FAIL "
        if not ok:
            passed = False
        print(f"  {status} {key}: {cur_val:.6g}  (ref {ref_val:.6g})")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-baseline", action="store_true",
                        help="Run all analyses and save results as the new baseline.")
    parser.add_argument("--only", metavar="NAME", nargs="+",
                        help="Restrict to named analyses (e.g. single_tank_ch2).")
    args = parser.parse_args()

    analyses = ANALYSES
    if args.only:
        analyses = [a for a in ANALYSES if a["name"] in args.only]
        if not analyses:
            sys.exit(f"No matching analyses for: {args.only}")

    rows = []
    for entry in analyses:
        result  = run_one(entry)
        metrics = extract_metrics(result)
        metrics["analysis"] = entry["name"]
        rows.append(metrics)

    if args.save_baseline:
        save_baseline(rows)
        return

    # ── Compare mode ─────────────────────────────────────────────────────────
    baseline = load_baseline()
    if not baseline:
        sys.exit(
            f"No baseline found at {BASELINE_CSV}.\n"
            "Run with --save-baseline first."
        )

    overall_pass = True
    print(f"\n{'='*60}")
    print("REGRESSION RESULTS")
    print(f"{'='*60}")
    for row in rows:
        name = row["analysis"]
        ref  = baseline.get(name)
        print(f"\n{name}")
        if ref is None:
            print("  NOT IN BASELINE — skipping")
            continue
        ok = compare(name, row, ref)
        overall_pass = overall_pass and ok
        print(f"  → {'PASS' if ok else 'FAIL'}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print(f"{'='*60}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
