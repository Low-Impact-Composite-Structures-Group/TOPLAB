"""
Numerical regression harness for the refactor.

    python regression/regression.py capture [name ...]   # write baselines
    python regression/regression.py compare [name ...]   # run + diff vs baselines

Each example is simulated and reduced to per-tank trajectories (mass, pressure,
temperature) resampled on a normalised time grid, so results stay comparable if
the solver's step selection changes. `simulate` is the only function that
touches toplab; update it when the entry point changes.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
EXAMPLES = {
    "single_tank_ch2": ROOT / "examples/single_tank_ch2/single_tank_ch2_config.yaml",
    "single_tank_slh2": ROOT / "examples/single_tank_slh2/single_tank_slh2_config.yaml",
    "single_tank_cch2": ROOT / "examples/single_tank_cch2/single_tank_cch2_config.yaml",
    "coupled_ch2_cch2": ROOT / "examples/coupled_ch2_cch2/coupled_ch2_cch2_config.yaml",
    "optimization_two_tank_cch2": ROOT / "examples/optimization_two_tank_cch2/two_tank_cch2_config.yaml",
    "coupled_ch2_lh2": ROOT / "examples/coupled_ch2_lh2/coupled_ch2_lh2_config.yaml",
}
N_SAMPLES = 60
RTOL = 1e-3


def simulate(config_path: Path) -> dict:
    from toplab.config import load_config
    from toplab.orchestration.system_orchestrator import SystemOrchestrator

    orchestrator = SystemOrchestrator(load_config(config_path), verbosity="quiet")
    results = orchestrator.run_simulation()
    data = results.get_combined_data()
    t = np.asarray(data["times"], dtype=float)
    grid = np.linspace(t[0], t[-1], N_SAMPLES)
    out = {"t_end": float(t[-1]), "n_tanks": int(results.n_tanks)}
    for key in ("masses", "pressures", "h2_temperatures"):
        out[key] = [np.interp(grid, t, row).tolist() for row in data[key]]
    return out


def compare(base: dict, new: dict) -> list:
    if base["n_tanks"] != new["n_tanks"]:
        return [f"n_tanks {base['n_tanks']} -> {new['n_tanks']}"]
    problems = []
    if not np.isclose(base["t_end"], new["t_end"], rtol=RTOL):
        problems.append(f"t_end {base['t_end']:.3f} -> {new['t_end']:.3f}")
    for key in ("masses", "pressures", "h2_temperatures"):
        for i, (b, n) in enumerate(zip(base[key], new[key])):
            b, n = np.asarray(b), np.asarray(n)
            err = np.abs(b - n).max() / max(np.abs(b).max(), 1e-12)
            if err > RTOL:
                problems.append(f"{key}[tank {i + 1}] max rel err {err:.2e}")
    return problems


def main(argv):
    mode = argv[1] if len(argv) > 1 else "compare"
    names = argv[2:] or list(EXAMPLES)
    BASELINE_DIR.mkdir(exist_ok=True)
    failed = False
    for name in names:
        start = time.time()
        try:
            new = simulate(EXAMPLES[name])
        except Exception as exc:
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            failed = True
            continue
        path = BASELINE_DIR / f"{name}.json"
        if mode == "capture":
            path.write_text(json.dumps(new))
            print(f"[captured] {name} ({time.time() - start:.0f}s) t_end={new['t_end']:.1f}s")
        else:
            problems = compare(json.loads(path.read_text()), new)
            print(f"[{'DIFF' if problems else 'OK'}] {name} ({time.time() - start:.0f}s)")
            for p in problems:
                print("    ", p)
            failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
