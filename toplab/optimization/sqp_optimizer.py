"""
Generic SQP optimiser for coupled multi-tank systems.

Parallels BaseSweepStudy in sweep_runner.py: core machinery lives here;
drivers in optimization/<study>/ are thin facades that supply a config path.
"""

import copy
import csv
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
from scipy.optimize import minimize

# Canonical design-variable order — never reorder
_ALL_VARS: List[str] = ["r1", "r2", "phi1", "phi2"]


# ── per-evaluation result ─────────────────────────────────────────────────────

@dataclass
class EvalResult:
    r1: float
    r2: float
    phi1: float = 0.0
    phi2: float = 0.0
    eta_g: float = 0.0
    eta_v: float = 0.0
    v_inner_m3: float = 0.0
    v_outer_m3: float = 0.0
    total_length_m: float = 0.0
    mission_completed: bool = False
    mission_ratio: float = 0.0
    error: Optional[str] = None

    @property
    def feasible(self) -> bool:
        return self.mission_completed and self.error is None


# ── config patcher ────────────────────────────────────────────────────────────

def _build_temp_config(
    base_raw: dict, r1: float, r2: float,
    phi1: float, phi2: float, config_dir: Path
) -> Path:
    """Deep-copy base config, patch node geometry, write to a temp YAML file."""
    raw = copy.deepcopy(base_raw)
    for node in raw.get("network", {}).get("nodes", []):
        if node.get("type") == "tank":
            nid = node["node_id"]
            if nid == 1:
                node["geometry"]["radius"] = float(r1)
                node["geometry"]["phi"]    = float(phi1)
            elif nid == 2:
                node["geometry"]["radius"] = float(r2)
                node["geometry"]["phi"]    = float(phi2)
    raw.setdefault("output", {}).update({"save_plots": False, "save_data": False, "silent": True})
    raw["dormancy_check"] = {"enabled": False, "duration_h": 24.0}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=config_dir, delete=False
    ) as fh:
        yaml.dump(raw, fh)
        return Path(fh.name)


# ── single-design evaluator ───────────────────────────────────────────────────

def evaluate_design(
    r1: float, r2: float, phi1: float, phi2: float,
    base_raw: dict, base_config_path: Path,
) -> EvalResult:
    """Run one simulation and return all metrics needed by the optimiser."""
    # Import here to avoid circular import at module load time
    from toplab.configuration.scenario_configuration import ScenarioConfig
    from toplab.orchestration.system_orchestrator import SystemOrchestrator

    result = EvalResult(r1=r1, r2=r2, phi1=phi1, phi2=phi2)
    result.total_length_m = r1 * (2.0 + phi1) + r2 * (2.0 + phi2)

    temp_path = _build_temp_config(base_raw, r1, r2, phi1, phi2, base_config_path.parent)
    try:
        config = ScenarioConfig.from_yaml(str(temp_path))
        orch = SystemOrchestrator(config, verbosity="quiet")

        # Outer volume is available from _cached_tank_properties after __init__ — no ODE needed.
        n_tanks = len(orch.tank_geometries)
        v_outer = 0.0
        v_inner = 0.0
        for i in range(n_tanks):
            props = orch.tank_system._cached_tank_properties[i]
            outer_vol = props.get("outer_volume")
            if outer_vol is None:
                outer_r = 0.5 * float(props["outer_diameter"])
                cyl_len = float(props.get("cylindrical_section_length", 0.0))
                outer_vol = (4.0 / 3.0) * math.pi * outer_r ** 3 + math.pi * outer_r ** 2 * cyl_len
            v_outer += float(outer_vol)
            v_inner += float(getattr(orch.tank_geometries[i], 'volume', props.get('volume', 0.0)))
        result.v_outer_m3 = v_outer
        result.v_inner_m3 = v_inner

        sim = orch.run_simulation()

        target_s = float(orch.tank_system.config.MISSION_DURATION)
        actual_s  = float(sim.times[-1])
        result.mission_ratio     = actual_s / target_s if target_s > 0.0 else 0.0
        result.mission_completed = actual_s >= 0.999 * target_s

        if result.mission_completed:
            total_fuel = total_struct = 0.0
            for i in range(n_tanks):
                state0 = sim.multi_tank_states[0].get_tank_state(i)
                props  = orch.tank_system._cached_tank_properties[i]
                total_fuel   += float(state0.fuel_mass)
                total_struct += float(props["liner_mass"]) + float(props["wall_mass"])
            total_mass = total_fuel + total_struct
            result.eta_g = total_fuel / total_mass if total_mass > 0.0 else 0.0
            result.eta_v = result.v_inner_m3 / result.v_outer_m3 if result.v_outer_m3 > 0.0 else 0.0

    except Exception as exc:
        result.error = str(exc)
    finally:
        temp_path.unlink(missing_ok=True)

    return result


# ── SQP optimiser ─────────────────────────────────────────────────────────────

class SQPOptimizer:
    """
    Generic SQP optimiser for a CH2+CCH2 coupled tank system.

    Reads a YAML config that specifies design variables (with enabled flags),
    constraints, and solver settings. Subclass or use directly from a driver.
    """

    def __init__(self, opt_config_path: Path):
        self.opt_config_path = opt_config_path
        with open(opt_config_path) as fh:
            self.cfg = yaml.safe_load(fh)

        base_rel = self.cfg["optimization"]["base_config"]
        self.base_config_path = (opt_config_path.parent / base_rel).resolve()
        with open(self.base_config_path) as fh:
            self.base_raw = yaml.safe_load(fh)

        dv = self.cfg["design_variables"]
        # Variables with enabled: true (default) form the active design vector
        self._active: List[str] = [v for v in _ALL_VARS if dv[v].get("enabled", True)]
        self._fixed:  List[str] = [v for v in _ALL_VARS if v not in self._active]

        # {(r1, r2, phi1, phi2): EvalResult}
        self._cache: Dict[Tuple[float, float, float, float], EvalResult] = {}
        self._eval_count = 0
        self._history: list = []

    # ── active vector → full (r1, r2, phi1, phi2) ────────────────────────────

    def _full_vec(self, x_active: np.ndarray) -> Tuple[float, float, float, float]:
        """Expand the reduced active vector to the full (r1, r2, phi1, phi2) tuple."""
        dv = self.cfg["design_variables"]
        vals: Dict[str, float] = {}
        for i, name in enumerate(self._active):
            vals[name] = float(x_active[i])
        for name in self._fixed:
            vals[name] = float(dv[name]["baseline"])
        return vals["r1"], vals["r2"], vals["phi1"], vals["phi2"]

    # ── cached evaluation ─────────────────────────────────────────────────────

    def _get(self, x: np.ndarray) -> EvalResult:
        r1, r2, phi1, phi2 = self._full_vec(x)
        key = (round(r1, 4), round(r2, 4), round(phi1, 3), round(phi2, 3))
        if key not in self._cache:
            self._eval_count += 1
            print(
                f"  [{self._eval_count:3d}]  r1={r1:.4f}m  r2={r2:.4f}m  "
                f"phi1={phi1:.3f}  phi2={phi2:.3f}",
                end="  ... ", flush=True,
            )
            t0  = time.perf_counter()
            res = evaluate_design(r1, r2, phi1, phi2, self.base_raw, self.base_config_path)
            elapsed = time.perf_counter() - t0
            tag = "OK" if res.feasible else ("INFEASIBLE" if not res.mission_completed else "ERROR")
            print(f"η_v={res.eta_v:.4f}  η_g={res.eta_g:.4f}  V={res.v_outer_m3:.3f}m³  "
                  f"L={res.total_length_m:.2f}m  {tag}  ({elapsed:.1f}s)")
            self._cache[key] = res
        return self._cache[key]

    # ── objective (minimise −η_v; infeasible → 0, i.e. worst possible) ───────

    def objective(self, x: np.ndarray) -> float:
        return -self._get(x).eta_v

    # ── inequality constraints (SLSQP: value ≥ 0) ────────────────────────────

    def con_length(self, x: np.ndarray) -> float:
        r1, r2, phi1, phi2 = self._full_vec(x)
        used = r1 * (2.0 + phi1) + r2 * (2.0 + phi2)
        return self.cfg["constraints"]["total_length_m"]["max"] - used

    def con_volume(self, x: np.ndarray) -> float:
        return (
            self.cfg["constraints"]["total_outer_volume_m3"]["max"]
            - self._get(x).v_outer_m3
        )

    # ── per-iterate callback ──────────────────────────────────────────────────

    def _callback(self, x: np.ndarray) -> None:
        r1, r2, phi1, phi2 = self._full_vec(x)
        res = self._get(x)
        self._history.append({
            "iter":           len(self._history),
            "r1": r1, "r2": r2, "phi1": phi1, "phi2": phi2,
            "eta_v":          res.eta_v,
            "eta_g":          res.eta_g,
            "v_inner_m3":     res.v_inner_m3,
            "v_outer_m3":     res.v_outer_m3,
            "total_length_m": res.total_length_m,
            "feasible":       res.feasible,
            "mission_ratio":  res.mission_ratio,
        })

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self) -> dict:
        # load config and build active vector, bounds, and finite-difference step sizes
        dv = self.cfg["design_variables"]

        # Build x0, bounds, and eps only over active variables
        x0     = np.array([dv[v]["baseline"] for v in self._active], dtype=float)
        bounds = [(dv[v]["min"], dv[v]["max"])  for v in self._active]

        raw_eps = self.cfg.get("solver", {}).get("finite_diff_step")
        if isinstance(raw_eps, list):
            all_eps = dict(zip(_ALL_VARS, raw_eps))
            eps = np.array([all_eps[v] for v in self._active], dtype=float)
        else:
            eps = float(raw_eps)

        constraints = [
            {"type": "ineq", "fun": self.con_length},
            {"type": "ineq", "fun": self.con_volume},
        ]
        slsqp_opts = {
            "maxiter": int(self.cfg.get("solver", {}).get("max_iter")),
            "ftol":    float(self.cfg.get("solver", {}).get("tol")),
            "eps":     eps,
            "disp":    True,
        }

        # header
        bar = "=" * 72
        print(f"\n{bar}")
        print(f"  CH2-CCH2 SQP — Maximise Volumetric Efficiency")
        print(f"{bar}")
        print(f"  Active variables ({len(self._active)}):")
        for v in self._active:
            print(f"    {v:<6} ∈ [{dv[v]['min']}, {dv[v]['max']}]   baseline={dv[v]['baseline']}")
        if self._fixed:
            print(f"  Fixed variables ({len(self._fixed)}):")
            for v in self._fixed:
                print(f"    {v:<6} = {dv[v]['baseline']}  (fixed)")
        r1_0, r2_0, phi1_0, phi2_0 = self._full_vec(x0)
        l0 = r1_0 * (2 + phi1_0) + r2_0 * (2 + phi2_0)
        print(f"  Constraints:")
        print(f"    Total length  ≤ {self.cfg['constraints']['total_length_m']['max']} m   "
              f"(baseline {l0:.2f} m)")
        print(f"    Outer volume  ≤ {self.cfg['constraints']['total_outer_volume_m3']['max']} m³")
        print(f"    Mission must complete (infeasible → η_g = 0 penalty)")
        print(f"  SLSQP: max_iter={slsqp_opts['maxiter']}  tol={slsqp_opts['ftol']}  "
              f"FD_step={eps}")
        print(f"{bar}\n")

        # baseline
        print("Evaluating baseline:")
        m0 = self._get(x0)
        self._history.append({
            "iter": -1, "r1": r1_0, "r2": r2_0, "phi1": phi1_0, "phi2": phi2_0,
            "eta_v": m0.eta_v, "eta_g": m0.eta_g,
            "v_inner_m3": m0.v_inner_m3, "v_outer_m3": m0.v_outer_m3,
            "total_length_m": m0.total_length_m,
            "feasible": m0.feasible, "mission_ratio": m0.mission_ratio,
        })

        # optimise
        print(f"\nRunning SLSQP...")
        t_start = time.perf_counter()
        opt = minimize(
            fun=self.objective, x0=x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options=slsqp_opts, callback=self._callback,
        )
        wall_s = time.perf_counter() - t_start

        # final report
        m_opt = self._get(opt.x)
        r1_opt, r2_opt, phi1_opt, phi2_opt = self._full_vec(opt.x)
        l_opt = r1_opt * (2 + phi1_opt) + r2_opt * (2 + phi2_opt)

        GREEN = "\033[92m"
        RESET = "\033[0m"

        print(f"\n{bar}")
        print(f"  RESULT  ({wall_s:.0f} s,  {self._eval_count} unique evaluations)")
        print(f"{bar}")
        print(f"  scipy status : {opt.message}")
        print(f"{GREEN}  Optimum      : r1={r1_opt:.4f} m   r2={r2_opt:.4f} m  "
              f"phi1={phi1_opt:.3f}   phi2={phi2_opt:.3f}{RESET}")
        print(f"  η_v          : {m_opt.eta_v:.4f}   "
              f"(Δ = {m_opt.eta_v - m0.eta_v:+.4f} vs baseline {m0.eta_v:.4f})")
        print(f"  η_g          : {m_opt.eta_g:.4f}   "
              f"(Δ = {m_opt.eta_g - m0.eta_g:+.4f} vs baseline {m0.eta_g:.4f})")
        print(f"  Total length : {l_opt:.3f} m   "
              f"(limit {self.cfg['constraints']['total_length_m']['max']} m)")
        print(f"  Outer volume : {m_opt.v_outer_m3:.3f} m³   "
              f"(limit {self.cfg['constraints']['total_outer_volume_m3']['max']} m³)")
        print(f"  Mission      : {'completed' if m_opt.mission_completed else 'INCOMPLETE'}")
        print(f"{bar}\n")

        self._save(opt, m0, m_opt, wall_s)
        return {"scipy_result": opt, "optimum": m_opt, "baseline": m0, "history": self._history}

    # ── output ────────────────────────────────────────────────────────────────

    def _save(self, opt, m0: EvalResult, m_opt: EvalResult, wall_s: float) -> None:
        # Output is relative to the optimisation config file, not this src module
        out_dir = (
            self.opt_config_path.parent
            / self.cfg.get("output", {}).get("results_dir", "output")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        hist_path = out_dir / "sqp_history.csv"
        if self._history:
            with open(hist_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self._history[0].keys()))
                writer.writeheader()
                writer.writerows(self._history)
            print(f"  History  → {hist_path}")

        cache_path = out_dir / "sqp_all_evals.csv"
        cache_rows = [
            {
                "r1": res.r1, "r2": res.r2, "phi1": res.phi1, "phi2": res.phi2,
                "eta_v": res.eta_v, "eta_g": res.eta_g,
                "v_inner_m3": res.v_inner_m3, "v_outer_m3": res.v_outer_m3,
                "total_length_m": res.total_length_m,
                "feasible": res.feasible, "mission_ratio": res.mission_ratio,
                "error": res.error or "",
            }
            for res in self._cache.values()
        ]
        if cache_rows:
            with open(cache_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(cache_rows[0].keys()))
                writer.writeheader()
                writer.writerows(cache_rows)
            print(f"  All evals → {cache_path}")

        r1_opt, r2_opt, phi1_opt, phi2_opt = self._full_vec(opt.x)
        l_opt = r1_opt * (2 + phi1_opt) + r2_opt * (2 + phi2_opt)
        summary_path = out_dir / "sqp_summary.txt"
        with open(summary_path, "w") as fh:
            fh.write("CH2-CCH2 SQP Optimisation Summary\n")
            fh.write("=" * 50 + "\n")
            fh.write(f"Status      : {opt.message}\n")
            fh.write(f"Wall time   : {wall_s:.0f} s\n")
            fh.write(f"Evaluations : {self._eval_count} unique\n")
            fh.write(f"Active vars : {', '.join(self._active)}\n")
            fh.write(f"Fixed vars  : {', '.join(self._fixed) if self._fixed else 'none'}\n\n")
            fh.write(f"Baseline : r1={m0.r1:.4f} m  r2={m0.r2:.4f} m  "
                     f"phi1={m0.phi1:.3f}  phi2={m0.phi2:.3f}  "
                     f"η_v={m0.eta_v:.4f}  η_g={m0.eta_g:.4f}  V={m0.v_outer_m3:.3f} m³\n")
            fh.write(f"Optimum  : r1={r1_opt:.4f} m  r2={r2_opt:.4f} m  "
                     f"phi1={phi1_opt:.3f}  phi2={phi2_opt:.3f}  "
                     f"η_v={m_opt.eta_v:.4f}  η_g={m_opt.eta_g:.4f}  V={m_opt.v_outer_m3:.3f} m³  "
                     f"L={l_opt:.3f} m\n")
            fh.write(f"Δη_v     : {m_opt.eta_v - m0.eta_v:+.4f}\n")
            fh.write(f"Δη_g     : {m_opt.eta_g - m0.eta_g:+.4f}\n")
        print(f"  Summary  → {summary_path}")
