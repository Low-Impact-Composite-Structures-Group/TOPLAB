#!/usr/bin/env python3
"""Two-tank CcH2 / ATR72 NSGA-II driver: Pareto front of eta_g, eta_v and venting performance."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toplab.optimization.nsga2_optimizer import NSGA2Optimizer, write_history, write_pareto

current_dir = Path(__file__).parent


def plot_front(result, path: Path) -> None:
    feasible = [(x, c) for x, c in result.history if c.feasible]
    fig, ax = plt.subplots(figsize=(7, 5))
    if feasible:
        ax.scatter([c.eta_g for _, c in feasible], [c.eta_v for _, c in feasible],
                   c="lightgray", s=25, label="feasible evaluations")
    front = ax.scatter([c.eta_g for c in result.pareto], [c.eta_v for c in result.pareto],
                       c=[c.eta_vent for c in result.pareto], cmap="viridis", s=60, label="Pareto set")
    fig.colorbar(front, ax=ax, label="venting performance eta_vent")
    ax.set_xlabel("gravimetric efficiency eta_g")
    ax.set_ylabel("volumetric efficiency eta_v")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def main():
    optimizer = NSGA2Optimizer(current_dir / "two_tank_cch2_nsga.yaml")
    result = optimizer.run()

    results_dir = current_dir / optimizer.cfg.get("output", {}).get("results_dir", "output/results")
    write_history(results_dir / "nsga2_history.csv", result)
    write_pareto(results_dir / "nsga2_pareto.csv", result)
    plot_front(result, results_dir / "nsga2_pareto.png")

    names = [v.name for v in result.variables]
    print(f"\nEvaluations: {len(result.history)}   Pareto-optimal designs: {len(result.pareto)}")
    for x, c in zip(result.pareto_x, result.pareto):
        print("  ", {n: round(float(v), 4) for n, v in zip(names, x)},
              f"eta_g={c.eta_g:.4f} eta_v={c.eta_v:.4f} eta_vent={c.eta_vent:.4f}")
    return result


if __name__ == "__main__":
    main()
