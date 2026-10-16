#!/usr/bin/env python3
"""Two-tank CcH2 / ATR72 optimisation driver: maximise eta_g over both tank radii with SLSQP."""

from pathlib import Path

from toplab.optimization.sqp_optimizer import SQPOptimizer, serialize_evaluation_history, serialize_single_objective

current_dir = Path(__file__).parent


def main():
    config_path = current_dir / "two_tank_cch2_sqp.yaml"
    optimizer = SQPOptimizer(config_path)
    objective = optimizer.cfg.get("objective", "eta_g")
    result = optimizer.solve_single_objective(objective)

    results_dir = current_dir / optimizer.cfg.get("output", {}).get("results_dir", "output/results")
    serialize_single_objective(result, results_dir / "sqp_result.csv")
    serialize_evaluation_history(optimizer, results_dir / "sqp_history.csv")

    names = [v.name for v in optimizer.variables]
    print("\nOptimum:", dict(zip(names, result.x)))
    print(f"{objective} = {getattr(result.candidate, objective):.4f}  "
          f"feasible={result.feasible}  nit={result.nit}  evaluations={result.unique_evaluations}")
    print(f"residuals g_dis, g_dorm, g_packaging = {result.physical_residuals}")
    return result


if __name__ == "__main__":
    main()
