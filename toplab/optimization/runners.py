"""Config-driven entry points: run an optimisation YAML, write CSVs and plots.

Command line::

    python -m toplab.optimization.runners sqp   path/to/sqp.yaml [--plot-only]
    python -m toplab.optimization.runners nsga2 path/to/nsga2.yaml [--plot-only]
    mpirun -n 16 python -m toplab.optimization.runners nsga2 path/to/nsga2.yaml --no-plot

Under MPI rank 0 drives NSGA-II and the other ranks evaluate designs (use at least 2 ranks).

Results go to ``output.results_dir`` (relative to the YAML):

* SQP: ``sqp_runs.csv`` (every solve), ``sqp_pareto.csv`` (when epsilon constraints are
  configured), ``sqp_evaluations.csv`` (every model evaluation), ``plots/*.png``.
* NSGA-II: ``nsga2_history.csv``, ``nsga2_pareto.csv``, ``plots/*.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from toplab.optimization.evaluation import load_problem, write_rows


def run_sqp(config_path: Path | str, plot: bool = True):
    """Run every configured objective (and epsilon grid) and write the result files."""
    from toplab.optimization.sqp_optimizer import SQPOptimizer, history_rows, run_rows

    optimizer = SQPOptimizer(config_path)
    results = optimizer.run()
    out = optimizer.problem.results_dir
    runs = [run for result in results for run in result.runs]
    write_rows(out / "sqp_runs.csv", run_rows(optimizer, runs))
    if optimizer.cfg.get("epsilon_constraints"):
        write_rows(out / "sqp_pareto.csv", run_rows(optimizer, [r for res in results for r in res.pareto]))
    write_rows(out / "sqp_evaluations.csv", history_rows(optimizer))

    names = [v.name for v in optimizer.variables]
    for result in results:
        feasible = sum(run.feasible for run in result.runs)
        print(f"\n[{result.objective}] {feasible}/{len(result.runs)} solves feasible, "
              f"{len(result.pareto)} non-dominated")
        for run in result.runs:
            design = {n: round(v, 5) for n, v in zip(names, run.x)}
            print(f"  eps={run.epsilon or '-'} feasible={run.feasible} "
                  f"{result.objective}={getattr(run.initial, result.objective):.4f}"
                  f"->{run.value:.4f} x={design}\n    {run.message}")
    print(f"\n{len(optimizer.history)} unique evaluations; results in {out}")
    if plot:
        plot_sqp(config_path)
    return results


def run_nsga2(config_path: Path | str, plot: bool = True):
    from toplab.optimization.nsga2_optimizer import NSGA2Optimizer, write_history, write_pareto

    optimizer = NSGA2Optimizer(config_path)
    result = optimizer.run()
    if result is None:  # MPI worker rank
        return None
    out = optimizer.problem.results_dir
    write_history(out / "nsga2_history.csv", result)
    write_pareto(out / "nsga2_pareto.csv", result)
    names = [v.name for v in result.variables]
    print(f"\nEvaluations: {len(result.history)}   Pareto designs: {len(result.pareto)}")
    for x, c in zip(result.pareto_x, result.pareto):
        design = {n: round(float(v), 5) for n, v in zip(names, x)}
        print("  ", design, {o: round(getattr(c, o), 4) for o in result.objectives}, f"feasible={c.feasible}")
    if plot:
        plot_nsga2(config_path)
    return result


def plot_sqp(config_path: Path | str) -> list[Path]:
    """Plot existing SQP CSVs without rerunning the optimisation."""
    from toplab.plotting.optimization_plots import plot_objective_vs_variables, plot_pareto, read_rows

    out = load_problem(config_path, evaluator=object()).results_dir
    written = plot_objective_vs_variables(out / "sqp_evaluations.csv", out / "plots")
    if (out / "sqp_pareto.csv").exists():
        pareto, history = read_rows(out / "sqp_pareto.csv"), read_rows(out / "sqp_evaluations.csv")
        for objective in dict.fromkeys(r["objective"] for r in pareto):
            front = [r for r in pareto if r["objective"] == objective]
            constrained = [k[4:] for k in front[0] if k.startswith("eps_") and front[0][k] != ""]
            written += plot_pareto(front, [r for r in history if r["objective"] == objective],
                                   [objective, *constrained], out / "plots", f"sqp_{objective}")
    return written


def plot_nsga2(config_path: Path | str) -> list[Path]:
    from toplab.optimization.nsga2_optimizer import NSGA2Settings
    from toplab.plotting.optimization_plots import plot_pareto, read_rows

    problem = load_problem(config_path, evaluator=object())
    objectives = (problem.cfg.get("nsga2") or {}).get("objectives", NSGA2Settings.objectives)
    out = problem.results_dir
    return plot_pareto(read_rows(out / "nsga2_pareto.csv"), read_rows(out / "nsga2_history.csv"),
                       objectives, out / "plots", "nsga2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=("sqp", "nsga2"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--plot-only", action="store_true", help="re-plot existing CSVs")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.plot_only:
        paths = (plot_sqp if args.method == "sqp" else plot_nsga2)(args.config)
        print("\n".join(str(p) for p in paths))
    else:
        (run_sqp if args.method == "sqp" else run_nsga2)(args.config, plot=not args.no_plot)


if __name__ == "__main__":
    main()
