"""Numerically pack equal circles inside a circular boundary.

Run, for example::

    python toplab/packaging/circle_packing.py --diameter 2.16 --count 4 --epsilon 0.05

The reported radius is the best solution found by a reproducible multistart
search, not a certificate of global optimality.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class PackingResult:
    outer_diameter: float
    epsilon: float
    radius: float
    centers: np.ndarray
    successful_starts: int


def pack_circles(diameter: float, count: int, epsilon: float, *,
                 restarts: int = 32, seed: int = 0) -> PackingResult:
    """Find a common radius with at least ``epsilon`` wall and pair clearance.

    SLSQP maximizes the radius from multiple deterministic and random starting
    positions. The optimization is scaled by the outer radius. A successful
    local solution is verified geometrically before being returned.
    """
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("diameter must be finite and positive")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    outer_radius = diameter / 2.0
    clearance = epsilon / outer_radius
    if clearance >= 1:
        raise ValueError("epsilon must be smaller than the outer radius")
    if count == 1:
        return PackingResult(diameter, epsilon, outer_radius - epsilon,
                             np.zeros((1, 2)), 1)

    limit = 1.0 - clearance
    i, j = np.triu_indices(count, 1)

    def constraints(z: np.ndarray) -> np.ndarray:
        positions = z[:-1].reshape(count, 2)
        radius = z[-1]
        delta = positions[i] - positions[j]
        wall = (limit - radius) ** 2 - np.sum(positions * positions, axis=1)
        pair = np.sum(delta * delta, axis=1) - (2 * radius + clearance) ** 2
        return np.concatenate((wall, pair))

    def jacobian(z: np.ndarray) -> np.ndarray:
        positions = z[:-1].reshape(count, 2)
        radius = z[-1]
        jac = np.zeros((count + len(i), 2 * count + 1))
        rows = np.arange(count)
        jac[rows, 2 * rows] = -2 * positions[:, 0]
        jac[rows, 2 * rows + 1] = -2 * positions[:, 1]
        jac[:count, -1] = -2 * (limit - radius)
        rows = count + np.arange(len(i))
        delta = 2 * (positions[i] - positions[j])
        jac[rows, 2 * i] = delta[:, 0]
        jac[rows, 2 * i + 1] = delta[:, 1]
        jac[rows, 2 * j] = -delta[:, 0]
        jac[rows, 2 * j + 1] = -delta[:, 1]
        jac[count:, -1] = -4 * (2 * radius + clearance)
        return jac

    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray] | None = None
    successful_starts = 0
    for start in range(restarts):
        if start < 2:
            ring_count = count if start == 0 else count - 1
            angles = 2 * np.pi * np.arange(ring_count) / ring_count
            positions = limit * 0.8 * np.column_stack((np.cos(angles), np.sin(angles)))
            if start == 1:
                positions = np.vstack((np.zeros((1, 2)), positions))
        else:
            angles = rng.uniform(0, 2 * np.pi, count)
            radial = limit * np.sqrt(rng.uniform(0, 1, count))
            positions = np.column_stack((radial * np.cos(angles), radial * np.sin(angles)))
        initial = np.concatenate((positions.ravel(), [limit / (2 + math.sqrt(count))]))
        solution = minimize(
            lambda z: -z[-1], initial,
            jac=lambda _z: np.r_[np.zeros(2 * count), -1.0],
            method="SLSQP",
            bounds=[(-limit, limit)] * (2 * count) + [(0.0, limit)],
            constraints={"type": "ineq", "fun": constraints, "jac": jacobian},
            options={"maxiter": 500, "ftol": 1e-11},
        )
        if not solution.success or not np.all(np.isfinite(solution.x)):
            continue
        positions = solution.x[:-1].reshape(count, 2)
        wall_radius = limit - np.linalg.norm(positions, axis=1).max()
        pair_radius = (np.linalg.norm(positions[i] - positions[j], axis=1).min() - clearance) / 2
        radius = min(float(solution.x[-1]), float(wall_radius), float(pair_radius))
        if radius <= 0 or solution.x[-1] - radius > 1e-7:
            continue
        successful_starts += 1
        if best is None or radius > best[0]:
            best = (radius, positions.copy())

    if best is None:
        raise RuntimeError(
            "No positive-radius feasible packing was found. The requested clearance may "
            "be impossible, or the numerical search may need more restarts.")
    return PackingResult(diameter, epsilon, best[0] * outer_radius,
                         best[1] * outer_radius, successful_starts)


def plot_packing(result: PackingResult):
    """Create an equal-aspect matplotlib figure for a packing result."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    outer_radius = result.outer_diameter / 2
    ax.add_patch(Circle((0, 0), outer_radius, fill=False, color="black", linewidth=2))
    for index, (x, y) in enumerate(result.centers, 1):
        ax.add_patch(Circle((x, y), result.radius, facecolor="steelblue",
                            edgecolor="black", alpha=0.6))
        ax.text(x, y, str(index), ha="center", va="center")
    padding = outer_radius * 0.08
    ax.set_xlim(-outer_radius - padding, outer_radius + padding)
    ax.set_ylim(-outer_radius - padding, outer_radius + padding)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{len(result.centers)} circles: r = {result.radius:.6g} m; "
                 f"minimum clearance = {result.epsilon:g} m")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diameter", type=float, required=True, help="outer diameter [m]")
    parser.add_argument("--count", type=int, required=True, help="number of equal circles")
    parser.add_argument("--epsilon", type=float, required=True, help="minimum clearance [m]")
    parser.add_argument("--restarts", type=int, default=32, help="number of solver starts (default: 32)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--save", type=Path, help="save the plot to this path")
    parser.add_argument("--no-show", action="store_true", help="do not open an interactive plot")
    args = parser.parse_args()
    try:
        result = pack_circles(args.diameter, args.count, args.epsilon,
                              restarts=args.restarts, seed=args.seed)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Best radius found: {result.radius:.9g} m (numerical; not guaranteed globally optimal)")
    if args.count == 1:
        print("Single circle: exact solution")
    else:
        print(f"Successful starts: {result.successful_starts}/{args.restarts}")
    for index, (x, y) in enumerate(result.centers, 1):
        print(f"  Circle {index}: x={x:.9g} m, y={y:.9g} m")
    fig = plot_packing(result)
    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Plot saved: {args.save}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
