"""Foam RC convergence figures (linear axes, equilibrated start).

Run from the TOPLAB root: ``python -m test.rc_circuit_validation.plot_foam_rc_convergence``
"""

from pathlib import Path

from matplotlib import pyplot as plt

from test.rc_circuit_validation.foam_rc_studies import (
    NODE_COUNTS,
    SAMPLE_TIMES,
    metadata,
    simulate,
)

FIGURE_DIRECTORY = Path("test/rc_circuit_validation/figures")


def main():
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for n in NODE_COUNTS:
        result = simulate(n)
        axes[0].plot(result.times / 60.0, result.heat_flow, label=f"N={n}")
        axes[1].plot(result.times / 60.0, result.cumulative_kj, label=f"N={n}")
    axes[0].set_ylabel("Heat flow into structure [W]")
    axes[1].set_ylabel("Cumulative heat into structure [kJ]")
    for axis in axes[:2]:
        axis.set_xlabel("Time [min]")
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, title="Insulation nodes")
    for time in SAMPLE_TIMES:
        axes[2].plot(
            NODE_COUNTS,
            [simulate(n).cumulative_at(time) for n in NODE_COUNTS],
            "o-",
            label=f"t = {time / 60:g} min",
        )
    axes[2].set_xlabel("Insulation nodes, N")
    axes[2].set_ylabel("Cumulative heat into structure [kJ]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    figure.suptitle(
        f"Foam near-equilibrium transient (equilibrated start, shell step)\n{metadata()}",
        fontsize=9,
    )
    figure.tight_layout()
    path = FIGURE_DIRECTORY / "foam_equilibrated_convergence.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
