"""Examples for the aft-fuselage packaging compaction search.

Run next to ``aft_placement_modified.py``.  Each case first solves the ordinary
placement problem at z_shift=0, then uses bisection to find the largest feasible
forward bulkhead shift and reports zeta_p = z_shift / L1.
"""
from __future__ import annotations

from dataclasses import dataclass

from aft_placement import (
    AftFuselageDimensions,
    maximize_packaging_compaction,
    plot_aft_placement,
)


@dataclass(frozen=True)
class Case:
    name: str
    outer_radii: tuple[float, ...]
    half_cyl_lengths: tuple[float, ...]


# Illustrative aft-fuselage geometry. Replace with project geometry as needed.
DIMS = AftFuselageDimensions(
    d1=4.0,
    d2=3.0,
    d3=1.4,
    l1=4.0,
    l2=3.0,
    l3=3.0,
    epsilon=0.05,
    psi_1=0.0,
    psi_2=0.0,
)

CASES = (
    Case(
        "single_small",
        outer_radii=(0.55,),
        half_cyl_lengths=(1.00,),
    ),
    Case(
        "two_equal",
        outer_radii=(0.55, 0.55),
        half_cyl_lengths=(1.00, 1.00),
    ),
    Case(
        "two_mixed",
        outer_radii=(0.70, 0.45),
        half_cyl_lengths=(1.20, 0.75),
    ),
    Case(
        "three_mixed",
        outer_radii=(0.65, 0.50, 0.40),
        half_cyl_lengths=(1.10, 0.90, 0.65),
    ),
    Case(
        "large_pair",
        outer_radii=(0.90, 0.80),
        half_cyl_lengths=(1.40, 1.20),
    ),
)


def run_case(case: Case, show_plot: bool = False) -> None:
    result = maximize_packaging_compaction(
        case.outer_radii,
        case.half_cyl_lengths,
        DIMS,
        z_tolerance=0.01,
        max_bisection_iterations=20,
        max_iterations=600,
        initial_step=0.15,
        minimum_step=0.01,
        n_axial_samples=40,
        n_circumferential_samples=16,
    )

    print(f"\n{'=' * 72}")
    print(case.name)
    print(f"  tanks       : {len(case.outer_radii)}")
    print(f"  feasible    : {result.feasible}")
    print(f"  z_shift     : {result.z_shift:.4f} m")
    print(f"  zeta_p      : {result.zeta_p:.4f}")
    print(f"  message     : {result.message}")

    for placement in result.placements:
        print(
            f"  tank {placement.tank_index + 1}: "
            f"center=({placement.x:.3f}, {placement.y:.3f}, {placement.z:.3f}) m, "
            f"max_violation={placement.max_violation:.3e} m"
        )

    if show_plot and result.feasible:
        fig = plot_aft_placement(result)
        fig.update_layout(
            title=(
                f"{case.name}: z_shift={result.z_shift:.3f} m, "
                f"zeta_p={result.zeta_p:.3f}"
            )
        )
        fig.show()


def main() -> None:
    for case in CASES:
        run_case(case, show_plot=True)

    # Uncomment to inspect one converged compact placement interactively.
    # run_case(CASES[2], show_plot=True)


if __name__ == "__main__":
    main()