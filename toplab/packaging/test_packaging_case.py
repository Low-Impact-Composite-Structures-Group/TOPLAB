"""Regression tests for the aft-fuselage packaging algorithm.

These cases are intentionally conservative: they use relatively small tanks
inside a comparatively large aft-fuselage volume, so they are expected a priori
to admit feasible placements. The goal is to test whether the packaging
algorithm reliably recovers those feasible arrangements as tank multiplicity
and size heterogeneity increase.

Run from the repository root, for example:

    python test_aft_packaging_cases.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from toplab.packaging.aft_placement import (
    AftFuselageDimensions,
    place_tanks_in_aft,
    plot_aft_placement,
)


@dataclass(frozen=True)
class PackagingCase:
    name: str
    outer_radii: tuple[float, ...]
    half_cyl_lengths: tuple[float, ...]
    description: str


# ---------------------------------------------------------------------------
# Test volume
# ---------------------------------------------------------------------------

# Deliberately generous relative to the cases below.
# Replace these values with your actual aft-fuselage volume when desired.
DIMS = AftFuselageDimensions(
    d1=2.0,
    d2=1.5,
    d3=0.5,
    l1=3.0,
    l2=2.0,
    l3=1.0,
    epsilon=0.05,
    psi_1=12.0,
    psi_2=10.0,
)


# ---------------------------------------------------------------------------
# Cases expected to be feasible
# ---------------------------------------------------------------------------

CASES: tuple[PackagingCase, ...] = (
    PackagingCase(
        name="single_medium",
        outer_radii=(0.50,),
        half_cyl_lengths=(1.00,),
        description="One medium capsule: basic sanity check.",
    ),
    PackagingCase(
        name="two_identical_medium",
        outer_radii=(0.45, 0.45),
        half_cyl_lengths=(0.80, 0.80),
        description=(
            "Two identical moderate-size tanks. They should fit comfortably "
            "with mostly longitudinal separation."
        ),
    ),
    PackagingCase(
        name="three_identical_small",
        outer_radii=(0.35, 0.35, 0.35),
        half_cyl_lengths=(0.60, 0.60, 0.60),
        description=(
            "Three identical small tanks. Tests multiplicity without strongly "
            "challenging radial clearance."
        ),
    ),
    PackagingCase(
        name="three_mixed",
        outer_radii=(0.50, 0.40, 0.30),
        half_cyl_lengths=(0.90, 0.70, 0.50),
        description=(
            "Three differently sized tanks. Tests largest-first ordering and "
            "heterogeneous capsule geometry."
        ),
    ),
    PackagingCase(
        name="four_small",
        outer_radii=(0.30, 0.30, 0.30, 0.30),
        half_cyl_lengths=(0.50, 0.50, 0.50, 0.50),
        description=(
            "Four small tanks. Tests whether the search can exploit both "
            "longitudinal and transverse displacement."
        ),
    ),
    PackagingCase(
        name="five_mixed_small",
        outer_radii=(0.35, 0.32, 0.30, 0.28, 0.25),
        half_cyl_lengths=(0.60, 0.55, 0.50, 0.45, 0.40),
        description=(
            "Five small heterogeneous tanks. Higher-multiplicity regression "
            "case that should still fit comfortably."
        ),
    ),
)


def _tank_total_length(radius: float, half_cyl_length: float) -> float:
    """Return total capsule length [m]."""
    return 2.0 * (radius + half_cyl_length)


def _print_case_header(case: PackagingCase) -> None:
    print("\n" + "=" * 78)
    print(f"CASE: {case.name}")
    print(case.description)
    print("-" * 78)

    for i, (radius, half_cyl) in enumerate(
        zip(case.outer_radii, case.half_cyl_lengths),
        start=1,
    ):
        print(
            f"Tank {i}: "
            f"R = {radius:.3f} m, "
            f"L = {_tank_total_length(radius, half_cyl):.3f} m"
        )


def _print_result(result) -> None:
    status = "PASS" if result.feasible else "FAIL"
    print(f"\nResult: {status}")
    print(result.message)

    for p in result.placements:
        x, y, z = p.center
        tx, ty, tz = p.axis

        print(
            f"  Tank {p.tank_index + 1}: "
            f"C = ({x: .3f}, {y: .3f}, {z: .3f}) m, "
            f"t = ({tx: .3f}, {ty: .3f}, {tz: .3f}), "
            f"z extent = "
            f"[{p.s_leftmost_pole:.3f}, {p.s_rightmost_pole:.3f}] m, "
            f"violation = {p.max_violation:.3e} m"
        )


def run_case(
    case: PackagingCase,
    *,
    save_plot: bool = False,
    plot_dir: Path | None = None,
):
    """Run one expected-feasible packaging case."""

    _print_case_header(case)

    result = place_tanks_in_aft(
        outer_radii=case.outer_radii,
        half_cyl_lengths=case.half_cyl_lengths,
        dims=DIMS,
        max_iterations=2000,
        initial_step=0.10,
        minimum_step=0.001,
        step_reduction=0.5,
        n_axial_samples=100,
        n_circumferential_samples=32,
    )

    _print_result(result)

    if save_plot:
        if plot_dir is None:
            plot_dir = Path("output") / "packaging_tests"

        plot_dir.mkdir(parents=True, exist_ok=True)

        fig = plot_aft_placement(result)
        html_path = plot_dir / f"{case.name}.html"
        fig.write_html(html_path)

        print(f"Saved plot: {html_path}")

    return result


def run_all_cases(
    cases: Sequence[PackagingCase] = CASES,
    *,
    save_plots: bool = False,
) -> dict[str, object]:
    """Run every predefined expected-feasible packaging case."""

    print("=" * 78)
    print("AFT-FUSELAGE PACKAGING ALGORITHM REGRESSION TEST")
    print("=" * 78)

    print(
        f"Fuselage: "
        f"d = ({DIMS.d1:.2f}, {DIMS.d2:.2f}, {DIMS.d3:.2f}) m, "
        f"l = ({DIMS.l1:.2f}, {DIMS.l2:.2f}, {DIMS.l3:.2f}) m, "
        f"epsilon = {DIMS.epsilon:.3f} m"
    )

    results: dict[str, object] = {}

    for case in cases:
        results[case.name] = run_case(
            case,
            save_plot=save_plots,
        )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    n_pass = 0

    for case in cases:
        result = results[case.name]
        status = "PASS" if result.feasible else "FAIL"

        if result.feasible:
            n_pass += 1

        max_violation = max(
            (p.max_violation for p in result.placements),
            default=0.0,
        )

        print(
            f"{case.name:<28} "
            f"{status:<6} "
            f"N = {len(case.outer_radii):<2d} "
            f"max violation = {max_violation:.3e} m"
        )

    print("-" * 78)
    print(f"{n_pass}/{len(cases)} expected-feasible cases passed.")

    if n_pass != len(cases):
        failed = [
            case.name
            for case in cases
            if not results[case.name].feasible
        ]
        raise RuntimeError(
            "Packaging algorithm failed expected-feasible cases: "
            + ", ".join(failed)
        )

    return results


if __name__ == "__main__":
    run_all_cases(save_plots=False)
