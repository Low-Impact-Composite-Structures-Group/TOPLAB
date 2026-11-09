"""Minimum insulation thickness that satisfies the dormancy venting requirement.

Bisects the insulation thickness of a 0.5 m radius, phi = 4 CcH2 tank (all other
inputs from ``cch2_baseline``) until the production dormancy check
(``run_dormancy_check``: zero outflow, 12 h, <= 0.1 % of the initial mass vented)
is just satisfied, for foam and for MLI (Modified Lockheed).

MLI properties follow the baseline YAML (500 uTorr, degradation factor 1,
4 mm shell, alpha_shell 5, emissivity_shell 0.05) except for the effective
density (30 kg/m^3) and a constant specific heat (1000 J/kg/K).

Run from ``TOPLAB/`` with ``python -m pytest test/test_min_insulation_thickness_dormancy.py -s``.
"""

import dataclasses
from pathlib import Path

import pytest

from toplab.config import load_config
from toplab.orchestration.run_analysis import run_dormancy_check

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "toplab-apps/analysis/novacom/slh2_baseline/config_slh2_baseline.yaml"
)
RADIUS = 0.5
PHI = 4.0
DURATION_H = 12.0
MAX_LOSS_PERCENT = 0.1
MLI_DENSITY = 30.0
MLI_SPECIFIC_HEAT = 1000.0
THICKNESS_BRACKET = (0.001, 1.0)  # [m]
TOLERANCE = 1e-4  # [m]


def insulation_block(model, thickness):
    block = {
        "model": model,
        "thickness": thickness,
        "shell_thickness": 0.004,
        "alpha_shell": 5.0,
        "emissivity_shell": 0.05,
    }
    if model == "modified_lockheed":
        block.update(
            pressure_torr= 1.0e-4,
            degradation_factor=1.0,  # the YAML key is misspelled "degration_factor" and ignored
            density=MLI_DENSITY,
            specific_heat=MLI_SPECIFIC_HEAT,
        )
    return block


def dormancy_loss_percent(model, thickness):
    baseline = load_config(str(CONFIG))
    tank = baseline.tanks[0]
    case = dataclasses.replace(
        baseline,
        tanks=[
            dataclasses.replace(
                tank,
                geometry={"phi": PHI, "radius": RADIUS},
                materials={**tank.materials, "insulation": insulation_block(model, thickness)},
            )
        ],
        dormancy_check={
            "enabled": True,
            "duration_h": DURATION_H,
            "maximum_loss_percent": MAX_LOSS_PERCENT,
        },
    )
    result = run_dormancy_check(case, silent=True)
    return result["tanks"][0]["loss_percent"]


def minimum_thickness(model, bracket=THICKNESS_BRACKET, tolerance=TOLERANCE):
    """Bisect for the thinnest insulation with loss <= MAX_LOSS_PERCENT.

    Returns ``(thickness, loss_at_thickness, loss_just_below, iterations)`` where
    ``thickness`` is the passing end of the final bracket.
    """
    low, high = bracket
    loss_low = dormancy_loss_percent(model, low)
    loss_high = dormancy_loss_percent(model, high)
    assert loss_low > MAX_LOSS_PERCENT, f"{model}: lower bound already passes"
    assert loss_high <= MAX_LOSS_PERCENT, f"{model}: upper bound still fails"
    iterations = 0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        loss_middle = dormancy_loss_percent(model, middle)
        if loss_middle <= MAX_LOSS_PERCENT:
            high, loss_high = middle, loss_middle
        else:
            low, loss_low = middle, loss_middle
        iterations += 1
    return high, loss_high, loss_low, iterations


@pytest.fixture(scope="module")
def results():
    found = {
        model: minimum_thickness(model) for model in ("foam", "modified_lockheed")
    }
    print(
        f"\nMinimum insulation thickness, {DURATION_H:g} h dormancy, "
        f"<= {MAX_LOSS_PERCENT} % vented, r = {RADIUS} m, phi = {PHI:g}"
    )
    for model, (thickness, loss, loss_below, iterations) in found.items():
        print(
            f"  {model:18s} {1000 * thickness:8.1f} mm  loss {loss:.4f} % "
            f"(just below: {loss_below:.4f} %), {iterations} bisections"
        )
    return found


@pytest.mark.parametrize("model", ("foam", "modified_lockheed"))
def test_bisected_thickness_brackets_the_requirement(results, model):
    thickness, loss, loss_below, _ = results[model]
    assert loss <= MAX_LOSS_PERCENT < loss_below
    assert thickness - TOLERANCE >= THICKNESS_BRACKET[0]
    # Independent re-evaluation on both sides of the answer.
    assert dormancy_loss_percent(model, thickness) <= MAX_LOSS_PERCENT
    assert dormancy_loss_percent(model, thickness - 1.5 * TOLERANCE) > MAX_LOSS_PERCENT


def test_mli_needs_less_insulation_than_foam(results):
    assert results["modified_lockheed"][0] < results["foam"][0]


def test_loss_decreases_with_thickness():
    for model in ("foam", "modified_lockheed"):
        losses = [dormancy_loss_percent(model, t) for t in (0.002, 0.005, 0.02, 0.2)]
        assert all(a >= b for a, b in zip(losses, losses[1:]))
