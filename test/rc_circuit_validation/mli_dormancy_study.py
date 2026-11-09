"""Investigation D: dormancy-mission comparison across node counts and capacitance.

Run ``python -m test.rc_circuit_validation.mli_dormancy_study`` from TOPLAB.
"""

import sys

import numpy as np

from test.rc_circuit_validation.mli_capacitance import (
    QUARTER_MIL,
    TabulatedSpecificHeat,
)
from test.rc_circuit_validation.mli_dormancy_harness import DormancyHarness

NODE_COUNTS = (1, 5, 10, 20, 50)
CASES = {  # label: (density kg/m^3, cp callable or None for 1000 J/kg/K)
    "rho300_cp1000 (production)": (None, None),
    "rho60_cp1000": (60.0, lambda T: 1000.0),
    "rho30_cp1000": (30.0, lambda T: 1000.0),
    "rho30_cpPET(T)*": (30.0, TabulatedSpecificHeat(QUARTER_MIL)),
    "rho60_cpPET(T)*": (60.0, TabulatedSpecificHeat(QUARTER_MIL)),
}
INITIALS = ("equilibrated", "post_refuel", "uniform_warm")


def run_all(cases=CASES, initials=INITIALS, node_counts=NODE_COUNTS):
    rows = []
    for label, (density, cp) in cases.items():
        harness = DormancyHarness(density=density, cp=cp)
        for initial in initials:
            for n in node_counts:
                r = harness.run(n, initial)
                rows.append(
                    dict(
                        case=label,
                        initial=initial,
                        nodes=n,
                        vented_kg=r.vented_mass,
                        vent_percent=100.0 * r.vented_mass / r.y[0, 0],
                        vent_onset_h=r.vent_onset_hours,
                        pressure_end_mpa=r.pressure[-1] / 1e6,
                        t_h2_end=r.y[1, -1],
                        heat_to_h2_kj=r.cumulative_h2_heat_kj[-1],
                        cpu=r.cpu_seconds,
                    )
                )
                print(format_row(rows[-1]), flush=True)
    return rows


def format_row(r):
    return (
        f"{r['case']:28s} {r['initial']:13s} N={r['nodes']:3d} "
        f"vent={r['vented_kg']:.4f} kg ({r['vent_percent']:.4f}%) "
        f"onset={r['vent_onset_h']:.2f} h P_end={r['pressure_end_mpa']:.4f} MPa "
        f"T_H2_end={r['t_h2_end']:.4f} K Q_H2={r['heat_to_h2_kj']:.1f} kJ "
        f"cpu={r['cpu']:.1f}s"
    )


if __name__ == "__main__":
    run_all()
