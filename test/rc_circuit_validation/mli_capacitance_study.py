"""Investigation C: MLI RC sensitivity to effective density and cp model.

Rows are plain dicts so the same data feeds tests, plots and the report.
Run ``python -m test.rc_circuit_validation.mli_capacitance_study`` from TOPLAB.
"""

import numpy as np

from test.rc_circuit_validation.mli_rc_studies import (
    SAMPLE_TIMES,
    simulate,
)

DENSITIES = (20.0, 40.0, 60.0, 300.0)
NODE_COUNTS = (1, 5, 10, 20, 50)
REFERENCE_NODES = 100
SOLVER = dict(method="LSODA", tolerances=(1e-8, 1e-10), grid="refined")


def run(node_count, density, initial, cp_model):
    shell = 288.15 if initial == "step" else 250.0
    return simulate(
        node_count,
        density,
        initial=initial,
        shell_temperature=shell,
        cp_model=cp_model,
        **SOLVER,
    )


def incremental_kj(result, time):
    """Cumulative heat minus the initial steady heat flow integrated in time."""
    return result.cumulative_at(time) - result.heat_flow[0] * time / 1000.0


def table(initial, cp_model, densities=DENSITIES, node_counts=NODE_COUNTS):
    rows = []
    for density in densities:
        reference = run(REFERENCE_NODES, density, initial, cp_model)
        for n in node_counts:
            result = run(n, density, initial, cp_model)
            for time in SAMPLE_TIMES:
                cum, cum_ref = result.cumulative_at(time), reference.cumulative_at(time)
                inc, inc_ref = incremental_kj(result, time), incremental_kj(reference, time)
                rows.append(
                    dict(
                        density=density,
                        nodes=n,
                        time=time,
                        cumulative=cum,
                        cumulative_error=cum / cum_ref - 1.0,
                        incremental=inc,
                        incremental_error=(inc - inc_ref) / abs(inc_ref)
                        if abs(inc_ref) > 1e-9
                        else float("nan"),
                        heat_flow=result.heat_flow_at(time),
                        heat_flow_error=result.heat_flow_at(time)
                        / reference.heat_flow_at(time)
                        - 1.0,
                        energy_residual=float(
                            result.stored_energy[-1]
                            - result.stored_energy[0]
                            - (
                                result.cumulative_kj[-1] * 0.0
                                + np.trapezoid(
                                    result.shell_heat_flow - result.heat_flow,
                                    result.times,
                                )
                            )
                        ),
                    )
                )
    return rows


def format_rows(rows, title):
    lines = [title, "rho  N   t[min]  Q[kJ]   dQ%    inc[kJ]  dInc%   q[W]   dq%"]
    for r in rows:
        lines.append(
            f"{r['density']:5.0f} {r['nodes']:4d} {r['time']/60:6.0f} "
            f"{r['cumulative']:9.3f} {100*r['cumulative_error']:8.2f} "
            f"{r['incremental']:9.3f} {100*r['incremental_error']:8.2f} "
            f"{r['heat_flow']:8.3f} {100*r['heat_flow_error']:8.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import time

    for initial in ("step", "steady"):
        for cp_model in ("const", "provisional_pet"):
            start = time.time()
            rows = table(initial, cp_model)
            print(format_rows(rows, f"== {initial} start, cp={cp_model}"))
            print(f"   ({time.time() - start:.0f} s)", flush=True)
