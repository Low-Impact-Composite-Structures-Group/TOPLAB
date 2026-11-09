"""Foam (Rohacell) RC spatial-convergence case from an equilibrated start.

Each N-node network starts at its own converged steady profile for a 288.15 K shell and a
60 K structure, then the shell boundary is held at ``T_SHELL_PERTURBED`` (an equilibrated
start with unchanged boundaries would have no transient). Structure and shell are fixed
boundaries; only the insulation nodes are integrated.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from test.rc_circuit_validation.test_insulation_rc_initialization import (
    _T_INSULATION,
    _T_SHELL_BOUNDARY,
    _T_STRUCTURE_BOUNDARY,
    _make_model,
    _steady_temperatures,
)
from toplab.thermodynamics.insulation_rc_diagnostic import InsulationRCModel

NODE_COUNTS = (1, 2, 5, 10, 20, 40)
T_SHELL_PERTURBED = 250.0
DURATION = 3.0 * 3600.0
SAMPLE_TIMES = (600.0, 3600.0, 10800.0)
OUTPUT_TIMES = np.linspace(0.0, DURATION, 721)
TOLERANCES = (1e-9, 1e-10)


@dataclass(frozen=True)
class FoamTransient:
    times: np.ndarray
    temperatures: np.ndarray  # (node_count, n_times), shell side first
    heat_flow: np.ndarray  # into the structure [W]
    cumulative_kj: np.ndarray
    energy_residual_kj: np.ndarray  # stored-energy change minus net heat in

    def cumulative_at(self, time):
        return float(np.interp(time, self.times, self.cumulative_kj))

    def heat_flow_at(self, time):
        return float(np.interp(time, self.times, self.heat_flow))


@lru_cache(maxsize=None)
def simulate(node_count, shell_temperature=T_SHELL_PERTURBED):
    rc_model = InsulationRCModel(_make_model(), node_count)
    start = _steady_temperatures(rc_model, _T_SHELL_BOUNDARY, _T_STRUCTURE_BOUNDARY)
    rtol, atol = TOLERANCES

    # Heat integrals are extra ODE states so they are integrated adaptively; quadrature on the
    # output grid is inaccurate for the thin-node flow spike right after the shell step.
    def rhs(_t, y):
        nodes = y[:node_count]
        flows = rc_model.heat_flows(shell_temperature, nodes, _T_STRUCTURE_BOUNDARY)
        derivative = rc_model.temperature_derivatives(
            shell_temperature, nodes, _T_STRUCTURE_BOUNDARY
        )
        return np.concatenate((derivative, [flows[0], flows[-1]]))

    solution = solve_ivp(
        rhs,
        (0.0, DURATION),
        np.concatenate((start, [0.0, 0.0])),
        method="DOP853",
        t_eval=OUTPUT_TIMES,
        rtol=rtol,
        atol=atol,
        max_step=300.0,
    )
    assert solution.success, solution.message
    nodes = solution.y[:node_count]
    shell_in, structure_out = solution.y[node_count], solution.y[node_count + 1]
    flows = np.asarray(
        [
            rc_model.heat_flows(shell_temperature, y, _T_STRUCTURE_BOUNDARY)[-1]
            for y in nodes.T
        ]
    )
    stored = np.asarray([rc_model.energy(y) for y in nodes.T])
    return FoamTransient(
        times=solution.t,
        temperatures=nodes,
        heat_flow=flows,
        cumulative_kj=structure_out / 1000.0,
        energy_residual_kj=(stored - stored[0] - (shell_in - structure_out)) / 1000.0,
    )


def metadata():
    return (
        f"Foam (Rohacell), {1000.0 * _T_INSULATION:g} mm; equilibrated at "
        f"{_T_SHELL_BOUNDARY:g} K shell / {_T_STRUCTURE_BOUNDARY:g} K structure, then shell "
        f"held at {T_SHELL_PERTURBED:g} K; 3 h; DOP853 rtol={TOLERANCES[0]:g}"
    )
