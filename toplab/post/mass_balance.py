"""Mass-balance validation of simulation results."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TankMassBalance:
    tank_index: int
    initial_mass: float      # kg
    final_mass: float        # kg
    net_inflow: float        # kg, integral of all recorded inflows minus outflows
    residual: float          # kg, (final - initial) - net_inflow
    relative_residual: float # residual / initial mass

    def passed(self, rtol: float) -> bool:
        return abs(self.relative_residual) <= rtol


def check_mass_balance(results) -> List[TankMassBalance]:
    """
    Per tank: m(t_end) - m(0) must equal the integral of recorded inflow - outflow
    - vent + coupling inflow - coupling outflow. Flows in the results are stored in g/s.
    """
    data = results.get_combined_data()
    t = np.asarray(data["times"], dtype=float)
    checks = []
    for i in range(results.n_tanks):
        net_gs = (data["inflow_rates"][i] - data["outflow_rates"][i] - data["vent_rates"][i]
                  + data["coupling_inflow_rates"][i] - data["coupling_outflow_rates"][i])
        net = float(np.trapz(net_gs, t)) / 1000.0
        m0, m1 = float(data["masses"][i][0]), float(data["masses"][i][-1])
        residual = (m1 - m0) - net
        checks.append(TankMassBalance(i, m0, m1, net, residual, residual / max(m0, 1e-12)))
    return checks
