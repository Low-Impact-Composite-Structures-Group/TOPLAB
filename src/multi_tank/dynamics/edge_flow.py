"""
EdgeFlow — directed mass-flow primitive for the multi-tank network graph.

Every flow in the system (discharge, venting, coupling, refuel) is represented
as an EdgeFlow.  The sign convention is absolute: ``mdot`` is always ≥ 0 and
the direction is given by ``from_node`` / ``to_node``.

Node index encoding
-------------------
* ≥ 0  — a tank in the system (matches the tank-list index used in TankSystem)
* -1   — the external environment (fuel cell / atmosphere / supply)

Edge types
----------
coupling   inter-tank mass transfer driven by a valve or coupling model
discharge  outflow to an external consumer (from_node = tank, to_node = -1)
vent       pressure-relief outflow      (from_node = tank, to_node = -1)
refuel     inflow from external supply  (from_node = -1,   to_node = tank)

Author: Dante Raso
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EdgeFlow:
    """A directed mass-flow across one edge of the tank network."""

    mdot: float
    """Mass flow rate [kg/s].  Always ≥ 0; direction encoded by from/to nodes."""

    h: float
    """Specific enthalpy of the flowing fluid at the boundary [J/kg]."""

    edge_type: str
    """One of: 'coupling', 'discharge', 'vent', 'refuel'."""

    from_node: int
    """Source node index (≥ 0 = tank index, -1 = external)."""

    to_node: int
    """Destination node index (≥ 0 = tank index, -1 = external)."""

    def is_inflow_for(self, tank_index: int) -> bool:
        """Return True when this edge delivers mass INTO the given tank."""
        return self.to_node == tank_index

    def is_outflow_for(self, tank_index: int) -> bool:
        """Return True when this edge removes mass FROM the given tank."""
        return self.from_node == tank_index

    def mass_contribution(self, tank_index: int) -> float:
        """Signed mdot contribution to tank ``tank_index`` [kg/s].

        Positive = mass entering the tank, negative = mass leaving.
        Returns 0.0 if this edge does not touch the tank.
        """
        if self.is_inflow_for(tank_index):
            return self.mdot
        if self.is_outflow_for(tank_index):
            return -self.mdot
        return 0.0

    def enthalpy_contribution(self, tank_index: int, h_tank: float) -> float:
        """Energy contribution to the enthalpy-flux term for ``tank_index`` [W].

        This is the ``mdot * (h_edge - h_tank)`` term that appears in the
        isochoric energy balance.  Outflows carry the tank's own enthalpy, so
        their contribution is exactly zero regardless of ``self.h``.
        """
        if self.is_inflow_for(tank_index):
            return self.mdot * (self.h - h_tank)
        if self.is_outflow_for(tank_index):
            return -self.mdot * (self.h - h_tank)
        return 0.0
