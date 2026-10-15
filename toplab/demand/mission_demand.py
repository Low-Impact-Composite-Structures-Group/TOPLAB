"""Demand loading from mission profiles."""

from __future__ import annotations

from typing import Type

from toplab.missions.mission_sections import FuelFlow, InFlow, OutFlow

from .demand import Demand, PiecewiseDemand


def _flow_samples(flow: FuelFlow) -> list:
    mf = flow.mass_flow
    return list(mf) if isinstance(mf, (list, tuple)) else [mf]


def demand_from_mission(mission, flow_class: Type[FuelFlow] = OutFlow) -> Demand:
    """
    Demand carried by the `flow_class` (OutFlow or InFlow) entries of a Mission.
    Where a section holds several flows of that class, the first one is used.
    """
    durations, values = [], []
    for section in mission.sections:
        flow = next((f for f in section.fuel_flows if isinstance(f, flow_class)), None)
        durations.append(section.duration)
        values.append(_flow_samples(flow) if flow is not None else [0.0])
    return PiecewiseDemand(durations, values)


def outflow_demand(mission) -> Demand:
    return demand_from_mission(mission, OutFlow)


def inflow_demand(mission) -> Demand:
    return demand_from_mission(mission, InFlow)
