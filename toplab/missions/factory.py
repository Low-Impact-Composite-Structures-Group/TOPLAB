"""Build :class:`Mission` objects from typed :class:`MissionSpec` entries."""

from __future__ import annotations

from pathlib import Path

from toplab.config.schema import MissionSpec
from toplab.missions.mission import Mission
from toplab.missions.mission_sections import InFlow, MissionSection, OutFlow

_NAMED_PROFILES = {
    name: getattr(Mission, name)
    for name in ("atr72", "triathlon", "rompokos")
    if hasattr(Mission, name)
}


def build_mission(spec: MissionSpec) -> Mission:
    profile = spec.profile.strip().lower()
    if profile in _NAMED_PROFILES:
        return _NAMED_PROFILES[profile]()
    if profile == "csv":
        return _csv_mission(spec)
    if profile in ("constant_flow", "sequential_constant_flow"):
        return _constant_flow_mission(spec)
    supported = ", ".join([*_NAMED_PROFILES, "csv", "constant_flow"])
    raise ValueError(f"Unknown mission profile: '{spec.profile}'. Supported profiles: {supported}.")


def _csv_mission(spec: MissionSpec) -> Mission:
    params = spec.parameters
    if not params.get("csv_file"):
        raise ValueError("CSV mission profile requires 'csv_file' in the mission 'parameters' block.")
    return Mission.from_csv(
        csv_path=str(Path(params["csv_file"])),
        cruise_altitude=params.get("cruise_altitude", 7010.0),
        standard_temperature=params.get("standard_temperature", 273.15),
        mach_number=params.get("mach_number", 0.0),
        phase=params.get("phase", "gas"),
        ambient_temperature=spec.ambient_temperature,
        column_name=params.get("column_name"),
    )


def _constant_flow_mission(spec: MissionSpec) -> Mission:
    opts = spec.options
    flow_rate = float(opts.get("flow_rate", 0.001))
    duration = float(opts.get("duration", 36000))
    key = opts.get("key", spec.type)

    kind = spec.type.lower()
    if kind == "refuel":
        from toplab.fluids.hydrogen_retrievers import SinglePhaseRequester
        # Dewar conditions; the cryopump enthalpy is recomputed dynamically.
        dewar = SinglePhaseRequester().get_hydrogen_properties(3e5, 20.4)
        flow = InFlow(abs(flow_rate), dewar)
    elif kind == "dormancy":
        flow = OutFlow(0.0, "gas")
    else:
        flow = OutFlow(-abs(flow_rate), "gas")

    # Ground level, zero Mach: tank-only operations.
    return Mission([MissionSection(duration, [flow], 0.0, 0.0, key, spec.ambient_temperature)])
