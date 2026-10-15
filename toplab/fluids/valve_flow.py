"""Adiabatic flow through tank -> pipe -> orifice valve -> pipe -> tank.

Model (steady, adiabatic, no shaft work, real-gas CoolProp properties of the fluid
leaving the source tank):

* Stagnation enthalpy is conserved along the whole path: h0 = h + v^2/2 = h_source.
* Each pipe removes stagnation pressure by Darcy-Weisbach, dp0 = (f L/D + K) rho v^2 / 2,
  with rho evaluated at the pipe's static state (low-Mach approximation).
* The orifice is an isentropic real-gas nozzle from its inlet stagnation state, scaled by
  a discharge coefficient and the valve opening. Choking is detected as the maximum of
  rho*v over the available pressure range, so no ideal-gas critical ratio is used.
* The mass flow is the single value for which the orifice flow equals the flow that the
  pipes allow, i.e. the pressure budget P1 - P2 = dp(pipe 1) + dp(orifice) + dp(pipe 2)
  closes. The kinetic energy of the jet is dissipated in the receiving tank, so the
  stream enters at h0.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional

import CoolProp.CoolProp as CP
from scipy.optimize import brentq, minimize_scalar

LAMINAR_RE = 2300.0


@lru_cache(maxsize=None)
def _state(fluid: str):
    # AbstractState avoids PropsSI's per-call string parsing, which dominates the cost here.
    return CP.AbstractState("HEOS", fluid)


@dataclass(frozen=True)
class PipeSpec:
    diameter_m: float
    length_m: float
    friction_factor: Optional[float] = None   # Darcy friction factor; None -> from roughness
    roughness_m: float = 1.5e-6
    loss_coefficient: float = 0.0             # additional minor losses K

    @property
    def area(self) -> float:
        return math.pi * self.diameter_m ** 2 / 4.0


@dataclass(frozen=True)
class ValvePath:
    orifice_diameter_m: float
    discharge_coefficient: float = 0.6
    upstream: Optional[PipeSpec] = None
    downstream: Optional[PipeSpec] = None

    @property
    def orifice_area(self) -> float:
        return math.pi * self.orifice_diameter_m ** 2 / 4.0


def _pipe_from_dict(data: Optional[Mapping[str, Any]], label: str) -> Optional[PipeSpec]:
    if not data:
        return None
    for key in ("diameter_m", "length_m"):
        if key not in data:
            raise ValueError(f"'{label}' requires '{key}'")
    friction = data.get("friction_factor")
    return PipeSpec(
        diameter_m=float(data["diameter_m"]),
        length_m=float(data["length_m"]),
        friction_factor=None if friction is None else float(friction),
        roughness_m=float(data.get("roughness_m", 1.5e-6)),
        loss_coefficient=float(data.get("loss_coefficient", 0.0)),
    )


def path_from_options(options: Mapping[str, Any]) -> ValvePath:
    """Build a ValvePath from an edge's ``valve`` / ``pipe_upstream`` / ``pipe_downstream`` options."""
    valve = options.get("valve") or {}
    if "orifice_diameter_m" in valve:
        diameter = float(valve["orifice_diameter_m"])
    elif "orifice_area_m2" in valve:
        diameter = math.sqrt(4.0 * float(valve["orifice_area_m2"]) / math.pi)
    else:
        raise ValueError("valve requires 'orifice_diameter_m' or 'orifice_area_m2'")
    return ValvePath(
        orifice_diameter_m=diameter,
        discharge_coefficient=float(valve.get("discharge_coefficient", 0.6)),
        upstream=_pipe_from_dict(options.get("pipe_upstream"), "pipe_upstream"),
        downstream=_pipe_from_dict(options.get("pipe_downstream"), "pipe_downstream"),
    )


def _friction_factor(pipe: PipeSpec, reynolds: float) -> float:
    if pipe.friction_factor is not None:
        return pipe.friction_factor
    if reynolds < LAMINAR_RE:
        return 64.0 / max(reynolds, 1e-9)
    rel = pipe.roughness_m / pipe.diameter_m
    return 0.25 / math.log10(rel / 3.7 + 5.74 / reynolds ** 0.9) ** 2  # Swamee-Jain


def _pipe_stagnation_loss(pipe: PipeSpec, mdot: float, p: float, h0: float, fluid: str) -> float:
    """Stagnation pressure lost in the pipe at mass flow ``mdot`` [Pa].

    Returns inf when the flow cannot pass the pipe (Mach >= 1, i.e. the pipe is choked).
    """
    st = _state(fluid)
    h = h0
    try:
        for _ in range(4):
            st.update(CP.HmassP_INPUTS, h, p)
            v = mdot / (st.rhomass() * pipe.area)
            h = h0 - 0.5 * v * v
        st.update(CP.HmassP_INPUTS, h, p)
        rho = st.rhomass()
        v = mdot / (rho * pipe.area)
        if v >= st.speed_sound():
            return math.inf
        mu = st.viscosity()
    except ValueError:
        return math.inf
    reynolds = rho * v * pipe.diameter_m / mu
    f = _friction_factor(pipe, reynolds)
    return (f * pipe.length_m / pipe.diameter_m + pipe.loss_coefficient) * 0.5 * rho * v * v


def _nozzle_mass_flux(p_in: float, h0: float, p_back: float, fluid: str) -> float:
    """Mass flux rho*v [kg/m^2/s] of an isentropic real-gas expansion p_in -> p_back (choked if limited)."""
    if p_back >= p_in:
        return 0.0
    st = _state(fluid)
    st.update(CP.HmassP_INPUTS, h0, p_in)
    s0 = st.smass()

    def flux(p: float) -> float:
        st.update(CP.PSmass_INPUTS, p, s0)
        return st.rhomass() * math.sqrt(max(0.0, 2.0 * (h0 - st.hmass())))

    result = minimize_scalar(lambda p: -flux(p), bounds=(p_back, p_in), method="bounded",
                             options={"xatol": 1e-4 * p_in})
    return max(-result.fun, flux(p_back))


def valve_mass_flow(path: ValvePath, fluid: str, p_source: float, temperature: float,
                    density: float, p_target: float, opening: float = 1.0) -> float:
    """Mass flow [kg/s] from the source tank state to a target tank at ``p_target``.

    ``opening`` in [0, 1] scales the orifice area.
    """
    if opening <= 0.0 or p_source <= p_target:
        return 0.0
    st = _state(fluid)
    st.update(CP.DmassT_INPUTS, density, temperature)
    h0 = st.hmass()
    area = path.discharge_coefficient * opening * path.orifice_area

    def orifice_flow(mdot: float) -> float:
        p_in = p_source
        if path.upstream is not None:
            p_in -= _pipe_stagnation_loss(path.upstream, mdot, p_source, h0, fluid)
        p_back = p_target
        if path.downstream is not None:
            p_back += _pipe_stagnation_loss(path.downstream, mdot, p_target, h0, fluid)
        if p_back >= p_in:
            return 0.0
        return area * _nozzle_mass_flux(p_in, h0, p_back, fluid)

    upper = orifice_flow(0.0)   # lossless limit: no pipe loss at zero flow
    if path.upstream is None and path.downstream is None:
        return upper
    if orifice_flow(upper) >= upper:
        return upper
    return brentq(lambda m: orifice_flow(m) - m, 0.0, upper, xtol=1e-9 * upper + 1e-15, rtol=1e-6)
