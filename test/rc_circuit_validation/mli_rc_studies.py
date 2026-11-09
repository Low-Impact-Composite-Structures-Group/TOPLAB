"""Shared model and cached transient studies for the MLI RC validation.

Used by ``test_insulation_rc_mli.py`` (assertions) and
``plot_mli_rc_convergence.py`` (figures) so both always use identical cases.
Only the validation-only ``InsulationRCModel`` is exercised; production code is
not modified.
"""

import dataclasses
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.optimize import root

from toplab.materials.insulation_models import make_modified_lockheed_model
from toplab.materials.nist_materials import NISTComposite, NISTMetal
from toplab.thermodynamics.insulation_rc_diagnostic import InsulationRCModel
from toplab.thermodynamics.isochoric_thermal_model import InsulatedTankThermalModel


L_CYLINDER = 1.5
T_SHELL_BOUNDARY = 288.15
T_STRUCTURE_BOUNDARY = 60.0
T_SHELL_PERTURBED = 250.0
MLI_THICKNESS = 0.02
DURATION = 3.0 * 3600.0
DEFAULT_DENSITY = make_modified_lockheed_model().density
DEFAULT_SPECIFIC_HEAT = 1000.0
SAMPLE_TIMES = (600.0, 3600.0, 10800.0)

PRODUCTION_TOLERANCES = (1e-6, 1e-9)
REFINED_TOLERANCES = (1e-9, 1e-10)
TIGHT_TOLERANCES = (1e-11, 1e-12)


def _refined_grid(duration=DURATION):
    return np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                np.geomspace(1.0e-4, 1.0, 121),
                np.geomspace(1.0, 60.0, 161),
                np.linspace(60.0, 600.0, 181),
                np.linspace(600.0, duration, 511),
                np.array([duration]),
            )
        )
    )


def _very_fine_grid(duration=DURATION):
    return np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                np.geomspace(1.0e-5, 1.0, 401),
                np.geomspace(1.0, 60.0, 481),
                np.linspace(60.0, 600.0, 541),
                np.linspace(600.0, duration, 1531),
            )
        )
    )


OUTPUT_GRIDS = {
    "refined": _refined_grid(),
    "very_fine": _very_fine_grid(),
}
UNIFORM_STEPS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
for _step in UNIFORM_STEPS:
    OUTPUT_GRIDS[f"uniform_{_step:g}s"] = np.arange(0.0, DURATION + 0.5 * _step, _step)


def layer_mass(density, inner_radius, outer_radius):
    volume = (
        math.pi * (outer_radius**2 - inner_radius**2) * L_CYLINDER
        + (4.0 * math.pi / 3.0) * (outer_radius**3 - inner_radius**3)
    )
    return density * volume


def make_model(insulation=None):
    insulation = insulation or make_modified_lockheed_model()
    liner = NISTMetal.aluminum_6061T6_nist()
    wall = NISTComposite.carbon_epoxy_nist()
    r_inner = 0.5
    r_structure = r_inner + 0.003 + 0.015
    r_shell = r_structure + MLI_THICKNESS
    return InsulatedTankThermalModel(
        tank_volume=math.pi * r_inner**3 * (4.0 / 3.0 + L_CYLINDER / r_inner),
        inner_surface_area=4.0 * math.pi * r_inner**2
        + 2.0 * math.pi * r_inner * L_CYLINDER,
        inner_diameter=2.0 * r_inner,
        r_structure=r_structure,
        r_shell=r_shell,
        cylinder_length=L_CYLINDER,
        liner_mass=layer_mass(liner.density, r_inner, r_inner + 0.003),
        wall_mass=layer_mass(wall.density, r_inner + 0.003, r_structure),
        foam_mass=layer_mass(insulation.density, r_structure, r_shell),
        shell_mass=layer_mass(liner.density, r_shell, r_shell + 0.002),
        ambient_temperature=T_SHELL_BOUNDARY,
        alpha_amb=5.0,
        emissivity_shell=0.05,
        liner_material=liner,
        wall_material=wall,
        shell_material=liner,
        insulation_model=insulation,
    )


def steady_temperatures(rc_model, shell_temperature=T_SHELL_BOUNDARY):
    initial = np.linspace(
        shell_temperature, T_STRUCTURE_BOUNDARY, rc_model.node_count + 2
    )[1:-1]

    def balance(temperatures):
        flows = rc_model.heat_flows(
            shell_temperature, temperatures, T_STRUCTURE_BOUNDARY
        )
        return flows[:-1] - flows[1:]

    solution = root(balance, initial, tol=1e-13)
    assert solution.success, solution.message
    return solution.x


@dataclass(frozen=True)
class Transient:
    times: np.ndarray
    temperatures: np.ndarray  # shape (node_count, n_times), shell side first
    shell_heat_flow: np.ndarray
    heat_flow: np.ndarray  # into the structure
    cumulative_kj: np.ndarray
    node_masses: np.ndarray
    node_radii: np.ndarray
    stored_energy: np.ndarray

    def mean_temperature(self):
        return self.node_masses @ self.temperatures / self.node_masses.sum()

    def cumulative_at(self, time):
        return float(np.interp(time, self.times, self.cumulative_kj))

    def heat_flow_at(self, time):
        return float(np.interp(time, self.times, self.heat_flow))


@lru_cache(maxsize=None)
def simulate(
    node_count,
    density=DEFAULT_DENSITY,
    initial="step",
    shell_temperature=T_SHELL_BOUNDARY,
    grid="refined",
    tolerances=REFINED_TOLERANCES,
    max_step=300.0,
    cp_model="const",
    method="DOP853",
):
    """Integrate the N-node MLI network for DURATION seconds.

    ``initial="step"``: every node at the shell temperature against the cold
    structure (stress test). ``initial="steady"``: start from the converged
    steady profile at the nominal boundaries, then hold ``shell_temperature``.
    """
    insulation = make_modified_lockheed_model(
        density=density, specific_heat=DEFAULT_SPECIFIC_HEAT
    )
    if cp_model == "provisional_pet":
        from test.rc_circuit_validation.mli_capacitance import (
            QUARTER_MIL,
            TabulatedSpecificHeat,
        )

        tabulated = TabulatedSpecificHeat(QUARTER_MIL)
        insulation = dataclasses.replace(insulation, specific_heat=tabulated)
    elif cp_model != "const":
        raise ValueError(cp_model)
    rc_model = InsulationRCModel(make_model(insulation), node_count)
    if initial == "step":
        start = np.full(node_count, T_SHELL_BOUNDARY)
    elif initial == "steady":
        start = steady_temperatures(rc_model)
    else:
        raise ValueError(initial)

    times = OUTPUT_GRIDS[grid]
    rtol, atol = tolerances
    solution = solve_ivp(
        lambda _t, y: rc_model.temperature_derivatives(
            shell_temperature, y, T_STRUCTURE_BOUNDARY
        ),
        (0.0, DURATION),
        start,
        method=method,
        t_eval=times,
        rtol=rtol,
        atol=atol,
        **({"max_step": max_step} if method == "DOP853" else {}),
    )
    assert solution.success, solution.message

    flows = np.asarray(
        [
            rc_model.heat_flows(shell_temperature, y, T_STRUCTURE_BOUNDARY)
            for y in solution.y.T
        ]
    )
    return Transient(
        times=solution.t,
        temperatures=solution.y,
        shell_heat_flow=flows[:, 0],
        heat_flow=flows[:, -1],
        cumulative_kj=cumulative_simpson(flows[:, -1], x=solution.t, initial=0.0)
        / 1000.0,
        node_masses=np.asarray(rc_model.node_masses),
        node_radii=np.asarray(rc_model.node_radii),
        stored_energy=(
            np.asarray([rc_model.energy(y) for y in solution.y.T])
            if cp_model == "const"
            else np.asarray(
                [
                    float(rc_model.node_masses @ tabulated.enthalpy(y))
                    for y in solution.y.T
                ]
            )
        ),
    )


def metadata(extra=""):
    return (
        f"MLI, {1000.0 * MLI_THICKNESS:g} mm; fixed structure "
        f"{T_STRUCTURE_BOUNDARY:g} K; DOP853"
        + (f"; {extra}" if extra else "")
    )
