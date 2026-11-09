"""Dormancy-mission harness for the N-node MLI RC network (validation only).

Reuses the production dormancy scaffolding (``load_config`` + dormancy
``MissionSpec`` + ``SystemOrchestrator``) to obtain the production tank, thermal
model, dynamic model and initial state. The right-hand side is then rebuilt with
N insulation states; hydrogen mass, temperature, pressure and venting still come
from the production ``dynamic_models[0].compute_state_derivatives``. With N=1
and the production insulation properties the right-hand side reproduces
``TankSystem._rhs`` (checked in the tests).

State: [m, T_H2, T_struct, T_ins_1 (shell side) .. T_ins_N (structure side), T_shell].
"""

import copy
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from toplab.config import MissionSpec, load_config
from toplab.orchestration.system_orchestrator import SystemOrchestrator
from toplab.thermodynamics.insulation_rc_diagnostic import InsulationRCModel
from toplab.system.state_management import MultiTankState

CONFIG = (
    Path(__file__).resolve().parents[3]
    / "toplab-apps/analysis/novacom/slh2_baseline/config_slh2_baseline.yaml"
)
DURATION = 12 * 3600.0


def build_case(duration=DURATION, insulation=None):
    """Production dormancy case; ``insulation`` overrides keys of the YAML insulation block."""
    cfg = load_config(str(CONFIG))
    if insulation:
        tank = cfg.tanks[0]
        materials = {
            **tank.materials,
            "insulation": {**tank.materials["insulation"], **insulation},
        }
        cfg = dataclasses.replace(
            cfg, tanks=[dataclasses.replace(tank, materials=materials)]
        )
    mission = MissionSpec(
        type="dormancy",
        profile="constant_flow",
        ambient_temperature=288.15,
        assigned_to_node=cfg.missions[0].assigned_to_node
        if cfg.missions
        else cfg.tank_ids[0],
        options={"flow_rate": 0.0, "duration": duration},
    )
    case = dataclasses.replace(
        cfg,
        missions=[mission],
        output={**cfg.output, "save_plots": False, "save_data": False, "silent": True},
        dormancy_check=None,
        solver={**cfg.solver, "max_step": 300.0},
    )
    return SystemOrchestrator(case, verbosity="quiet", sizing_config=cfg)


@dataclass
class DormancyResult:
    times: np.ndarray
    y: np.ndarray
    pressure: np.ndarray
    heat_to_h2: np.ndarray
    vent_flow: np.ndarray
    cumulative_h2_heat_kj: np.ndarray
    cpu_seconds: float
    node_count: int
    node_masses: np.ndarray
    heat_to_structure: np.ndarray = None

    @property
    def vented_mass(self):
        return float(self.y[0, 0] - self.y[0, -1])

    @property
    def vent_onset_hours(self):
        idx = np.nonzero(self.vent_flow > 0.0)[0]
        return float(self.times[idx[0]] / 3600.0) if idx.size else float("nan")


class DormancyHarness:
    def __init__(self, density=None, cp=None, duration=DURATION, thickness=None):
        self.orchestrator = build_case(
            duration,
            {"thickness": thickness} if thickness is not None else None,
        )
        self.system = self.orchestrator.tank_system
        self.duration = duration
        production = self.system.thermal_models[0]
        self.production_thermal = production
        self.thermal = copy.copy(production)
        if density is not None:
            scale = density / production.insulation_model.density
            model = dataclasses.replace(production.insulation_model, density=density)
            self.thermal.m_foam = production.m_foam * scale
        else:
            model = production.insulation_model
        if cp is not None:
            model = dataclasses.replace(model, specific_heat=cp)
        self.thermal.insulation_model = model
        self.dynamic = self.system.dynamic_models[0]
        self.tank = self.system.tanks[0]
        self.fluids = [self.system._coolprop_fluids[0]]
        self.initial_state = self.system.create_initial_state()[:5]

    def rc_model(self, node_count):
        return InsulationRCModel(self.thermal, node_count)

    def steady_profile(self, rc_model, structure_temperature, shell_temperature=None):
        shell = (
            self.initial_state[4] if shell_temperature is None else shell_temperature
        )
        n = rc_model.node_count
        guess = np.linspace(shell, structure_temperature, n + 2)[1:-1]

        def balance(temperatures):
            flows = rc_model.heat_flows(shell, temperatures, structure_temperature)
            return flows[:-1] - flows[1:]

        if n == 1:
            return np.array(
                [
                    self.thermal.determine_initial_insulation_temperature(
                        structure_temperature, shell
                    )
                ]
            )
        solution = root(balance, guess, tol=1e-12)
        assert solution.success, solution.message
        return solution.x

    def starting_vector(self, rc_model, insulation="equilibrated"):
        m, t_h2, t_struct, _, t_shell = self.initial_state
        if insulation == "equilibrated":
            nodes = self.steady_profile(rc_model, t_struct)
        elif insulation == "post_refuel":
            nodes = self.steady_profile(rc_model, POST_REFUEL_STRUCTURE_TEMPERATURE)
        elif insulation == "uniform_warm":
            nodes = np.full(rc_model.node_count, t_shell)
        else:
            raise ValueError(insulation)
        return np.concatenate(([m, t_h2, t_struct], nodes, [t_shell]))

    def derivatives(self, rc_model, t, y, with_diagnostics=False):
        n = rc_model.node_count
        m, t_h2, t_struct = y[0], y[1], y[2]
        nodes, t_shell = y[3 : 3 + n], y[3 + n]
        lumped = float(rc_model.node_masses @ nodes / rc_model.node_masses.sum())
        state = MultiTankState.from_state_vector(
            np.array([m, t_h2, t_struct, lumped, t_shell]),
            [self.tank],
            t,
            fluids=self.fluids,
        ).tank_states[0]

        flows = rc_model.heat_flows(t_shell, nodes, t_struct)
        q_structure_to_h2 = self.thermal.compute_structure_to_h2_heat_flux(t, state)
        bounded = max(4.0, min(t_struct, 400.0))
        structure_capacity = self.thermal.m_liner * float(
            self.thermal.liner_material.determine_specific_heat(bounded)
        ) + self.thermal.m_wall * float(
            self.thermal.wall_material.determine_specific_heat(bounded)
        )
        shell_bounded = max(4.0, min(t_shell, 400.0))
        shell_capacity = self.thermal.m_shell * float(
            self.thermal.shell_material.determine_specific_heat(shell_bounded)
        )
        d_struct = (flows[-1] - q_structure_to_h2) / structure_capacity
        d_shell = (
            self.thermal.compute_ambient_to_shell_heat_flux(t_shell) - flows[0]
        ) / shell_capacity
        d_nodes = rc_model.temperature_derivatives(t_shell, nodes, t_struct)
        derivatives = self.dynamic.compute_state_derivatives(
            time=t,
            state=state,
            edge_flows=[],
            Q_structure_to_h2=q_structure_to_h2,
            dT_structure_dt=d_struct,
            dT_insulation_dt=float(rc_model.node_masses @ d_nodes / rc_model.node_masses.sum()),
            dT_shell_dt=d_shell,
            Q_discharge=0.0,
            tank_index=0,
        )
        dy = np.concatenate(
            (
                [derivatives.fuel_mass_derivative, derivatives.h2_temperature_derivative, d_struct],
                d_nodes,
                [d_shell],
            )
        )
        if with_diagnostics:
            return dy, dict(
                pressure=state.pressure,
                q_structure_to_h2=q_structure_to_h2,
                vent_flow=derivatives.vent_flow,
            )
        return dy

    def run(self, node_count, insulation="equilibrated", output_step=600.0):
        rc_model = self.rc_model(node_count)
        y0 = self.starting_vector(rc_model, insulation)
        times = np.arange(0.0, self.duration + 0.5 * output_step, output_step)
        start = time.perf_counter()
        solution = solve_ivp(
            lambda t, y: self.derivatives(rc_model, t, y),
            (0.0, self.duration),
            y0,
            method="LSODA",
            t_eval=times,
            rtol=1e-6,
            atol=1e-9,
            max_step=300.0,
        )
        cpu = time.perf_counter() - start
        assert solution.success, solution.message
        diagnostics = [
            self.derivatives(rc_model, t, y, with_diagnostics=True)[1]
            for t, y in zip(solution.t, solution.y.T)
        ]
        heat = np.array([d["q_structure_to_h2"] for d in diagnostics])
        return DormancyResult(
            times=solution.t,
            y=solution.y,
            pressure=np.array([d["pressure"] for d in diagnostics]),
            heat_to_h2=heat,
            vent_flow=np.array([d["vent_flow"] for d in diagnostics]),
            cumulative_h2_heat_kj=np.concatenate(
                ([0.0], np.cumsum(0.5 * (heat[1:] + heat[:-1]) * np.diff(solution.t)))
            )
            / 1000.0,
            cpu_seconds=cpu,
            node_count=node_count,
            node_masses=rc_model.node_masses,
            heat_to_structure=np.array(
                [rc_model.heat_flows(y[3 + node_count], y[3 : 3 + node_count], y[2])[-1] for y in solution.y.T]
            ),
        )


QUASI_STEADY_MODELS = ("B", "C")


def _quasi_steady_methods():
    """Models B and C (validation harness): no insulation thermal state; state = [m, T_H2, T_struct, T_shell]."""

    def derivatives(self, model, t, y, with_diagnostics=False):
        m, t_h2, t_struct, t_shell = y
        thermal = self.thermal
        if model == "C":
            q_blanket = thermal.compute_blanket_heat_flow(t_shell, t_struct)
        elif model == "B":
            t_ins = thermal.reconstruct_insulation_temperature(t_shell, t_struct)
            q_blanket = thermal.compute_shell_to_insulation_heat_flux(t_shell, t_ins)
        else:
            raise ValueError(model)
        state = MultiTankState.from_state_vector(
            np.array([m, t_h2, t_struct, 0.5 * (t_shell + t_struct), t_shell]),
            [self.tank],
            t,
            fluids=self.fluids,
        ).tank_states[0]
        q_structure_to_h2 = thermal.compute_structure_to_h2_heat_flux(t, state)
        structure_capacity, shell_capacity = self._capacities(t_struct, t_shell)
        d_struct = (q_blanket - q_structure_to_h2) / structure_capacity
        d_shell = (thermal.compute_ambient_to_shell_heat_flux(t_shell) - q_blanket) / shell_capacity
        derivatives = self.dynamic.compute_state_derivatives(
            time=t,
            state=state,
            edge_flows=[],
            Q_structure_to_h2=q_structure_to_h2,
            dT_structure_dt=d_struct,
            dT_insulation_dt=0.0,
            dT_shell_dt=d_shell,
            Q_discharge=0.0,
            tank_index=0,
        )
        dy = np.array(
            [
                derivatives.fuel_mass_derivative,
                derivatives.h2_temperature_derivative,
                d_struct,
                d_shell,
            ]
        )
        if with_diagnostics:
            return dy, dict(
                pressure=state.pressure,
                q_structure_to_h2=q_structure_to_h2,
                q_blanket=q_blanket,
                vent_flow=derivatives.vent_flow,
            )
        return dy

    def run_quasi_steady(self, model, insulation="equilibrated", output_step=600.0):
        m, t_h2, t_struct, _, t_shell = self.initial_state
        y0 = np.array([m, t_h2, t_struct, t_shell])
        times = np.arange(0.0, self.duration + 0.5 * output_step, output_step)
        start = time.perf_counter()
        solution = solve_ivp(
            lambda t, y: self.quasi_steady_derivatives(model, t, y),
            (0.0, self.duration),
            y0,
            method="LSODA",
            t_eval=times,
            rtol=1e-6,
            atol=1e-9,
            max_step=300.0,
        )
        cpu = time.perf_counter() - start
        assert solution.success, solution.message
        diagnostics = [
            self.quasi_steady_derivatives(model, t, y, with_diagnostics=True)[1]
            for t, y in zip(solution.t, solution.y.T)
        ]
        heat = np.array([d["q_structure_to_h2"] for d in diagnostics])
        return DormancyResult(
            times=solution.t,
            y=solution.y[[0, 1, 2, 3]],
            pressure=np.array([d["pressure"] for d in diagnostics]),
            heat_to_h2=heat,
            vent_flow=np.array([d["vent_flow"] for d in diagnostics]),
            cumulative_h2_heat_kj=np.concatenate(
                ([0.0], np.cumsum(0.5 * (heat[1:] + heat[:-1]) * np.diff(solution.t)))
            )
            / 1000.0,
            cpu_seconds=cpu,
            node_count=0,
            node_masses=np.zeros(0),
            heat_to_structure=np.array([d["q_blanket"] for d in diagnostics]),
        )

    def capacities(self, t_struct, t_shell):
        thermal = self.thermal
        bounded = max(4.0, min(t_struct, 400.0))
        structure = thermal.m_liner * float(
            thermal.liner_material.determine_specific_heat(bounded)
        ) + thermal.m_wall * float(thermal.wall_material.determine_specific_heat(bounded))
        shell = thermal.m_shell * float(
            thermal.shell_material.determine_specific_heat(max(4.0, min(t_shell, 400.0)))
        )
        return structure, shell

    return derivatives, run_quasi_steady, capacities


(
    DormancyHarness.quasi_steady_derivatives,
    DormancyHarness.run_quasi_steady,
    DormancyHarness._capacities,
) = _quasi_steady_methods()


def run_production(thermal_mode="transient_lumped", thickness=None, density=None, duration=DURATION, output_step=600.0):
    """Run the production dormancy case (optionally with the quasi-steady MLI mode)."""
    insulation = {"thermal_mode": thermal_mode}
    if thickness is not None:
        insulation["thickness"] = thickness
    if density is not None:
        insulation["density"] = density
    orchestrator = build_case(duration, insulation)
    start = time.perf_counter()
    results = orchestrator.run_simulation()
    cpu = time.perf_counter() - start
    states = [s.get_tank_state(0) for s in results.multi_tank_states]
    mass = np.array([s.fuel_mass for s in states])
    return dict(
        times=np.asarray(results.times),
        mass=mass,
        vented=mass[0] - mass,
        h2_temperature=np.array([s.h2_temperature for s in states]),
        structure_temperature=np.array([s.structure_temperature for s in states]),
        shell_temperature=np.array([s.shell_temperature for s in states]),
        insulation_temperature=np.array([s.insulation_temperature for s in states]),
        pressure=np.array([s.pressure for s in states]),
        cpu_seconds=cpu,
        thermal_model=orchestrator.tank_system.thermal_models[0],
        states=states,
    )


# Assumption: a warm structure (150 K) before refuelling; the insulation holds
# the corresponding steady profile when the tank is suddenly chilled to 25 K.
POST_REFUEL_STRUCTURE_TEMPERATURE = 150.0
