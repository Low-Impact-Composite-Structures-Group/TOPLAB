from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from CoolProp.CoolProp import PropsSI

from toplab.coupling.inter_tank_coupling import InterTankCoupling
from toplab.configuration.scenario_configuration import ScenarioConfig
from toplab.orchestration.system_orchestrator import SystemOrchestrator


class ConstantFlowBenchmarkCoupling(InterTankCoupling):
    def __init__(self, source_idx: int, target_idx: int, flow_rate: float):
        super().__init__(source_idx, target_idx, "constant_flow_benchmark")
        self.flow_rate = flow_rate
        self.source_tank = source_idx
        self.target_tank = target_idx

    def evaluate(self, time_s, source_tank, dest_tank):
        return True

    def calculate_flow_rate(self, t: float, tank_states):
        return self.flow_rate

    def calculate_flow(self, source_state, target_state, t: float):
        if source_state.fuel_mass <= self.flow_rate * 10.0:
            return 0.0
        return self.flow_rate


def build_component_benchmark_config(component_type: str) -> dict:
    flow_rate = 0.05
    duration_s = 250.0

    edge = {
        "edge_id": f"{component_type}_benchmark",
        "connection_type": "pressure_compensation",
        "from_node": 1,
        "to_node": 2,
        "activation_conditions": {
            "pressure_open_bar": 1000.0,
            "pressure_close_bar": 1001.0,
        },
        "flow_parameters": {
            "max_flow_rate_kg_s": flow_rate,
            "orifice_diameter_m": 0.02,
        },
        "peripheral_components": [],
    }

    if component_type == "compressor":
        source_pressure = 100e5
        source_temperature = 290.0
        target_pressure = 50e5
        target_temperature = 190.0
        ambient_temperature = 290.0
        edge["peripheral_components"] = [
            {
                "type": "compressor",
                "parameters": {
                    "efficiency": 1.0,
                    "outlet_pressure": 120e5,
                },
            }
        ]
    elif component_type == "cryopump":
        source_pressure = 20e5
        source_temperature = 35.0
        target_pressure = 12e5
        target_temperature = 26.0
        # Ambient set to source temperature so T_shell,0 ≈ source temperature,
        # making the insulation driving force negligible over the test duration.
        ambient_temperature = 36.0
        edge["peripheral_components"] = [
            {
                "type": "cryopump",
                "parameters": {
                    "reservoir_pressure": 3e5,
                    "efficiency": 0.78,
                },
            }
        ]
    elif component_type == "ideal_heat_exchanger":
        source_pressure = 120e5
        source_temperature = 260.0
        target_pressure = 50e5
        target_temperature = 180.0
        ambient_temperature = 290.0
        edge["peripheral_components"] = [
            {
                "type": "ideal_heat_exchanger",
                "parameters": {
                    "target_temperature": 315.0,
                    "pressure_drop": 0.0,
                },
            }
        ]
    else:
        raise ValueError(f"Unsupported benchmark component type: {component_type}")

    config_dict = {
        "analysis": {
            "name": f"{component_type} benchmark",
            "description": "Peripheral-component benchmark in multi-tank DAE context",
            "version": "1.0",
        },
        "network": {
            "nodes": [],
            "edges": [edge],
        },
        "mission": {
            "type": "discharge",
            "profile": "constant_flow",
            "flow_rate": flow_rate,
            "duration": duration_s,
            "ambient_temperature": ambient_temperature,
            "assigned_to_node": 2,
        },
        "physics": {
            "fluid_properties": {
                "use_coolprop": True,
                "coolprop_fluid": "H2",
            },
            "orifice_flow": {
                "discharge_coefficient": 0.8,
                "atmospheric_pressure": 101325.0,
            },
            "choked_flow": {
                "enable_choked_flow": True,
            },
            "safety_limits": {
                "max_mass_transfer_fraction": 0.2,
                "minimum_pressure_pa": 1000.0,
                "max_flow_rate_kg_s": flow_rate,
            },
            "numerical": {
                "pressure_tolerance_pa": 100.0,
                "flow_rate_tolerance_kg_s": 1e-8,
                "property_update_frequency": 1,
            },
        },
        "solver": {
            "method": "LSODA",
            "rtol": 1e-5,
            "atol": 1e-7,
            "time_step": 1.0,
            "max_step": 1.0,
        },
        "output": {
            "identifier": f"{component_type}_benchmark",
            "save_plots": False,
            "save_data": False,
            "plots": {},
        },
    }

    for node_id, radius, pressure, temperature in [
        (1, 1.5, source_pressure, source_temperature),
        (2, 0.35, target_pressure, target_temperature),
    ]:
        config_dict["network"]["nodes"].append(
            {
                "node_id": node_id,
                "type": "tank",
                "description": f"Tank {node_id}",
                "fluid": "CH2",
                "geometry": {"radius": radius},
                "initial_conditions": {
                    "pressure": pressure,
                    "temperature": temperature,
                    "density": None,
                },
                "operating_limits": {
                    "minimum_pressure": 5e5,
                    "venting_pressure": 200e5,
                },
                "materials": {
                    "liner": {
                        "nist_path": "aluminum_6061T6_nist",
                        "thickness": 0.001,
                    },
                    "composite": {
                        "nist_path": "carbon_epoxy_nist",
                        "winding_angle": 54.7,
                    },
                    "insulation": {
                        "thickness": 0.01,
                        "shell_thickness": 0.002,
                        "alpha_amb": 10.0,
                        "emissivity": 0.05,
                    },
                    "safety_margin": 1.25,
                },
                "stopping_criteria": {
                    "minimum_density": 0.0001,
                    "use_density_stopping_events": False,
                },
                "plotting": {},
            }
        )

    config_dict["coupling_rules"] = [
        {
            "coupling_id": edge["edge_id"],
            "coupling_type": edge["connection_type"],
            "participants": {
                "source": edge["from_node"],
                "target": edge["to_node"],
            },
            "activation_conditions": deepcopy(edge["activation_conditions"]),
            "flow_parameters": deepcopy(edge["flow_parameters"]),
            "peripheral_components": deepcopy(edge["peripheral_components"]),
        }
    ]

    return config_dict


def run_component_benchmark(component_type: str) -> dict:
    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(build_component_benchmark_config(component_type), handle)
        config_path = handle.name

    try:
        config = ScenarioConfig.from_yaml(config_path)
        orchestrator = SystemOrchestrator(config)

        original_valve = orchestrator.tank_system.coupling_valves[0]
        assert len(original_valve.component_chain) == 1

        benchmark_coupling = ConstantFlowBenchmarkCoupling(0, 1, 0.05)
        benchmark_coupling.set_component_chain(original_valve.component_chain)
        orchestrator.tank_system.coupling_valves = [benchmark_coupling]

        results = orchestrator.run_simulation(
            "LSODA",
            {
                "time_step": 1.0,
                "rtol": 1e-5,
                "atol": 1e-7,
                "max_step": 1.0,
                "max_simulation_time": 250.0,
            },
        )

        source_initial = results.multi_tank_states[0].get_tank_state(0)
        target_initial = results.multi_tank_states[0].get_tank_state(1)
        source_final = results.multi_tank_states[-1].get_tank_state(0)
        target_final = results.multi_tank_states[-1].get_tank_state(1)

        rho_final = target_final.fuel_mass / target_final.tank.volume
        h_target = PropsSI("Hmass", "T", target_final.h2_temperature, "Dmass", rho_final, "PARAHYD")

        if component_type == "compressor":
            expected_component_name = "Compressor"
            s_in = PropsSI("Smass", "P", source_final.pressure, "T", source_final.h2_temperature, "PARAHYD")
            h_expected = PropsSI("Hmass", "P", 120e5, "Smass", s_in, "PARAHYD")
        elif component_type == "cryopump":
            expected_component_name = "CryoPumpModel"
            h1 = PropsSI("H", "P", 3e5, "Q", 0, "PARAHYD")
            s1 = PropsSI("S", "P", 3e5, "Q", 0, "PARAHYD")
            h2s = PropsSI("H", "P", target_final.pressure, "S", s1, "PARAHYD")
            h_expected = h1 + (h2s - h1) / 0.78
        else:
            expected_component_name = "IdealHeatExchanger"
            h_expected = PropsSI("Hmass", "P", source_final.pressure, "T", 315.0, "PARAHYD")

        t_expected = PropsSI("T", "Dmass", rho_final, "Hmass", h_expected, "PARAHYD")

        return {
            "component_name": type(original_valve.component_chain[0]).__name__,
            "expected_component_name": expected_component_name,
            "initial_target_mass": target_initial.fuel_mass,
            "final_target_mass": target_final.fuel_mass,
            "initial_source_mass": source_initial.fuel_mass,
            "final_source_mass": source_final.fuel_mass,
            "target_temperature": target_final.h2_temperature,
            "expected_temperature": t_expected,
            "target_enthalpy": h_target,
            "expected_enthalpy": h_expected,
            "coupling_flow_history": orchestrator.tank_system.coupling_flow_history,
        }
    finally:
        Path(config_path).unlink(missing_ok=True)


def build_yaml_smoke_config() -> dict:
    return {
        "analysis": {
            "name": "yaml smoke",
            "description": "Native YAML peripheral-component smoke",
            "version": "1.0",
        },
        "network": {
            "nodes": [
                {
                    "node_id": 1,
                    "type": "tank",
                    "description": "Source",
                    "fluid": "CH2",
                    "geometry": {"radius": 1.0},
                    "initial_conditions": {
                        "pressure": 120e5,
                        "temperature": 260.0,
                        "density": None,
                    },
                    "operating_limits": {
                        "minimum_pressure": 5e5,
                        "venting_pressure": 200e5,
                    },
                    "materials": {
                        "liner": {
                            "nist_path": "aluminum_6061T6_nist",
                            "thickness": 0.001,
                        },
                        "composite": {
                            "nist_path": "carbon_epoxy_nist",
                            "winding_angle": 54.7,
                        },
                        "insulation": {
                            "thickness": 0.01,
                            "shell_thickness": 0.002,
                            "alpha_amb": 10.0,
                            "emissivity": 0.05,
                        },
                        "safety_margin": 1.25,
                    },
                    "stopping_criteria": {
                        "minimum_density": 0.01,
                        "use_density_stopping_events": False,
                    },
                    "plotting": {},
                },
                {
                    "node_id": 2,
                    "type": "tank",
                    "description": "Target",
                    "fluid": "CH2",
                    "geometry": {"radius": 0.5},
                    "initial_conditions": {
                        "pressure": 50e5,
                        "temperature": 180.0,
                        "density": None,
                    },
                    "operating_limits": {
                        "minimum_pressure": 5e5,
                        "venting_pressure": 200e5,
                    },
                    "materials": {
                        "liner": {
                            "nist_path": "aluminum_6061T6_nist",
                            "thickness": 0.001,
                        },
                        "composite": {
                            "nist_path": "carbon_epoxy_nist",
                            "winding_angle": 54.7,
                        },
                        "insulation": {
                            "thickness": 0.01,
                            "shell_thickness": 0.002,
                            "alpha_amb": 10.0,
                            "emissivity": 0.05,
                        },
                        "safety_margin": 1.25,
                    },
                    "stopping_criteria": {
                        "minimum_density": 0.01,
                        "use_density_stopping_events": False,
                    },
                    "plotting": {},
                },
            ],
            "edges": [
                {
                    "edge_id": "yaml_hx_smoke",
                    "connection_type": "pressure_compensation",
                    "from_node": 1,
                    "to_node": 2,
                    "activation_conditions": {
                        "pressure_open_bar": 55.0,
                        "pressure_close_bar": 56.0,
                    },
                    "flow_parameters": {
                        "max_flow_rate_kg_s": 0.02,
                        "orifice_diameter_m": 0.01,
                    },
                    "peripheral_components": [
                        {
                            "type": "ideal_heat_exchanger",
                            "parameters": {
                                "target_temperature": 315.0,
                                "pressure_drop": 0.0,
                            },
                        }
                    ],
                }
            ],
        },
        "mission": {
            "type": "discharge",
            "profile": "constant_flow",
            "flow_rate": 0.01,
            "duration": 40.0,
            "ambient_temperature": 290.0,
            "assigned_to_node": 2,
        },
        "physics": {
            "fluid_properties": {
                "use_coolprop": True,
                "coolprop_fluid": "H2",
            },
            "orifice_flow": {
                "discharge_coefficient": 0.8,
                "atmospheric_pressure": 101325.0,
            },
            "choked_flow": {
                "enable_choked_flow": True,
            },
            "safety_limits": {
                "max_mass_transfer_fraction": 0.2,
                "minimum_pressure_pa": 1000.0,
                "max_flow_rate_kg_s": 0.02,
            },
            "numerical": {
                "pressure_tolerance_pa": 100.0,
                "flow_rate_tolerance_kg_s": 1e-8,
                "property_update_frequency": 1,
            },
        },
        "solver": {
            "method": "LSODA",
            "rtol": 1e-5,
            "atol": 1e-7,
            "time_step": 1.0,
            "max_step": 1.0,
        },
        "output": {
            "identifier": "yaml_hx_smoke",
            "save_plots": False,
            "save_data": False,
            "plots": {},
        },
    }


def write_yaml_smoke_config(path) -> None:
    path.write_text(yaml.safe_dump(build_yaml_smoke_config()))
