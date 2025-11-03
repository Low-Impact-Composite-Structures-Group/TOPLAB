import os
import copy
import numpy
import yaml
import pickle

from plotting.figures import SingleFigure, Line

from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.dynamics.stopping_criteria import StoppingCriteriaFactory
from src.efficiencies.efficiency_computers import GravimetricEfficiencyComputerFactory, VolumetricEfficiencyComputerFactory
from src.insulation.factory import InsulationFactory
from src.multistep_methods.linear_multistep_methods import MultistepMethodFactory
from src.thermodynamics.external_models import ExternalModelFactory
from src.thermodynamics.internal_models import InternalModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.materials.materials import MaterialFactory
from src.mission.mission import Mission, MissionFactory
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps
from src.thermodynamics.tank_states import InitialConditions, OperationalEnvelope

def min_max_to_list(min: float, max: float, step: float) -> list:
    return list(numpy.arange(min, max + step, step))

def pad_with_nan(data: list[list[float]]) -> numpy.ndarray:
    max_len = max(len(series) for series in data)
    padded = numpy.full((len(data), max_len), numpy.nan)
    for i, series in enumerate(data):
        padded[i, :len(series)] = series
    return padded

def perform_analysis(config: dict):
    tank_radii = min_max_to_list(**config["tank"]["dimensions"]["radii"])
    number_of_tanks = min_max_to_list(**config["tank"]["number_of_tanks"])

    mission = MissionFactory().create_mission_from_file(**config["mission"])
    initial_state = InitialConditions(**config["initial_conditions"])
    operational_envelope = OperationalEnvelope(**config["operational_envelope"])
    insulation = InsulationFactory().create_insulation(**config["insulation"])
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )

    stopping_criteria = [
        StoppingCriteriaFactory().create_criterion(criterion)
        for criterion in config["stopping_criteria"]
    ]

    multistep_method = MultistepMethodFactory().create_method(**config["multistep_method"])
    dynamic_model_factory = FormulationModelSelector().get_dynamic_model(config["dynamic_model_formulation"])
    internal_model = InternalModelFactory().create_model(config["thermal_model_formulation"]["internal"])
    external_model = ExternalModelFactory().create_model(config["thermal_model_formulation"]["external"])
    thermal_model = ThermodynamicModel(internal_model, external_model, insulation)
    heat_flux_factor = config["heat_flux_factor"]
    thermal_capacity_convergence = config["thermal_capacity_convergence"]

    fuel_mass = mission.required_fuel
    fuel_density = initial_state.hydrogen.density if hasattr(initial_state.hydrogen, "density") else initial_state.hydrogen.liquid.density
    fuel_volume = fuel_mass / fuel_density

    operating_pressure = operational_envelope.max_pressure or initial_state.pressure

    results = {
        "tank_radii": tank_radii,
        "insulation": insulation,
        "data": [],
    }

    for no_tanks in number_of_tanks:
        row = {
            "number_of_tanks": no_tanks,
            "tanks": [],
            "tank_states": [],
        }
        for radius in tank_radii:
            # new_mission = Mission([
            #     copy.deepcopy(section)._replace(fuel_flows=[flow._replace(mass_flow=flow.mass_flow / no_tanks) for flow in section.fuel_flows])
            #     for section in mission.sections
            # ])

            # Define the new mission, by defining the lower fuel flow per tank
            sections = list()
            for section in mission.sections:
                section = copy.deepcopy(section)
                section.fuel_flows[0].mass_flow /= no_tanks
                sections.append(section)
            new_mission = Mission(sections)

            tank_volume = fuel_volume / no_tanks
            tank_length = CylindricalTankSphericalCaps.length_from_radius_and_volume(radius, tank_volume)
            if tank_length is None:
                break

            total_length = tank_length + 2 * radius
            tank = CylindricalTankSphericalCaps(radius, total_length, material, operating_pressure)
            tank_state = MissionAnalysis().perform_analysis(
                tank,
                initial_state,
                new_mission,
                stopping_criteria,
                operational_envelope,
                multistep_method,
                dynamic_model_factory,
                thermal_model,
                heat_flux_factor,
                thermal_capacity_convergence=thermal_capacity_convergence
            )

            row["tanks"].append(tank)
            row["tank_states"].append(tank_state)

        results["data"].append(row)

    return results

def extract_data(data: dict, store_path: str):

    grav_factory = GravimetricEfficiencyComputerFactory()
    vol_factory = VolumetricEfficiencyComputerFactory()

    grav_computer = grav_factory.create_efficiency_computer(data["config"]["efficiency_computers"]["gravimetric"])
    vol_computer = vol_factory.create_efficiency_computer(data["config"]["efficiency_computers"]["volumetric"])

    vol_data = []
    grav_data = []
    for row in data["data"]:
        vols = [vol_computer.compute_efficiency(tank, data["insulation"]) for tank in row["tanks"]]
        gravs = [
            grav_computer.compute_efficiency(tank, data["insulation"], state.first_state)
            for tank, state in zip(row["tanks"], row["tank_states"])
        ]
        vol_data.append(vols)
        grav_data.append(gravs)

    numpy.savez_compressed(
        store_path,
        tank_radii=data["tank_radii"],
        number_of_tanks=[row["number_of_tanks"] for row in data["data"]],
        volumetric=pad_with_nan(vol_data),
        gravimetric=pad_with_nan(grav_data)
    )

def plot_results(store_path: str, fig_path: str, config: dict, extensions: list[str] = ["png", "eps"]):
    data = numpy.load(store_path)
    x_ticks = min_max_to_list(**config["x_ticks"])
    y_ticks = min_max_to_list(**config["y_ticks"])

    for key in ["volumetric", "gravimetric"]:
        lines = [
            Line(
                data["tank_radii"][:len(vals)],
                vals,
                f"{int(n)} tanks",
                marker=None
            )
            for vals, n in zip(data[key], data["number_of_tanks"])
        ]
        fig = SingleFigure(
            lines,
            config["x_label"],
            f"{key.capitalize()} Efficiency [-]",
            x_ticks=x_ticks,
            y_ticks=y_ticks,
        )
        for ext in extensions:
            fig.save(f"{fig_path}_{key}.{ext}")

        fig.show()

    return fig
