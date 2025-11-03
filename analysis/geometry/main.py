import numpy
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

import plotting.plot_style

from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.dynamics.stopping_criteria import StoppingCriteriaFactory
from src.efficiencies.efficiency_computers import GravimetricEfficiency, VolumetricEfficiency
from src.insulation.factory import InsulationFactory
from src.multistep_methods.linear_multistep_methods import MultistepMethodFactory
from src.thermodynamics.external_models import ExternalModelFactory
from src.thermodynamics.internal_models import InternalModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.materials.materials import MaterialFactory
from src.mission.mission import MissionFactory
from src.tank_design.tank_shapes import TankDimensions, TankFactory
from src.thermodynamics.tank_states import InitialConditions, OperationalEnvelope

def min_max_to_list(min: float, max: float, step: float) -> list:
    return list(numpy.arange(min, max+step, step))

def perform_analysis(config: dict, update: bool = False, parallel: bool = True):
    tank_radii = min_max_to_list(**config["tank"]["dimensions"]["radii"])
    tank_lengths = min_max_to_list(**config["tank"]["dimensions"]["body_length"])

    mission = MissionFactory().create_mission_from_list(config["mission"])
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

    tank_pressure = operational_envelope.max_pressure or initial_state.pressure
    heat_flux_factor = config["heat_flux_factor"]
    thermal_capacity_convergence = config["thermal_capacity_convergence"]

    tank_factory = TankFactory()
    tank_type = config["tank"]["type"]

    def run_sim(radius, length):
        dimensions = TankDimensions(radius, length)
        tank = tank_factory.create_tank(tank_type, dimensions, material, tank_pressure)
        return MissionAnalysis().perform_analysis(
            tank,
            initial_state,
            mission,
            stopping_criteria,
            operational_envelope,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor,
            thermal_capacity_convergence,
            user_update=update,
        )

    grid = list(product(tank_radii, tank_lengths))

    if parallel:
        tank_states = Parallel(n_jobs=-1, backend="loky")(delayed(run_sim)(r, l) for r, l in tqdm(grid))
    else:
        tank_states = [run_sim(r, l) for r, l in tqdm(grid)]

    return {
        "tank_states": numpy.reshape(tank_states, (len(tank_radii), len(tank_lengths))),
        "radii": tank_radii,
        "lengths": tank_lengths,
        "config": config
    }

def extract_data(data: dict, store_path: str):
    config = data["config"]

    insulation = InsulationFactory().create_insulation(**data["config"]["insulation"])
    material = MaterialFactory().create_material(
        data["config"]["material"]["type"], data["config"]["material"]["args"]
    )

    grav = GravimetricEfficiency()
    vol = VolumetricEfficiency()


    tank_factory = TankFactory()
    tank_type = config["tank"]["type"]

    tanks = [
        [
            tank_factory.create_tank(tank_type, TankDimensions(r, l), material, state.max_pressure)
            # CylindricalTankSphericalCaps(r, l + 2 * r, material, state.max_pressure)
            for state, l in zip(row, data["lengths"])
        ]
        for row, r in zip(data["tank_states"], data["radii"])
    ]

    grav_eff = [
        [grav.compute_efficiency(t, insulation, s.first_state) for t, s in zip(t_row, s_row)]
        for t_row, s_row in zip(tanks, data["tank_states"])
    ]

    vol_eff = [
        [vol.compute_efficiency(t, insulation) for t in t_row]
        for t_row in tanks
    ]

    tank_volumes = numpy.array([
        [t.volume for t in t_row] for t_row in tanks
    ])

    mission_factory = MissionFactory()
    initial_state = InitialConditions(**config["initial_conditions"])
    fuel_density = initial_state.hydrogen.density if hasattr(initial_state.hydrogen, "density") else initial_state.hydrogen.liquid.density

    references = config["reference_missions"]

    ref_volumes = [
        mission_factory.create_mission_from_file(reference['file'], "liquid").required_fuel / fuel_density
        for reference in references
    ]
    ref_lines = [
        tank_volumes / volume
        for volume in ref_volumes
    ]

    ref_labels = [
        ref["label"]
        for ref in references
    ]

    numpy.savez_compressed(
        store_path,
        radii=data["radii"],
        lengths=data["lengths"],
        grav_eff=grav_eff,
        vol_eff=vol_eff,
        volumes=tank_volumes,
        ref_volumes=ref_lines,
        ref_labels=ref_labels
    )

def plot_results(store_path: str, fig_path: str, config: dict, extensions: list[str] = ["eps", "png"]):
    data = numpy.load(store_path)
    X, Y = numpy.meshgrid(data["lengths"], data["radii"])

    fig, ax = plt.subplots()
    grav = ax.contourf(X, Y, data["grav_eff"])
    cbar = fig.colorbar(grav)
    cbar.set_label(config["colour_bar_label"])


    # Plot volumetric efficiency lines
    vol_eff_levels = min_max_to_list(**config["volumetric_efficiency_levels"])
    print(vol_eff_levels)
    print(data["vol_eff"])
    lines = ax.contour(X, Y, data["vol_eff"], 10, colors="black", linestyles="dashed", levels=vol_eff_levels)
    ax.clabel(lines, inline=True)

    # Format axis
    xticks = min_max_to_list(**config["x_axis"]["ticks"])
    yticks = min_max_to_list(**config["y_axis"]["ticks"])
    ax.set_xlabel(config["x_axis"]["label"])
    ax.set_xticks(xticks)
    ax.set_xlim((xticks[0], xticks[-1]))
    ax.set_yticks(yticks)
    ax.set_ylim((yticks[0], yticks[-1]))
    ax.set_ylabel(config["y_axis"]["label"])

    for ref_levels, label in zip(data["ref_volumes"], data["ref_labels"]):
        contour = ax.contour(X, Y, ref_levels, levels=[1], colors="k")
        try:
            ax.clabel(contour, inline=1, fmt=label)
        except TypeError:
            print(f"{label} has no data in range")

    fig.tight_layout()

    for format in extensions:
        fig.savefig(f"{fig_path}.{format}")

    return fig

