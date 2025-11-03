import os
import numpy
import yaml
import copy
from scipy.optimize import minimize_scalar

from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.efficiencies.efficiency_computers import GravimetricEfficiency
from src.insulation.factory import InsulationFactory
from src.multistep_methods.linear_multistep_methods import MultistepMethodFactory
from src.thermodynamics.external_models import ExternalModelFactory
from src.thermodynamics.internal_models import InternalModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from plotting.figures import Line, SingleFigure
from src.materials.materials import MaterialFactory
from src.mission.mission import MissionFactory
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps, TankDimensions, radius_from_volume_sphere
from src.thermodynamics.tank_states import InitialConditions, OperationalEnvelope


def min_max_to_list(min: float, max: float, step: float) -> list:
    return list(numpy.arange(min, max+step, step))


def perform_analysis(config: dict):


    tank_radius = config["mission"].pop("fuselage_radius")
    tank_radius = 2.8

    # Define the mission
    mission_factory = MissionFactory()
    mission = mission_factory.create_mission_from_file(**config["mission"])

    # Define the initial state of the tank
    initial_state = InitialConditions(**config["initial_conditions"])

    # Define the target conditions
    operational_envelope = OperationalEnvelope(**config["operational_envelope"])

    # Define required fuel
    fuel_mass = mission.required_fuel
    initial_fuel = initial_state.hydrogen
    # initial_density = initial_fuel.liquid.density if hasattr(initial_fuel, "liquid") else initial_fuel.density
    initial_density = initial_fuel.density if hasattr(initial_fuel, "density") else initial_fuel.liquid.density
    fuel_volume = fuel_mass / initial_density

    # Define the material of the tank
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )
    
    tank_length = CylindricalTankSphericalCaps.length_from_radius_and_volume(
            tank_radius, fuel_volume
        )
    print(tank_length)
    tank_dimensions = TankDimensions(
        tank_radius,
        tank_length
    )
    tank = CylindricalTankSphericalCaps(
        radius=tank_dimensions.radius,
        total_length=tank_dimensions.body_length+2*tank_dimensions.radius,
        material=material,
        operating_pressure=config["initial_conditions"]["pressure"],
    )

    stopping_criteria = config["stopping_criteria"]

    multistep_method = MultistepMethodFactory().create_method(**config["multistep_method"])


    # Define the dynamic mode formulation
    dynamic_model_factory = FormulationModelSelector().get_dynamic_model(config["dynamic_model_formulation"])

    # Define the thermal model
    internal_model = InternalModelFactory().create_model(
        config["thermal_model_formulation"]["internal"]
    )
    external_model = ExternalModelFactory().create_model(
        config["thermal_model_formulation"]["external"]
    )

    # Define the heat flux factor, to account for piping and such
    heat_flux_factor = config["heat_flux_factor"]

    # When doing efficiency analysis, tank mass is of importance
    # when analysing a single mission this is somewhat negligible
    thermal_capacity_convergence = config["thermal_capacity_convergence"]

    # Define insulation
    insulation_type = config["insulation"]["type"]
    insulation_args = config["insulation"]["args"]
    insulation_thicknesses = min_max_to_list(**insulation_args["thickness"])

    def fun(insulation_thickness):
        insulation = InsulationFactory().create_insulation(insulation_type, {**insulation_args, "thickness": insulation_thickness})
        thermal_model = ThermodynamicModel(internal_model, external_model, insulation)
        new_tank = copy.deepcopy(tank)
        # Analyse mission
        data = MissionAnalysis().perform_analysis(
            new_tank,
            initial_state,
            mission,
            stopping_criteria,
            operational_envelope,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor,
            thermal_capacity_convergence,
        )

        new_tank.set_operating_pressure(2 * data.max_pressure)
        gravimetric_efficiency = GravimetricEfficiency().compute_efficiency(
            new_tank, insulation, data.first_state
        )

        return -gravimetric_efficiency
    
    optimal = minimize_scalar(fun, bounds=(1e-3, 0.2), method="bounded", tol=1e-3)
    optimal_thickness = optimal.x
    optimal_efficiency = -fun(optimal_thickness)

    efficiencies = list()
    for thickness in insulation_thicknesses:
        insulation = InsulationFactory().create_insulation(insulation_type, {**insulation_args, "thickness": thickness})
        thermal_model = ThermodynamicModel(internal_model, external_model, insulation)
        new_tank = copy.deepcopy(tank)
        # Analyse mission
        data = MissionAnalysis().perform_analysis(
            new_tank,
            initial_state,
            mission,
            stopping_criteria,
            operational_envelope,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor,
            thermal_capacity_convergence,
        )

        print(data.max_pressure * 1e-5)

        new_tank.set_operating_pressure(2 * data.max_pressure)
        gravimetric_efficiency = GravimetricEfficiency().compute_efficiency(
            new_tank, insulation, data.first_state
        )
        efficiencies.append(gravimetric_efficiency)
        print(gravimetric_efficiency)

    fig = SingleFigure(
        [Line(
            numpy.array(insulation_thicknesses) * 1e3,
            efficiencies,
            "Brute",
            marker=None
        ),
        Line([optimal_thickness * 1e3], [optimal_efficiency], "Optimal", marker=None)
        ],
        "Insulation Thickness [mm]",
        "Gravimetric Efficiency [-]",
    )
    fig.show()


def main():
    file_name = "main.YAML"

    dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(dir, file_name)

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    perform_analysis(config)


if __name__ == "__main__":
    main()


# End
