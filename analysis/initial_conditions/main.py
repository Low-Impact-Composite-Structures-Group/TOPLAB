import numpy

from plotting.figures import Line, SingleFigure
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.efficiencies.efficiency_computers import GravimetricEfficiencyComputerFactory, VolumetricEfficiencyComputerFactory
from src.insulation.factory import InsulationFactory
from src.multistep_methods.linear_multistep_methods import MultistepMethodFactory
from src.thermodynamics.external_models import ExternalModelFactory
from src.thermodynamics.internal_models import InternalModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.materials.materials import MaterialFactory
from src.mission.mission import MissionFactory
from src.tank_design.tank_shapes import CylindricalTankSphericalCaps, TankDimensions, radius_from_volume_sphere
from src.thermodynamics.tank_states import InitialConditions, OperationalEnvelope


def min_max_to_list(min: float, max: float, step: float) -> list:
    return list(numpy.arange(min, max+step, step))


def pad_with_nan(data: list[list[float]]) -> numpy.ndarray:
    max_len = max(len(series) for series in data)
    padded = numpy.full((len(data), max_len), numpy.nan)
    for i, series in enumerate(data):
        padded[i, :len(series)] = series
    return padded


def perform_analysis(config: dict):

    """Config Extraction and Model Initialisation"""

    # Define fuselage radii and labels for plotting
    tank_radius = config["mission"].pop("fuselage_radius")

    # Define the mission
    mission_factory = MissionFactory()
    mission = mission_factory.create_mission_from_file(**config["mission"])


    # Define the initial state and operational envelope of the tank
    initial_states = [
        InitialConditions(**initial)
        for initial in config["initial_conditions"]
    ]

    operational_envelope = OperationalEnvelope(**config["operational_envelope"])

    # Define required fuel for the different missions, required for tank sizing
    fuel_mass = mission.required_fuel

    # Define tank dimensions, based on the required fuel volume
    fuel_volume_margin = config["fuel_volume_margin"]

    # Define the material of the tank
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )

    # Define the tanks for the different missions
    tanks = []
    for initial in initial_states:
        fuel = initial.hydrogen
        density = fuel.density if hasattr(fuel, "density") else fuel.liquid.density
        volume =  fuel_mass / density
        tank_radius = min(radius_from_volume_sphere(volume), tank_radius)
        tank_length = CylindricalTankSphericalCaps.length_from_radius_and_volume(
            tank_radius, fuel_volume_margin * volume
        )
        dimensions = TankDimensions(tank_radius, tank_length)
        tank = CylindricalTankSphericalCaps(
            radius=dimensions.radius,
            total_length=dimensions.body_length + 2 * dimensions.radius,
            material=material,
            operating_pressure=initial.pressure,
        )
        tanks.append(tank)


    stopping_criteria = config["stopping_criteria"]

    multistep_method = MultistepMethodFactory().create_method(**config["multistep_method"])


    # Define the dynamic mode formulation
    dynamic_model_factory = FormulationModelSelector().get_dynamic_model(config["dynamic_model_formulation"])

    # Define the thermal model
    insulation = InsulationFactory().create_insulation(**config["insulation"])
    internal_model = InternalModelFactory().create_model(
        config["thermal_model_formulation"]["internal"]
    )
    external_model = ExternalModelFactory().create_model(
        config["thermal_model_formulation"]["external"]
    )
    thermal_model = ThermodynamicModel(internal_model, external_model, insulation)

    # Define the heat flux factor, to account for piping and such
    heat_flux_factor = config["heat_flux_factor"]

    # When doing efficiency analysis, tank mass is of importance
    # when analysing a single mission this is somewhat negligible
    thermal_capacity_convergence = config["thermal_capacity_convergence"]



    """Computations"""

    # Analyse mission
    tank_states = [
        MissionAnalysis().perform_analysis(
            tank,
            initial,
            mission,
            stopping_criteria,
            operational_envelope,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor,
            thermal_capacity_convergence,
        )
        for tank, initial in zip(tanks, initial_states)
    ]

    # Store data for pickling
    data = {
        "tank_states": tank_states,
        "initial_states": initial_states,
        "tanks": tanks,
        "labels": [state.pressure for state in initial_states],
    }

    return data


def extract_data(data: dict, store_path: str):

    config = data["config"]
    grav_eff_computer = GravimetricEfficiencyComputerFactory().create_efficiency_computer(config["efficiency_computers"]["gravimetric"])
    vol_eff_computer = VolumetricEfficiencyComputerFactory().create_efficiency_computer(config["efficiency_computers"]["volumetric"])

    insulation = InsulationFactory().create_insulation(
        config["insulation"]["type"], config["insulation"]["args"]
    )

    gravimetric_efficiencies = [
        grav_eff_computer.compute_efficiency(tank, insulation, state.first_state)
        for tank, state in zip(data["tanks"], data["tank_states"])
    ]
    volumetric_efficiencies = [
        vol_eff_computer.compute_efficiency(tank, insulation)
        for tank in data["tanks"]
    ]
    initial_pressures = [
        state.first_state.pressure
        for state in data["tank_states"]
    ]

    numpy.savez_compressed(
        store_path,
        gravimetric_efficiencies=gravimetric_efficiencies,
        volumetric_efficiencies=volumetric_efficiencies,
        initial_pressures=initial_pressures,
    )
    

def plot_results(store_path: str, fig_path: str, config: dict, extensions: list[str]=["eps", "png"]) -> SingleFigure:

    # Extract ticks and labels
    x_label = config["x"]["label"]
    y_label = config["y"]["label"]
    x_ticks = min_max_to_list(**config["x"]["ticks"])
    y_ticks = min_max_to_list(**config["y"]["ticks"])
    
    # Load numpy data
    data = numpy.load(store_path)

    # Convert pressure to bar
    pa2bar = 1e-5
    pressures = numpy.array(data["initial_pressures"]) * pa2bar

    grav_eff_line = Line(pressures, data["gravimetric_efficiencies"], "Gravimetric", marker=None)
    vol_eff_line = Line(pressures, data["volumetric_efficiencies"], "Volumetric", marker=None)
    fig = SingleFigure([grav_eff_line, vol_eff_line], x_label, y_label, x_ticks=x_ticks, y_ticks=y_ticks)

    for extension in extensions:
        path = f"{fig_path}.{extension}"
        fig.save(path)

    return fig


# End
