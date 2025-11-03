import numpy

from plotting.figures import Line, SingleFigure
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
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
    fuselage_radii = [
        mission.pop("fuselage_radius")
        for mission in config["missions"]
    ]
    labels = [
        mission.pop("label")
        for mission in config["missions"]
    ]

    # Define the mission
    mission_factory = MissionFactory()
    missions = [
        mission_factory.create_mission_from_file(**mission)
        for mission in config["missions"]
    ]

    # Define the initial state and operational envelope of the tank
    initial_state = InitialConditions(**config["initial_conditions"])
    operational_envelope = OperationalEnvelope(**config["operational_envelope"])

    # Define required fuel for the different missions, required for tank sizing
    fuel_masses = [mission.required_fuel for mission in missions]
    initial_fuel = initial_state.hydrogen
    initial_density = initial_fuel.density if hasattr(initial_fuel, "density") else initial_fuel.liquid.density
    fuel_volumes = [
        fuel_mass / initial_density
        for fuel_mass in fuel_masses
    ]

    # Define tank dimensions, based on the required fuel volume
    fuel_volume_margin = config["fuel_volume_margin"]

    # Define the material of the tank
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )

    # Define the tanks for the different missions
    tanks = []
    for fuel_volume, fuselage_radius in zip(fuel_volumes, fuselage_radii):
        tank_radius = min(radius_from_volume_sphere(fuel_volume), fuselage_radius)
        tank_length = CylindricalTankSphericalCaps.length_from_radius_and_volume(
            tank_radius, fuel_volume_margin * fuel_volume
        )
        dimensions = TankDimensions(tank_radius, tank_length)
        tank = CylindricalTankSphericalCaps(
            radius=dimensions.radius,
            total_length=dimensions.body_length + 2 * dimensions.radius,
            material=material,
            operating_pressure=config["initial_conditions"]["pressure"],
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
        for tank, mission in zip(tanks, missions)
    ]

    # Store data for pickling
    data = {
        "tank_states": tank_states,
        "missions": missions,
        "tanks": tanks,
        "labels": labels,
    }

    return data


def extract_data(data: dict, store_path: str):

    # Unpack required data from tank states
    times = [
        tank_states.timesteps_in_hours
        for tank_states in data["tank_states"]
    ]
    pressures = [
        tank_states.pressures_in_bar
        for tank_states in data["tank_states"]
    ]
    temperatures = [
        tank_states.temperatures
        for tank_states in data["tank_states"]
    ]
    print("*"*10, "Note fluxes are changed in sign","*"*10)
    fluxes = [
        numpy.array(tank_states.required_fluxes_in_MW) * -1
        for tank_states in data["tank_states"]
    ]

    numpy.savez_compressed(
        store_path,
        times=pad_with_nan(times),
        pressure=pad_with_nan(pressures),
        temperature=pad_with_nan(temperatures),
        flux=pad_with_nan(fluxes),
        labels=data["labels"],
    )
    

def plot_results(store_path: str, fig_path: str, config: dict, extensions: list[str]=["eps", "png"]) -> SingleFigure:

    # Define variables that are to be plotted
    variables = ["pressure", "temperature", "flux"]
    
    # Load numpy data
    data = numpy.load(store_path)

    # Extract time steps (in hours)
    time_ticks = min_max_to_list(**config["time_ticks"])

    # Loop over desire variables and plot
    for variable in variables:
        lines = [
            Line(times, y_vals, label)
            for label, times, y_vals in zip(data["labels"], data["times"], data[variable])
        ]
        fig = SingleFigure(
            lines,
            config["time_label"],
            config[f"{variable}_label"],
            x_ticks=time_ticks,
            y_ticks=min_max_to_list(**config[f"{variable}_ticks"])
        )

        for extension in extensions:
            fig_name = f"{fig_path}_{variable}.{extension}"
            fig.save(fig_name)

    return fig


# End
