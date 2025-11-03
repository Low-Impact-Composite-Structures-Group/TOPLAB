import numpy
import copy
import pickle

from plotting.figures import Line, SingleFigure
from src.dynamics.dynamic_analysis import MissionAnalysis, SwitchPhaseDrainingAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.dynamics.stopping_criteria import StoppingCriteriaFactory
from src.efficiencies.efficiency_computers import GravimetricEfficiencyComputerFactory
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


def perform_analysis(config: dict):

    """Config Loading"""

    # Define the mission
    mission_factory = MissionFactory()
    mission = mission_factory.create_mission_from_list(config["missions"])
    
    # Define the initial state of the tank
    initial_state = InitialConditions(**config["initial_conditions"])

    # Define the target conditions
    operational_envelope = OperationalEnvelope(**config["operational_envelope"])

    # Define the material of the tank
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )

    # Define multistep method, for time stepping
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

    # Define tank dimensions, based on the required fuel volume
    tank_radius = config["tank"]["dimensions"]["radius"]
    tank_lengths = config["tank"]["dimensions"]["body_lengths"]

    # Define the stopping criteria, when to end the simulation
    # Ex: min mass or fill
    stopping_criteria = [
        StoppingCriteriaFactory().create_criterion(criterion)
        for criterion in config["stopping_criteria"]
    ]

    # Tank definition
    tank_type = config["tank"]["type"]
    tank_factory = TankFactory()


    """Computations"""

    # The operational envelope is only used for switch draining, the normal
    # analysis has no restrictions on the envelope (pressure - temperature)
    no_envelope = copy.deepcopy(operational_envelope)
    no_envelope.max_pressure = None

    # Initialise for data storage
    results = {
        "switch": list(),
        "normal": list(),
        "lengths": list(),
    }

    # Iterate over tank dimensions and compute tank states
    for tank_length in tank_lengths:

        # Define tank with provided dimensions
        dimensions = TankDimensions(tank_radius, tank_length)
        tank = tank_factory.create_tank(
            tank_type, dimensions, material, operational_envelope.max_pressure
        )

        # Analyse missions first switch drain then normal
        switch_tank_states = SwitchPhaseDrainingAnalysis().perform_analysis(
            tank,
            initial_state,
            mission,
            operational_envelope,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor,
        )
        normal_tank_states = MissionAnalysis().perform_analysis(
            tank,
            initial_state, 
            mission,
            stopping_criteria,
            no_envelope,
            multistep_method,
            dynamic_model_factory,
            thermal_model,
            heat_flux_factor,
            thermal_capacity_convergence=True,
        )

        # Store data
        results["lengths"].append(tank_length)
        results["normal"].append(normal_tank_states)
        results["switch"].append(switch_tank_states)

    return results


def extract_data(pickle_path: str, store_path: str):

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Unpack config, cause used multiple times
    config = data["config"]

    # Define insulation
    insulation = InsulationFactory().create_insulation(**config["insulation"])

    # Define the material of the tank
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )

    # Unpack radius cause used multiple times
    radius = config["tank"]["dimensions"]["radius"]

    # Define the efficiency computer
    computer_factory = GravimetricEfficiencyComputerFactory()
    efficiency_computer = computer_factory.create_efficiency_computer(
        config["efficiency_computers"]["gravimetric"]
    )

    # Tank definition
    tank_type = config["tank"]["type"]
    tank_factory = TankFactory()


    # Compute efficiencies for switch draining and normal draining
    switch_eff = [
        efficiency_computer.compute_efficiency(
            tank_factory.create_tank(tank_type, TankDimensions(radius, length), material, state.max_pressure),
            insulation,
            state.first_state,
        )
        for state, length in zip(data["switch"], data["lengths"])
    ]
    normal_eff = [
        efficiency_computer.compute_efficiency(
            tank_factory.create_tank(tank_type, TankDimensions(radius, length), material, state.max_pressure),
            insulation,
            state.first_state,
        )
        for state, length in zip(data["normal"], data["lengths"])
    ]

    # Store data
    numpy.savez_compressed(
        store_path,
        lengths=data["lengths"],
        switch=switch_eff,
        normal=normal_eff
    )

def plot_results(store_path: str, fig_path: str, config, extensions: list[str]=["png", "eps"]):

    # Load numpy data
    data = numpy.load(store_path)

    # Define plot lines
    switch_line = Line(data["lengths"], data["switch"], "Switch", marker=None)
    normal_line = Line(data["lengths"], data["normal"], "Normal", marker=None)

    # Define ticks
    xticks = min_max_to_list(**config["length_ticks"])
    yticks = min_max_to_list(**config["efficiency_ticks"])

    # Plot and save figure
    fig = SingleFigure(
        [switch_line, normal_line],
        config["x_label"],
        config["y_label"],
        x_ticks=xticks,
        y_ticks=yticks
    )
    for extension in extensions:
        file = f"{fig_path}.{extension}"
        fig.save(file)

    return fig




# def main():
#     file_name = "main.YAML"
#     pickle_file = "switch_drain_analysis.pkl"
#     store_file = "switch_drain_analysis.npz"
#     fig_file = "switch_drain_analysis"
#     data_path = os.path.join("data", "switch_draining")


#     # Computed paths
#     pickle_path = os.path.join(data_path, pickle_file)
#     store_path = os.path.join(data_path, store_file)
#     fig_path = os.path.join(data_path, fig_file)

#     dir = os.path.dirname(os.path.realpath(__file__))

#     file_path = os.path.join(dir, file_name)
    

#     with open(file_path, "r") as file:
#         config = yaml.safe_load(file)

#     data = perform_analysis(config)

#     data["config"] = config

#     with open(pickle_path, "wb") as f:
#         pickle.dump(data, f)

#     extract_data(pickle_path, store_path)

#     fig = plot_results(store_path, fig_path, config["plotting"])

#     fig.show()






# if __name__ == "__main__":
#     main()


# End
