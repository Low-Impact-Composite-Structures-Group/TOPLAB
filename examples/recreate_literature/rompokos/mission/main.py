import os
import yaml
import matplotlib.pyplot as plt

from plotting.plot_tank_states import plot_tank_fill, plot_thermo_mechanical_loading
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.dynamics.stopping_criteria import StoppingCriteriaFactory
from src.insulation.factory import InsulationFactory
from src.materials.materials import MaterialFactory
from src.mission.mission import Mission, MissionFactory
from src.multistep_methods.linear_multistep_methods import MultistepMethodFactory
from src.tank_design.tank_shapes import TankFactory, TankDimensions
from src.thermodynamics.external_models import ExternalModelFactory
from src.thermodynamics.internal_models import InternalModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.thermodynamics.tank_states import OperationalEnvelope, InitialConditions


def perform_analysis():

    dir = os.path.dirname(os.path.realpath(__file__))

    file_name = "main.YAML"

    file_path = os.path.join(dir, file_name)

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # Define the state of the fuel tank
    initial_conditions = InitialConditions(**config["initial_conditions"])

    # Define the tank material
    material = MaterialFactory().create_material(**config["material"])

    # Define the fuel tank
    tank = config["tank"]
    tank_type = tank["type"]
    tank_dimensions = TankDimensions(**tank["dimensions"])
    tank = TankFactory().create_tank(tank_type, tank_dimensions, material, initial_conditions.pressure)

    # Define the target conditions
    operating_envelope = OperationalEnvelope(**config["operational_envelope"])

    # Define insulation and thermodynamic model
    insulation = InsulationFactory().create_insulation(**config["insulation"])

    # Define the mission
    mission = MissionFactory().create_mission_from_file(**config["mission"])

    # Define timestep method
    multistep_method = MultistepMethodFactory().create_method(**config["multistep_method"])

    criterion_factory = StoppingCriteriaFactory()
    stopping_criteria = [
        criterion_factory.create_criterion(criterion)
        for criterion in config["stopping_criteria"]
    ]

    # Define the heat flux factor, used to account for piping losses
    heat_flux_factor = config["heat_flux_factor"]

    # Define the dynamic mode formulation
    dynamic_model_factory = FormulationModelSelector().get_dynamic_model(config["dynamic_model_formulation"])

    # Define the thermal model
    internal_model = InternalModelFactory().create_model(
        config["thermal_model_formulation"]["internal"]
    )
    external_model = ExternalModelFactory().create_model(
        config["thermal_model_formulation"]["external"]
    )
    thermal_model = ThermodynamicModel(internal_model, external_model, insulation)

    tank_states = MissionAnalysis().perform_analysis(
        tank,
        initial_conditions,
        mission,
        stopping_criteria,
        operating_envelope,
        multistep_method,
        dynamic_model_factory,
        thermal_model,
        heat_flux_factor,
    )

    y1ticks = [i / 10 for i in range(14, 23)]
    y2ticks = [i / 10 for i in range(210, 251, 5)]
    xticks = [i for i in range(0, 25, 4)]

    plot_thermo_mechanical_loading(
        tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    y1ticks = [i for i in range(0, int(10001), int(2e3))]
    y2ticks = [i for i in range(0, 101, 20)]
    plot_tank_fill(
        tank_states,
        xticks,
        y1ticks,
    )
    plt.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
