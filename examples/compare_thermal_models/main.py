import os
import yaml

from plotting.figures import SingleFigure, Line
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.dynamics.stopping_criteria import StoppingCriteriaFactory
from src.insulation.factory import InsulationFactory
from src.materials.materials import MaterialFactory
from src.mission.mission import Mission
from src.multistep_methods.linear_multistep_methods import MultistepMethodFactory
from src.tank_design.tank_shapes import TankFactory, TankDimensions
from src.thermodynamics.external_models import ExternalModelFactory
from src.thermodynamics.internal_models import InternalModelFactory
from src.thermodynamics.thermodynamic_models import ThermodynamicModel
from src.thermodynamics.tank_states import OperationalEnvelope, InitialConditions


def main():

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
    mission = Mission.from_list(config["mission"])

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
    dynamic_model_factory = FormulationModelSelector()
    dynamic_model = dynamic_model_factory.get_dynamic_model(config["dynamic_model_formulation"])

    # Define the thermal model
    thermal_models = config["thermal_model_formulation"]

    lines = list()
    for thermal_model in thermal_models:

        model = ThermodynamicModel(
            InternalModelFactory().create_model(thermal_model["internal"]),
            ExternalModelFactory().create_model(thermal_model["external"]),
            insulation
        )

        label = (
            thermal_model["internal"].split("_")[0].capitalize()
            + " - "
            + thermal_model["external"].split("_")[0].capitalize()
        )
        
        tank_states = MissionAnalysis().perform_analysis(
            tank,
            initial_conditions,
            mission,
            stopping_criteria,
            operating_envelope,
            multistep_method,
            dynamic_model,
            model,
            heat_flux_factor,
        )

        pressures = tank_states.pressures_in_bar
        times = tank_states.timesteps_in_hours

        line = Line(times, pressures, label)
        lines.append(line)

    y1ticks = [i / 10 for i in range(14, 23)]
    xticks = [i for i in range(0, 20, 2)]

    fig = SingleFigure(lines, "Time [hours]", "Pressure [bar]", x_ticks=xticks, y_ticks=y1ticks)
    fig.show()


if __name__ == "__main__":
    main()


# End
