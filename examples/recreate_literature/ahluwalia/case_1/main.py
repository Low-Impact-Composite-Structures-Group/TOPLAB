import os
import yaml
import matplotlib.pyplot as plt

from plotting.plot_tank_states import plot_tank_fill, plot_thermo_mechanical_loading
from src.dynamics.dynamic_analysis import MissionAnalysis
from src.dynamics.dynamic_models.factory import FormulationModelSelector
from src.dynamics.stopping_criteria import StoppingCriteriaFactory
from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.insulation.factory import InsulationFactory
from src.materials.materials import MaterialFactory
from src.mission.mission import Mission
from src.mission.mission_sections import InFlow, MissionSection
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

    # Define stopping criteria
    criterion_factory = StoppingCriteriaFactory()
    stopping_criteria = [
        criterion_factory.create_criterion(criterion)
        for criterion in config["stopping_criteria"]
    ]

    # Define the material of the tank
    material = MaterialFactory().create_material(
        config["material"]["type"], config["material"]["args"]
    )

    # Define the target conditions
    target_conditions = OperationalEnvelope(**config["operational_envelope"])
    
    # Define the fuel tank
    tank = config["tank"]
    tank_type = tank["type"]
    tank_dimensions = TankDimensions(**tank["dimensions"])
    tank = TankFactory().create_tank(tank_type, tank_dimensions, material, initial_conditions.pressure)

    # Define the dynamic mode formulation
    dynamic_model_factory = FormulationModelSelector().get_dynamic_model(config["dynamic_model_formulation"])

    # Define timestep method
    multistep_method = MultistepMethodFactory().create_method(**config["multistep_method"])

    # Define insulation and thermodynamic model
    insulation = InsulationFactory().create_insulation(**config["insulation"])

    # Define the thermal model
    internal_model = InternalModelFactory().create_model(
        config["thermal_model_formulation"]["internal"]
    )
    external_model = ExternalModelFactory().create_model(
        config["thermal_model_formulation"]["external"]
    )
    thermal_model = ThermodynamicModel(internal_model, external_model, insulation)

    heat_flux_factor = config["heat_flux_factor"]

    # Define refuelling conditions
    mission_sect = config["mission"][0]
    fuel_flow = mission_sect["fuel_flow"]
    mission_section = MissionSection(
            mission_sect["duration"],
            [
                InFlow(
                    fuel_flow["mass_flow"],
                    SinglePhaseRequester().get_hydrogen_properties(
                        fuel_flow["pressure"], fuel_flow["temperature"]
                    )
                )
            ],
            mission_sect["altitude"],
            mission_sect["mach_number"]
        )
    mission = Mission([mission_section])

    tank_states = MissionAnalysis().perform_analysis(
        tank, initial_conditions, mission, stopping_criteria, target_conditions, multistep_method, dynamic_model_factory, thermal_model, heat_flux_factor
    )

    y1ticks = None
    y2ticks = None
    xticks = None
    plot_thermo_mechanical_loading(
        tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    y1ticks = None
    y2ticks = None
    plot_tank_fill(
        tank_states,
        xticks,
        y1ticks,
        y2ticks
    )
    plt.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
