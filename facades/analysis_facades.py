from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Union
from itertools import cycle

from src.dynamics.dynamic_analysis import (MissionAnalysis,
                                           SwitchMissionAnalysis)
from src.dynamics.dynamic_model_factories import (DynamicModelFactory,
                                                  SwitchCaseFactory)
from src.dynamics.stopping_criteria import (EMPTY_LIMIT, LowerPressureReached,
                                            MaxPressureReached, NoFuelMass,
                                            StoppingCriterion, TankIsEmpty,
                                            TargetFillReached,
                                            TargetMassReached, TargetDensityReached)
from src.efficiencies.tank_performance import TankPerformance
from src.mission.mission import Mission
from src.mission.mission_sections import MissionSection, InFlow, OutFlow
from src.multistep_methods.linear_multistep_methods import EulerMethod
from src.tank_design.tank_shapes import Tank, TankFactory
from src.thermodynamics.external_models import ForcedConvectionModel, NaturalConvectionModel
from src.thermodynamics.internal_models import SingleZoneModel
from src.thermodynamics.tank_states import (InitialState, TankStates,
                                            TargetState, TankState)
from src.thermodynamics.thermodynamic_models import ThermodynamicModel

# The lower mass limit is to be used for draining analysis og gas tanks
LOWER_MASS_LIMIT = 1 # TODO: find suitable lower limit


# Factories and constants to be used in the analysis
#TODO: refactor to use yaml input for these constants using dependency injection
# to pass the configuration to the classes and functions that need it
TIMESTEP = 10
MULTISTEP_METHOD = EulerMethod(TIMESTEP)
DYNAMIC_MODEL_FACTORY = DynamicModelFactory()
INTERNAL_MODEL = SingleZoneModel()
EXTERNAL_MODEL = NaturalConvectionModel()
HEAT_FLUX_FACTOR = 1


class Material(Protocol):
    ...


class Insulation(Protocol):
    ...


@dataclass
class TankDimensions:
    radius: float
    body_length: float

@dataclass
class GenericTankDimensions(TankDimensions):
    a: float
    b: float

@dataclass
class OperatingEnvelope:
    max_pressure: float
    min_pressure: float
    min_temperature: float


@dataclass
class InitialConditions:
    pressure: float
    temperature: float
    fill: float
    multi_flow: bool = False  # Used for dual tank analysis

@dataclass
class TargetConditions:
    fuel_mass: float
    fill: float
    density: float = None


class AnalysisFacade(Protocol):

    @classmethod
    def analyse(cls) -> TankPerformance:
        ...

    @classmethod
    def _define_tank(
        cls,
        tank_dimensions: Union[TankDimensions, GenericTankDimensions],
        material: Material,
        target_state: OperatingEnvelope,
        initial_state: InitialState
    ) -> Tank:
        # TODO: this is sloppy... find a better way to do this
        if hasattr(tank_dimensions, 'a') and hasattr(tank_dimensions, 'b'):
            return TankFactory.create_tank(
                tank_dimensions.radius,
                tank_dimensions.body_length,
                material,
                cls._define_operating_pressure(target_state, initial_state),
                tank_dimensions.a,
                tank_dimensions.b
            )
        else:
            return TankFactory.create_tank(
                tank_dimensions.radius,
                tank_dimensions.body_length,
                material,
                cls._define_operating_pressure(target_state, initial_state)
            )

    @staticmethod
    def _define_operating_pressure(
        target_state: TargetState,
        initial_state: InitialState
    ) -> float:
        if target_state.max_pressure is not None:
            return target_state.max_pressure
        return initial_state.pressure

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope
    ) -> TargetState:
        fill = None
        target_conditions = TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            fill,
            LOWER_MASS_LIMIT
        )
        return target_conditions

    @staticmethod
    def _define_thermal_model(
        insulation: Insulation,
        constant_heat_flux: float = None  # Optional constant heat flux
    ) -> ThermodynamicModel:
        return ThermodynamicModel(
            INTERNAL_MODEL, EXTERNAL_MODEL, insulation, constant_heat_flux=constant_heat_flux
        )

    @staticmethod
    def _define_initial_state(
        initial_conditions: InitialConditions
    ) -> InitialState:
        return InitialState(
            initial_conditions.pressure,
            initial_conditions.temperature,
            initial_conditions.fill,
            multi_flow=initial_conditions.multi_flow
        )


class DrainingAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        fuel_mass_flow: float,
        fuel_flow_state: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope
    ) -> TankPerformance:
        print(tank_dimensions, datetime.now().strftime("%H:%M:%S"))
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            cls._define_mission(fuel_mass_flow, fuel_flow_state),
            cls._define_stopping_criteria(),
            cls._define_target_conditions(operating_envelope),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @staticmethod
    def _define_mission(
        fuel_mass_flow: float,
        fuel_flow_state: float
    ) -> Mission:
        mission_sections = [
            MissionSection.draining(
                fuel_mass_flow, fuel_flow_state
            )
        ]
        return Mission(mission_sections)

    @staticmethod
    def _define_stopping_criteria() -> list[StoppingCriterion]:
        return [NoFuelMass(), TankIsEmpty()]


@dataclass
class ParallelDrainingAnalysis(AnalysisFacade):
    radius: float
    body_length: float
    material: Material
    insulation: Insulation
    fuel_mass_flow: float
    fuel_phase_flow: str
    initial_state: InitialConditions
    operating_envelope: OperatingEnvelope

    def __post_init__(self):
        self.tank = self._define_tank(
            TankDimensions(self.radius, self.body_length),
            self.material,
            self.operating_envelope,
            self.initial_state
        )


    def analyse(self):
        tank_states = MissionAnalysis.perform_analysis(
            self.tank,
            self.initial_state,
            self._define_mission(
                self.fuel_mass_flow, self.fuel_phase_flow
            ),
            self._define_stopping_criteria(),
            self._define_target_conditions(self.operating_envelope),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            self._define_thermal_model(self.insulation),
            HEAT_FLUX_FACTOR
        )

        return tank_states.max_pressure

    @staticmethod
    def _define_mission(
        fuel_mass_flow: float,
        fuel_flow_state: float
    ) -> Mission:
        mission_sections = [
            MissionSection.draining(
                fuel_mass_flow, fuel_flow_state
            )
        ]
        return Mission(mission_sections)

    @staticmethod
    def _define_stopping_criteria() -> list[StoppingCriterion]:
        return [NoFuelMass(), TankIsEmpty()]


class MissionAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        constant_heat_flux: float = None
    ) -> TankPerformance:
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            cls._define_target_conditions(operating_envelope),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation, constant_heat_flux),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return list()


class FillingAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions,
    ) -> TankPerformance:
        MULTISTEP_METHOD.timestep = 10
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
        tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            cls._define_target_conditions(
                operating_envelope, target_conditions
            ),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return [TargetFillReached(), TargetMassReached()]

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TargetState:
        target_conditions = TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            target_conditions.fill,
            target_conditions.fuel_mass
        )
        return target_conditions


class SwitchPhaseDrainingAnalysis(DrainingAnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        fuel_mass_flow: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope
    ) -> TankPerformance:

        # Define the initial state and the fuel tank
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )

        # Set up iterations
        tank_states = TankStates(list(), MULTISTEP_METHOD.timestep)
        max_changes = 100
        for _ in range(max_changes):

            # Compute new tank states
            tank_states += SwitchMissionAnalysis.perform_analysis(
                tank,
                initial_state,
                cls._define_mission(
                    fuel_mass_flow,
                    cls._define_flow_state(
                        operating_envelope, tank_states
                    )
                ),
                cls._define_stopping_criteria(),
                cls._define_target_conditions(operating_envelope),
                MULTISTEP_METHOD,
                SwitchCaseFactory(),
                cls._define_thermal_model(insulation),
                HEAT_FLUX_FACTOR
            )

            # Verify if tank has been drained
            if cls._tank_is_drained(tank_states):
                return TankPerformance(tank, insulation, tank_states)

            # Update the initial state for the new iteration
            initial_state = cls._define_initial_state(
                tank_states.last_state
            )

        raise ValueError(
            "Exceeded maximum iterations is switch drain analysis..."
        )

    @classmethod
    def _tank_is_drained(
        cls, tank_states: TankStates
    ) -> bool:
        return (
            tank_states.last_state.fuel_mass <= LOWER_MASS_LIMIT
            or tank_states.last_fill < EMPTY_LIMIT
        )

    @classmethod
    def _define_flow_state(
        cls,
        operating_envelope: OperatingEnvelope,
        tank_states: TankStates
    ) -> str:
        if len(tank_states.states) == 0:
            return "liquid"
        if tank_states.last_pressure < operating_envelope.min_pressure:
            return "liquid"
        elif tank_states.last_pressure > operating_envelope.max_pressure:
            return "gas"
        ValueError("Tank state out of bound for operating envelope...")

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return [
            NoFuelMass(),
            TankIsEmpty(),
            MaxPressureReached(),
            LowerPressureReached()
        ]

class DensityAnalysisFacade(AnalysisFacade):

    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TankPerformance:
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")
        initial_state = cls._define_initial_state(initial_conditions)
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )
        tank_states = MissionAnalysis.perform_analysis(
            tank,
            initial_state,
            mission,
            cls._define_stopping_criteria(),
            cls._define_target_conditions(operating_envelope, target_conditions),
            MULTISTEP_METHOD,
            DYNAMIC_MODEL_FACTORY,
            cls._define_thermal_model(insulation),
            HEAT_FLUX_FACTOR
        )
        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return [TargetDensityReached()]

    @staticmethod
    def _define_target_conditions(
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions
    ) -> TargetState:
        return TargetState(
            operating_envelope.max_pressure,
            operating_envelope.min_pressure,
            operating_envelope.min_temperature,
            target_conditions.fill,
            target_conditions.fuel_mass,
            target_conditions.density
        )



class InOutTankAnalysisFacade(AnalysisFacade):
    @classmethod
    def analyse(
        cls,
        tank_dimensions: TankDimensions,
        material: Material,
        insulation: Insulation,
        mission: Mission,
        constant_heat_flux: float,
        initial_conditions: InitialConditions,
        operating_envelope: OperatingEnvelope,
        target_conditions: TargetConditions,
    ) -> TankPerformance:
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")
        # Ensure multi_flow is True for this analysis
        initial_state = InitialState(
            initial_conditions.pressure,
            initial_conditions.temperature,
            initial_conditions.fill,
            multi_flow=True
        )
        tank = cls._define_tank(
            tank_dimensions, material, operating_envelope, initial_state
        )

        tank_states = TankStates([], MULTISTEP_METHOD.timestep)

        # Iterate through all mission sections
        for section in mission.sections:
            # 1. Solve tank for this section with both inflow and outflow
            tank_states_section = MissionAnalysis.perform_analysis(
                tank,
                initial_state,
                Mission([section]),
                cls._define_stopping_criteria(),
                cls._define_target_conditions(operating_envelope),
                MULTISTEP_METHOD,
                DYNAMIC_MODEL_FACTORY,
                cls._define_thermal_model(insulation, constant_heat_flux),
                HEAT_FLUX_FACTOR
            )

            tank_states += tank_states_section
            # Propagate multi_flow=True
            initial_state = InitialState(
                tank_states.last_pressure,
                tank_states.last_temperature,
                tank_states.last_fill,
                multi_flow=True
            )

        return TankPerformance(tank, insulation, tank_states)

    @classmethod
    def _define_stopping_criteria(cls) -> list[StoppingCriterion]:
        return list()


class MultiTankAnalysisFacade(AnalysisFacade):
    @classmethod
    def analyse(
        cls,
        tank_configurations: list,  # List of tank configs (dimensions, materials, etc.)
        mission: Mission,           # Single mission that applies to the system
        interaction_rules: dict,    # Rules for tank interactions
        initial_conditions: list,   # Initial conditions for each tank
        operating_envelopes: list,  # Operating envelopes for each tank
        target_conditions: list,    # Target conditions for each tank
    ) -> list[TankPerformance]:
        """Analyze multiple interacting tanks simultaneously."""
        print(f"Timestep: {MULTISTEP_METHOD.timestep} seconds")

        # Initialize global flow tracking lists and time tracking
        global_inflow_1_list = []
        global_outflow_1_list = []
        global_inflow_2_list = []
        global_outflow_2_list = []
        global_time_points = []
        current_time_offset = 0  # Start at time zero

        # Create tank objects
        tanks = []
        for i, config in enumerate(tank_configurations):
            initial_state = cls._define_initial_state(initial_conditions[i])
            tank = cls._define_tank(
                config["dimensions"],
                config["material"],
                operating_envelopes[i],
                initial_state
            )
            tanks.append(tank)

        # Initialize tank states and results
        all_tank_states = [TankStates([], MULTISTEP_METHOD.timestep) for _ in tanks]

        # Iterate through all mission sections
        for section_idx, section in enumerate(mission.sections):
            print(f"Calculating mission section {section.fuel_flow_key or section_idx}")

            # Set up section states
            current_states = []
            for i, tank in enumerate(tanks):
                if len(all_tank_states[i].states) == 0:
                    # First section - use initial conditions
                    current_states.append(InitialState(
                        initial_conditions[i].pressure,
                        initial_conditions[i].temperature,
                        initial_conditions[i].fill,
                        multi_flow=True
                    ))
                else:
                    # Use last state from previous section
                    last_state = all_tank_states[i].last_state
                    current_states.append(InitialState(
                        last_state.pressure,
                        last_state.temperature,
                        all_tank_states[i].last_fill,
                        multi_flow=True
                    ))

            # Perform time-stepping for this section with proper time offset
            section_states, section_flows = cls._analyse_section_multi_tank(
                tanks,
                current_states,
                section,
                interaction_rules,
                operating_envelopes,
                target_conditions,
                tank_configurations,
                current_time_offset  # Use actual time offset in seconds
            )

            # Update the time offset for the next section
            if len(section_flows['time_points']) > 0:
                current_time_offset = section_flows['time_points'][-1] + MULTISTEP_METHOD.timestep

            # Concatenate the flow lists
            global_inflow_1_list.extend(section_flows['inflow_1'])
            global_outflow_1_list.extend(section_flows['outflow_1'])
            global_inflow_2_list.extend(section_flows['inflow_2'])
            global_outflow_2_list.extend(section_flows['outflow_2'])
            global_time_points.extend(section_flows['time_points'])

            # Add states to results
            for i, states in enumerate(section_states):
                all_tank_states[i] += states

        # Create performance objects
        performances = []
        for i, tank in enumerate(tanks):
            performances.append(TankPerformance(
                tank,
                tank_configurations[i]["insulation"],
                all_tank_states[i]
            ))

        # Store the flows in a class variable for access from plotting functions
        cls.flow_data = {
            'time': global_time_points,
            'tank1_inflow': global_inflow_1_list,
            'tank1_outflow': global_outflow_1_list,
            'tank2_inflow': global_inflow_2_list,
            'tank2_outflow': global_outflow_2_list
        }

        return performances

    @classmethod
    def _analyse_section_multi_tank(
        cls,
        tanks: list,
        initial_states: list,
        mission_section: MissionSection,
        interaction_rules: dict,
        operating_envelopes: list,
        target_conditions: list,
        tank_configurations: list,
        time_offset: int = 0  # Added parameter for time tracking
    ) -> tuple[list[TankStates], dict]:
        # Initialize tank states for this section
        all_states = []

        # Initialize flow tracking for this section
        inflow_1_list = []
        outflow_1_list = []
        inflow_2_list = []
        outflow_2_list = []
        time_points = []

        # Create initial TankState for each tank
        for i, tank in enumerate(tanks):
            initial_tank_state = TankState(
                tank,
                initial_states[i].temperature,
                initial_states[i].pressure,
                initial_states[i].compute_fuel_mass(tank.volume),
                multi_flow=True
            )
            states = TankStates([initial_tank_state], MULTISTEP_METHOD.timestep)
            all_states.append(states)

        # Number of steps for this section
        steps = int(mission_section.duration / MULTISTEP_METHOD.timestep)

        # Time-stepping loop
        for step in range(steps):
            current_time = time_offset + step * MULTISTEP_METHOD.timestep
            time_points.append(current_time)

            # Get current states for all tanks
            current_states = [states.last_state for states in all_states]

            # Compute interactions between tanks based on rules
            flow_adjustments = cls._compute_tank_interactions(
                current_states,
                interaction_rules,
                operating_envelopes,
                mission_section,
                previous_flow_rate=getattr(interaction_rules, "previous_flow_rate", None),
                step_index=step,  # Pass current step index
                total_steps=steps  # Pass total steps
            )

            # Track flows for the entire simulation
            tank1_inflow = 0.0
            tank1_outflow = 0.0
            tank2_inflow = 0.0
            tank2_outflow = 0.0

            for tank_idx, adjusts in flow_adjustments.items():
                for flow_type, value in adjusts.items():
                    if tank_idx == 0 and flow_type == "inflow":
                        tank1_inflow = value
                        inflow_1_list.append(value)      # Changed: was incorrectly using outflow_1_list
                    if tank_idx == 0 and flow_type == "outflow":
                        tank1_outflow = value
                        outflow_1_list.append(value)     # Correct
                    if tank_idx == 1 and flow_type == "inflow":
                        tank2_inflow = value
                        inflow_2_list.append(value)      # Correct
                    if tank_idx == 1 and flow_type == "outflow":
                        tank2_outflow = value
                        outflow_2_list.append(value)     # Correct

            # If no adjustment was made for a tank/flow, add zero
            if len(inflow_1_list) < len(time_points):
                inflow_1_list.append(0.0)
            if len(outflow_1_list) < len(time_points):
                outflow_1_list.append(0.0)
            if len(inflow_2_list) < len(time_points):
                inflow_2_list.append(0.0)
            if len(outflow_2_list) < len(time_points):
                outflow_2_list.append(0.0)

            # Compute derivatives and update states for each tank
            new_states = []

            all_adjusted_flows = []
            for i, tank in enumerate(tanks):
                # Create thermal model for this tank
                thermal_model = cls._define_thermal_model(
                    tank_configurations[i]["insulation"],
                    tank_configurations[i].get("heat_flux", None)
                )

                # Compute heat flux and temperatures
                heat_flux, temperatures = thermal_model.compute_heat_flux(
                    tank, current_states[i], mission_section
                )

                # Get dynamic model
                dynamic_model = DYNAMIC_MODEL_FACTORY.get_dynamic_model(
                    current_states[i], cls._define_target_conditions(operating_envelopes[i])
                )

                # Apply flow adjustments based on interactions
                adjusted_flows = cls._adjust_flows(
                    mission_section.fuel_flows,
                    flow_adjustments.get(i, {}),
                    current_states[i]
                )

                # Compute derivatives
                derivatives = current_states[i].compute_state_derivatives(
                    dynamic_model,
                    adjusted_flows,
                    heat_flux * HEAT_FLUX_FACTOR,
                    tank.compute_thermal_capacity(temperatures[0])
                )

                # Compute new state values
                new_temp = MULTISTEP_METHOD.compute_new_value(
                    [derivatives.temperature], current_states[i].temperature
                )
                new_pressure = MULTISTEP_METHOD.compute_new_value(
                    [derivatives.pressure], current_states[i].pressure
                )
                new_mass = current_states[i].fuel_mass + (derivatives.gas_mass + derivatives.liquid_mass) * MULTISTEP_METHOD.timestep

                # Create new state
                new_state = TankState(
                    tank,
                    new_temp,
                    new_pressure,
                    new_mass,
                    multi_flow=True
                )
                new_states.append(new_state)

            # Add new states to results
            for i, new_state in enumerate(new_states):
                all_states[i].add_tank_state(new_state)

        # Return both the tank states and the flow data
        flow_data = {
            'time_points': time_points,
            'inflow_1': inflow_1_list,
            'outflow_1': outflow_1_list,
            'inflow_2': inflow_2_list,
            'outflow_2': outflow_2_list
        }

        return all_states, flow_data

    @classmethod
    def _compute_tank_interactions(
        cls,
        current_states: list,
        interaction_rules: dict,
        operating_envelopes: list,
        mission_section: MissionSection = None,
        previous_flow_rate: float = None,
        step_index: int = None,
        total_steps: int = None
    ) -> dict:
        """Compute flow adjustments with gradual transitions."""
        flow_adjustments = {}

        # Mission-based interaction (Tank 1 supplies what Tank 2 needs)
        if interaction_rules.get("interaction_type") == "mission_based":
            tank2_idx = interaction_rules.get("consumer_tank_idx", 1)
            tank1_idx = interaction_rules.get("reservoir_tank_idx", 0)
            safety_margin = interaction_rules.get("safety_margin", 1.05)

            # Calculate total outflow from Tank 2 for this section
            total_outflow = 0.0
            for flow in mission_section.fuel_flows:
                if not hasattr(flow, "hydrogen"):  # It's an OutFlow
                    if isinstance(flow.mass_flow, list):
                        # Interpolate
                        from src.dynamics.dynamic_analysis import MissionSectionAnalysis
                        interpolated_flow = MissionSectionAnalysis.interpolate_mass_flows(
                            flow.mass_flow, step_index, total_steps
                        )
                        total_outflow += interpolated_flow
                    else:
                        total_outflow += flow.mass_flow

            # Apply safety margin and make positive (as it will be negated for outflow)
            transfer_flow_rate = abs(total_outflow) * safety_margin

            # Only transfer if needed
            if transfer_flow_rate > 0.001:  # Small threshold to avoid tiny flows
                # Initialize dictionaries
                if tank1_idx not in flow_adjustments:
                    flow_adjustments[tank1_idx] = {}
                if tank2_idx not in flow_adjustments:
                    flow_adjustments[tank2_idx] = {}

                # Apply gradual transition if previous flow rate exists
                if previous_flow_rate is not None:
                    # Limit rate of change to 0.01 kg/s per timestep
                    max_change = 0.01
                    if abs(transfer_flow_rate - previous_flow_rate) > max_change:
                        if transfer_flow_rate > previous_flow_rate:
                            transfer_flow_rate = previous_flow_rate + max_change
                        else:
                            transfer_flow_rate = previous_flow_rate - max_change

                # Store current flow rate for next call
                interaction_rules["previous_flow_rate"] = transfer_flow_rate

                # Tank 1 supplies (negative for outflow)
                flow_adjustments[tank1_idx]["outflow"] = -transfer_flow_rate
                # Tank 2 receives (positive for inflow)
                flow_adjustments[tank2_idx]["inflow"] = transfer_flow_rate

                # Tank 2 discharges to supply the mission
                flow_adjustments[tank2_idx]["outflow"] = -total_outflow  # Use the original outflow demand

        # Pressure-based interaction (Transfer when pressure drops)
        elif interaction_rules.get("pressure_based_flow", False):
            tank2_idx = interaction_rules.get("consumer_tank_idx", 1)
            tank1_idx = interaction_rules.get("reservoir_tank_idx", 0)

            pressure_threshold = interaction_rules.get("pressure_threshold",
                                                    operating_envelopes[tank2_idx].max_pressure * 1.1)

            if current_states[tank2_idx].pressure < pressure_threshold:
                # Increase flow from Tank 1 to Tank 2
                if tank1_idx not in flow_adjustments:
                    flow_adjustments[tank1_idx] = {}
                if tank2_idx not in flow_adjustments:
                    flow_adjustments[tank2_idx] = {}

                flow_rate = interaction_rules.get("flow_rate", 0.05)  # kg/s
                flow_adjustments[tank1_idx]["outflow"] = -flow_rate  # Negative for outflow
                flow_adjustments[tank2_idx]["inflow"] = flow_rate    # Positive for inflow

        return flow_adjustments

    @classmethod
    def _adjust_flows(
        cls,
        original_flows: list,
        adjustments: dict,
        current_state: TankState
    ) -> list:
        """Create tank-specific flows based on mission and tank role."""
        # For consumer tank (Tank 2) - Use original mission flows
        if "consumer" in adjustments:
            return original_flows

        # For reservoir tank (Tank 1) - Only use flows from tank interactions
        adjusted_flows = []

        # Add outflow from reservoir to consumer if needed
        if "outflow" in adjustments and abs(adjustments["outflow"]) > 0.001:
            from src.mission.mission_sections import OutFlow
            # Negative value for outflow
            adjusted_flows.append(OutFlow(-adjustments["outflow"], "gas"))

        return adjusted_flows


def main():
    pass


if __name__ == "__main__":
    main()


# End
