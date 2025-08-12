from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.mission.mission import Mission
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.mission.mission_sections import OutFlow, MissionSection
import numpy as np


HOURS_TO_SECONDS = 3600.0
# Add scenario-specific timesteps
REFUEL_TIMESTEP = 2.0    # Smaller timestep for refuel (faster dynamics)
DISCHARGE_TIMESTEP = 10.0  # Standard timestep for discharge
DORMANCY_TIMESTEP = 60.0  # Larger timestep for dormancy (slower dynamics)

##########################
## COMMON CONFIGURATION ##
##########################

# system architecture

architecture = {
    "num_tanks": 3,
    "tank_ids": [0, 1, 2],
    "tank_names": ["LH2", "HPGH2", "CCH2"],
    "tank_aliases": ["Tank 1", "Tank 2", "Tank 3"],
    # fuel enters tank 1, which feeds into tank 2, which feeds into tank 3, which supplies the mission
    "connectivity": {
        "Tank 1": ["Tank 2"],
        "Tank 2": ["Tank 3"],
        "Tank 3": []
    }
}




####################################
## REFUEL ANALYSIS CONFIGURATION ##
####################################

# Define Tank 1 parameters (reservoir)
p_init_1_refuel = 5e+7  # Pa
t_init_1_refuel = 70  # K
fill_1_refuel = 0.0 # no liquid
p_max_1_refuel = 5.0e+8  # Pa
p_min_1_refuel = None  # Pa
ambient_heat_load_1_refuel = 2000.0  # W/m²
mass_fraction_1_refuel = 0.0 # analog to fill, but for gas wrt mass

# Define Tank 2 parameters (consumer)
p_init_2_refuel = 4e+7  # Pa
t_init_2_refuel =  70 # K
fill_2_refuel = 0.0 # no liquid
p_max_2_refuel = 5.0e+8  # Pa
p_min_2_refuel = None  # Pa
ambient_heat_load_2_refuel = 2000.0  # W/m²
mass_fraction_2_refuel = 0.1

# mission details for refuel
duration_hours_refuel = 1.0  # Duration of refuel in hours
altitude_refuel = 0.0  # Altitude in meters
fuel_flow_refuel = 0.1  # Fuel flow rate in kg/s

# get refuel hydrogen properties
refuel_hydrogen = SinglePhaseRequester().get_hydrogen_properties(p_init_1_refuel, t_init_1_refuel)

refuel_mission = Mission([
    MissionSection(
        duration_hours_refuel * HOURS_TO_SECONDS,
        [
            OutFlow(-fuel_flow_refuel, "gas")  # NEGATIVE OutFlow = INFLOW to system
        ],
        altitude_refuel,
        0.0,
        "Refuelling"
    )
])

# Initial conditions
initial_conditions_1_refuel = InitialConditions(p_init_1_refuel, t_init_1_refuel, fill_1_refuel, multi_flow=True, mass_fraction=mass_fraction_1_refuel)
initial_conditions_2_refuel = InitialConditions(p_init_2_refuel, t_init_2_refuel, fill_2_refuel, multi_flow=True, mass_fraction=mass_fraction_2_refuel)

# No transfer between tanks during refuel (keep fuel in Tank 1)
refuel_interaction_rules = {
    "type": "conditional",
    "max_flow_rate": 0.0,  # No flow between tanks
    "active_at_start": False,
    "conditions": [],
    "default_flow": 0.0
}

######################################
## DISCHARGE ANALYSIS CONFIGURATION ##
######################################

# Define Tank 1 parameters (reservoir) - 100kg, 300K, 500 bar
p_init_1 = 5e+7  # Pa (500 bar)
t_init_1 = 300  # K
fill_1 = 0.0  # no liquid
p_max_1 = 5.0e+8  # Pa
p_min_1 = 1500000  # Pa
ambient_heat_load_1 = 2000.0  # W/m²
mass_fraction_1 = 0.5
# Define Tank 2 parameters (consumer) - 200kg, 70K, 400 bar
p_init_2 = 4e+7  # Pa (400 bar)
t_init_2 = 70  # K
fill_2 = 0.0  # no liquid
p_max_2 = 5.0e+8  # Pa
p_min_2 = 1500000  # Pa
ambient_heat_load_2 = 2000.0  # W/m²
mass_fraction_2 = 0.45

# Get mission details
mission = Mission.atr72()
# Add after creating the mission
# print(f"Mission created: {mission.__class__.__name__}")
# print(f"Number of sections: {len(mission.sections)}")
for i, section in enumerate(mission.sections):
    print(f"Section {i+1}: {section.duration/3600:.4f}h")
print(f"Total duration: {sum(s.duration for s in mission.sections)/3600:.4f}h")
total_fuel_mass = mission.required_fuel
print(f"Total fuel mass required for mission: {total_fuel_mass:.2f} kg")

# Calculate appropriate radius based on required mass
hydrogen_requester = SinglePhaseRequester()
hydrogen_props = hydrogen_requester.get_hydrogen_properties(p_init_1, t_init_1)
VOLUME_MARGIN = 1.5 # make the tank 10% larger than the required volume
required_volume = VOLUME_MARGIN*(mission.required_fuel / hydrogen_props.density)
radius_1 = (3 * required_volume / (4 * np.pi))**(1/3)
radius_2 = radius_1  # Same radius for both tanks
print(f"Calculated tank radius: {radius_1:.2f} m")

# Instantiate tank objects
tank_material = Composite.carbon(np.radians(55))
tank_dimensions_1 = TankDimensions(radius_1, 0.0)  # Spherical tank
tank_dimensions_2 = TankDimensions(radius_2, 0.0)  # Spherical tank

# Insulation for both tanks
insulation_thickness = 0.05  # m
insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

# Operating envelopes
operating_window_1 = OperatingEnvelope(p_max_1, p_min_1, None)
operating_window_2 = OperatingEnvelope(p_max_2, p_min_2, None)

# Initial conditions
initial_conditions_1 = InitialConditions(p_init_1, t_init_1, fill_1, multi_flow=True, mass_fraction=mass_fraction_1)
initial_conditions_2 = InitialConditions(p_init_2, t_init_2, fill_2, multi_flow=True, mass_fraction=mass_fraction_2)

# Define tank configurations for MultiTankAnalysisFacade
tank_configs = [
    {
        "dimensions": tank_dimensions_1,
        "material": tank_material,
        "insulation": insulation,
        "heat_flux": ambient_heat_load_1
    },
    {
        "dimensions": tank_dimensions_2,
        "material": tank_material,
        "insulation": insulation,
        "heat_flux": ambient_heat_load_2
    }
]

interaction_rules = {
    "type": "mission_based",
    "max_flow_rate": 0.1,  # kg/s - limit maximum flow between tanks
    "active_at_start": True,
    "safety_factor": 0.8
}

# Define interaction rules
# interaction_rules = {
# "type": "conditional",
# "max_flow_rate": 0.1,  # kg/s - limit maximum flow between tanks
# "active_at_start": True,
# "conditions": [
#     {
#         "type": "time_after",
#         "tank_idx": 1,        # Monitor Tank 2 (consumer)
#         "threshold": 0.1*3600,
#         "use_mission_flow": True,
#         "safety_factor": 0.8  # Same as before
#     }
# ],
# "default_flow": 0.0  # No flow until condition is met
# }

#####################################
## DORMANCY ANALYSIS CONFIGURATION ##
#####################################

# DORMANCY ANALYSIS CONFIGURATION
duration_hours = 24.0  # Duration of dormancy in hours
altitude = 0.0  # Altitude in meters

# Create a dormancy mission with a single section
dormancy_mission = Mission([
    Mission.dormancy_section(
        duration=duration_hours,
        altitude=altitude,
        fuel_flow=0.0,  # Will be forced to zero anyway
        throttle=0.0,   # Will be forced to zero anyway
        phase="gas",    # Dummy value, not used
        mach_number=0.0
    )
])

# Define Tank 1 parameters for dormancy - 100kg, 300K, 200 bar
p_init_1_dormancy = 2e+7  # Pa (200 bar)
t_init_1_dormancy = 300   # K
fill_1_dormancy = 0.0     # no liquid
mass_fraction_1_dormancy = 0.11  # Will give approximately 100kg (adjust after first run if needed)

# Define Tank 2 parameters for dormancy - 50kg, 70K, 20 bar
p_init_2_dormancy = 2e+6  # Pa (20 bar)
t_init_2_dormancy = 70    # K
fill_2_dormancy = 0.0     # no liquid
mass_fraction_2_dormancy = 0.11  # Will give approximately 50kg (adjust after first run if needed)

# Initial conditions for dormancy
initial_conditions_1_dormancy = InitialConditions(p_init_1_dormancy, t_init_1_dormancy, fill_1_dormancy,
                                                 multi_flow=True, mass_fraction=mass_fraction_1_dormancy)
initial_conditions_2_dormancy = InitialConditions(p_init_2_dormancy, t_init_2_dormancy, fill_2_dormancy,
                                                 multi_flow=True, mass_fraction=mass_fraction_2_dormancy)

dormancy_interaction_rules = {
    "type": "mission_based",
    "max_flow_rate": 0.0,  # No flow between tanks
    "active_at_start": False,  # Disable flow
    "safety_factor": 0.0
}
