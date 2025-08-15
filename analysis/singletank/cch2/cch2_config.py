from src.fluids.hydrogen_retrievers import SinglePhaseRequester
from src.mission.mission import Mission
from src.insulation.foam_insulations import ConstantFoamInsulation
from src.materials.materials import Composite
from facades.analysis_facades import OperatingEnvelope, TankDimensions, InitialConditions, TargetConditions, MultiTankAnalysisFacade
from src.mission.mission_sections import OutFlow, MissionSection
import numpy as np


HOURS_TO_SECONDS = 3600.0
# Add scenario-specific timesteps
REFUEL_TIMESTEP = 0.5    # Smaller timestep for refuel (faster dynamics)
DISCHARGE_TIMESTEP = 5.0  # Standard timestep for discharge
DORMANCY_TIMESTEP = 60.0  # Larger timestep for dormancy (slower dynamics)

##########################
## COMMON CONFIGURATION ##
##########################



####################################
## REFUEL ANALYSIS CONFIGURATION ##
####################################

# Define Tank 1 parameters (reservoir)
p_init_refuel = 15e+5  # Pa
t_init_refuel = 70  # K
fill_refuel = 0.0 # no liquid
p_max_refuel = 5.0e+8  # Pa
p_min_refuel = None  # Pa
ambient_heat_load_refuel = 0.0  # W/m²
mass_fraction_refuel = 0.0 # analog to fill, but for gas wrt mass


# mission details for refuel
duration_hours_refuel = 0.5  # Duration of refuel in hours
altitude_refuel = 0.0  # Altitude in meters
fuel_flow_refuel = 0.1  # Fuel flow rate in kg/s

# get refuel hydrogen properties
refuel_hydrogen = SinglePhaseRequester().get_hydrogen_properties(p_init_refuel, t_init_refuel)

# Set multi_flow flag to True in initial conditions
initial_conditions_refuel = InitialConditions(p_init_refuel, t_init_refuel, fill_refuel,
                                             multi_flow=True, mass_fraction=mass_fraction_refuel)

# Make sure the mission uses an OutFlow with negative mass flow (which indicates inflow)
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
operating_window_refuel = OperatingEnvelope(p_max_refuel, p_min_refuel, None)


######################################
## DISCHARGE ANALYSIS CONFIGURATION ##
######################################

# Define Tank 1 parameters (reservoir) - 100kg, 300K, 500 bar
p_init_disch = 4e+7  # Pa
t_init_disch = 70  # K
fill_disch = 0.0  # no liquid
p_max_disch = 5.0e+8  # Pa
p_min_disch = 1500000  # Pa
ambient_heat_load_disch = 2000.0  # W/m²
mass_fraction_disch = 1.0

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
hydrogen_props = hydrogen_requester.get_hydrogen_properties(p_init_disch, t_init_disch)
VOLUME_MARGIN = 1.7 # make the tank 10% larger than the required volume
required_volume = VOLUME_MARGIN*(mission.required_fuel / hydrogen_props.density)
radius_1 = (3 * required_volume / (4 * np.pi))**(1/3)
print(f"Calculated tank radius: {radius_1:.2f} m")

# Instantiate tank objects
tank_material = Composite.carbon(np.radians(55))
tank_dimensions = TankDimensions(radius_1, 0.0)  # Spherical tank

# Insulation for both tanks
insulation_thickness = 0.05  # m
insulation = ConstantFoamInsulation.rohacell(insulation_thickness)

# Operating envelopes
operating_window_disch = OperatingEnvelope(p_max_disch, p_min_disch, None)

# Initial conditions
initial_conditions_disch = InitialConditions(p_init_disch, t_init_disch, fill_disch, multi_flow=False, mass_fraction=mass_fraction_disch)

# Define tank configurations for MultiTankAnalysisFacade
tank_config = [
    {
        "dimensions": tank_dimensions,
        "material": tank_material,
        "insulation": insulation,
        "heat_flux": ambient_heat_load_disch
    }
]


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
p_init_dormancy = 2e+7  # Pa (200 bar)
t_init_dormancy = 300   # K
fill_dormancy = 0.0     # no liquid
mass_fraction_dormancy = 0.11  # Will give approximately 100kg (adjust after first run if needed)

# Define Tank 2 parameters for dormancy - 50kg, 70K, 20 bar
p_init_2_dormancy = 2e+6  # Pa (20 bar)
t_init_2_dormancy = 70    # K
fill_2_dormancy = 0.0     # no liquid
mass_fraction_2_dormancy = 0.11  # Will give approximately 50kg (adjust after first run if needed)

# Initial conditions for dormancy
initial_conditions_dormancy = InitialConditions(p_init_dormancy, t_init_dormancy, fill_dormancy,
                                                 multi_flow=True, mass_fraction=mass_fraction_dormancy)
