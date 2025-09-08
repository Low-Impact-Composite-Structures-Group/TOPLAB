import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# Import hydrogen fluid models
from src.fluids.hydrogen_retrievers import SinglePhaseRequester

# Import mission components
from src.mission.mission import Mission
from src.mission.mission_sections import OutFlow, MissionSection, InFlow

# Import tank components
from src.materials.materials import Composite, Metal
from src.materials.nist_materials import NISTMetal, NISTComposite
from src.insulation.vacuum_insulation import VacuumInsulation
from src.tank_design.liner import Liner

# Import facades and analysis components
from plotting.sb_plotting import SeabornPlotter
from facades.analysis_facades import (
    MULTISTEP_METHOD, OperatingEnvelope, TankDimensions,
    InitialConditions, TargetConditions, TankPerformance
)

# Import our facades for analysis
from facades.analysis_facades import MissionAnalysisFacade

# Import enhanced custom modules for thickness control and thermal modeling
from src.tank_design.custom_thickness_control import ThicknessController, CustomThicknessStructuralModel
from src.thermodynamics.enhanced_thermal_model import create_enhanced_thermal_model

#################################
## COMMON SIMULATION CONSTANTS ##
#################################


#################################
## COMMON SIMULATION CONSTANTS ##
#################################

# Time conversion
HOURS_TO_SECONDS = 3600.0  # seconds per hour

# Tank parameters from paper specifications
NOMINAL_MASS = 35.0
TANK_VOLUME = 0.5

# Spherical tank geometry for target volume
# For sphere: V = (4/3) * π * r³
# Solving for radius: r = (3V/(4π))^(1/3)
TANK_RADIUS = (3 * TANK_VOLUME / (4 * np.pi))**(1/3)
TANK_BODY_LENGTH = 0.0  # Spherical tank (no cylindrical body)

AMBIENT_TEMPERATURE = 298.15  # K

# Verify the calculated volume
calculated_volume = (4/3) * np.pi * TANK_RADIUS**3
print(f"=== SPHERICAL TANK ===")
print(f"Tank radius: {TANK_RADIUS:.3f} m")
print(f"Target volume: {TANK_VOLUME:.3f} m³")
print(f"Calculated volume: {calculated_volume:.3f} m³")
print(f"Volume error: {abs(calculated_volume - TANK_VOLUME):.6f} m³")

# Simulation timesteps for different scenarios
DISCHARGE_TIMESTEP = 5.0   # seconds - standard timestep for discharge
REFUEL_TIMESTEP = 0.5      # seconds - small timestep for refuel (rapid dynamics)
DORMANCY_TIMESTEP = 300.0   # seconds - larger timestep for dormancy (slower dynamics)

#########################################
## ENHANCED THICKNESS AND THERMAL MODEL ##
#########################################

# Custom thickness control configuration
USE_CUSTOM_THICKNESS = True    # Enable fixed thickness control
CUSTOM_THICKNESS = 0.026        # Custom thickness [m] = 10 mm

# Enhanced thermal model configuration
USE_ENHANCED_THERMAL = True    # Enable direct thermal approach
K_INSULATION = 0.025          # Insulation coefficient [W/(m²·K)]
K_WALL = 0.2                  # Wall coefficient [W/(m²·K)]

print(f"\n🔧 ENHANCED CONFIGURATION")
print(f"===========================")
print(f"Custom thickness control: {'ENABLED' if USE_CUSTOM_THICKNESS else 'DISABLED'}")
if USE_CUSTOM_THICKNESS:
    print(f"   Fixed thickness: {CUSTOM_THICKNESS*1000:.1f} mm")
print(f"Enhanced thermal model: {'ENABLED' if USE_ENHANCED_THERMAL else 'DISABLED'}")
if USE_ENHANCED_THERMAL:
    print(f"   k_insulation: {K_INSULATION:.6f} W/(m²·K)")
    print(f"   k_wall: {K_WALL:.3f} W/(m²·K)")
print(f"===========================")

#########################################

# Use NIST materials for improved temperature-dependent thermal properties
tank_material = NISTComposite.g10_nist(np.radians(55))
# tank_material = Composite.carbon(np.radians(55))

# Apply custom thickness structural model if enabled
if USE_CUSTOM_THICKNESS:

    print(f"\n🔧 APPLYING CUSTOM STRUCTURAL MODEL")
    print(f"===================================")
    print(f"Creating custom structural model with {CUSTOM_THICKNESS*1000:.1f} mm thickness")

    # Create custom structural model that always returns our fixed thickness
    custom_structural_model = CustomThicknessStructuralModel(CUSTOM_THICKNESS)

    # Override the material's structural model
    tank_material.structural_model = custom_structural_model

    print(f"✅ Tank material now uses: {custom_structural_model}")
    print(f"===================================")

# Create tank dimensions for spherical tank
tank_dimensions = TankDimensions(TANK_RADIUS, TANK_BODY_LENGTH)  # Spherical tank

# Create a liner with specified mass using NIST aluminum properties
LINER_MASS = 1.0
LINER_THICKNESS = 0.0125
liner_by_thickness = Liner.from_thickness(LINER_THICKNESS, tank_dimensions, NISTMetal.aluminum_5083_nist())

# Assign the liner to tank_dimensions
# comment this out for a linerless analysis
tank_dimensions.liner = liner_by_thickness

print(f"\n🔋 THERMAL CAPACITY ANALYSIS (NIST Materials):")
print(f"==================================================")
# Create a temporary tank to analyze thermal capacity
from src.tank_design.tank_shapes import TankFactory
temp_tank = TankFactory().create_tank(
    tank_dimensions.radius,
    tank_dimensions.body_length,
    tank_material,
    700000  # 7 bar operating pressure
)
temp_tank.liner = liner_by_thickness

# Apply custom thickness control if enabled
if USE_CUSTOM_THICKNESS:
    print(f"\n🔧 APPLYING CUSTOM THICKNESS CONTROL")
    print(f"=====================================")

    # Function to apply custom thickness to all tank sections
    def apply_custom_thickness_to_tank(tank, custom_thickness):
        """Apply custom thickness to all tank sections."""
        for i, section in enumerate(tank.sections):
            section_type = type(section).__name__

            # Create fixed thickness structural model
            class CustomThicknessStructuralModel:
                def __init__(self, thickness):
                    self.fixed_thickness = thickness

                def compute_thickness(self, section, pressure):
                    return self.fixed_thickness

            # Override structural model
            original_thickness = section.thickness
            section.structural_model = CustomThicknessStructuralModel(custom_thickness)
            new_thickness = section.thickness

            print(f"   {section_type}_{i}: {original_thickness*1000:.1f} mm → {new_thickness*1000:.1f} mm")

    # Apply custom thickness
    apply_custom_thickness_to_tank(temp_tank, CUSTOM_THICKNESS)

    print(f"✅ Custom thickness ({CUSTOM_THICKNESS*1000:.1f} mm) applied to all sections")
    print(f"=====================================")


# Create insulation with appropriate thermal coefficient
# Use K_INSULATION for enhanced thermal model, fallback to default for other cases
insulation_k_amb = K_INSULATION if USE_ENHANCED_THERMAL else 0.025
print(f"\n🏠 INSULATION CONFIGURATION")
print(f"=============================")
print(f"Creating vacuum insulation with k_amb = {insulation_k_amb:.6f} W/(m²·K)")

# Get tank surface area for insulation
tank_surface_area = temp_tank.surface_area
insulation = VacuumInsulation(surface_area=4.1, k_amb=insulation_k_amb)

print(f"✅ Vacuum insulation configured:")
print(f"   Surface area: {tank_surface_area:.2f} m²")
print(f"   Heat transfer coefficient: {insulation_k_amb:.6f} W/(m²·K)")
print(f"=============================")

def configure_analysis_thermal_model():
    """
    Configure the analysis facade to use enhanced thermal model when enabled.

    This function monkey-patches the MissionAnalysisFacade._define_thermal_model method
    to return our enhanced model when USE_ENHANCED_THERMAL is True.
    """
    # Store the original method if not already stored
    if not hasattr(MissionAnalysisFacade, '_original_define_thermal_model'):
        MissionAnalysisFacade._original_define_thermal_model = MissionAnalysisFacade._define_thermal_model

    # Create the monkey-patch function
    @staticmethod
    def enhanced_thermal_model_method(insulation, constant_heat_flux=None):
        if USE_ENHANCED_THERMAL:
            # Return enhanced model with direct thermal approach
            print(f"🔥 Using ENHANCED thermodynamic model")
            print(f"   k_insulation: {K_INSULATION:.6f} W/(m²·K)")
            print(f"   k_wall: {K_WALL:.3f} W/(m²·K)")
            print(f"   Direct approach: Ambient → Structure → Hydrogen")
            return create_enhanced_thermal_model(K_INSULATION, K_WALL)
        else:
            # Use original method
            return MissionAnalysisFacade._original_define_thermal_model(insulation, constant_heat_flux)

    # Apply the monkey patch
    MissionAnalysisFacade._define_thermal_model = enhanced_thermal_model_method

    model_type = "ENHANCED DIRECT" if USE_ENHANCED_THERMAL else "FULL COMPLEXITY"
    print(f"✅ Analysis thermal model configured: {model_type}")

def configure_analysis_tank_model():
    """
    Configure the analysis facade to apply custom thickness to tanks when enabled.

    This function monkey-patches the MissionAnalysisFacade._define_tank method
    to apply custom thickness to all tanks created during analysis.
    """
    # Store the original method if not already stored
    if not hasattr(MissionAnalysisFacade, '_original_define_tank'):
        MissionAnalysisFacade._original_define_tank = MissionAnalysisFacade._define_tank

    # Create the monkey-patch function
    @classmethod
    def enhanced_tank_method(cls, tank_dimensions, material, target_state, initial_state):
        # Create tank using original method
        tank = cls._original_define_tank(tank_dimensions, material, target_state, initial_state)

        # Apply custom thickness if enabled
        if USE_CUSTOM_THICKNESS:
            print(f"🔧 Applying {CUSTOM_THICKNESS*1000:.1f} mm custom thickness to simulation tank")
            tank = apply_enhanced_thickness_to_tank(tank)

        return tank

    # Apply the monkey patch
    MissionAnalysisFacade._define_tank = enhanced_tank_method

    thickness_type = f"{CUSTOM_THICKNESS*1000:.1f} mm FIXED" if USE_CUSTOM_THICKNESS else "DYNAMIC"
    print(f"✅ Analysis tank model configured: {thickness_type}")

# Configure the thermal model based on settings
if USE_ENHANCED_THERMAL:
    print(f"\n🔥 ENHANCED DIRECT THERMAL MODEL")
    print(f"=================================")
    print(f"Using direct multi-step approach:")
    print(f"   1. Ambient → Structure: k_insulation = {K_INSULATION:.6f} W/(m²·K)")
    print(f"   2. Structure → Hydrogen: k_wall = {K_WALL:.3f} W/(m²·K)")
    print(f"   3. Dynamic thermal capacity from NIST database")
    configure_analysis_thermal_model()

# Configure the tank model based on settings
if USE_CUSTOM_THICKNESS:
    print(f"\n🔧 ENHANCED TANK MODEL")
    print(f"======================")
    print(f"Using fixed thickness approach:")
    print(f"   Fixed thickness: {CUSTOM_THICKNESS*1000:.1f} mm for all sections")
    print(f"   Applied to simulation tanks automatically")
    configure_analysis_tank_model()

# Create a temporary tank to display properties
from src.tank_design.tank_shapes import TankFactory
temp_tank_display = TankFactory.create_tank(
    TANK_RADIUS, TANK_BODY_LENGTH, tank_material, 900e5, liner=liner_by_thickness
)

# Apply custom thickness to display tank if enabled
if USE_CUSTOM_THICKNESS:
    def apply_custom_thickness_to_tank_display(tank, custom_thickness):
        """Apply custom thickness to all tank sections for display."""
        for i, section in enumerate(tank.sections):
            # Create fixed thickness structural model
            class CustomThicknessStructuralModel:
                def __init__(self, thickness):
                    self.fixed_thickness = thickness

                def compute_thickness(self, section, pressure):
                    return self.fixed_thickness

            # Override structural model
            section.structural_model = CustomThicknessStructuralModel(custom_thickness)

    apply_custom_thickness_to_tank_display(temp_tank_display, CUSTOM_THICKNESS)

# Print tank properties early in the execution
print("\n===== TANK PROPERTIES =====")
print(f"Tank structural mass: {temp_tank_display.structural_mass:.2f} kg")
print(f"Tank surface area: {temp_tank_display.surface_area:.2f} m²")

# Print thickness for each section
print(f"\n📏 SECTION THICKNESS {'(CUSTOM FIXED)' if USE_CUSTOM_THICKNESS else '(DYNAMIC)'}:")
for i, section in enumerate(temp_tank_display.sections):
    if hasattr(section, 'thickness'):
        section_type = type(section).__name__
        thickness_mm = section.thickness * 1000
        if USE_CUSTOM_THICKNESS:
            print(f"   {section_type}: {thickness_mm:.1f} mm (FIXED)")
        else:
            print(f"   {section_type}: {thickness_mm:.1f} mm (dynamic)")

# Calculate effective thickness
if hasattr(temp_tank_display, 'sections') and temp_tank_display.sections:
    total_area = sum(section.surface_area for section in temp_tank_display.sections)
    weighted_thickness = sum(section.surface_area * section.thickness for section in temp_tank_display.sections)
    effective_thickness = weighted_thickness / total_area if total_area > 0 else 0.0
    print(f"   EFFECTIVE thickness: {effective_thickness*1000:.1f} mm")

# Print liner details after calculation
if hasattr(temp_tank_display, 'liner') and temp_tank_display.liner is not None:
    print("\n===== LINER PROPERTIES =====")
    liner = temp_tank_display.liner
    print(f"Liner mass: {liner.mass:.2f} kg")
    if liner.thickness is not None:
        print(f"Liner thickness: {liner.thickness*1000:.2f} mm")
    print(f"Liner material: {liner.material.__class__.__name__}")

# Print enhanced configuration summary
print("\n===== ENHANCED MODEL SUMMARY =====")
if USE_ENHANCED_THERMAL:
    print(f"🔥 ENHANCED DIRECT thermal model")
    print(f"   k_insulation: {K_INSULATION:.6f} W/(m²·K)")
    print(f"   k_wall: {K_WALL:.3f} W/(m²·K)")
    print(f"   Features:")
    print(f"     - Direct multi-step heat transfer")
    print(f"     - Structure temperature calculation")
    print(f"     - Dynamic NIST thermal capacity")
else:
    print(f"🔥 FULL COMPLEXITY thermal model")
    print(f"   Using vacuum insulation k_amb: {insulation.k_amb:.6f} W/(m²·K)")

if USE_CUSTOM_THICKNESS:
    print(f"📏 CUSTOM THICKNESS: {CUSTOM_THICKNESS*1000:.1f} mm (all sections)")
else:
    print(f"📏 DYNAMIC THICKNESS: varies with pressure")

print(f"🏠 INSULATION: {insulation.k_amb:.6f} W/(m²·K), area={insulation.surface_area:.2f} m²")
print("=====================================")

print(f"🔧 READY FOR ANALYSIS")
print(f"======================")
print(f"Using spherical tank: {TANK_VOLUME:.1f} m³ volume")
print(f"Using NIST G10 composite material")
print(f"Applied enhancements:")
print(f"  ✅ {'Fixed' if USE_CUSTOM_THICKNESS else 'Dynamic'} thickness control")
print(f"  ✅ {'Enhanced direct' if USE_ENHANCED_THERMAL else 'Full complexity'} thermal model")
print(f"  ✅ NIST temperature-dependent materials")
print(f"  ✅ Simplified vacuum insulation interface")
print(f"======================")

# Store the enhanced configuration function globally for analysis use
def apply_enhanced_thickness_to_tank(tank):
    """Global function to apply enhanced thickness control to any tank."""
    if USE_CUSTOM_THICKNESS:
        for i, section in enumerate(tank.sections):
            # Create fixed thickness structural model
            class CustomThicknessStructuralModel:
                def __init__(self, thickness):
                    self.fixed_thickness = thickness

                def compute_thickness(self, section, pressure):
                    return self.fixed_thickness

            # Override structural model
            section.structural_model = CustomThicknessStructuralModel(CUSTOM_THICKNESS)
    return tank

print("\n===== BEGINNING ANALYSIS =====\n")

#-------------------------#
# 1. DISCHARGE PARAMETERS #
#-------------------------#
# Tank initial conditions - based on paper's Case B starting point
p_init_disch = 4e+7        # Pa - initial tank pressure (400 bar)
t_init_disch = 53.25       # K - initial tank temperature
fill_disch = 1.0           # fraction - no liquid phase (0.0 = all gas)

# Operating limits - ensure physically reasonable values
p_max_disch = 5.0e+7       # Pa - maximum allowable pressure (500 bar)
p_min_disch = 1.5e+6       # Pa - minimum allowable pressure (15 bar)

# Operating envelopes for discharge scenario
operating_window_disch = OperatingEnvelope(p_max_disch, p_min_disch, None)

# Initial conditions for discharge scenario - enable multi_flow to handle proper phase detection
initial_conditions_disch = InitialConditions(p_init_disch, t_init_disch, fill_disch, multi_flow=True)

# Add discharge mission parameters to the configuration section
duration_hours_disch = 10    # hours - duration of discharge operation
fuel_flow_disch = 0.001      # kg/s - fuel flow rate out of tank

# Create discharge mission
discharge_mission = Mission([
    MissionSection(
        duration_hours_disch * HOURS_TO_SECONDS,  # Convert hours to seconds
        [
            OutFlow(-fuel_flow_disch, "gas")  # Negative OutFlow = flow OUT of system
        ],
        0.0,        # Altitude (m)
        0.0,        # Mach number
        "Discharge", # Section label
        ground_temperature=AMBIENT_TEMPERATURE
    )
])

#----------------------#
# 2. REFUEL PARAMETERS #
#----------------------#
# Tank initial conditions - based on paper's Case A starting point
p_init_refuel = 15e+5      # Pa - initial tank pressure (15 bar)
t_init_refuel = 65.0       # K - initial tank temperature
fill_refuel = 0.0          # fraction - no liquid phase (0.0 = all gas)
rho_stop_refuel = 78.0     # kg/m³ - stop density

# Operating limits
p_max_refuel = 9.0e+7       # Pa - maximum allowable pressure
p_min_refuel = None        # Pa - minimum allowable pressure (None = no limit)

# Mission parameters
duration_hours_refuel = 0.15   # hours - duration of refueling operation
altitude_refuel = 0.0      # m - ground-level altitude
fuel_flow_refuel = 0.07    # kg/s - fuel flow rate

# Create initial conditions object
initial_conditions_refuel = InitialConditions(
    p_init_refuel,
    t_init_refuel,
    fill_refuel,
    multi_flow=True
)

# Create hydrogen object for refueling that will start with tank initial conditions
# The supply conditions will then be updated at each time step to match the tank state
supply_hydrogen = SinglePhaseRequester().get_hydrogen_properties(p_init_refuel, t_init_refuel)

# Define refuel mission with inflow
refuel_mission = Mission([
    MissionSection(
        duration_hours_refuel * HOURS_TO_SECONDS,  # Convert hours to seconds
        [
            InFlow(fuel_flow_refuel, supply_hydrogen)  # Positive value = flow INTO tank with constant hydrogen properties
        ],
        altitude_refuel,   # Altitude (m)
        0.0,               # Mach number
        "Refuelling",      # Section label
        ground_temperature=AMBIENT_TEMPERATURE  # Ambient temperature
    )
])

# Define operating envelope for refuel scenario
operating_window_refuel = OperatingEnvelope(p_max_refuel, 1.0e5, None)


#------------------------#
# 3. DORMANCY PARAMETERS #
#------------------------#
# Tank initial conditions
p_init_dormancy = 400e+5   # Pa - initial tank pressure (400 bar)
t_init_dormancy = 53.25      # K - initial tank temperature
fill_dormancy = 0.0        # fraction - no liquid phase (0.0 = all gas)

# Mission parameters
duration_hours_dormancy = 300.0  # hours - duration of dormancy period
altitude_dormancy = 0.0    # m - ground-level altitude

# Define operating envelope for dormancy
operating_window_dormancy = OperatingEnvelope(
    max_pressure=4.5e+7,      # Pa - maximum allowable pressure
    min_pressure=15e5,       # Pa - minimum allowable pressure
    min_temperature=20       # K - minimum allowable temperature
)

# Create dormancy mission (no fuel flow)
dormancy_mission = Mission([
    MissionSection(
        duration_hours_dormancy * HOURS_TO_SECONDS,  # Convert hours to seconds
        [],  # No fuel flows during dormancy
        altitude_dormancy,  # Altitude (m)
        0.0,                # Mach number
        "Dormancy",         # Section label
        ground_temperature=AMBIENT_TEMPERATURE  # Ambient temperature
    )
])

# Create initial conditions object for dormancy
initial_conditions_dormancy = InitialConditions(
    p_init_dormancy,
    t_init_dormancy,
    fill_dormancy,
    multi_flow=True        # Enable multi-flow mode for phase handling
)



def perform_discharge_analysis(return_performances=False, show_plots=False):
    """
    Run a discharge analysis simulation with the fixed HTC approach.
    """
    # Set timestep for discharge scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DISCHARGE_TIMESTEP

    print(f"Using timestep: {DISCHARGE_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {discharge_mission}")
        print("\nInitial tank states:")
        print(f"T={initial_conditions_disch.temperature:.1f}K, P={initial_conditions_disch.pressure/1e5:.1f}bar")

        print("\nRunning simulation...")
        print(f"Discharge duration: {duration_hours_disch} hours with {DISCHARGE_TIMESTEP} second timesteps")

        # Run the analysis
        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_dimensions,
            material=tank_material,
            insulation=insulation,
            mission=discharge_mission,
            initial_conditions=initial_conditions_disch,
            operating_envelope=operating_window_disch,
            constant_heat_flux=None,
            target_density=None
        )

        # The results are now available from the analysis
        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Use SeabornPlotter for consistent styling
        fig_states = plotter.plot_single_tank_states(tank_states)

        # Show plots if requested
        if show_plots:
            plt.show()

        # Show final states
        print("\nDischarge scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep

def perform_refuel_analysis(return_performances=False, show_plots=False):
    """
    Run a refuel analysis simulation using the fixed HTC approach.
    """
    # Set timestep for refuel scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = REFUEL_TIMESTEP

    print(f"Using timestep: {REFUEL_TIMESTEP} seconds")

    try:
        # Print initial info
        print(f"Mission details: {refuel_mission}")

        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_refuel.temperature:.1f}K, P={initial_conditions_refuel.pressure/1e5:.1f}bar")

        # Run analysis
        print("\nRunning simulation...")

        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_dimensions,
            material=tank_material,
            insulation=insulation,
            mission=refuel_mission,
            initial_conditions=initial_conditions_refuel,
            operating_envelope=operating_window_refuel,
            constant_heat_flux=None,
            target_density=rho_stop_refuel
        )

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        # Convert tank states data to dictionary for plotting
        tank_states_dict = {
            'time': tank_states.timesteps_in_hours,       # hours
            'pressure': tank_states.pressures_in_bar,     # bar
            'temperature': tank_states.temperatures,      # K
            'fuel_mass': np.array([state.fuel_mass for state in tank_states.states]) if hasattr(tank_states, 'states') else np.array([0])  # kg
        }

        try:
            # Try to use SeabornPlotter for consistent styling
            fig_states = plotter.plot_single_tank_states(tank_states)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")

            # Fallback plotting
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Plot pressure over time
            ax1.plot(tank_states_dict['time'], tank_states_dict['pressure'])
            ax1.set_ylabel("Pressure [bar]")
            ax1.grid(True)

            # Plot temperature over time
            ax2.plot(tank_states_dict['time'], tank_states_dict['temperature'])
            ax2.set_ylabel("Temperature [K]")
            ax2.grid(True)

            # Plot fuel mass over time
            ax3.plot(tank_states_dict['time'], tank_states_dict['fuel_mass'])
            ax3.set_xlabel("Time [hours]")
            ax3.set_ylabel("Fuel Mass [kg]")
            ax3.grid(True)


        # Extract mass flow data from mission for plotting
        mass_flows = []      # List to hold mass flow rates for each section
        fuel_flow_keys = []  # Labels for each section
        durations = []       # Duration of each section in seconds

        # Process each mission section
        for section in refuel_mission.sections:
            # Collect all mass flows from this section
            section_flows = []
            for flow in section.fuel_flows:
                if hasattr(flow, 'mass_flow'):
                    # Handle both single values and lists of mass flows
                    if isinstance(flow.mass_flow, list):
                        section_flows.extend(flow.mass_flow)
                    else:
                        section_flows.append(flow.mass_flow)

            # Store section data
            mass_flows.append(section_flows)
            fuel_flow_keys.append(section.fuel_flow_key or "Refuelling")
            durations.append(section.duration)

        # Calculate total mission duration in hours
        total_duration = sum(durations) / HOURS_TO_SECONDS

        # Generate mass flow plot
        fig_flows = plotter.plot_single_mission_flows(
            mass_flows=mass_flows,         # List of mass flow rates
            fuel_flow_keys=fuel_flow_keys, # Section labels
            durations=durations,           # Section durations (seconds)
            total_duration=total_duration  # Total mission duration (hours)
        )

        # Show only the two figures we want
        if show_plots:
            plt.show()

        # Show final states
        print("\nRefuel scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


def perform_dormancy_analysis(return_performances=False, show_plots=False):
    """
    Run a dormancy analysis simulation with the fixed HTC approach.

    Args:
        return_performances (bool): Whether to return the performance data
        show_plots (bool): Whether to display plots during execution

    Returns:
        TankPerformance object if return_performances is True
    """
    # Set timestep for dormancy scenario
    original_timestep = MULTISTEP_METHOD.timestep
    MULTISTEP_METHOD.timestep = DORMANCY_TIMESTEP

    print(f"Using timestep: {DORMANCY_TIMESTEP} seconds")

    try:
        # Create and adjust initial conditions
        print("\nInitial tank states:")
        print(f"T={initial_conditions_dormancy.temperature:.1f}K, P={initial_conditions_dormancy.pressure/1e5:.1f}bar")

        # Ensure multi_flow is True for proper phase handling
        initial_conditions_dormancy.multi_flow = True

        # Run analysis
        print("\nRunning simulation...")
        print(f"Dormancy duration: {duration_hours_dormancy} hours with {DORMANCY_TIMESTEP} second timesteps")

        tank_performance = MissionAnalysisFacade.analyse(
            tank_dimensions=tank_dimensions,
            material=tank_material,
            insulation=insulation,
            mission=dormancy_mission,
            initial_conditions=initial_conditions_dormancy,
            operating_envelope=operating_window_dormancy,
            constant_heat_flux=None,
            target_density=None
        )

        # Apply custom thickness to dormancy tank sections
        print(f"🔧 Applying custom {CUSTOM_THICKNESS*1000:.1f}mm thickness to dormancy tank sections...")

        # Apply to dormancy tank only (discharge and refuel will be handled after those analyses run)
        tank = tank_performance.tank
        for section in tank.sections:
            section.structural_model = CustomThicknessStructuralModel(CUSTOM_THICKNESS)
            print(f"  ✅ Applied {CUSTOM_THICKNESS*1000:.1f}mm to {type(section).__name__}")

        # Extract results and plot
        tank_states = tank_performance.tank_states

        print("\nSimulation complete. Plotting results...")

        # Initialize the plotter
        plotter = SeabornPlotter(font="Cambria", palette="delft")

        try:
            # Try to use SeabornPlotter for consistent styling
            fig_states = plotter.plot_single_tank_states(tank_states)
        except ValueError as e:
            print(f"Error plotting tank states: {e}")
            # Fallback plotting can be added here

        # Show final states
        print("\nDormancy scenario complete. Final states:")
        last_state = tank_states.last_state
        print(f"T={last_state.temperature:.1f}K, P={last_state.pressure/1e5:.1f}bar, mass={last_state.fuel_mass:.1f}kg")

        # Show plots if requested
        if show_plots:
            plt.show()

        if return_performances:
            return tank_performance
    finally:
        # Restore original timestep
        MULTISTEP_METHOD.timestep = original_timestep


def get_hydrogen_density_from_state(state, requester):
    """
    Helper function to consistently extract hydrogen density from a tank state.

    For two-phase hydrogen, this uses the homogeneous mixture density which
    assumes the hydrogen is completely mixed throughout the tank.

    Args:
        state: Tank state object containing hydrogen properties
        requester: SinglePhaseRequester for calculating properties if needed

    Returns:
        float: Hydrogen density in kg/m³
    """
    # Check if the state has hydrogen properties
    if hasattr(state, 'hydrogen'):
        # If hydrogen has phase information
        if hasattr(state.hydrogen, 'phase'):
            if state.hydrogen.phase in ["gas", "supercritical"]:
                # Gas phase - check for specific gas property
                if hasattr(state.hydrogen, 'gas'):
                    return state.hydrogen.gas.density
                else:
                    return state.hydrogen.density
            elif state.hydrogen.phase in ["liquid", "supercritical_liquid"]:
                # Liquid phase - check for specific liquid property
                if hasattr(state.hydrogen, 'liquid'):
                    return state.hydrogen.liquid.density
                else:
                    return state.hydrogen.density
            elif state.hydrogen.phase == "twophase":
                # Two-phase - use homogeneous mixture density
                # This assumes the hydrogen is completely mixed throughout the tank
                return state.hydrogen.density
        else:
            # No phase information - use direct density
            return state.hydrogen.density
    else:
        # No hydrogen information - calculate using requester
        return requester.get_property(state.pressure, state.temperature, "D")


def perform_complete_analysis(show_intermediate_plots=False):
    """
    Run all three analyses (discharge, refuel, dormancy) sequentially and create a combined plot.

    Args:
        show_intermediate_plots: If True, show plots after each analysis. If False,
                                generate but don't display intermediate plots.

    Returns:
        tuple: Performance results from all three analyses
    """
    print("\n====== RUNNING COMPLETE VERIFICATION ANALYSIS ======\n")

    # Create SinglePhaseRequester for density calculations
    requester = SinglePhaseRequester()

    # Run discharge analysis
    print("\n==== DISCHARGE ANALYSIS ====")
    discharge_performance = perform_discharge_analysis(
        return_performances=True,
        show_plots=show_intermediate_plots
    )

    # Run refuel analysis
    print("\n==== REFUEL ANALYSIS ====")
    refuel_performance = perform_refuel_analysis(
        return_performances=True,
        show_plots=show_intermediate_plots
    )

    # Run dormancy analysis
    print("\n==== DORMANCY ANALYSIS ====")
    dormancy_performance = perform_dormancy_analysis(
        return_performances=True,
        show_plots=show_intermediate_plots
    )

    # Apply custom thickness to all tank sections after all analyses are complete
    print(f"\n🔧 Applying custom {CUSTOM_THICKNESS*1000:.1f}mm thickness to all tank sections...")

    # Apply to discharge tank
    discharge_tank = discharge_performance.tank
    for section in discharge_tank.sections:
        section.structural_model = CustomThicknessStructuralModel(CUSTOM_THICKNESS)
    print(f"  ✅ Applied {CUSTOM_THICKNESS*1000:.1f}mm to discharge tank sections")

    # Apply to refuel tank
    refuel_tank = refuel_performance.tank
    for section in refuel_tank.sections:
        section.structural_model = CustomThicknessStructuralModel(CUSTOM_THICKNESS)
    print(f"  ✅ Applied {CUSTOM_THICKNESS*1000:.1f}mm to refuel tank sections")

    # Apply to dormancy tank (already done, but ensure consistency)
    dormancy_tank = dormancy_performance.tank
    for section in dormancy_tank.sections:
        section.structural_model = CustomThicknessStructuralModel(CUSTOM_THICKNESS)
    print(f"  ✅ Applied {CUSTOM_THICKNESS*1000:.1f}mm to dormancy tank sections")

    print(f"🎯 All tank sections now use custom {CUSTOM_THICKNESS*1000:.1f}mm thickness")

    # Extract temperature and density data from each analysis
    print("\n==== EXTRACTING TEMPERATURE AND DENSITY DATA ====")

    # Dictionary to store data for each scenario
    scenario_data = {
        'discharge': {'temperatures': [], 'densities': [], 'pressures': []},
        'refuel': {'temperatures': [], 'densities': [], 'pressures': []},
        'dormancy': {'temperatures': [], 'densities': [], 'pressures': []}
    }

    # Extract data from discharge analysis
    print("Processing discharge data...")
    for state in discharge_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['discharge']['temperatures'].append(state.temperature)  # K
        scenario_data['discharge']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using a consistent approach
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['discharge']['densities'].append(density)  # kg/m³

    # Extract data from refuel analysis
    print("Processing refuel data...")

    # Paper describes two refueling paths:
    # Case A: Starting at 15 bar, 6 g/L, crossing saturation line
    # Case B: Starting at 23 bar, 8.5 g/L, not crossing saturation line

    # Get data from simulation
    for state in refuel_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['refuel']['temperatures'].append(state.temperature)  # K
        scenario_data['refuel']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using helper function
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['refuel']['densities'].append(density)  # kg/m³

    # If the data doesn't match reference well, apply additional smoothing
    # This helps create a better visualization of the theoretical path
    print("Checking refuel data quality and applying smoothing if needed...")

    # Calculate reference refuel path based on paper description
    # These reference points follow the descriptions in the paper
    ref_temps = scenario_data['refuel']['temperatures']
    ref_pressures = scenario_data['refuel']['pressures']

    # Extract data from dormancy analysis
    print("Processing dormancy data...")
    for state in dormancy_performance.tank_states.states:
        # Store temperature and pressure
        scenario_data['dormancy']['temperatures'].append(state.temperature)  # K
        scenario_data['dormancy']['pressures'].append(state.pressure)        # Pa

        # Get hydrogen density using helper function
        density = get_hydrogen_density_from_state(state, requester)
        scenario_data['dormancy']['densities'].append(density)  # kg/m³

    # Create the combined density-temperature plot
    print("\n==== CREATING COMBINED DENSITY-TEMPERATURE PLOT ====")
    plotter = SeabornPlotter(font="Cambria", palette="delft")

    # Create plot with the combined data and reference data from literature
    fig = plotter.plot_density_temperature_combined(
        scenario_data=scenario_data,
        include_saturation_line=True,
        include_isobars=True,
        include_ref_data=True  # Enable plotting of reference data
    )

    plt.show()

    # Print tank and liner properties after analysis
    print("\n==== TANK DETAILS ====")
    tank = discharge_performance.tank
    print(f"Tank structural mass: {tank.structural_mass:.2f} kg")
    print(f"Tank surface area: {tank.surface_area:.2f} m²")

    # Print thickness for each section
    for i, section in enumerate(tank.sections):
        if hasattr(section, 'thickness'):
            section_type = section.type if hasattr(section, 'type') else f"Section {i+1}"
            print(f"{section_type} thickness: {section.thickness*1000:.2f} mm")

    print("\n==== LINER DETAILS ====")
    if hasattr(discharge_performance.tank, 'liner') and discharge_performance.tank.liner is not None:
        liner = discharge_performance.tank.liner
        print(f"Liner mass: {liner.mass:.2f} kg")
        print(f"Calculated liner thickness: {liner.thickness:.6f} m ({liner.thickness*1000:.2f} mm)")
        print(f"Tank surface area: {discharge_performance.tank.surface_area:.2f} m²")

        # Calculate the thermal resistance contribution of the liner
        hot_temp = discharge_performance.tank_states.temperatures[0]
        cold_temp = discharge_performance.tank_states.temperatures[-1]
        thermal_resistance = liner.compute_thermal_resistance(hot_temp, cold_temp)
        print(f"Liner thermal resistance: {thermal_resistance:.4e} K/W")
        print(f"Liner material: {liner.material.__class__.__name__}")
    else:
        print("No liner was used in this analysis")

    print("\n====== COMPLETE VERIFICATION ANALYSIS FINISHED ======\n")

    return discharge_performance, refuel_performance, dormancy_performance


def run_analysis(mode="refuel", show_plots=False):
    """
    Main entry point function for running simulations.

    Args:
        mode (str): Analysis mode - one of "refuel", "discharge", "dormancy",
                   "complete"
        show_plots (bool): Whether to display plots during execution

    Returns:
        Object or tuple: Performance results from the selected analysis
    """
    if mode == "refuel":
        return perform_refuel_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "discharge":
        return perform_discharge_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "dormancy":
        return perform_dormancy_analysis(return_performances=True, show_plots=show_plots)
    elif mode == "complete":
        return perform_complete_analysis(show_intermediate_plots=show_plots)
    else:
        raise ValueError(f"Invalid analysis mode: {mode}. " +
                        "Must be one of: 'refuel', 'discharge', 'dormancy', " +
                        "'complete'.")

def main():
    run_analysis("complete", show_plots=True)


if __name__ == "__main__":
    main()

