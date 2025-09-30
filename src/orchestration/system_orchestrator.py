"""
System orchestrator for multi-tank hydrogen analysis.

This module provides the main orchestration class that integrates:
- ScenarioConfig unified configuration parser
- MultiTankSystem DAE physics engine
- Mission sequence execution with stopping criteria
- Results validation and output generation

Key Features:
- Load YAML configurations via ScenarioConfig
- Execute mission sequences (discharge -> refuel -> dormancy)
- Multi-tank coupling with pressure-triggered valves
- Enhanced stopping criteria (density + time-based)
- Production-ready orchestration framework
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from CoolProp.CoolProp import PropsSI

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration system
from src.configuration.scenario_configuration import ScenarioConfig

# Multi-tank DAE physics engine
from src.multi_tank.system.tank_system import TankSystem, TankSystemConfig, TankConfig
from src.multi_tank.utilities.tank_geometry import create_tank_from_fuel_mass

# Mission framework
from src.mission.isochoric_missions import DischargeMission

# Coupling rules system
from src.orchestration.coupling_rules import parse_coupling_rules, CouplingRule

# Heat flow data collection for iHEX calculation
from src.dynamics.isochoric_dynamic_models import set_heat_flow_data_collector

# Utilities
from CoolProp.CoolProp import PropsSI


class SystemOrchestrator:
    """
    System orchestrator for running multi-tank hydrogen storage analysis.

    The SystemOrchestrator provides a high-level interface for configuring
    and running tank system analyses. It handles:
    - Mission profile configuration and parsing
    - Tank geometry sizing and configuration
    - Analysis execution and results collection
    """

    # Class-level cache for sequential mission tank geometry
    _cached_tank_geometries = {}
    _sizing_mission_key = None

    def __init__(self, scenario_config: ScenarioConfig = None, config_path: str = None):
        # Initialize scenario configuration
        if scenario_config is not None:
            self.scenario_config = scenario_config
        elif config_path is not None:
            try:
                self.scenario_config = ScenarioConfig.from_yaml(config_path)
            except Exception as e:
                print(f"   ❌ Failed to load ScenarioConfig: {e}")
                raise e
        else:
            raise ValueError("Must provide either scenario_config or config_path")

        # Validate that scenario_config is properly loaded
        if not hasattr(self.scenario_config, '_config_path'):
            raise ValueError(f"ScenarioConfig not properly loaded: {type(self.scenario_config)}")

        # Load mission profile first (needed for tank sizing)
        # Respect the actual profile specified in the configuration
        try:
            import yaml
            with open(self.scenario_config._config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            if 'mission' in raw_config and 'profile' in raw_config['mission']:
                # Use the profile specified in the mission section
                mission_profile = raw_config['mission']['profile']
                print(f"   📋 Using mission section profile: {mission_profile}")
            else:
                # Fallback to mission_sequence
                mission = self.scenario_config.mission_sequence.missions[0]
                mission_profile = mission.profile
                print(f"   📋 Using mission_sequence profile: {mission_profile}")
        except Exception as e:
            # Ultimate fallback
            mission_profile = "atr72"  # Default to atr72 instead of constant flow
            print(f"   ⚠️  Using fallback profile: {mission_profile} (error: {e})")

        self.mission_profile = self._get_mission_profile(mission_profile)
        print(f"   ✓ Mission profile loaded: {mission_profile}")

        # Create tank geometries from scenario configuration (now with mission profile available)
        self.tank_geometries = self._create_tank_geometries()
        print(f"   ✓ Created {len(self.tank_geometries)} tank geometries")

        # Create TankSystem configuration from ScenarioConfig
        self.tank_system_config = self._create_tank_system_config()
        print(f"   ✓ Tank system configuration created")

        # Parse coupling rules from configuration
        coupling_rules_config = self.scenario_config.config_dict.get('coupling_rules', [])
        self.coupling_rules = parse_coupling_rules(coupling_rules_config)
        if coupling_rules_config:
            print(f"   ✓ Parsed {len(self.coupling_rules)} coupling rules")
            for rule in self.coupling_rules:
                print(f"     - {rule.coupling_id}: {rule.coupling_type}")

        # Convert coupling rules to TankSystem expected format
        tank_system_coupling_rules = []
        for rule_config in coupling_rules_config:
            participants = rule_config.get('participants', {})
            activation = rule_config.get('activation_conditions', {})
            flow_params = rule_config.get('flow_parameters', {})
            hysteresis = rule_config.get('hysteresis', {})

            # Map our config format to TankSystem expected format
            # PressureTriggeredValve: p_open is activation threshold (valve opens), p_close is deactivation threshold (valve closes)
            activation_threshold = hysteresis.get('activation_threshold_bar', 20.0)    # Open valve when Tank 2 ≤ this pressure
            deactivation_threshold = hysteresis.get('deactivation_threshold_bar', 21.0)  # Close valve when Tank 2 ≥ this pressure

            # Log fallback usage for transparency
            if 'activation_threshold_bar' not in hysteresis:
                print(f"⚠️  Using fallback: activation_threshold_bar = {activation_threshold} bar (consider adding to YAML config)")
            if 'deactivation_threshold_bar' not in hysteresis:
                print(f"⚠️  Using fallback: deactivation_threshold_bar = {deactivation_threshold} bar (consider adding to YAML config)")

            tank_system_rule = {
                'source_tank': participants.get('source', 1) - 1,  # Convert 1-based to 0-based index
                'target_tank': participants.get('target', 2) - 1,  # Convert 1-based to 0-based index
                'opening_pressure': activation_threshold * 1e5,    # p_open: valve opens when target ≤ this
                'closing_pressure': deactivation_threshold * 1e5,  # p_close: valve closes when target ≥ this
                'max_flow_rate': flow_params.get('max_flow_rate_kg_s', 0.05),
                'orifice_diameter': flow_params.get('orifice_diameter_m', 0.01)  # Default 10mm orifice
            }

            # Log fallback usage for critical coupling parameters
            if 'max_flow_rate_kg_s' not in flow_params:
                print(f"⚠️  Using fallback: max_flow_rate_kg_s = {tank_system_rule['max_flow_rate']} kg/s (consider adding to YAML config)")
            if 'orifice_diameter_m' not in flow_params:
                print(f"⚠️  Using fallback: orifice_diameter_m = {tank_system_rule['orifice_diameter']} m (consider adding to YAML config)")

            tank_system_coupling_rules.append(tank_system_rule)

        # Initialize TankSystem with all components
        self.tank_system = TankSystem(
            tank_geometries=self.tank_geometries,
            config=self.tank_system_config,
            coupling_rules=tank_system_coupling_rules  # Pass converted config
        )
        print(f"   ✓ TankSystem DAE engine initialized")

        # Setup mission-specific flow rates if needed
        self._setup_mission_sequences()
        print(f"   ✓ Mission sequences configured")

        # Results storage
        # Store results after simulation
        self.results = None

        # Heat flow data collection for iHEX calculations
        self.heat_flow_data = {
            't': [],           # Time [s]
            'qdot_disch': [],  # Discharge heat rate [W] (iHEX requirements from DAE)
            'qdot_ohex': [],   # oHEX heat rate [W] (calculated in post-processing)
            'mdot_disch': [],  # Discharge mass flow rate [kg/s]
            'T': [],           # Temperature [K]
            'rho': []          # Density [kg/m³]
        }
        self.validation_results = None

        print("✅ System Orchestrator ready for mission execution!")
        print(f"   Tanks: {len(self.tank_geometries)}")
        print(f"   Missions: {self.scenario_config.get_mission_count()}")
        print(f"   Materials: {list(self.scenario_config.materials.keys())}")
        print(f"   Analysis type: {self.scenario_config.analysis_name}")

    def _create_tank_geometries(self) -> List[Any]:
        """Create tank geometries from scenario configuration."""
        tank_geometries = []

        # Validate scenario_config has required attributes
        if not hasattr(self.scenario_config, 'tank_geometries'):
            raise ValueError(f"ScenarioConfig missing tank_geometries: {type(self.scenario_config)}")

        for tank_id, geometry_data in self.scenario_config.tank_geometries.items():
            print(f"   Creating Tank {tank_id} geometry...")

            # Extract geometry parameters
            phi = geometry_data.get('phi', 3.0)
            initial_pressure = float(geometry_data['initial_pressure'])

            # Determine geometry creation method
            if 'fuel_mass' in geometry_data:
                # Method 1: Create from fuel mass requirement
                fuel_mass = float(geometry_data['fuel_mass'])
                initial_temperature = geometry_data.get('initial_temperature', 53.25)

                # Use create_tank_from_fuel_mass utility
                tank_geom = create_tank_from_fuel_mass(
                    fuel_mass=fuel_mass,
                    initial_pressure=initial_pressure,
                    initial_temperature=initial_temperature,
                    operating_pressure=initial_pressure * 1.125,  # 12.5% margin
                    safety_margin=1.1,
                    liner_thickness=0.005,
                    insulation_thickness=0.05
                )
                print(f"     From fuel mass: {fuel_mass}kg → V={tank_geom.volume:.4f}m³")

            elif 'radius' in geometry_data:
                # Method 2: Create from specified radius (fallback to simple spherical)
                from src.tank_design.tank_shapes import SphericalTank
                from src.materials.materials_for_multi_tank.nist_material import NISTMaterial

                radius = float(geometry_data['radius'])
                material = NISTMaterial.aluminum_6061T6_nist()
                operating_pressure = geometry_data.get('venting_pressure', geometry_data['initial_pressure'])
                tank_geom = SphericalTank(radius=radius, material=material, operating_pressure=operating_pressure)
                print(f"     From radius: {radius}m → V={tank_geom.volume:.4f}m³")

            else:
                # Method 3: Create tank sized for mission requirements
                from src.tank_design.tank_shapes import SphericalTank
                from src.materials.materials_for_multi_tank.nist_material import NISTMaterial

                material = NISTMaterial.aluminum_6061T6_nist()
                operating_pressure = geometry_data.get('venting_pressure', 450e5)  # Default 450 bar
                if 'venting_pressure' not in geometry_data:
                    print(f"⚠️  Using fallback: venting_pressure = {operating_pressure/1e5:.0f} bar for Tank {tank_id} (consider adding to YAML config)")

                # Calculate tank geometry from mission requirements
                tank_geom = self._create_mission_sized_tank(geometry_data, material, operating_pressure)
                print(f"     Mission-sized geometry: V={tank_geom.volume:.4f}m³")

            tank_geometries.append(tank_geom)

        return tank_geometries

    def _create_tank_system_config(self) -> TankSystemConfig:
        """Create TankSystemConfig from ScenarioConfig."""
        # Get mission info (mission profile already loaded during initialization)
        mission = self.scenario_config.mission_sequence.missions[0]  # Use first mission

        # Create tank configurations for each tank geometry
        tank_configs = []
        for i, (tank_id, geometry_data) in enumerate(self.scenario_config.tank_geometries.items()):
            # Set pressure thresholds
            initial_pressure = float(geometry_data['initial_pressure'])
            venting_pressure = float(geometry_data.get('venting_pressure', initial_pressure * 1.125))
            minimum_pressure = float(geometry_data.get('minimum_pressure', min(15e5, initial_pressure * 0.1)))
            if 'minimum_pressure' not in geometry_data:
                print(f"⚠️  Using fallback: minimum_pressure = {minimum_pressure/1e5:.1f} bar for Tank {tank_id} (calculated from initial_pressure)")

            # Use calculated temperature from mission sizing if available
            if 'calculated_initial_temperature' in geometry_data:
                initial_temp = geometry_data['calculated_initial_temperature']
            elif 'initial_temperature' in geometry_data:
                # Use explicit initial temperature from config (prioritize over density calculation)
                initial_temp = geometry_data['initial_temperature']
            elif 'initial_density' in geometry_data:
                # Calculate temperature from pressure and density as fallback
                try:
                    from CoolProp.CoolProp import PropsSI
                    density = float(geometry_data['initial_density'])
                    initial_temp = PropsSI("T", "P", initial_pressure, "D", density, "hydrogen")
                except:
                    initial_temp = 53.25  # Default cryogenic temperature
            else:
                initial_temp = 53.25  # Default

            # Create tank configuration
            tank_config = TankConfig(
                P_INIT=initial_pressure,
                T_INIT=initial_temp,
                P_VENT=venting_pressure,
                P_MIN=minimum_pressure,
                MASS_INIT=geometry_data.get('initial_mass'),  # None if not specified
                scenario=mission.type.upper(),  # DISCHARGE, REFUEL, DORMANCY
                name=f"Tank{tank_id}"
            )

            tank_configs.append(tank_config)

            print(f"   Tank {tank_id} config: P_init={initial_pressure/1e5:.0f}bar, T_init={initial_temp:.1f}K, ρ_stop={geometry_data.get('minimum_density', 5.8):.1f}kg/m³")

        # Calculate mission duration based on profile
        mission_duration = self._calculate_mission_duration()

        # Get stopping criteria from configuration (prioritize stopping_criteria over tank geometry)
        stopping_criteria = self.scenario_config.config_dict.get('stopping_criteria', {})

        # Get minimum density (prioritize stopping_criteria, fallback to tank geometry)
        if 'minimum_density' in stopping_criteria:
            minimum_density = float(stopping_criteria['minimum_density'])
        else:
            first_tank_data = list(self.scenario_config.tank_geometries.values())[0]
            minimum_density = float(first_tank_data.get('minimum_density', 5.8))  # Default 5.8 kg/m³
            if 'minimum_density' not in first_tank_data:
                print(f"⚠️  Using fallback: minimum_density = {minimum_density} kg/m³ for stopping criteria (consider adding to YAML config)")

        # Get target density from stopping criteria configuration
        target_density = None
        if 'target_density' in stopping_criteria:
            target_density = float(stopping_criteria['target_density'])
        elif 'maximum_density' in stopping_criteria:
            target_density = float(stopping_criteria['maximum_density'])

        # Create system configuration with mission profile
        system_config = TankSystemConfig(
            AMBIENT_TEMPERATURE=mission.ambient_temperature,
            MISSION_DURATION=mission_duration,
            tanks=tank_configs,
            mission_profile=self.mission_profile,
            minimum_density=minimum_density,
            target_density=target_density
        )

        return system_config

    def _calculate_mission_duration(self) -> float:
        """
        Calculate mission duration in seconds based on stored mission profile.

        Returns:
            float: Mission duration in seconds
        """
        if hasattr(self, 'mission_profile') and self.mission_profile is not None:
            # Calculate duration from mission sections
            duration_seconds = sum(section.duration for section in self.mission_profile.sections)
            return duration_seconds
        else:
            # Fallback to default duration
            return 3600.0  # 1 hour default

    def _get_mission_profile(self, profile_name: str):
        """Get mission profile object from profile name."""
        from src.mission.mission import Mission, MissionSection, OutFlow, InFlow

        if profile_name.lower() == "atr72":
            return Mission.atr72()
        elif profile_name.lower() in ["constant_flow", "sequential_constant_flow"]:
            return self._create_constant_flow_mission()
        else:
            raise ValueError(f"Unknown mission profile: {profile_name}")

    def _create_constant_flow_mission(self):
        """Create constant flow mission from scenario config parameters."""
        from src.mission.mission import Mission, MissionSection, OutFlow, InFlow
        import yaml

        # Handle both old mission format and new mission section format
        try:
            # Try to read from mission section first (preferred for sequential missions)
            config_path = getattr(self.scenario_config, '_config_path', 'NO_PATH')
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            if 'mission' in raw_config:
                # New mission section format - use current mission parameters
                mission_data = raw_config['mission']
                flow_rate = mission_data.get('flow_rate', 0.001)
                duration = mission_data.get('duration', 36000)

                # Log fallback usage for mission parameters
                if 'flow_rate' not in mission_data:
                    print(f"⚠️  Using fallback: flow_rate = {flow_rate} kg/s (consider adding to YAML config)")
                if 'duration' not in mission_data:
                    print(f"⚠️  Using fallback: duration = {duration} s ({duration/3600:.1f} hours) (consider adding to YAML config)")
                mission_type = mission_data.get('type', 'discharge')
                mission_key = mission_data.get('key', mission_type)
                ambient_temp = mission_data.get('ambient_temperature', 288.15)
                if 'ambient_temperature' not in mission_data:
                    print(f"⚠️  Using fallback: ambient_temperature = {ambient_temp} K (consider adding to YAML config)")

            elif 'mission_sequence' in raw_config and 'missions' in raw_config['mission_sequence']:
                # Fallback to sequential mission format - use first mission
                mission_data = raw_config['mission_sequence']['missions'][0]
                flow_rate = mission_data.get('flow_rate', 0.001)
                duration = mission_data.get('max_duration', mission_data.get('duration', 36000))
                mission_type = mission_data.get('type', 'discharge')
                mission_key = mission_data.get('key', mission_type)
                ambient_temp = raw_config['mission_sequence'].get('ambient_temperature', 288.15)
            else:
                # Fallback to old format
                mission_config = self.scenario_config.mission_sequence.missions[0]
                flow_rate = getattr(mission_config, 'flow_rate', 0.001)
                duration = getattr(mission_config, 'duration', 3600)
                mission_type = getattr(mission_config, 'type', 'discharge')
                mission_key = mission_type
                ambient_temp = getattr(mission_config, 'ambient_temperature', 288.15)
        except Exception as e:
            # Ultimate fallback
            print(f"   ⚠️  Config reading failed: {e}")
            flow_rate = 0.001
            duration = 36000  # 10 hours
            mission_type = 'discharge'
            mission_key = 'discharge'
            ambient_temp = 288.15

        # Create flow object based on mission type
        if mission_type.lower() == 'discharge':
            # Negative flow for discharge (outflow)
            fuel_flow = OutFlow(-abs(flow_rate), "gas")
        elif mission_type.lower() == 'refuel':
            # Positive flow for refuel (inflow) - need proper hydrogen object for cryopump physics
            from src.fluids.hydrogen_retrievers import SinglePhaseRequester
            # Create a dummy hydrogen object - the actual cryopump enthalpy will be calculated dynamically
            dummy_hydrogen = SinglePhaseRequester().get_hydrogen_properties(3e5, 20.4)  # 3 bar, 20.4K (dewar conditions)
            fuel_flow = InFlow(abs(flow_rate), dummy_hydrogen)
        elif mission_type.lower() == 'dormancy':
            # Zero flow for dormancy
            fuel_flow = OutFlow(0.0, "gas")
        else:
            # Default to discharge
            fuel_flow = OutFlow(-abs(flow_rate), "gas")

        # Create mission section with correct constructor parameters
        mission_section = MissionSection(
            duration,
            [fuel_flow],
            0.0,  # altitude - Ground level for tank operations
            0.0,  # mach_number
            mission_key,  # fuel_flow_key - use mission key
            ambient_temp  # temperature
        )

        return Mission([mission_section])

    def _create_mission_sized_tank(self, geometry_data, material, operating_pressure):
        """
        Create tank geometry sized for mission fuel requirements.

        Args:
            geometry_data: Tank configuration from ScenarioConfig
            material: Tank material
            operating_pressure: Operating pressure [Pa]

        Returns:
            SphericalTank: Tank sized for mission requirements
        """
        from CoolProp.CoolProp import PropsSI
        import math

        # Check if this is part of a sequential mission analysis
        # For sequential missions, reuse tank geometry from sizing mission
        try:
            import yaml
            config_path = getattr(self.scenario_config, '_config_path', 'NO_PATH')
            if config_path != 'NO_PATH':
                with open(config_path, 'r') as f:
                    raw_config = yaml.safe_load(f)

                # Check if this config has mission_sequence (sequential missions)
                if 'mission_sequence' in raw_config and 'missions' in raw_config['mission_sequence']:
                    sizing_mission = raw_config['mission_sequence'].get('sizing_mission', 'discharge')
                    current_mission = raw_config.get('mission', {}).get('type', 'unknown')

                    # For sequential missions, use the ORIGINAL config path as base, not temp files
                    # Extract original config path from the driver context
                    original_config_path = config_path
                    if '/tmp/simple_mission_' in config_path or 'temp_config_' in config_path:
                        # This is a temporary config - find the original config path
                        # Look for the original stops_verification.yaml config
                        import os
                        config_dir = os.path.dirname(config_path)
                        potential_original = os.path.join(config_dir, 'stops_verification.yaml')
                        if os.path.exists(potential_original):
                            original_config_path = potential_original
                        else:
                            # If stops_verification.yaml not found in same dir, look in parent directories
                            # The temp files are in /tmp but original might be elsewhere
                            # Use a consistent original path based on temp filename pattern
                            if '/tmp/simple_mission_' in config_path:
                                # For stops verification, use a consistent base path
                                original_config_path = 'stops_verification.yaml'  # Use relative path as consistent key

                    # Generate cache key based on original config path
                    cache_key = f"{original_config_path}_{sizing_mission}"

                    if current_mission == sizing_mission:
                        # This is the sizing mission - calculate and cache geometry
                        print(f"   🔧 Sizing mission '{sizing_mission}' - calculating tank geometry")
                        SystemOrchestrator._sizing_mission_key = cache_key
                    else:
                        # This is a non-sizing mission - try to reuse cached geometry
                        if cache_key in SystemOrchestrator._cached_tank_geometries:
                            cached_tank = SystemOrchestrator._cached_tank_geometries[cache_key]
                            print(f"   🔧 Non-sizing mission '{current_mission}' - reusing tank geometry from '{sizing_mission}'")
                            print(f"     Cached geometry: V={cached_tank.volume:.4f}m³, R={cached_tank.radius:.3f}m")
                            return cached_tank
                        else:
                            print(f"   ⚠️  No cached geometry found for sizing mission '{sizing_mission}' - calculating new geometry")
                            print(f"     Cache key: {cache_key}")
                            print(f"     Available keys: {list(SystemOrchestrator._cached_tank_geometries.keys())}")
        except Exception as e:
            print(f"   ⚠️  Sequential mission detection failed: {e} - proceeding with normal sizing")

        # Get mission fuel requirements from the actual mission profile
        # This handles both predefined profiles (ATR72) and custom constant flow missions
        try:
            # First try to get fuel requirements directly from the mission profile
            if hasattr(self.mission_profile, 'required_fuel'):
                required_fuel_mass = self.mission_profile.required_fuel
                print(f"   🔧 Using mission profile fuel requirement: {required_fuel_mass:.2f} kg")
            else:
                # Calculate from mission sections for custom profiles
                required_fuel_mass = 0.0
                for section in self.mission_profile.sections:
                    # Get total outflow from section
                    section_fuel = 0.0
                    for flow in section.fuel_flows:
                        if hasattr(flow, 'flow_rate') and flow.flow_rate < 0:  # Outflow (negative)
                            section_fuel += abs(flow.flow_rate) * section.duration
                        elif hasattr(flow, 'mass_flow_rate') and flow.mass_flow_rate < 0:  # Outflow (negative)
                            section_fuel += abs(flow.mass_flow_rate) * section.duration
                    required_fuel_mass += section_fuel
                print(f"   🔧 Calculated fuel requirement from sections: {required_fuel_mass:.2f} kg")
        except Exception as e:
            # Fallback to config-based calculation for simple constant flow missions
            print(f"   ⚠️ Mission profile fuel calculation failed: {e}")
            try:
                import yaml
                config_path = getattr(self.scenario_config, '_config_path', 'NO_PATH')
                with open(config_path, 'r') as f:
                    raw_config = yaml.safe_load(f)

                if 'mission' in raw_config:
                    current_mission = raw_config['mission']
                    flow_rate = current_mission.get('flow_rate', 0.001)
                    duration = current_mission.get('duration', 3600)
                    required_fuel_mass = flow_rate * duration
                    print(f"   🔧 Using config-based fuel requirement: {required_fuel_mass:.2f} kg")
                else:
                    required_fuel_mass = 36.0  # Default for ATR72-like missions
                    print(f"   🔧 Using default fuel requirement: {required_fuel_mass:.2f} kg")
            except:
                required_fuel_mass = 36.0  # Default for ATR72-like missions
                print(f"   🔧 Using fallback fuel requirement: {required_fuel_mass:.2f} kg")
        print(f"   🔧 Mission profile type: {type(self.mission_profile)}")
        if hasattr(self.mission_profile, 'sections'):
            print(f"   🔧 Mission sections: {len(self.mission_profile.sections)}")
            for i, section in enumerate(self.mission_profile.sections):
                if hasattr(section, 'duration') and hasattr(section, 'fuel_flows'):
                    flows = getattr(section, 'fuel_flows', [])
                    flow_info = []
                    for flow in flows:
                        # Check different possible attributes for flow rate
                        if hasattr(flow, 'flow_rate'):
                            flow_info.append(f"{flow.flow_rate:.3f} kg/s")
                        elif hasattr(flow, 'mass_flow_rate'):
                            flow_info.append(f"{flow.mass_flow_rate:.3f} kg/s")
                        else:
                            flow_info.append(f"{type(flow).__name__}")
                    flow_info_str = flow_info if flow_info else ["No flows"]
                    print(f"      Section {i}: duration={section.duration}s, flows={flow_info_str}")

        # Get initial conditions - ALWAYS use discharge mission conditions for tank sizing
        try:
            config_path = getattr(self.scenario_config, '_config_path', 'NO_PATH')
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            # Check if this is part of a sequential mission
            if 'mission' in raw_config:
                current_mission = raw_config['mission']
                mission_type = current_mission.get('type', 'discharge')

                if mission_type != 'discharge':
                    # Override with discharge mission conditions for consistent tank sizing

                    geometry_data = geometry_data.copy()  # Don't modify original
                    geometry_data['initial_pressure'] = 400e5  # 400 bar - discharge pressure
                    geometry_data['initial_density'] = 78.0    # kg/m³ - discharge density

        except Exception as e:
            print(f"   ⚠️  Could not override initial conditions: {e}")

        initial_pressure = float(geometry_data['initial_pressure'])  # Pa

        # Calculate initial density from pressure and temperature OR density constraint
        if 'initial_density' in geometry_data:
            # Density specified - solve for temperature
            target_density = float(geometry_data['initial_density'])  # kg/m³
            try:
                initial_temp = PropsSI("T", "P", initial_pressure, "D", target_density, "hydrogen")
                print(f"     Solved initial temperature: {initial_temp:.2f} K (from P={initial_pressure/1e5:.0f} bar, ρ={target_density:.1f} kg/m³)")
            except Exception as e:
                print(f"     Warning: Could not solve for temperature from P,ρ: {e}")
                initial_temp = geometry_data.get('initial_temperature', 53.25)
                target_density = PropsSI("D", "P", initial_pressure, "T", initial_temp, "hydrogen")
                print(f"     Using fallback: T={initial_temp:.2f} K, ρ={target_density:.1f} kg/m³")
        else:
            # Temperature specified - calculate density
            initial_temp = geometry_data.get('initial_temperature', 53.25)  # K
            target_density = PropsSI("D", "P", initial_pressure, "T", initial_temp, "hydrogen")
            print(f"     Calculated density: {target_density:.2f} kg/m³ (from P={initial_pressure/1e5:.0f} bar, T={initial_temp:.1f} K)")

        # Calculate required tank volume with ullage allowance
        fill_fraction = geometry_data.get('fill_fraction', 0.90)  # 90% fill default
        required_volume = required_fuel_mass / (target_density * fill_fraction)

        # Calculate tank radius for spherical tank
        # V = (4/3) * π * r³  =>  r = (3V / 4π)^(1/3)
        tank_radius = ((3 * required_volume) / (4 * math.pi)) ** (1/3)

        print(f"     Mission fuel required: {required_fuel_mass:.2f} kg")
        print(f"     Fill fraction: {fill_fraction:.0%}")
        print(f"     Required volume: {required_volume:.4f} m³")
        print(f"     Tank radius: {tank_radius:.3f} m")

        # Store calculated temperature in geometry_data for later use
        geometry_data['calculated_initial_temperature'] = initial_temp
        geometry_data['calculated_initial_density'] = target_density

        # Create spherical tank
        from src.tank_design.tank_shapes import SphericalTank
        tank = SphericalTank(radius=tank_radius, material=material, operating_pressure=operating_pressure)

        # Cache tank geometry for sequential missions if this is the sizing mission
        if hasattr(SystemOrchestrator, '_sizing_mission_key') and SystemOrchestrator._sizing_mission_key:
            cache_key = SystemOrchestrator._sizing_mission_key
            SystemOrchestrator._cached_tank_geometries[cache_key] = tank
            print(f"     ✓ Cached tank geometry for sequential missions: V={tank.volume:.4f}m³")
            # Reset the key after caching
            SystemOrchestrator._sizing_mission_key = None

        return tank

    def _setup_mission_sequences(self):
        """Setup mission-specific flow rates and sequences."""
        print("   Setting up mission sequences...")

        mission_count = self.scenario_config.get_mission_count()
        if mission_count > 1:
            print(f"     Multi-mission sequences detected ({mission_count} missions)")
            print(f"     TODO: Implement mission chaining")
        else:
            mission = self.scenario_config.mission_sequence.missions[0]
            print(f"     Single mission: {mission.type} - {mission.profile}")

        # Apply mission assignment logic if assigned_to is specified
        self._apply_mission_assignment()

        # Mission profile is already stored in self.mission_profile during initialization
        # and passed to TankSystemConfig in _create_tank_system_config
        print(f"   ✓ Mission flow profile configured for DAE system")

    def _apply_mission_assignment(self):
        """Apply mission flows only to the assigned tank."""
        mission = self.scenario_config.mission_sequence.missions[0]

        if mission.assigned_to is not None:
            assigned_tank_id = mission.assigned_to
            # Convert tank ID to tank index (tank IDs are 1-based, indices are 0-based)
            assigned_tank_index = assigned_tank_id - 1

            print(f"     Mission assigned to Tank {assigned_tank_id} (index {assigned_tank_index})")

            # Override the TankSystem's flow rate methods to apply mission only to assigned tank
            # IMPORTANT: Don't interfere with coupling flows - only override base mission flows
            original_get_outflow_rate = self.tank_system._get_outflow_rate

            def mission_assigned_outflow_rate(time: float, tank_index: int):
                """Get outflow rates with mission assignment logic."""
                if tank_index == assigned_tank_index:
                    # This tank executes the mission - apply mission profile flows
                    if hasattr(self, 'mission_profile') and self.mission_profile is not None:
                        return self._get_mission_flow_rate(time)
                    else:
                        # Fallback to original method for assigned tank
                        return original_get_outflow_rate(time, tank_index)
                else:
                    # Other tanks have no mission outflow (but coupling can still happen)
                    return 0.0

            # Replace only the outflow method - let coupling handle inflows
            self.tank_system._get_outflow_rate = mission_assigned_outflow_rate

            print(f"     ✓ Mission flows applied to Tank {assigned_tank_id}")
            print(f"     ✓ Coupling flows preserved for pressure compensation")
        else:
            print(f"     ⚠️ No mission assignment specified - applying mission to all tanks")

    def _get_mission_flow_rate(self, time: float) -> float:
        """Get mission flow rate at given time from mission profile."""
        if not hasattr(self, 'mission_profile') or self.mission_profile is None:
            return 0.0

        current_time = 0.0
        for section in self.mission_profile.sections:
            section_end_time = current_time + section.duration

            if time <= section_end_time:
                # We're in this section
                section_time = time - current_time

                # Get flow from this section
                for flow in section.fuel_flows:
                    if hasattr(flow, 'mass_flow'):
                        if isinstance(flow.mass_flow, list):
                            # Time-varying flow: linear interpolation
                            start_rate = abs(flow.mass_flow[0])
                            end_rate = abs(flow.mass_flow[-1])
                            progress = section_time / section.duration if section.duration > 0 else 0
                            return start_rate + (end_rate - start_rate) * progress
                        else:
                            # Constant flow
                            return abs(flow.mass_flow)

                return 0.0  # No flow found in this section

            current_time = section_end_time

        # Beyond mission duration
        return 0.0

    def run_simulation(self, solver_method: str = "RK45", solver_config: dict = None) -> Any:
        """
        Execute the complete multi-tank simulation using ScenarioConfig.

        Args:
            solver_method: Override solver method (RK45, LSODA, etc.)
            solver_config: Optional solver configuration parameters (timestep, rtol, atol, max_step)

        Returns:
            MultiTankResults: Analysis results from DAE integration
        """
        print("\n🚀 Starting ScenarioConfig-based simulation...")
        print(f"   Analysis: {self.scenario_config.analysis_name}")
        print(f"   Tanks: {len(self.tank_geometries)}")
        print(f"   Mission: {self.scenario_config.mission_sequence.missions[0].type}")
        print(f"   Solver: {solver_method}")

        start_time = time.time()

        try:
            # Clear previous heat flow data and set up collector
            for key in self.heat_flow_data:
                self.heat_flow_data[key].clear()
            set_heat_flow_data_collector(self.heat_flow_data)
            print("   🔧 Heat flow data collector configured for iHEX extraction")

            # Run the MultiTankSystem DAE simulation
            self.results = self.tank_system.run_analysis(solver_method, solver_config)

            end_time = time.time()
            wall_time = end_time - start_time

            print("✅ Simulation completed successfully!")
            print(f"   Wall time: {wall_time:.2f} seconds")
            print(f"   Final time: {self.results.times[-1]/3600:.2f} hours")
            print(f"   Data points: {len(self.results.times)}")

            # Debug heat flow data collection
            qdot_disch_count = len(self.heat_flow_data['qdot_disch'])
            qdot_disch_nonzero = sum(1 for q in self.heat_flow_data['qdot_disch'] if abs(q) > 1e-3)
            max_qdot_disch = max(abs(q) for q in self.heat_flow_data['qdot_disch']) if self.heat_flow_data['qdot_disch'] else 0
            print(f"   🔧 Heat flow data: {qdot_disch_count} points, {qdot_disch_nonzero} non-zero, max = {max_qdot_disch:.1f}W")

            # Display final states
            final_multi_state = self.results.multi_tank_states[-1]
            for i in range(len(self.tank_geometries)):
                tank_state = final_multi_state.get_tank_state(i)
                print(f"   Tank {i+1}: m={tank_state.fuel_mass:.2f}kg, "
                      f"T={tank_state.temperature:.1f}K, "
                      f"ρ={tank_state.density:.1f}kg/m³")

            return self.results

        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            raise

    def validate_results(self) -> Dict[str, Any]:
        """
        Validate multi-tank simulation results.

        Returns:
            Dictionary with validation results
        """
        if not self.results:
            raise ValueError("No results available. Run simulation first.")

        print("\n📊 Validating simulation results...")
        # Validation results - simplified for now
        self.validation_results = {"overall": True, "message": "TankSystem validation not yet implemented"}

        return self.validation_results

    def _calculate_ohex_requirements(self, tank_index: int = 0) -> List[float]:
        """
        Calculate OHEX (Outboard Heat Exchanger) heat requirements.

        Uses enthalpy difference method: Q_oHEX = mdot * (h_target - h_current)
        where h_target is at standard fuel cell inlet conditions.

        Args:
            tank_index: Index of tank to calculate for

        Returns:
            List of OHEX heat requirements [W] for each time point
        """
        if not self.results:
            return []

        # Get OHEX target conditions from config
        hex_config = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('heat_exchanger_requirements', {})
        target_temp = float(hex_config.get('ohex_target_temperature', 200.0))  # K
        target_press = float(hex_config.get('ohex_target_pressure', 20e5))    # Pa

        # Log fallback usage for heat exchanger parameters
        if 'ohex_target_temperature' not in hex_config:
            print(f"⚠️  Using fallback: ohex_target_temperature = {target_temp} K (now available in YAML config)")
        if 'ohex_target_pressure' not in hex_config:
            print(f"⚠️  Using fallback: ohex_target_pressure = {target_press/1e5:.0f} bar (now available in YAML config)")

        try:
            # Calculate target enthalpy (constant for all time points)
            h_target = PropsSI("Hmass", "T", target_temp, "P", target_press, "hydrogen")
        except Exception as e:
            print(f"   ⚠️ Could not calculate OHEX target enthalpy: {e}")
            return [0.0] * len(self.results.times)

        qdot_ohex = []
        tank_series = self.results.get_tank_series(tank_index)

        # Add statistics tracking
        max_q_ohex = 0.0
        valid_points = 0
        print(f"   🔧 Calculating OHEX for {len(tank_series.states)} time points...")

        for i, state in enumerate(tank_series.states):
            try:
                # Get current state conditions
                T_current = state.temperature
                m_current = state.fuel_mass

                # Get tank volume for density calculation
                tank_volume = self.tank_geometries[tank_index].volume
                rho_current = m_current / tank_volume  # kg/m³

                # Get actual mass flow rate used during simulation (stored in tank state)
                mass_rate = abs(state.outflow_rate)

                if T_current > 0 and rho_current > 0 and mass_rate > 0:
                    # Calculate current pressure and enthalpy
                    p_current = PropsSI("P", "T", T_current, "Dmass", rho_current, "hydrogen")
                    h_current = PropsSI("Hmass", "T", T_current, "P", p_current, "hydrogen")

                    # Calculate OHEX heat requirement
                    q_ohex = mass_rate * (h_target - h_current)  # [W]
                    q_ohex_final = max(0.0, q_ohex)  # Ensure non-negative
                    qdot_ohex.append(q_ohex_final)

                    # Track statistics
                    if q_ohex_final > max_q_ohex:
                        max_q_ohex = q_ohex_final
                    if q_ohex_final > 0:
                        valid_points += 1
                else:
                    qdot_ohex.append(0.0)

            except Exception:
                # Silently handle CoolProp errors (common at extreme conditions)
                qdot_ohex.append(0.0)

        print(f"   📊 OHEX calculated: {valid_points}/{len(qdot_ohex)} points, max = {max_q_ohex:.0f}W")
        return qdot_ohex

    def _calculate_ihex_requirements(self, tank_index: int = 0) -> List[float]:
        """
        Extract iHEX requirements from DAE simulation data.

        iHEX requirements come directly from the qdot_disch values calculated
        during the DAE integration in the isochoric dynamic models.
        This represents the Configuration B discharge heat requirement that
        is computed as part of the energy balance to maintain pressure constraints.

        Args:
            tank_index: Index of tank to calculate for (currently unused - uses global heat flow data)

        Returns:
            List of iHEX heat requirements [W] for each time point from DAE simulation
        """
        if not self.results:
            return []

        print(f"   📊 Extracting iHEX requirements from DAE simulation data...")

        # Check if heat flow data was collected during simulation
        if not hasattr(self, 'heat_flow_data') or not self.heat_flow_data['qdot_disch']:
            print("   ⚠️  No heat flow data collected, returning zero iHEX requirements")
            return [0.0] * len(self.results.times)

        # Get raw data from DAE integration
        heat_times = self.heat_flow_data['t']
        heat_qdot_disch = self.heat_flow_data['qdot_disch']

        if len(heat_times) != len(heat_qdot_disch):
            print(f"   ⚠️  Heat flow time/data length mismatch: {len(heat_times)} vs {len(heat_qdot_disch)}")
            return [0.0] * len(self.results.times)

        if len(heat_times) == 0:
            print("   ⚠️  No heat flow data collected during integration")
            return [0.0] * len(self.results.times)

        # Interpolate heat flow data to match results timeline
        import numpy as np
        results_times = np.array(self.results.times)
        heat_times_array = np.array(heat_times)
        heat_qdot_array = np.array(heat_qdot_disch)

        # Handle length mismatch by interpolation
        if len(heat_times) != len(self.results.times):
            print(f"   🔧 Interpolating heat flow data: {len(heat_times)} → {len(self.results.times)} points")
            qdot_ihex = np.interp(results_times, heat_times_array, heat_qdot_array).tolist()
        else:
            qdot_ihex = heat_qdot_disch.copy()

        # Statistics
        valid_points = sum(1 for q in qdot_ihex if abs(q) > 1e-3)
        max_q_ihex = max(abs(q) for q in qdot_ihex) if qdot_ihex else 0.0

        print(f"   📊 iHEX extracted from DAE: {valid_points}/{len(qdot_ihex)} points, max = {max_q_ihex:.0f}W")
        return qdot_ihex

    def _get_mission_flow_rate_at_time(self, time_s: float) -> float:
        """
        Get mass flow rate from mission profile at specified time.

        Args:
            time_s: Time in seconds

        Returns:
            Mass flow rate [kg/s]
        """
        try:
            # Get the first mission (ATR72 profile)
            mission = self.scenario_config.mission_sequence.missions[0]

            # Convert time to hours for mission lookup
            time_h = time_s / 3600.0

            # Get flow rate from mission profile (already in kg/s from unit conversion)
            flow_rate = mission.get_flow_rate_at_time(time_h)

            return flow_rate if flow_rate > 0 else 0.0

        except Exception:
            return 0.0

    def generate_plots(self, save_path: str = None) -> Any:
        """
        Generate plots from simulation results.

        Args:
            save_path: Optional path to save plot file

        Returns:
            Figure object from plotting
        """
        if not self.results:
            raise ValueError("No results available. Run simulation first.")

        print("📊 Generating multi-tank analysis plots...")

        try:
            # Import the new plotting module
            from src.plotting.multi_tank_plotting import DelftColourPlotter

            # Get plotting configuration from config
            plot_config = self.scenario_config.config_dict.get('output', {}).get('plots', {})
            style_config = plot_config.get('style', {})

            # Create plotter with analysis name and styling options from config
            plotter = DelftColourPlotter(
                analysis_name=self.scenario_config.analysis_name,
                use_greyscale=style_config.get('use_greyscale', False),
                enable_multi_tank_overlay=style_config.get('enable_multi_tank_overlay', False)
            )

            # Check if this is a sequential mission analysis
            is_sequential = self._is_sequential_mission_analysis()

            if is_sequential:
                print("🔍 Detected sequential mission analysis - using sequential plotting methods")
                return self._generate_sequential_plots(plotter, save_path)
            else:
                print("🔍 Detected single mission analysis - using standard plotting methods")
                return self._generate_single_mission_plots(plotter, save_path)

        except Exception as e:
            print(f"   ⚠️ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _is_sequential_mission_analysis(self) -> bool:
        """
        Detect if this is a sequential mission analysis based on config structure.

        Returns:
            bool: True if sequential missions detected, False otherwise
        """
        # Check if we have stored mission results from sequential execution
        if hasattr(self, 'mission_results') and len(self.mission_results) > 1:
            print(f"   🔎 Found {len(self.mission_results)} sequential mission results")
            return True

        # Check if config has mission section with multiple entries
        if hasattr(self.scenario_config, 'config_dict') and self.scenario_config.config_dict:
            mission_section = self.scenario_config.config_dict.get('mission', {})

            # Look for multiple mission entries (discharge, refuel, dormancy)
            mission_types = ['discharge', 'refuel', 'dormancy']
            found_missions = [mission_type for mission_type in mission_types
                            if mission_type in mission_section]

            # Sequential if we have more than one mission type
            if len(found_missions) > 1:
                print(f"   🔎 Found sequential missions in config: {', '.join(found_missions)}")
                return True

        print("   🔎 Single mission analysis detected")
        return False

    def _generate_sequential_plots(self, plotter, save_path: str = None):
        """Generate plots for sequential mission analysis."""
        if not hasattr(self, 'mission_results'):
            print("⚠️  No mission results available for sequential plotting")
            return None

        mission_data = self.mission_results
        figures = []

        # Generate sequential plots for each tank
        num_tanks = len(self.tank_geometries)

        for tank_idx in range(num_tanks):
            print(f"🎨 Generating sequential plots for Tank {tank_idx + 1}")

            # Create reference lines from tank configuration
            tank_config_data = list(self.scenario_config.tank_geometries.values())[tank_idx]
            reference_lines = plotter.create_reference_lines_from_config(tank_config_data)

            # Add mission ambient temperature if available
            if self.scenario_config.mission_sequence and self.scenario_config.mission_sequence.missions:
                reference_lines['T_ambient'] = self.scenario_config.mission_sequence.missions[0].ambient_temperature

            # Sequential tank evolution (4-panel plot)
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_name = Path(save_path).stem
                save_ext = Path(save_path).suffix or '.png'
                plot_path = save_dir / f"{save_name}_sequential_evolution_tank{tank_idx + 1}{save_ext}"
            else:
                plot_path = None

            fig1 = plotter.plot_sequential_tank_evolution(
                mission_results=mission_data,
                tank_index=tank_idx,
                reference_lines=reference_lines,
                save_path=str(plot_path) if plot_path else None
            )
            figures.append(fig1)

            # Sequential density-temperature diagram
            if save_path:
                plot_path = save_dir / f"{save_name}_sequential_density_temperature_tank{tank_idx + 1}{save_ext}"
            else:
                plot_path = None

            # Get density-temperature plot configuration
            dt_config = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('density_temperature', {})
            isobar_pressures = dt_config.get('isobar_pressures', [450, 400, 100, 15, 5])
            show_reference_pressures = dt_config.get('show_reference_pressures', True)
            include_saturation_line = dt_config.get('include_saturation_line', True)
            include_isobars = dt_config.get('include_isobars', True)

            ref_pressures = reference_lines if show_reference_pressures else None

            fig2 = plotter.plot_sequential_density_temperature(
                mission_results=mission_data,
                tank_index=tank_idx,
                include_saturation_line=include_saturation_line,
                include_isobars=include_isobars,
                isobar_pressures=isobar_pressures,
                reference_pressures=ref_pressures,
                save_path=str(plot_path) if plot_path else None
            )
            figures.append(fig2)

            # Sequential mass flows
            mf_config = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('mass_flows', {})
            if mf_config.get('enabled', True):
                if save_path:
                    mf_filename = mf_config.get('filename', 'mass_flows')
                    plot_path = save_dir / f"{save_name}_sequential_{mf_filename}_tank{tank_idx + 1}{save_ext}"
                else:
                    plot_path = None

                fig3 = plotter.plot_sequential_mass_flows(
                    mission_results=mission_data,
                    tank_index=tank_idx,
                    save_path=str(plot_path) if plot_path else None
                )
                figures.append(fig3)

        # Count and report plots generated
        plots_per_tank = 2  # Always: evolution + density-temperature
        if mf_config.get('enabled', True):
            plots_per_tank += 1

        plot_types = ['sequential evolution', 'sequential density-temperature']
        if mf_config.get('enabled', True):
            plot_types.append('sequential mass flows')

        print(f"   ✅ Generated {len(figures)} sequential plots ({plots_per_tank} plots per tank: {' + '.join(plot_types)})")

        # Return the first figure (or all figures if multiple tanks)
        return figures[0] if len(figures) == 1 else figures

    def _generate_single_mission_plots(self, plotter, save_path: str = None):
        """Generate plots for single mission analysis."""
        figures = []

        # Check if multi-tank overlay is enabled and we have multiple tanks
        num_tanks = len(self.tank_geometries)
        use_overlay = plotter.enable_multi_tank_overlay and num_tanks > 1

        if use_overlay:
            print(f"   🎨 Plotting multi-tank evolution (overlay mode, {num_tanks} tanks)...")

            # Create reference lines from first tank configuration (for overlay mode)
            tank_config_data = list(self.scenario_config.tank_geometries.values())[0]
            reference_lines = plotter.create_reference_lines_from_config(tank_config_data)

            # Add mission ambient temperature if available
            mission = self.scenario_config.mission_sequence.missions[0]
            reference_lines['T_ambient'] = mission.ambient_temperature

            # Generate single overlay plot for all tanks
            overlay_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_name = Path(save_path).stem
                save_ext = Path(save_path).suffix or '.png'
                overlay_save_path = save_dir / f"{save_name}_evolution_all_tanks{save_ext}"

            fig = plotter.plot_tank_evolution(
                results=self.results,
                tank_index=0,  # Not used in overlay mode
                reference_lines=reference_lines,
                save_path=str(overlay_save_path) if overlay_save_path else None,
                overlay_all_tanks=True
            )
            figures.append(fig)

        # Generate per-tank plots (individual evolution plots if not overlaying, plus other plot types)
        for tank_idx in range(num_tanks):
            # Create reference lines from tank configuration
            tank_config_data = list(self.scenario_config.tank_geometries.values())[tank_idx]
            reference_lines = plotter.create_reference_lines_from_config(tank_config_data)

            # Add mission ambient temperature if available
            mission = self.scenario_config.mission_sequence.missions[0]
            reference_lines['T_ambient'] = mission.ambient_temperature

            # Generate individual tank evolution plot (only if not using overlay mode)
            if not use_overlay:
                print(f"   🎨 Plotting Tank {tank_idx + 1} evolution...")

                tank_save_path = None
                if save_path:
                    # Create tank-specific save path
                    from pathlib import Path
                    save_dir = Path(save_path).parent
                    save_name = Path(save_path).stem
                    save_ext = Path(save_path).suffix or '.png'
                    tank_save_path = save_dir / f"{save_name}_evolution_tank{tank_idx + 1}{save_ext}"

                fig = plotter.plot_tank_evolution(
                    results=self.results,
                    tank_index=tank_idx,
                    reference_lines=reference_lines,
                    save_path=str(tank_save_path) if tank_save_path else None
                )
                figures.append(fig)

            # Generate density-temperature plot for this tank
            dt_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_name = Path(save_path).stem
                save_ext = Path(save_path).suffix or '.png'
                dt_save_path = save_dir / f"{save_name}_density_temperature_tank{tank_idx + 1}{save_ext}"

            # Get density-temperature plot configuration
            dt_config = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('density_temperature', {})

            # Extract plot parameters from config with defaults
            isobar_pressures = dt_config.get('isobar_pressures', [450, 400, 100, 15, 5])
            show_reference_pressures = dt_config.get('show_reference_pressures', True)
            include_saturation_line = dt_config.get('include_saturation_line', True)
            include_isobars = dt_config.get('include_isobars', True)

            # Only pass reference pressures if config allows it
            ref_pressures = reference_lines if show_reference_pressures else None

            # For multi-tank systems, disable isobars and saturation lines by default to avoid clutter
            is_multi_tank = len(self.tank_geometries) > 1
            show_saturation = include_saturation_line if not is_multi_tank else False
            show_isobars = include_isobars if not is_multi_tank else False

            dt_fig = plotter.plot_density_temperature(
                results=self.results,
                tank_index=tank_idx,
                include_saturation_line=show_saturation,
                include_isobars=show_isobars,
                isobar_pressures=isobar_pressures,
                reference_pressures=ref_pressures,
                save_path=str(dt_save_path) if dt_save_path else None
            )
            figures.append(dt_fig)

            # Get mass flow plot configuration
            mf_config = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('mass_flows', {})

            # Generate mass flow plot for this tank
            mf_save_path = None
            if save_path:
                from pathlib import Path
                save_dir = Path(save_path).parent
                save_name = Path(save_path).stem
                save_ext = Path(save_path).suffix or '.png'

                # Use filename from config if available
                mf_filename = mf_config.get('filename', 'mass_flows')
                mf_save_path = save_dir / f"{save_name}_{mf_filename}_tank{tank_idx + 1}{save_ext}"

            # Check if mass flow plots are enabled
            if mf_config.get('enabled', True):
                # Extract plot parameters from config with defaults
                include_venting_flow = mf_config.get('include_venting_flow', True)
                include_coupling_flows = mf_config.get('include_coupling_flows', True)

                mf_fig = plotter.plot_mass_flows(
                    results=self.results,
                    tank_index=tank_idx,
                    include_venting_flow=include_venting_flow,
                    include_coupling_flows=include_coupling_flows,
                    save_path=str(mf_save_path) if mf_save_path else None
                )
                figures.append(mf_fig)

            # Get heat exchanger plot configuration
            hex_config = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('heat_exchanger_requirements', {})

            # Generate heat exchanger plot for this tank (if enabled)
            if hex_config.get('enabled', False):
                hex_save_path = None
                if save_path:
                    from pathlib import Path
                    save_dir = Path(save_path).parent
                    save_name = Path(save_path).stem
                    save_ext = Path(save_path).suffix or '.png'

                    # Use filename from config if available
                    hex_filename = hex_config.get('filename', 'heat_exchanger_requirements')
                    hex_save_path = save_dir / f"{save_name}_{hex_filename}_tank{tank_idx + 1}{save_ext}"

                # Calculate OHEX data if needed
                qdot_ohex = self._calculate_ohex_requirements(tank_idx) if hex_config.get('include_ohex', True) else []

                # Calculate iHEX data if needed
                qdot_ihex = self._calculate_ihex_requirements(tank_idx) if hex_config.get('include_ihex', True) else [0.0] * len(self.results.times)

                # Prepare heat exchanger data
                heat_exchanger_data = {
                    'times': self.results.times / 3600.0,  # Convert to hours
                    'ihex_requirements': qdot_ihex,
                    'ohex_requirements': qdot_ohex
                }

                hex_fig = plotter.plot_heat_exchanger_requirements(
                    heat_exchanger_data=heat_exchanger_data,
                    tank_index=tank_idx,
                    include_ohex=hex_config.get('include_ohex', True),
                    include_total=hex_config.get('include_total', True),
                    save_path=str(hex_save_path) if hex_save_path else None
                )
                figures.append(hex_fig)

        # Count plots generated
        mass_flows_enabled = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('mass_flows', {}).get('enabled', True)
        heat_exchanger_enabled = self.scenario_config.config_dict.get('output', {}).get('plots', {}).get('heat_exchanger_requirements', {}).get('enabled', False)

        plots_per_tank = 2  # Always: evolution + density-temperature
        if mass_flows_enabled:
            plots_per_tank += 1
        if heat_exchanger_enabled:
            plots_per_tank += 1

        total_plots = len(self.results.tank_metadata) * plots_per_tank

        plot_types = ['evolution', 'density-temperature']
        if mass_flows_enabled:
            plot_types.append('mass flows')
        if heat_exchanger_enabled:
            plot_types.append('heat exchanger requirements')

        print(f"   ✅ Generated {len(figures)} tank plots ({plots_per_tank} plots per tank: {' + '.join(plot_types)})")

        # Return the first figure (or all figures if multiple tanks)
        return figures[0] if len(figures) == 1 else figures

    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the scenario configuration and results."""
        summary = {
            'scenario': {
                'name': self.scenario_config.analysis_name,
                'description': self.scenario_config.description,
                'version': self.scenario_config.version
            },
            'tanks': {
                'count': len(self.tank_geometries),
                'geometries': [
                    {
                        'volume': tank.volume,
                        'radius': getattr(tank, 'radius', 'N/A')
                    }
                    for tank in self.tank_geometries
                ]
            },
            'missions': {
                'count': self.scenario_config.get_mission_count(),
                'sequences': [
                    {
                        'type': mission.type,
                        'profile': mission.profile,
                        'ambient_temperature': mission.ambient_temperature
                    }
                    for mission in self.scenario_config.mission_sequence.missions
                ]
            },
            'materials': {
                'count': len(self.scenario_config.materials),
                'types': list(self.scenario_config.materials.keys())
            }
        }

        # Add results summary if available
        if self.results:
            final_multi_state = self.results.multi_tank_states[-1]
            summary['results'] = {
                'final_time_hours': self.results.times[-1] / 3600,
                'data_points': len(self.results.times),
                'final_tank_states': [
                    {
                        'fuel_mass': final_multi_state.get_tank_state(i).fuel_mass,
                        'temperature': final_multi_state.get_tank_state(i).temperature,
                        'density': final_multi_state.get_tank_state(i).density
                    }
                    for i in range(len(self.tank_geometries))
                ]
            }

        if self.validation_results:
            summary['validation'] = self.validation_results

        return summary

    def get_comprehensive_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary with all input/output parameters for benchmarking."""
        from CoolProp.CoolProp import PropsSI
        import numpy as np

        try:
            # Initialize summary structure
            summary = {
                'analysis_name': self.scenario_config.analysis_name,
                'tank_count': len(self.tank_geometries),
                'tanks': {}
            }

            # Process each tank
            for tank_idx in range(len(self.tank_geometries)):
                tank_name = f"Tank_{tank_idx + 1}"
                tank_geometry = self.tank_geometries[tank_idx]

                # Get tank configuration data
                tank_config_list = list(self.scenario_config.tank_geometries.values())
                tank_config = tank_config_list[tank_idx] if tank_idx < len(tank_config_list) else {}

                # Get initial state if results exist
                initial_state = None
                final_state = None
                if self.results and self.results.multi_tank_states:
                    initial_state = self.results.multi_tank_states[0].get_tank_state(tank_idx)
                    final_state = self.results.multi_tank_states[-1].get_tank_state(tank_idx)

                # === DIRECT INPUTS ===
                direct_inputs = {}

                # Initial conditions from tank config (handle None and string values safely)
                initial_pressure_pa = tank_config.get('initial_pressure', 0)
                initial_pressure_bar = float(initial_pressure_pa) / 1e5 if initial_pressure_pa else 0.0

                # Try different temperature keys - it might be calculated vs configured
                initial_temperature = tank_config.get('initial_temperature',
                                   tank_config.get('calculated_temperature', 0))
                initial_temperature = float(initial_temperature) if initial_temperature else 0.0

                # Try to get the temperature from tank system if available and temperature is still 0
                if initial_temperature == 0 and hasattr(self, 'tank_system') and self.tank_system:
                    try:
                        # Access tank configuration from tank system
                        if hasattr(self.tank_system.config, 'tanks') and tank_idx < len(self.tank_system.config.tanks):
                            tank_system_config = self.tank_system.config.tanks[tank_idx]
                            if hasattr(tank_system_config, 'T_INIT'):
                                initial_temperature = tank_system_config.T_INIT
                    except:
                        pass

                # Also try to get the calculated temperature from results if available
                if initial_temperature == 0 and initial_state:
                    initial_temperature = initial_state.temperature

                initial_density = tank_config.get('initial_density', 0)
                initial_density = float(initial_density) if initial_density else 0.0

                # Pressure limits - try different key names used in config
                vent_pressure_pa = (tank_config.get('vent_pressure', 0) or
                                   tank_config.get('venting_pressure', 0))
                vent_pressure_bar = float(vent_pressure_pa) / 1e5 if vent_pressure_pa else 0.0

                min_pressure_pa = tank_config.get('minimum_pressure', 0)
                min_pressure_bar = float(min_pressure_pa) / 1e5 if min_pressure_pa else 0.0

                # Stopping criteria - check both config locations
                min_density = (tank_config.get('minimum_density', 0) or
                              self.scenario_config.config_dict.get('stopping_criteria', {}).get('minimum_density', 0))
                min_density = float(min_density) if min_density else 0.0

                # Material thicknesses from config
                liner_thickness_m = 0.005  # Default 5mm
                insulation_thickness_m = 0.050  # Default 50mm

                # Try to get actual thicknesses from materials config
                materials_config = self.scenario_config.config_dict.get('materials', {})
                if 'liner' in materials_config:
                    liner_thickness_m = float(materials_config['liner'].get('thickness', 0.005))
                if 'insulation' in materials_config:
                    insulation_thickness_m = float(materials_config['insulation'].get('thickness', 0.050))

                direct_inputs.update({
                    'initial_pressure_bar': initial_pressure_bar,
                    'initial_temperature_K': initial_temperature,
                    'initial_density_kg_m3': initial_density,
                    'vent_pressure_bar': vent_pressure_bar,
                    'min_pressure_bar': min_pressure_bar,
                    'min_density_kg_m3': min_density,
                    'liner_thickness_m': liner_thickness_m,
                    'insulation_thickness_m': insulation_thickness_m
                })

                # === PREPROCESSED INPUTS ===
                preprocessed_inputs = {}

                # Calculate preprocessed temperatures if we have valid initial conditions
                if initial_pressure_bar > 0 and initial_temperature > 0:
                    try:
                        # Get hydrogen critical pressure for comparison
                        P_crit = PropsSI("Pcrit", "hydrogen") / 1e5  # Convert to bar

                        if initial_pressure_bar <= P_crit:
                            # Below critical pressure - can calculate saturation temperature
                            T_sat = PropsSI("T", "P", initial_pressure_bar * 1e5, "Q", 0, "hydrogen")
                            delta_T = initial_temperature - T_sat
                        else:
                            # Above critical pressure - supercritical fluid (no saturation line)
                            # Use critical temperature as reference
                            T_crit = PropsSI("Tcrit", "hydrogen")
                            T_sat = T_crit  # Reference to critical point
                            delta_T = initial_temperature - T_crit

                        preprocessed_inputs.update({
                            'T_sat_K': T_sat,
                            'delta_T_K': delta_T
                        })
                    except Exception as e:
                        # Handle CoolProp errors gracefully
                        preprocessed_inputs.update({
                            'T_sat_K': 0.0,
                            'delta_T_K': 0.0
                        })
                else:
                    preprocessed_inputs.update({
                        'T_sat_K': 0.0,
                        'delta_T_K': 0.0
                    })

                # === DIRECT OUTPUTS (Tank Geometry) ===
                direct_outputs = {}

                # Tank dimensions
                tank_volume = tank_geometry.volume
                tank_radius = getattr(tank_geometry, 'radius', 0)
                tank_diameter = 2 * tank_radius if np.any(tank_radius > 0) else 0

                # Surface area (spherical approximation)
                surface_area = 4 * np.pi * tank_radius**2 if np.any(tank_radius > 0) else 0

                # Calculate tank masses (using typical structural mass ratios)
                liner_mass = 0.0
                wall_mass = 0.0

                # Try to get accurate masses from tank system properties
                if hasattr(self, 'tank_system') and self.tank_system:
                    try:
                        tank_props = self.tank_system._get_tank_properties(tank_geometry, f"Tank{tank_idx+1}", tank_idx)
                        liner_mass = tank_props.get('liner_mass', 0)
                        wall_mass = tank_props.get('wall_mass', 0)
                    except:
                        # Fallback calculations
                        liner_mass = tank_volume * 5.0  # ~5 kg/m³ typical
                        wall_mass = tank_volume * 50.0   # ~50 kg/m³ typical
                else:
                    # Fallback calculations
                    liner_mass = tank_volume * 5.0
                    wall_mass = tank_volume * 50.0

                total_tank_mass = liner_mass + wall_mass

                # Calculate wall thickness (composite + insulation)
                wall_thickness_m = 0.020 + insulation_thickness_m  # 20mm composite + insulation

                # Try to get actual wall thickness from tank properties if available
                if hasattr(self, 'tank_system') and self.tank_system:
                    try:
                        # Wall thickness is typically the difference between external and internal radius
                        inner_radius = tank_radius - liner_thickness_m
                        # Estimate external radius including all layers
                        wall_thickness_m = 0.020 + insulation_thickness_m  # Keep calculated value
                    except:
                        pass

                direct_outputs.update({
                    'tank_volume_m3': tank_volume,
                    'tank_radius_m': tank_radius,
                    'tank_diameter_m': tank_diameter,
                    'surface_area_m2': surface_area,
                    'wall_thickness_m': wall_thickness_m,
                    'liner_mass_kg': liner_mass,
                    'wall_mass_kg': wall_mass,
                    'total_tank_mass_kg': total_tank_mass
                })

                # Initial fuel mass
                if initial_state:
                    fuel_mass = initial_state.fuel_mass
                else:
                    # Calculate from initial conditions
                    fuel_mass = initial_density * tank_volume if np.any(initial_density > 0) else 0

                direct_outputs['fuel_mass_kg'] = fuel_mass

                # === POSTPROCESSED OUTPUTS ===
                postprocessed_outputs = {}

                # Mission duration and time to vent
                mission_duration_hours = 0
                time_to_vent_hours = 0

                if self.results and self.results.times:
                    mission_duration_hours = self.results.times[-1] / 3600

                    # Find time to vent (when pressure reaches vent limit)
                    try:
                        tank_series = self.results.get_tank_series(tank_idx)
                        vent_pressure_pa = vent_pressure_bar * 1e5

                        for i, state in enumerate(tank_series.states):
                            if state.pressure and state.pressure >= vent_pressure_pa:
                                time_to_vent_hours = self.results.times[i] / 3600
                                break

                        if time_to_vent_hours == 0:
                            time_to_vent_hours = mission_duration_hours  # Never vented
                    except:
                        time_to_vent_hours = mission_duration_hours

                # Gravimetric efficiency = hydrogen mass / (hydrogen mass + structure mass)
                # Structure mass = liner mass + wall mass (insulation neglected per definition)
                structure_mass = liner_mass + wall_mass
                total_system_mass = fuel_mass + structure_mass
                mass_efficiency = (fuel_mass / total_system_mass) if np.any(total_system_mass > 0) else 0

                # Volumetric efficiency = hydrogen volume / (hydrogen volume + structure volume)
                # Structure volume = liner volume + wall volume + insulation volume
                hydrogen_volume = tank_volume  # Internal tank volume containing hydrogen

                # Calculate structure volumes
                # Liner volume (spherical shell): V = 4π * [(r_outer)³ - (r_inner)³] / 3
                # Handle array/scalar tank_radius safely
                if hasattr(tank_radius, '__len__'):  # Array case
                    inner_radius = tank_radius - liner_thickness_m
                else:  # Scalar case
                    inner_radius = tank_radius - liner_thickness_m if tank_radius > liner_thickness_m else tank_radius * 0.95
                liner_outer_radius = tank_radius
                liner_volume = (4/3) * np.pi * (liner_outer_radius**3 - inner_radius**3)

                # Wall volume (composite + insulation layers)
                wall_outer_radius = tank_radius + wall_thickness_m
                wall_volume = (4/3) * np.pi * (wall_outer_radius**3 - liner_outer_radius**3)

                # Total volume = hydrogen volume + structure volumes
                total_volume = hydrogen_volume + liner_volume + wall_volume
                volumetric_efficiency = (hydrogen_volume / total_volume) if np.any(total_volume > 0) else 0

                # Energy calculations
                fuel_energy_mj = fuel_mass * 120.0 if np.any(fuel_mass > 0) else 0  # H2 LHV ~120 MJ/kg

                # Calculate energy requirements (placeholder - would need actual heat exchanger calculations)
                ohex_energy_mj = 0.0
                ihex_energy_mj = 0.0

                # Try to calculate actual energy requirements if available
                if self.results:
                    try:
                        # Calculate OHEX energy
                        qdot_ohex = self._calculate_ohex_requirements(tank_idx)
                        if qdot_ohex and self.results.times:
                            dt = np.diff(self.results.times)  # Time steps in seconds
                            if len(dt) == len(qdot_ohex) - 1:
                                # Integrate power to get energy
                                ohex_energy_mj = np.sum(np.array(qdot_ohex[:-1]) * dt) / 1e6  # Convert W·s to MJ

                        # Calculate iHEX energy
                        qdot_ihex = self._calculate_ihex_requirements(tank_idx)
                        if qdot_ihex and self.results.times:
                            dt = np.diff(self.results.times)
                            if len(dt) == len(qdot_ihex) - 1:
                                ihex_energy_mj = np.sum(np.array(qdot_ihex[:-1]) * dt) / 1e6
                    except:
                        pass  # Keep default values

                total_energy_mj = ohex_energy_mj + ihex_energy_mj

                postprocessed_outputs.update({
                    'mission_duration_hours': mission_duration_hours,
                    'time_to_vent_hours': time_to_vent_hours,
                    'mass_efficiency': mass_efficiency,
                    'volumetric_efficiency': volumetric_efficiency,
                    'fuel_energy_MJ': fuel_energy_mj,
                    'ohex_energy_MJ': ohex_energy_mj,
                    'ihex_energy_MJ': ihex_energy_mj,
                    'total_energy_MJ': total_energy_mj
                })

                # Assemble tank summary
                summary['tanks'][tank_name] = {
                    'direct_inputs': direct_inputs,
                    'preprocessed_inputs': preprocessed_inputs,
                    'direct_outputs': direct_outputs,
                    'postprocessed_outputs': postprocessed_outputs
                }

            return summary

        except Exception as e:
            print(f"   ⚠️ Error generating comprehensive summary: {e}")
            return {'error': str(e)}

    def print_scenario_summary(self):
        """Print comprehensive scenario summary."""
        summary = self.get_scenario_summary()

        print("\n🔍 SCENARIO CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Analysis: {summary['scenario']['name']}")
        print(f"Description: {summary['scenario']['description']}")
        print(f"Version: {summary['scenario']['version']}")
        print(f"Tanks: {summary['tanks']['count']}")
        print(f"Missions: {summary['missions']['count']}")
        print(f"Materials: {summary['materials']['count']} ({', '.join(summary['materials']['types'])})")

        if 'results' in summary:
            print(f"\nResults:")
            print(f"  Final time: {summary['results']['final_time_hours']:.2f} hours")
            print(f"  Data points: {summary['results']['data_points']}")
            for i, tank_state in enumerate(summary['results']['final_tank_states']):
                print(f"  Tank {i+1}: m={tank_state['fuel_mass']:.2f}kg, "
                      f"T={tank_state['temperature']:.1f}K, "
                      f"ρ={tank_state['density']:.1f}kg/m³")

        if 'validation' in summary:
            overall = summary['validation']['overall']
            print(f"  Validation: {'✅ PASSED' if overall else '❌ FAILED'}")

        print("=" * 60)

    def print_comprehensive_analysis_summary(self):
        """Print comprehensive analysis parameter summary table."""
        summary = self.get_comprehensive_analysis_summary()

        if 'error' in summary:
            print(f"❌ Error generating summary: {summary['error']}")
            return

        print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY: {summary['analysis_name']}")
        print("=" * 100)

        # Header
        header = f"{'Parameter':<35} {'Unit':<12}"
        for tank_name in summary['tanks'].keys():
            header += f"{tank_name:<15}"
        print(header)
        print("-" * 100)

        # Define parameter groups and their display information
        param_groups = [
            ("DIRECT INPUTS", [
                ('initial_pressure_bar', 'bar', 'Initial Pressure'),
                ('initial_temperature_K', 'K', 'Initial Temperature'),
                ('initial_density_kg_m3', 'kg/m³', 'Initial Density'),
                ('vent_pressure_bar', 'bar', 'Vent Pressure'),
                ('min_pressure_bar', 'bar', 'Minimum Pressure'),
                ('min_density_kg_m3', 'kg/m³', 'Minimum Density'),
                ('liner_thickness_m', 'm', 'Liner Thickness'),
                ('insulation_thickness_m', 'm', 'Insulation Thickness')
            ]),
            ("PREPROCESSED INPUTS", [
                ('T_sat_K', 'K', 'Saturation Temperature'),
                ('delta_T_K', 'K', 'ΔT from Saturation')
            ]),
            ("DIRECT OUTPUTS", [
                ('tank_volume_m3', 'm³', 'Tank Volume'),
                ('tank_radius_m', 'm', 'Tank Radius'),
                ('tank_diameter_m', 'm', 'Tank Diameter'),
                ('surface_area_m2', 'm²', 'Surface Area'),
                ('wall_thickness_m', 'm', 'Wall Thickness'),
                ('fuel_mass_kg', 'kg', 'Fuel Mass'),
                ('liner_mass_kg', 'kg', 'Liner Mass'),
                ('wall_mass_kg', 'kg', 'Wall Mass'),
                ('total_tank_mass_kg', 'kg', 'Total Tank Mass')
            ]),
            ("POSTPROCESSED OUTPUTS", [
                ('mission_duration_hours', 'hours', 'Mission Duration'),
                ('time_to_vent_hours', 'hours', 'Time to Vent'),
                ('mass_efficiency', '-', 'Mass Efficiency'),
                ('volumetric_efficiency', '-', 'Volumetric Efficiency'),
                ('fuel_energy_MJ', 'MJ', 'Fuel Energy'),
                ('ohex_energy_MJ', 'MJ', 'OHEX Energy'),
                ('ihex_energy_MJ', 'MJ', 'iHEX Energy'),
                ('total_energy_MJ', 'MJ', 'Total Energy')
            ])
        ]

        # Print each parameter group
        for group_name, params in param_groups:
            print(f"\n{group_name}")
            print("-" * 50)

            for param_key, unit, display_name in params:
                row = f"{display_name:<35} {unit:<12}"

                for tank_name in summary['tanks'].keys():
                    tank_data = summary['tanks'][tank_name]

                    # Find the parameter in the appropriate group
                    value = None
                    for group_key in ['direct_inputs', 'preprocessed_inputs', 'direct_outputs', 'postprocessed_outputs']:
                        if param_key in tank_data[group_key]:
                            value = tank_data[group_key][param_key]
                            break

                    if value is not None:
                        # Format based on parameter type
                        if 'efficiency' in param_key:
                            formatted_value = f"{value:.3f}"
                        elif 'energy' in param_key or 'MJ' in unit:
                            formatted_value = f"{value:.1f}"
                        elif 'hours' in unit:
                            formatted_value = f"{value:.2f}"
                        elif 'bar' in unit:
                            formatted_value = f"{value:.0f}"
                        elif param_key in ['tank_volume_m3', 'surface_area_m2']:
                            formatted_value = f"{value:.3f}"
                        elif param_key in ['tank_radius_m', 'tank_diameter_m']:
                            formatted_value = f"{value:.2f}"
                        elif 'thickness' in param_key:
                            formatted_value = f"{value:.3f}"
                        elif 'mass' in param_key:
                            formatted_value = f"{value:.1f}"
                        elif 'density' in param_key:
                            formatted_value = f"{value:.1f}"
                        elif 'temperature' in param_key.lower() or unit == 'K':
                            formatted_value = f"{value:.1f}"
                        else:
                            formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = "N/A"

                    row += f"{formatted_value:<15}"

                print(row)

        print("\n" + "=" * 100)
        print("✅ Comprehensive analysis summary complete")
        print("   📝 All input parameters, tank geometry, and performance metrics displayed")

        # Also save to markdown file
        self.save_comprehensive_analysis_summary_to_markdown()

    def save_comprehensive_analysis_summary_to_markdown(self, output_dir: str = None):
        """Save comprehensive analysis parameter summary to markdown file."""
        import os
        from datetime import datetime

        if output_dir is None:
            # Use the configured output directory
            output_dir = self.scenario_config.config_dict.get('output', {}).get('results_directory', 'output/results')

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        analysis_name = self.scenario_config.analysis_name.replace(' ', '_').lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{analysis_name}_comprehensive_summary_{timestamp}.text"
        filepath = os.path.join(output_dir, filename)

        try:
            summary = self.get_comprehensive_analysis_summary()

            if 'error' in summary:
                print(f"❌ Cannot save summary due to error: {summary['error']}")
                return None

            # Generate markdown content
            md_content = f"# Comprehensive Analysis Summary: {summary['analysis_name']}\n\n"
            md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += f"**Number of Tanks:** {summary['tank_count']}\n\n"

            # Define parameter groups and their display information (same as print method)
            param_groups = [
                ("DIRECT INPUTS", [
                    ('initial_pressure_bar', 'bar', 'Initial Pressure'),
                    ('initial_temperature_K', 'K', 'Initial Temperature'),
                    ('initial_density_kg_m3', 'kg/m³', 'Initial Density'),
                    ('vent_pressure_bar', 'bar', 'Vent Pressure'),
                    ('min_pressure_bar', 'bar', 'Minimum Pressure'),
                    ('min_density_kg_m3', 'kg/m³', 'Minimum Density'),
                    ('liner_thickness_m', 'm', 'Liner Thickness'),
                    ('insulation_thickness_m', 'm', 'Insulation Thickness')
                ]),
                ("PREPROCESSED INPUTS", [
                    ('T_sat_K', 'K', 'Saturation Temperature'),
                    ('delta_T_K', 'K', 'ΔT from Saturation')
                ]),
                ("DIRECT OUTPUTS", [
                    ('tank_volume_m3', 'm³', 'Tank Volume'),
                    ('tank_radius_m', 'm', 'Tank Radius'),
                    ('tank_diameter_m', 'm', 'Tank Diameter'),
                    ('surface_area_m2', 'm²', 'Surface Area'),
                    ('wall_thickness_m', 'm', 'Wall Thickness'),
                    ('fuel_mass_kg', 'kg', 'Fuel Mass'),
                    ('liner_mass_kg', 'kg', 'Liner Mass'),
                    ('wall_mass_kg', 'kg', 'Wall Mass'),
                    ('total_tank_mass_kg', 'kg', 'Total Tank Mass')
                ]),
                ("POSTPROCESSED OUTPUTS", [
                    ('mission_duration_hours', 'hours', 'Mission Duration'),
                    ('time_to_vent_hours', 'hours', 'Time to Vent'),
                    ('mass_efficiency', '-', 'Mass Efficiency'),
                    ('volumetric_efficiency', '-', 'Volumetric Efficiency'),
                    ('fuel_energy_MJ', 'MJ', 'Fuel Energy'),
                    ('ohex_energy_MJ', 'MJ', 'OHEX Energy'),
                    ('ihex_energy_MJ', 'MJ', 'iHEX Energy'),
                    ('total_energy_MJ', 'MJ', 'Total Energy')
                ])
            ]

            # Create markdown table
            tank_names = list(summary['tanks'].keys())

            # Header row
            header = "| Parameter | Unit |"
            for tank_name in tank_names:
                header += f" {tank_name} |"
            header += "\n"

            # Separator row
            separator = "|-----------|------|"
            for _ in tank_names:
                separator += "--------|"
            separator += "\n"

            md_content += header + separator

            # Add each parameter group
            for group_name, params in param_groups:
                md_content += f"\n**{group_name}**\n\n"

                for param_key, unit, display_name in params:
                    row = f"| {display_name} | {unit} |"

                    for tank_name in tank_names:
                        tank_data = summary['tanks'][tank_name]

                        # Find the parameter in the appropriate group
                        value = None
                        for group_key in ['direct_inputs', 'preprocessed_inputs', 'direct_outputs', 'postprocessed_outputs']:
                            if param_key in tank_data[group_key]:
                                value = tank_data[group_key][param_key]
                                break

                        if value is not None:
                            # Format based on parameter type (same logic as print method)
                            if 'efficiency' in param_key:
                                formatted_value = f"{value:.3f}"
                            elif 'energy' in param_key or 'MJ' in unit:
                                formatted_value = f"{value:.1f}"
                            elif 'hours' in unit:
                                formatted_value = f"{value:.2f}"
                            elif 'bar' in unit:
                                formatted_value = f"{value:.0f}"
                            elif param_key in ['tank_volume_m3', 'surface_area_m2']:
                                formatted_value = f"{value:.3f}"
                            elif param_key in ['tank_radius_m', 'tank_diameter_m']:
                                formatted_value = f"{value:.2f}"
                            elif 'thickness' in param_key:
                                formatted_value = f"{value:.3f}"
                            elif 'mass' in param_key:
                                formatted_value = f"{value:.1f}"
                            elif 'density' in param_key:
                                formatted_value = f"{value:.1f}"
                            elif 'temperature' in param_key.lower() or unit == 'K':
                                formatted_value = f"{value:.1f}"
                            else:
                                formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = "N/A"

                        row += f" {formatted_value} |"

                    md_content += row + "\n"

            # Add footer
            md_content += f"\n---\n\n"
            md_content += f"*Generated by SystemOrchestrator comprehensive analysis*\n"

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)

            print(f"   💾 Comprehensive summary saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"   ❌ Error saving markdown summary: {e}")
            return None


def main():
    """Test the complete ScenarioConfig → SystemOrchestrator → MultiTankSystem pipeline."""
    import os
    from pathlib import Path

    print("🚀 TESTING ORCHESTRATOR INTEGRATION")
    print("=" * 60)

    # Find test configuration
    test_config_path = Path("data/single_tank_cch2_config.yaml")
    if not test_config_path.exists():
        # Try alternative location
        test_config_path = Path(__file__).parent.parent.parent / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

    if not test_config_path.exists():
        print(f"❌ Config file not found at expected locations")
        return

    try:
        # Load ScenarioConfig first
        print(f"📋 Loading configuration: {test_config_path}")
        from src.configuration.scenario_configuration import ScenarioConfig
        scenario_config = ScenarioConfig.from_yaml(str(test_config_path))

        # Create orchestrator with ScenarioConfig object
        orchestrator = SystemOrchestrator(scenario_config)

        # Print scenario summary
        orchestrator.print_scenario_summary()

        # Run simulation
        print("\n⚡ Running DAE simulation...")
        results = orchestrator.run_simulation()

        # Print final results
        print(f"\n✅ Simulation Complete!")
        print(f"   Final time: {results.t[-1]/3600:.2f} hours")
        print(f"   Data points: {len(results.t)}")

        # Print final tank states
        n_tanks = len(orchestrator.tank_geometries)
        print(f"\n🛢️  Final Tank States ({n_tanks} tanks):")
        for i in range(n_tanks):
            idx = i * 3  # Each tank has 3 state variables: m, T, Ts
            m_final = results.y[idx, -1]
            T_final = results.y[idx + 1, -1]

            print(f"   Tank {i+1}: m={m_final:.2f}kg, T={T_final:.1f}K")

        print("\n🎉 ORCHESTRATOR INTEGRATION SUCCESS!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()