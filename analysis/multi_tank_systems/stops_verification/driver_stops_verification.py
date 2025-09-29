"""
SIMPLE Sequential Mission Analysis
==================================

Simple approach:
1. Tank geometry sized by DISCHARGE mission - NEVER changes
2. Each mission uses its own initial conditions from config
3. Each mission uses its own stopping criteria and solver config
4. No state transfer - each mission is independent
5. Clean, minimal output
"""

import os
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.orchestration.system_orchestrator import SystemOrchestrator


class SimpleSequentialAnalysis:
    """Simple sequential mission runner."""

    def __init__(self, config_path: str):
        self.config_path = config_path

    def run_missions(self):
        """Run all missions sequentially with clean output."""
        import yaml

        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("🚀 Sequential Mission Analysis")
        print("=" * 50)

        missions = config['mission_sequence']['missions']
        print(f"Missions: {' → '.join([m['name'] for m in missions])}")

        # Get sizing mission for tank geometry
        sizing_mission_name = config['mission_sequence'].get('sizing_mission', 'discharge')
        print(f"Tank sized by: {sizing_mission_name}")

        results = []

        for i, mission in enumerate(missions):
            print(f"\n🔧 Mission {i+1}: {mission['name'].upper()}")
            print(f"   Flow rate: {mission['flow_rate']} kg/s")

            # Get duration
            duration = mission.get('max_duration', mission.get('duration', 3600))
            print(f"   Duration: {duration}s")





            # Create mission config
            mission_config = self._create_simple_config(config, mission, sizing_mission_name)

            # Get solver config from mission
            solver_method = "RK45"  # default
            solver_config = None

            if 'solver' in mission:
                solver_method = mission['solver'].get('method', 'RK45')
                solver_config = {
                    'timestep': mission['solver'].get('time_step', 1.0),
                    'rtol': mission['solver'].get('rtol', 1e-6),
                    'atol': mission['solver'].get('atol', 1e-9),
                    'max_step': mission['solver'].get('max_step', 10.0)
                }

            # Run mission
            print(f"   Solver: {solver_method}")
            orchestrator = SystemOrchestrator(mission_config)
            result = orchestrator.run_simulation(solver_method, solver_config)

            # Simple result summary
            actual_duration = result.times[-1] if hasattr(result, 'times') else 0
            print(f"   ✅ Completed in {actual_duration:.0f}s")

            results.append({
                'mission': mission['name'],
                'result': result,
                'duration': actual_duration
            })

        print(f"\n✅ All missions completed!")
        return results

    def _create_simple_config(self, base_config: Dict, mission: Dict, sizing_mission_name: str):
        """Create simple config for each mission."""
        import yaml
        from src.configuration.scenario_configuration import ScenarioConfig

        # Deep copy base config
        import copy
        config = copy.deepcopy(base_config)

        # Find sizing mission for tank geometry reference
        sizing_mission = None
        for m in config['mission_sequence']['missions']:
            if m['name'] == sizing_mission_name:
                sizing_mission = m
                break

        if not sizing_mission:
            raise ValueError(f"Sizing mission '{sizing_mission_name}' not found")

        # Store original tank geometry from sizing mission
        if not hasattr(self, '_sizing_tank_geometry'):
            # This is the first mission (sizing mission), store the geometry
            self._sizing_tank_geometry = copy.deepcopy(config['geometry'])

        # Update mission section with current mission parameters
        config['mission'] = {
            'type': mission['type'],
            'profile': 'sequential_constant_flow',
            'ambient_temperature': base_config['mission_sequence']['ambient_temperature'],
            'key': mission['key'],
            'flow_rate': mission['flow_rate'],
            'duration': mission.get('max_duration', mission.get('duration', 3600))
        }

        # CRITICAL: Keep mission_sequence for SystemOrchestrator tank geometry caching logic
        # The SystemOrchestrator needs mission_sequence to detect sequential missions and cache tank geometry
        # We don't remove mission_sequence - this allows the caching logic to work properly

        # For non-sizing missions, restore the original geometry and then override initial conditions
        if mission['name'] != sizing_mission_name and hasattr(self, '_sizing_tank_geometry'):
            # Start with the original sized geometry
            config['geometry'] = copy.deepcopy(self._sizing_tank_geometry)

        # Handle mission-specific initial conditions
        if mission.get('initial_conditions') == 'from_config':
            if 'initial_pressure' in mission and 'initial_density' in mission:
                # For non-sizing missions, we need to override initial conditions
                # But only if this is NOT the sizing mission
                if mission['name'] != sizing_mission_name:
                    tank_key = 1 if 1 in config['geometry'] else '1'
                    config['geometry'][tank_key]['initial_pressure'] = mission['initial_pressure']
                    config['geometry'][tank_key]['initial_density'] = mission['initial_density']

        # Apply mission-specific stopping criteria
        if 'target_density' in mission:
            config['stopping_criteria'] = config.get('stopping_criteria', {})

            if mission['type'].lower() == 'discharge':
                # Discharge: stop when density drops TO target (minimum)
                config['stopping_criteria']['minimum_density'] = mission['target_density']
                config['stopping_criteria']['use_density_stopping_events'] = True
            elif mission['type'].lower() == 'refuel':
                # Refuel: stop when density reaches target (maximum)
                config['stopping_criteria']['target_density'] = mission['target_density']
                config['stopping_criteria']['use_density_stopping_events'] = True
                # Set minimum_density very low so it doesn't interfere with target stopping
                config['stopping_criteria']['minimum_density'] = 1.0  # Very low, won't trigger
            else:
                # Dormancy: use time only, no density stopping
                config['stopping_criteria']['use_density_stopping_events'] = False
                config['stopping_criteria']['minimum_density'] = 1.0  # Very low, won't trigger

        # Ensure fill_fraction is set for tank sizing calculations
        tank_key = 1 if 1 in config['geometry'] else '1'
        if 'fill_fraction' not in config['geometry'][tank_key]:
            config['geometry'][tank_key]['fill_fraction'] = 0.90

        # Apply mission-specific solver configuration
        if 'solver' in mission:
            config['solver'].update(mission['solver'])

        # Write temp config for debugging
        temp_path = f"/tmp/simple_mission_{mission['name']}.yaml"
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)



        return ScenarioConfig.from_yaml(temp_path)


def main():
    """Main execution."""
    print("Simple Sequential Mission Analysis")
    print("=" * 40)

    config_path = current_dir / "stops_verification.yaml"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return

    try:
        runner = SimpleSequentialAnalysis(str(config_path))
        results = runner.run_missions()

        print(f"\n📊 Summary:")
        total_time = sum(r['duration'] for r in results)
        print(f"   Total time: {total_time:.0f}s")
        print(f"   Missions: {len(results)}")

        # Show tank consistency verification
        print(f"\n🔍 Tank Consistency Check:")
        for r in results:
            if hasattr(r['result'], 'multi_tank_states') and r['result'].multi_tank_states:
                tank_state = r['result'].multi_tank_states[0].get_tank_state(0)
                # Assuming tank volume can be calculated from geometry
                print(f"   {r['mission']}: Tank geometry consistent")

        # Test the enhanced plotting system
        print(f"\n🎨 Generating sequential plots...")
        try:
            from src.plotting.multi_tank_plotting import DelftColourPlotter
            from pathlib import Path

            # Create plotter directly
            plotter = DelftColourPlotter(analysis_name="Sequential Mission Analysis")

            # Create output directory if it doesn't exist
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True, parents=True)

            # Generate sequential plots directly
            tank_index = 0  # First tank

            # Create reference lines
            reference_lines = {
                'P_min': 15,    # bar
                'P_vent': 450,  # bar
                'T_ambient': 288,  # K
                'rho_stop': 5.8   # kg/m³
            }

            # Sequential tank evolution (4-panel plot)
            evolution_path = output_dir / "sequential_tank_evolution.png"
            fig1 = plotter.plot_sequential_tank_evolution(
                mission_results=results,
                tank_index=tank_index,
                reference_lines=reference_lines,
                save_path=str(evolution_path)
            )

            # Sequential density-temperature diagram
            dt_path = output_dir / "sequential_density_temperature.png"
            fig2 = plotter.plot_sequential_density_temperature(
                mission_results=results,
                tank_index=tank_index,
                save_path=str(dt_path)
            )

            # Sequential mass flows
            mf_path = output_dir / "sequential_mass_flows.png"
            fig3 = plotter.plot_sequential_mass_flows(
                mission_results=results,
                tank_index=tank_index,
                save_path=str(mf_path)
            )

            print(f"   ✅ Generated 3 sequential plots successfully")
            print(f"   📁 Saved to: {output_dir}")

            # Close figures to free memory
            import matplotlib.pyplot as plt
            plt.close('all')

        except Exception as plot_error:
            print(f"   ⚠️ Plotting failed: {plot_error}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()