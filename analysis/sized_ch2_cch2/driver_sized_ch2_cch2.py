#!/usr/bin/env python3
"""
Coupled CH2-CCH2 Multi-Tank Analysis Driver

Coupled gaseous and cryo-compressed hydrogen storage system with pressure
compensation coupling. Optionally checks whether dormancy causes venting.
"""

import copy
import sys
import tempfile
from pathlib import Path

import yaml

# Add parent directories for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

from src.orchestration.run_analysis import run_analysis


def _dormancy_check(config_path: Path) -> None:
    """
    Run a zero-outflow dormancy simulation from the same initial conditions as
    discharge and report whether venting occurs within the configured window.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    dormancy_cfg = raw.get('dormancy_check', {})
    if not dormancy_cfg.get('enabled', False):
        return

    duration_h = float(dormancy_cfg.get('duration_h', 24.0))

    print(f"\n{'=' * 80}")
    print(f"DORMANCY CHECK  ({duration_h:.1f} h)  — same initial conditions as discharge")
    print('=' * 80)

    # Build a dormancy variant: copy all node/material/physics config unchanged
    # so initial conditions, geometry, and HTC are identical to the discharge run.
    dc = copy.deepcopy(raw)
    original_mission = raw.get('mission', {})
    dc['mission'] = {
        'type': 'dormancy',
        'profile': 'constant_flow',
        'ambient_temperature': original_mission.get('ambient_temperature', 288.15),
        # assigned_to_node required by mission-assignment logic; zero flow so choice is irrelevant
        'assigned_to_node': original_mission.get('assigned_to_node', 1),
        'flow_rate': 0.0,
        'duration': duration_h * 3600.0,
    }
    # Suppress all file output for this auxiliary run
    dc.setdefault('output', {}).update({'save_plots': False, 'save_data': False, 'silent': True})
    dc.pop('dormancy_check', None)

    # Write temp config to the same directory so relative CSV/material paths still resolve
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', dir=config_path.parent, delete=False
    ) as tf:
        yaml.dump(dc, tf)
        temp_path = Path(tf.name)

    try:
        from src.configuration.scenario_configuration import ScenarioConfig
        from src.orchestration.system_orchestrator import SystemOrchestrator

        config = ScenarioConfig.from_yaml(str(temp_path))
        orchestrator = SystemOrchestrator(config, verbosity='quiet')
        results = orchestrator.run_simulation()
    finally:
        temp_path.unlink(missing_ok=True)

    # Per-tank venting report
    any_venting = False
    tank_configs = list(orchestrator.scenario_config.tank_geometries.values())

    for idx in range(len(orchestrator.tank_geometries)):
        tc = tank_configs[idx]
        p_vent_pa = float(tc.get('venting_pressure', 0.0))
        p_vent_bar = p_vent_pa / 1e5

        pressures = [ms.get_tank_state(idx).pressure for ms in results.multi_tank_states]
        peak_bar = max(pressures) / 1e5

        vent_time_s = next(
            (results.times[i] for i, ms in enumerate(results.multi_tank_states)
             if ms.get_tank_state(idx).pressure >= p_vent_pa),
            None,
        )

        tag = f"Tank {idx + 1}"
        if vent_time_s is not None:
            any_venting = True
            print(f"  {tag}: VENTING at {vent_time_s / 3600:.2f} h  "
                  f"(P_vent = {p_vent_bar:.0f} bar, peak = {peak_bar:.1f} bar)")
        else:
            print(f"  {tag}: no venting  "
                  f"(P_vent = {p_vent_bar:.0f} bar, peak = {peak_bar:.1f} bar)")

    verdict = "VENTING DETECTED" if any_venting else "No venting"
    print(f"  → {verdict} within {duration_h:.1f} h dormancy window.")


def main():
    """Execute coupled CH2-CCH2 multi-tank analysis."""
    config_path = current_dir / "sized_ch2_cch2_config.yaml"
    result = run_analysis(
        config_path=config_path,
        analysis_name="Coupled CH2-CCH2 Multi-Tank System",
        show_material_props=False,
        verbose=False,
    )
    _dormancy_check(config_path)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise



