import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "src"))

from materials.materials_for_multi_tank.nist_material import NISTMaterial, get_material_by_nist_path


@dataclass
class MissionConfig:
    type: str
    profile: str
    ambient_temperature: float
    parameters: Dict[str, Any]
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        mission_data = config_dict.get('mission', {})
        mission_type = mission_data.get('type')
        profile = mission_data.get('profile')
        ambient_temp = mission_data.get('ambient_temperature', 288.15)
        parameters = mission_data.get('parameters', {})

        return cls(
            type=mission_type,
            profile=profile,
            ambient_temperature=ambient_temp,
            parameters=parameters
        )


@dataclass
class MissionSequenceConfig:
    missions: List[MissionConfig]

    @classmethod
    def from_dict(cls, config_dict: dict):
        mission_data = config_dict.get('mission', {})

        if 'sequence' in mission_data:
            missions = []
            for i, seq_mission in enumerate(mission_data['sequence']):
                mission_config = {
                    'mission': {
                        'type': seq_mission.get('type'),
                        'profile': seq_mission.get('profile'),
                        'ambient_temperature': seq_mission.get('ambient_temperature', 288.15),
                        'parameters': seq_mission.get('parameters', {})
                    }
                }
                missions.append(MissionConfig.from_dict(mission_config))
            return cls(missions=missions)
        else:
            single_mission = MissionConfig.from_dict(config_dict)
            return cls(missions=[single_mission])

    def validate(self):
        if not self.missions:
            raise ValueError("At least one mission is required")
        for mission in self.missions:
            if not mission.type or not mission.profile:
                raise ValueError("Mission type and profile are required")


class ScenarioConfig:

    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.analysis_name = config_dict.get('analysis_name', 'Unnamed Analysis')
        self.description = config_dict.get('description', '')
        self.version = config_dict.get('version', '1.0')

        self.mission_sequence = MissionSequenceConfig.from_dict(config_dict)
        self.materials = self._parse_materials()
        self.tank_geometries = self._parse_tank_geometries()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        scenario = cls(config_dict)
        scenario._config_path = str(yaml_path)  # Store config path for SystemOrchestrator
        scenario.validate()
        return scenario

    def _parse_materials(self) -> Dict[str, NISTMaterial]:
        materials = {}
        materials_data = self.config_dict.get('materials', {})

        for material_name, material_data in materials_data.items():
            if material_name == 'safety_margin':
                continue

            nist_path = material_data.get('nist_path')
            if nist_path:
                try:
                    materials[material_name] = get_material_by_nist_path(nist_path)
                except ValueError as e:
                    print(f"Warning: Could not load NIST material {nist_path}: {e}")

        return materials

    def _parse_tank_geometries(self) -> Dict[int, Dict[str, Any]]:
        geometries = {}
        geometry_data = self.config_dict.get('geometry', {})

        for tank_id, geom_data in geometry_data.items():
            geometries[int(tank_id)] = geom_data

        return geometries

    def validate(self):
        self.mission_sequence.validate()

        if not self.tank_geometries:
            raise ValueError("At least one tank geometry is required")

    def get_tank_count(self) -> int:
        return len(self.tank_geometries)

    def get_mission_count(self) -> int:
        return len(self.mission_sequence.missions)

    def summary(self) -> str:
        lines = []
        lines.append(f"Scenario: {self.analysis_name}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Version: {self.version}")
        lines.append(f"Tanks: {self.get_tank_count()}")
        lines.append(f"Missions: {self.get_mission_count()}")
        lines.append(f"Materials: {list(self.materials.keys())}")
        return "\n".join(lines)


def main():
    test_config_path = Path(__file__).parent.parent.parent / "analysis" / "multi_tank_systems" / "single_tank_cch2" / "single_tank_cch2_config.yaml"

    if not test_config_path.exists():
        print(f"Test configuration file not found: {test_config_path}")
        return False

    try:
        print("Testing ScenarioConfig with single_tank_cch2_config.yaml...")

        scenario = ScenarioConfig.from_yaml(test_config_path)

        print("Configuration parsed successfully!")
        print("=" * 60)
        print(scenario.summary())
        print("=" * 60)

        print("\nNIST Materials:")
        for name, material in scenario.materials.items():
            specific_heat = material.get_specific_heat(300.0)
            print(f"  - {name}: {material.__class__.__name__}, Cp(300K) = {specific_heat:.2f} J/kg/K")

        print("\nMission Details:")
        for i, mission in enumerate(scenario.mission_sequence.missions, 1):
            print(f"  Mission {i}: {mission.type} - {mission.profile}")
            if hasattr(mission, 'parameters') and 'stopping_criteria' in mission.parameters:
                criteria = mission.parameters['stopping_criteria']
                print(f"    Stopping criteria: {list(criteria.keys())}")

        print("\nTank Geometry:")
        for tank_id, geometry in scenario.tank_geometries.items():
            initial_pressure = float(geometry['initial_pressure']) if isinstance(geometry['initial_pressure'], str) else geometry['initial_pressure']
            print(f"  Tank {tank_id}: phi={geometry['phi']}, P_init={initial_pressure/1e5:.0f} bar")

        return True

    except Exception as e:
        print(f"ScenarioConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)