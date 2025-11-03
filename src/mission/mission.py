from __future__ import annotations

import yaml
import os

from dataclasses import dataclass

from src.mission.mission_sections import MissionSection, OutFlow


@dataclass
class Mission:
    sections: list[MissionSection]

    @property
    def required_fuel(self) -> float:
        return sum([
            sum([
                abs(flow.mass_flow) * section.duration
                for flow in section.fuel_flows
            ])
            for section in self.sections
        ])
    

class MissionFactory:

    def create_mission_from_list(self, mission_sections: list[dict]) -> Mission:
        return Mission([
        MissionSection(
            fuel_flows=[OutFlow(**flow) for flow in section.pop("fuel_flows")],
            **section
        )
        for section in mission_sections
    ])

    def create_mission_from_file(self, file_name: str, fuel_flow_phase: str) -> Mission:
        mission_data = self._load_mission_from_yaml(
            self._define_file_path(file_name)
        )
        updated_sections = self._update_fuel_flow_phase(
            fuel_flow_phase, mission_data
        )

        return self.create_mission_from_list(updated_sections)

    def _update_fuel_flow_phase(self, fuel_phase_flow, mission_data):
        updated_sections = [
            {
                **section,
                "fuel_flows": [
                    {**flow, "phase": fuel_phase_flow}
                    for flow in section["fuel_flows"]
                ]
            }
            for section in mission_data["mission"]
        ]
        
        return updated_sections

    def _define_file_path(self, mission: str) -> str:
        file_name = f"{mission}.YAML"
        dir_path = os.path.dirname(__file__)
        return os.path.join(dir_path, "reference_missions", file_name)

    def _load_mission_from_yaml(self, path: str) -> Mission:
        with open(path, 'r') as file:
            return yaml.safe_load(file)

def main():
    pass


if __name__ == "__main__":
    main()


# End
