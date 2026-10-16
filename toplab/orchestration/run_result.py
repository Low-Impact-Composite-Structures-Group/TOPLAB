"""Read-only bundle of everything post-processing and reporting need from a finished run."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from toplab.config import AnalysisConfig


@dataclass
class RunResult:
    config: AnalysisConfig
    network: Any
    results: Any
    designs: List[Any]
    tank_geometries: List[Any]
    couplings: List[Any]
    streams: Dict[int, List[Any]]
    mission_profile: Any
    tank_system: Any
    validation_results: Optional[Dict[str, Any]] = None
