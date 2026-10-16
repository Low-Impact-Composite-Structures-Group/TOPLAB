"""iHEX heat is recorded per tank at accepted points, aligned with the result times."""

from pathlib import Path

import numpy as np

from toplab.config import load_config
from toplab.orchestration.system_orchestrator import SystemOrchestrator
from toplab.post.hex import calculate_ihex_requirements

CONFIG = Path(__file__).parent.parent / "examples/single_tank_cch2/single_tank_cch2_config.yaml"


def test_ihex_history_aligns_with_results_and_run_result_works():
    orchestrator = SystemOrchestrator(load_config(CONFIG), verbosity="quiet")
    results = orchestrator.run_simulation()
    history = orchestrator.tank_system.discharge_heat_history
    assert history.shape == (1, len(results.times))

    run = orchestrator.run_result()
    qdot = np.asarray(calculate_ihex_requirements(run, 0))
    assert qdot.shape == (len(results.times),)
    assert np.allclose(qdot, history[0])
    assert qdot.max() > 0.0
