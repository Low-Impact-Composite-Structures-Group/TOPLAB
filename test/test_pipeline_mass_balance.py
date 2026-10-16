"""End-to-end: coupled example runs through the hybrid solver and conserves mass."""

from pathlib import Path

from toplab.config import load_config
from toplab.orchestration.system_orchestrator import SystemOrchestrator
from toplab.post.mass_balance import check_mass_balance

CONFIG = Path(__file__).parent.parent / "examples/coupled_ch2_cch2/coupled_ch2_cch2_config.yaml"


def test_coupled_example_conserves_mass_and_builds_run_result():
    orchestrator = SystemOrchestrator(load_config(CONFIG), verbosity="quiet")
    results = orchestrator.run_simulation()

    checks = check_mass_balance(results)
    assert len(checks) == 2
    assert all(c.passed(1e-3) for c in checks)

    run = orchestrator.run_result()
    assert run.results is results
    assert len(run.designs) == len(run.tank_geometries) == 2
