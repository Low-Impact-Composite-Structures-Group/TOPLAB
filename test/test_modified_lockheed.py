import math
from pathlib import Path

import pytest

from toplab.materials.insulation_models import ROHACELL, insulation_model_from_config
from toplab.materials.modified_lockheed import (
    modified_lockheed_heat_flux,
    modified_lockheed_heat_flux_components,
)


def test_config_selection_and_overrides():
    assert insulation_model_from_config({}) is ROHACELL
    assert insulation_model_from_config({"model": "foam"}) is ROHACELL
    assert insulation_model_from_config({"model": "rohacell"}) is ROHACELL
    assert insulation_model_from_config({"type": "foam"}) is ROHACELL
    with pytest.raises(ValueError, match="not working and is unsupported"):
        insulation_model_from_config({"model": "mli"})
    with pytest.raises(ValueError):
        insulation_model_from_config({"model": "foo"})


def test_cch2_modified_lockheed_study_builds_tank_system():
    from toplab.config import load_config
    from toplab.orchestration.system_orchestrator import SystemOrchestrator

    config_path = (
        Path(__file__).resolve().parents[2]
        / "toplab-apps/analysis/mli_study/cch2/input.yaml"
    )
    orchestrator = SystemOrchestrator(load_config(config_path), verbosity="quiet")

    thermal_model = orchestrator.tank_system.thermal_models[0]
    assert thermal_model.insulation_model.name == "MLI (Modified Lockheed)"
    assert thermal_model.r_shell > thermal_model.r_structure
    assert all(math.isfinite(value) for value in orchestrator.tank_system.create_initial_state())
    results = orchestrator.run_simulation()
    assert results.times[-1] > 0.0


def test_modified_lockheed_zero_temperature_difference_has_zero_components():
    components = modified_lockheed_heat_flux_components(100.0, 100.0, 0.02)
    assert components.q_solid == 0.0
    assert components.q_radiation == 0.0
    assert components.q_gas == 0.0
    assert components.q_total == 0.0


def test_modified_lockheed_thickness_sets_continuous_layer_count_and_reduces_heat_flux():
    thin = modified_lockheed_heat_flux_components(300.0, 20.0, 0.01)
    thick = modified_lockheed_heat_flux_components(300.0, 20.0, 0.02)
    assert thin.number_of_layers == pytest.approx(16.0)
    assert thick.number_of_layers == pytest.approx(32.0)
    assert thick.q_total < thin.q_total


def test_modified_lockheed_components_include_solid_and_radiation_at_low_pressure():
    components = modified_lockheed_heat_flux_components(
        300.0, 20.0, 0.02, pressure_torr=1.0e-12
    )
    assert components.q_solid > 0.0
    assert components.q_radiation > 0.0
    assert components.q_gas < components.q_solid * 1.0e-5
    assert components.q_total == pytest.approx(
        components.q_solid + components.q_radiation + components.q_gas
    )


def test_modified_lockheed_gas_component_scales_linearly_with_pressure():
    pressures = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    gas_fluxes = [
        modified_lockheed_heat_flux_components(
            300.0, 20.0, 0.02, pressure_torr=pressure
        ).q_gas
        for pressure in pressures
    ]
    for pressure, gas_flux in zip(pressures, gas_fluxes):
        assert gas_flux == pytest.approx(gas_fluxes[0] * pressure / pressures[0])


def test_modified_lockheed_degradation_factor_scales_total_heat_flux():
    ideal = modified_lockheed_heat_flux(300.0, 20.0, 0.02)
    degraded = modified_lockheed_heat_flux(
        300.0, 20.0, 0.02, degradation_factor=3.0
    )
    assert degraded == pytest.approx(3.0 * ideal)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"T_hot": 0.0}, "greater than 0 K"),
        ({"T_cold": 301.0}, "greater than or equal"),
        ({"thickness": 0.0}, "thickness"),
        ({"layer_density": 0.0}, "layer_density"),
        ({"pressure_torr": -1.0}, "pressure_torr"),
        ({"degradation_factor": 0.0}, "degradation_factor"),
    ],
)
def test_modified_lockheed_rejects_invalid_inputs(kwargs, message):
    args = {"T_hot": 300.0, "T_cold": 20.0, "thickness": 0.02}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        modified_lockheed_heat_flux(**args)


def test_modified_lockheed_configuration_exposes_correlation_parameters():
    model = insulation_model_from_config(
        {
            "model": "modified_lockheed",
            "layer_density": 20.0,
            "pressure_torr": 2.0e-5,
            "degradation_factor": 2.0,
            "specific_heat": 900.0,
        }
    )
    assert model.name == "MLI (Modified Lockheed)"
    assert model.specific_heat(77.0) == 900.0
    assert model.heat_flux is not None
    expected = modified_lockheed_heat_flux(
        300.0,
        20.0,
        0.02,
        layer_density=20.0,
        pressure_torr=2.0e-5,
        degradation_factor=2.0,
    )
    assert model.heat_flux(300.0, 20.0, 0.02) == pytest.approx(expected)


def test_cch2_modified_lockheed_reports_ihex_energy_at_pressure_hold():
    import numpy as np
    from pathlib import Path
    from toplab.config import load_config
    from toplab.orchestration.system_orchestrator import SystemOrchestrator

    path = Path(__file__).parent.parent / "examples/single_tank_cch2/single_tank_cch2_config.yaml"
    text = path.read_text().replace("      insulation:\n", "      insulation:\n        model: modified_lockheed\n", 1)
    assert "modified_lockheed" in text
    tmp = path.with_name("_tmp_test_ihex.yaml")  # relative paths resolve next to the YAML
    tmp.write_text(text)
    try:
        orchestrator = SystemOrchestrator(load_config(tmp), verbosity="quiet")
    finally:
        tmp.unlink()
    result = orchestrator.run_simulation()
    history = orchestrator.tank_system.discharge_heat_history
    assert history.max() > 0
    # assert np.trapezoid(history[0], np.asarray(result.times)) > 0


def test_emissivity_keys_disambiguate_shell_and_mli():
    from toplab.materials import modified_lockheed
    from toplab.materials.insulation_models import insulation_model_from_config

    def flux(cfg):
        model = insulation_model_from_config({"model": "modified_lockheed", **cfg})
        return model.heat_flux(80.0, 20.0, 0.01)

    default = flux({})
    assert flux({"emissivity": 0.5}) == default  # legacy shell key, ignored by the MLI
    assert flux({"emissivity": 0.5, "emissivity_shell": 0.05}) != default
    assert flux({"emissivity_mli": 0.5}) == flux({"emissivity": 0.5, "emissivity_shell": 0.05})
    assert modified_lockheed.DEFAULT_EMISSIVITY != 0.5


def test_shell_keys_new_and_legacy_names_build_identical_systems():
    from pathlib import Path
    from toplab.config import load_config
    from toplab.orchestration.system_orchestrator import SystemOrchestrator

    path = Path(__file__).parent.parent / "examples/single_tank_cch2/single_tank_cch2_config.yaml"
    legacy = path.read_text()
    assert "alpha_amb:" in legacy
    renamed = legacy.replace("alpha_amb:", "alpha_shell:").replace("emissivity:", "emissivity_shell:")

    def shell_props(text, name):
        tmp = path.with_name(name)  # relative paths resolve next to the YAML
        tmp.write_text(text)
        try:
            model = SystemOrchestrator(load_config(tmp), verbosity="quiet").tank_system.thermal_models[0]
        finally:
            tmp.unlink()
        return model.alpha_amb, model.eps_shell

    assert shell_props(legacy, "legacy.yaml") == shell_props(renamed, "renamed.yaml")
