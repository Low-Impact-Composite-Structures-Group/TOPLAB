import pytest

from toplab.materials.insulation_models import ROHACELL, insulation_model_from_config
from toplab.materials.mli_properties import integrated_thermal_conductivity


def test_integral_positive_for_hot_above_cold_and_odd():
    q = integrated_thermal_conductivity(55.0, 300.0)
    assert q > 0.0
    assert integrated_thermal_conductivity(300.0, 55.0) == pytest.approx(-q)
    assert integrated_thermal_conductivity(100.0, 100.0) == 0.0


def test_integral_matches_numeric_integration_of_k_eq():
    A, B = -1.134e-5, 2.616e-10
    n = 20000
    lo, hi = 55.0, 300.0
    dT = (hi - lo) / n
    num = sum((A * T + B * T ** 3) * dT for T in (lo + (i + 0.5) * dT for i in range(n)))
    assert integrated_thermal_conductivity(lo, hi, A, B) == pytest.approx(num, rel=1e-6)


def test_config_selection_and_overrides():
    assert insulation_model_from_config({}) is ROHACELL
    mli = insulation_model_from_config({"model": "mli", "density": 250.0, "specific_heat": 900.0})
    assert mli.density == 250.0 and mli.specific_heat(77.0) == 900.0
    with pytest.raises(ValueError):
        insulation_model_from_config({"model": "foo"})


def test_reproduces_fit_table_heat_leak():
    # Fit row 1: Th=300 K, Tc=55 K, G=323.56 -> Q ~ 11.6 W
    G = 323.5635
    assert G * integrated_thermal_conductivity(55.0, 300.0) == pytest.approx(11.6, rel=0.02)


def test_reproduces_fit_row2_lh2_case():
    # Fit row 2: Th=300 K, Tc=25 K, G=218.83 -> Q ~ 5.0 W
    G = 218.8338
    assert G * integrated_thermal_conductivity(25.0, 300.0) == pytest.approx(5.0, rel=0.02)
