from toplab.dynamics.isochoric_dynamic_models import TwoPhaseIsochoricModel


def test_two_phase_configuration_uses_pressure_hysteresis():
    model = TwoPhaseIsochoricModel(p_min=600000.0, p_vent=2000000.0)

    assert model._determine_configuration(570000.0) == "B"
    assert model._determine_configuration(600000.0) == "B"
    assert model._determine_configuration(615000.0) == "B"
    assert model._determine_configuration(640000.0) == "A"