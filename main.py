


from analysis.compare_dynamic_models import perform_analysis as compare_dynamic_models
from analysis.compare_thermal_models import perform_analysis as compare_thermal_models
from analysis.recreate_ahluwalia_fill_analysis import perform_analysis as fill_analysis
from analysis.recreate_lin_energy_derivative import perform_analysis as pressure_derivative
from analysis.recreate_lin_pressure_rise import perform_analysis as lin_pressure_rise
from analysis.study_gasphase_tank import perform_analysis as mission_gasphase
from analysis.study_missions_gas import perform_analysis as mission_twophase_gas
from analysis.study_missions_liquid import perform_analysis as mission_twophase_liquid


def main():
    analysis = [
        compare_dynamic_models,
        compare_thermal_models,
        fill_analysis,
        pressure_derivative,
        lin_pressure_rise,
        mission_gasphase,
        # mission_twophase_gas,
        # mission_twophase_liquid,
    ]
    for study in analysis:
        study()


if __name__ == "__main__":
    main()


# End
