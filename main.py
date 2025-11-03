


from analysis_deprecated.compare_dynamic_models import perform_analysis as compare_dynamic_models
from analysis_deprecated.compare_thermal_models import perform_analysis as compare_thermal_models
from analysis_deprecated.recreate_ahluwalia_fill_analysis import perform_analysis as fill_analysis
from analysis_deprecated.recreate_lin_energy_derivative import perform_analysis as pressure_derivative
from analysis_deprecated.study_gasphase_tank import perform_analysis as geometry_gasphase
from analysis_deprecated.study_missions_gas import perform_analysis as mission_twophase_gas
from analysis_deprecated.study_missions_liquid import perform_analysis as mission_twophase_liquid


def main():
    analysis = [
        mission_twophase_liquid,
        compare_dynamic_models,
        compare_thermal_models,
        fill_analysis,
        pressure_derivative,
        geometry_gasphase,
        # mission_twophase_gas,

    ]
    for study in analysis:
        study()


if __name__ == "__main__":
    main()


# End
