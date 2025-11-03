from .study_geometry_old import analyse_two_phase_tank


def perform_analysis(directory: str):
    analyse_two_phase_tank(directory, fuel_flow_state="gas")


def main():
    pass


if __name__ == "__main__":
    main()


# End
