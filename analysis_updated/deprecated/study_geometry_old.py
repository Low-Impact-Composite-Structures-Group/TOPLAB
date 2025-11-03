from typing import Protocol

from examples import save_results

from facades.analysis_facades import GasPhaseTankAnalysisFacade, TwoPhaseTankAnalysisFacade



BODY_LENGTHS = [i / 100 for i in range(0, 601, 25)]
RADII = [i / 100 for i in range(25, 301, 10)]


class Performance(Protocol):
    ...


def analyse_two_phase_tank(
    directory: str,
    fuel_flow_state: str = "liquid"
) -> list[list[Performance]]:

    performances = [
        [
            TwoPhaseTankAnalysisFacade.analyse(
                radius, body_length, fuel_flow_state=fuel_flow_state
            )
            for body_length in BODY_LENGTHS
        ]
        for radius in RADII
    ]

    save_results.save_results(RADII, BODY_LENGTHS, performances, directory)

    return performances

def analyse_gas_phase_tank(
    directory: str,
) -> list[list[Performance]]:

    performances = [
        [
            GasPhaseTankAnalysisFacade.analyse(radius, body_length)
            for body_length in BODY_LENGTHS
        ]
        for radius in RADII
    ]

    save_results.save_results(RADII, BODY_LENGTHS, performances, directory)

    return performances


def main():
    pass


if __name__ == "__main__":
    main()


# End
