

import matplotlib.pylab as plt

import plotting.plot_style
from src.fluids.hydrogen_retrievers import TwoPhaseRequester
from src.fluids.energy_derivative_computer import EnergyDerivativeComputer


def perform_analysis():

    computer = EnergyDerivativeComputer()
    requester = TwoPhaseRequester()

    for pressure in [103, 138, 172]:
        hydrogen = requester.get_hydrogen_properties(
            pressure * 1e3, None
        )
        fills = [i / 100 for i in range(7, 100, 1)]
        derivatives = [
            computer.compute_energy_derivative(hydrogen, fill)
            for fill in fills
        ]
        densities = [
            computer.compute_density(hydrogen, fill)
            for fill in fills
        ]
        plt.plot(
            densities, derivatives,
            marker="",
            label=f"{int(pressure)} kPa"
        )

    xticks = [i for i in range(0, 81, 10)]
    yticks = [i / 100 for i in range(4, 25, 4)]
    
    plt.legend()
    plt.ylabel(r"Energy Derivative [kPa$\cdot$m$^3$/kJ]")
    plt.xlabel("Average Density [kg/m$^3$]")
    plt.xlim((xticks[0], xticks[-1]))
    plt.xticks(xticks)
    plt.ylim((yticks[0], yticks[-1]))
    plt.yticks(yticks)
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()