
import numpy as np
import matplotlib.pyplot as plt
from typing import Protocol


class TankStates(Protocol):
    volume: float
    gravimetric_efficiency: float
    volumetric_efficiency: float

def plot_geometric_study(
    radii: list[float],
    body_lengths: list[float],
    tanks: list[list[TankStates]]
):
    X, Y = np.meshgrid(body_lengths, radii)

    # Extract desired variables
    gravimetric_efficiencies = [
        [tank.gravimetric_efficiency for tank in row]
        for row in tanks
    ]
    volumetric_efficiencies = [
        [tank.volumetric_efficiency for tank in row]
        for row in tanks
    ]
    volumes = [
        [tank.volume for tank in row]
        for row in tanks
    ]

    fig, ax = plt.subplots()
    graf_effs = ax.contourf(X, Y, gravimetric_efficiencies)
    cbar = fig.colorbar(graf_effs)
    cbar.set_label("Gravimetric efficiency [-]")
    voll_effs = ax.contour(
        X, Y, volumetric_efficiencies, 10,
        colors="black", linestyles="dashed"
    )
    ax.clabel(voll_effs, inline=True)
    vol_lines = ax.contour(
        X, Y, volumes, 8,
        colors="orange"
    )
    ax.clabel(vol_lines, inline=True)
    ax.set_xlabel("Tank body length [m]")
    ax.set_ylabel("Tank radius [m]")

    lines = [
        voll_effs.legend_elements()[0][0],
        vol_lines.legend_elements()[0][0]
    ]
    labels = ["Vol. eff. [-]", r"Tank vol. [m$^3$]"]
    # plt.legend(lines, labels, loc="south outside")
    plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True,ncol=5)
    plt.tight_layout()
    plt.show()