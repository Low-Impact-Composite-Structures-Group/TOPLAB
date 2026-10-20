"""Inspect the NASA Modified Lockheed MLI correlation over cryogenic ranges."""

from pathlib import Path

import matplotlib.pyplot as plt

from toplab.materials.modified_lockheed import (
    DEFAULT_LAYER_DENSITY,
    DEFAULT_PRESSURE_TORR,
    modified_lockheed_heat_flux_components,
)


T_HOT = 300.0
LAYER_DENSITY = DEFAULT_LAYER_DENSITY
PRESSURE_TORR = DEFAULT_PRESSURE_TORR
THICKNESSES_M = (0.005, 0.0075, 0.010, 0.020, 0.030, 0.050)
DEGRADATION_FACTORS = (0.5, 1.0, 2.0, 5.0)
COLD_TEMPERATURES_K = (20.0, 40.0, 60.0, 80.0)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"


def main():
    print("NASA Modified Lockheed MLI heat-flux sweep")
    print(
        f"Th={T_HOT:.1f} K, P={PRESSURE_TORR:g} torr, "
        f"Nstar={LAYER_DENSITY:g} layers/cm, epsilon=0.031, Df=1.0"
    )
    print(
        f"{'Tc [K]':>8} {'t_ins [mm]':>11} {'Ns':>8} "
        f"{'q_solid':>12} {'q_radiation':>14} {'q_gas':>12} {'q_total':>12}"
    )

    figure, axes = plt.subplots()
    for T_cold in COLD_TEMPERATURES_K:
        heat_fluxes = []
        for thickness in THICKNESSES_M:
            components = modified_lockheed_heat_flux_components(
                T_hot=T_HOT,
                T_cold=T_cold,
                thickness=thickness,
                layer_density=LAYER_DENSITY,
                pressure_torr=PRESSURE_TORR,
            )
            print(
                f"{T_cold:8.1f} {thickness * 1000:11.1f} "
                f"{components.number_of_layers:8.2f} "
                f"{components.q_solid:12.6g} "
                f"{components.q_radiation:14.6g} "
                f"{components.q_gas:12.6g} "
                f"{components.q_total:12.6g}"
            )
            heat_fluxes.append(components.q_total)
        axes.plot(
            [thickness * 1000 for thickness in THICKNESSES_M],
            heat_fluxes,
            marker="o",
            label=f"Tc = {T_cold:g} K",
        )

    axes.set_xlabel("Insulation thickness [mm]")
    axes.set_ylabel("Heat flux [W/m²]")
    axes.set_title("Modified Lockheed heat flux vs. MLI thickness")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    thickness_figure_path = OUTPUT_DIRECTORY / "modified_lockheed_thickness_sweep.png"
    figure.savefig(thickness_figure_path, dpi=160)

    print("\nResidual-gas pressure sensitivity")
    print("Th=300 K, Tc=20 K, t_ins=20 mm, Nstar=16 layers/cm")
    print(
        f"{'P [torr]':>12} {'q_solid':>12} {'q_radiation':>14} "
        f"{'q_gas':>12} {'q_total':>12}"
    )
    pressures = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    pressure_components = []
    for pressure_torr in pressures:
        components = modified_lockheed_heat_flux_components(
            T_hot=300.0,
            T_cold=20.0,
            thickness=0.020,
            layer_density=16.0,
            pressure_torr=pressure_torr,
        )
        print(
            f"{pressure_torr:12.1e} {components.q_solid:12.6g} "
            f"{components.q_radiation:14.6g} {components.q_gas:12.6g} "
            f"{components.q_total:12.6g}"
        )
        pressure_components.append(components)

    pressure_figure, pressure_axes = plt.subplots()
    for name, values in (
        ("Solid conduction", [item.q_solid for item in pressure_components]),
        ("Radiation", [item.q_radiation for item in pressure_components]),
        ("Residual gas", [item.q_gas for item in pressure_components]),
        ("Total", [item.q_total for item in pressure_components]),
    ):
        pressure_axes.plot(pressures, values, marker="o", label=name)
    pressure_axes.set_xscale("log")
    pressure_axes.set_yscale("log")
    pressure_axes.set_xlabel("Interstitial pressure [torr]")
    pressure_axes.set_ylabel("Heat flux [W/m²]")
    pressure_axes.set_title("Modified Lockheed component pressure sensitivity")
    pressure_axes.grid(True, which="both", alpha=0.3)
    pressure_axes.legend()
    pressure_figure.tight_layout()
    pressure_figure_path = OUTPUT_DIRECTORY / "modified_lockheed_pressure_sensitivity.png"
    pressure_figure.savefig(pressure_figure_path, dpi=160)

    print("\nDegradation-factor sensitivity")
    print("Th=300 K, Tc=20 K, t_ins=20 mm, Nstar=16 layers/cm")
    print(
        f"{'Df [-]':>8} {'q_solid':>12} {'q_radiation':>14} "
        f"{'q_gas':>12} {'q_total':>12}"
    )

    degradation_components = []

    for degradation_factor in DEGRADATION_FACTORS:
        components = modified_lockheed_heat_flux_components(
            T_hot=300.0,
            T_cold=20.0,
            thickness=0.020,
            layer_density=16.0,
            pressure_torr=PRESSURE_TORR,
            degradation_factor=degradation_factor,
        )

        print(
            f"{degradation_factor:8.1f} "
            f"{components.q_solid:12.6g} "
            f"{components.q_radiation:14.6g} "
            f"{components.q_gas:12.6g} "
            f"{components.q_total:12.6g}"
        )

        degradation_components.append(components)

    # Plot only AFTER all 10 degradation factors have been evaluated
    degradation_figure, degradation_axes = plt.subplots()

    for name, values in (
        ("Solid conduction", [item.q_solid for item in degradation_components]),
        ("Radiation", [item.q_radiation for item in degradation_components]),
        ("Residual gas", [item.q_gas for item in degradation_components]),
        ("Total", [item.q_total for item in degradation_components]),
    ):
        degradation_axes.plot(
            DEGRADATION_FACTORS,
            values,
            marker="o",
            label=name,
        )

    degradation_axes.set_xlabel("Degradation factor [-]")
    degradation_axes.set_ylabel("Heat flux [W/m²]")
    degradation_axes.set_title("Modified Lockheed degradation-factor sensitivity")
    degradation_axes.grid(True, alpha=0.3)
    degradation_axes.legend()
    degradation_figure.tight_layout()

    degradation_figure_path = (
        OUTPUT_DIRECTORY / "modified_lockheed_degradation_sensitivity.png"
    )
    degradation_figure.savefig(degradation_figure_path, dpi=160)

    print(
        f"\nSaved figures:"
        f"\n  {thickness_figure_path}"
        f"\n  {pressure_figure_path}"
        f"\n  {degradation_figure_path}"
    )

    plt.show()

    print(f"\nSaved figures:\n  {thickness_figure_path}\n  {pressure_figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
