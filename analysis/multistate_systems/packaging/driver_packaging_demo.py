#!/usr/bin/env python3
"""Standalone demo for composite-volume packaging and orthogonal routing."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure project root is importable when running directly.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.multistate.packaging import (
    AxialSegment,
    CapsulePrimitive,
    CompositeAxisymmetricVolume,
    GridRouter,
    PackagingLayout,
    Point3D,
    RegularPrismPrimitive,
    SpherePrimitive,
    components_volume,
    packaged_volume,
    packaging_efficiency,
    piping_volume,
    plot_layout,
)


def build_demo_layout() -> tuple[PackagingLayout, list[Point3D], list[Point3D], list[Point3D]]:
    """Create layout and routes for capsule->cube->sphere->prism chain."""
    # Approximate outer mold line from drawing [mm -> m]:
    # 1) Cylinder: L=2067, D=2160
    # 2) Truncated cone: L=1800, D=2160 -> D=1770
    # 3) Truncated cone: L=3407, D=1770 -> D=328
    volume = CompositeAxisymmetricVolume(
        segments=(
            AxialSegment(length=2.067, radius_start=1.08, radius_end=1.08),
            AxialSegment(length=1.800, radius_start=1.08, radius_end=0.885),
            AxialSegment(length=3.407, radius_start=0.885, radius_end=0.164),
        )
    )
    layout = PackagingLayout(volume)

    layout.add_component(
        component_id="CH2_tank",
        primitive=SpherePrimitive(radius=0.5, connection_axis="z"),
        center=Point3D(0.0, 0.0, 4.8),
        sampling_step=0.05,
    )
    layout.add_component(
        component_id="HEX",
        primitive=RegularPrismPrimitive(width=0.4, depth=0.4, height=0.4, connection_axis="z"),
        center=Point3D(0.0, 0.0, 3.6),
        sampling_step=0.05,
    )
    layout.add_component(
        component_id="CcCH2_tank",
        primitive=CapsulePrimitive(cylinder_length=1.0, radius=1.0, axis="z"),
        center=Point3D(0.0, 0.0, 1.5),
        sampling_step=0.2,
    )
    layout.add_component(
        component_id="FC_stack",
        primitive=RegularPrismPrimitive(width=0.1, depth=0.2, height=0.1, connection_axis="z"),
        center=Point3D(0.0, 0.0, 5.9),
        sampling_step=0.05,
    )

    tank_to_cube_start = layout.get_connection_point("CcCH2_tank", 1)
    tank_to_cube_end = layout.get_connection_point("HEX", 0)

    cube_to_sphere_start = layout.get_connection_point("HEX", 1)
    cube_to_sphere_end = layout.get_connection_point("CH2_tank", 0)

    sphere_to_prism_start = layout.get_connection_point("CH2_tank", 1)
    sphere_to_prism_end = layout.get_connection_point("FC_stack", 0)

    tank_to_cube_start_dir = layout.get_connection_direction("CcCH2_tank", 1)
    tank_to_cube_end_dir = layout.get_connection_direction("HEX", 0)

    cube_to_sphere_start_dir = layout.get_connection_direction("HEX", 1)
    cube_to_sphere_end_dir = layout.get_connection_direction("CH2_tank", 0)

    sphere_to_prism_start_dir = layout.get_connection_direction("CH2_tank", 1)
    sphere_to_prism_end_dir = layout.get_connection_direction("FC_stack", 0)

    router = GridRouter(grid_step=0.05, turn_penalty=0.12)
    route_tank_to_cube = router.route(
        tank_to_cube_start,
        tank_to_cube_end,
        volume=layout.volume,
        obstacles=[layout.get_component("CH2_tank"), layout.get_component("FC_stack")],
        pipe_radius=0.04,
        start_direction=tank_to_cube_start_dir,
        end_direction=tank_to_cube_end_dir,
        min_straight_length=0.1,
    )
    route_cube_to_sphere = router.route(
        cube_to_sphere_start,
        cube_to_sphere_end,
        volume=layout.volume,
        obstacles=[layout.get_component("CcCH2_tank"), layout.get_component("FC_stack")],
        pipe_radius=0.04,
        start_direction=cube_to_sphere_start_dir,
        end_direction=cube_to_sphere_end_dir,
        min_straight_length=0.1,
    )
    route_sphere_to_prism = router.route(
        sphere_to_prism_start,
        sphere_to_prism_end,
        volume=layout.volume,
        obstacles=[layout.get_component("CcCH2_tank"), layout.get_component("HEX")],
        pipe_radius=0.04,
        start_direction=sphere_to_prism_start_dir,
        end_direction=sphere_to_prism_end_dir,
        min_straight_length=0.1,
    )

    return layout, route_tank_to_cube, route_cube_to_sphere, route_sphere_to_prism


def main() -> None:
    layout, route_1, route_2, route_3 = build_demo_layout()
    routes = [route_1, route_2, route_3]
    pipe_radius = 0.04

    print("Placed component quadrants:")
    for component_id, quadrant in layout.get_component_quadrants().items():
        print(f"  - {component_id}: {quadrant}")

    print(f"Orthogonal route points (CcCH2_tank->HEX): {len(route_1)}")
    print(f"Orthogonal route points (HEX->CH2_tank): {len(route_2)}")
    print(f"Orthogonal route points (CH2_tank->FC_stack): {len(route_3)}")

    vol_components = components_volume(layout)
    vol_piping = piping_volume(routes, pipe_radius=pipe_radius)
    vol_total_packed = packaged_volume(layout, routes, pipe_radius=pipe_radius)
    vol_outer = layout.volume.volume
    eta_packaging = packaging_efficiency(layout, routes, pipe_radius=pipe_radius)

    print("\nPackaging volume summary:")
    print(f"  - Components volume: {vol_components:.6f} m^3")
    print(f"  - Piping volume: {vol_piping:.6f} m^3")
    print(f"  - Packed volume (components + piping): {vol_total_packed:.6f} m^3")
    print(f"  - Outer geometry volume: {vol_outer:.6f} m^3")
    print(f"  - Packaging efficiency: {eta_packaging:.4%}")

    fig, _ = plot_layout(
        layout,
        sample_step=0.12,
        route_point_sets=[route_1, route_2, route_3],
        show_labels=False,
        show_quadrants=True,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
