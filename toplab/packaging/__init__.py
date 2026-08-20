"""Packaging layout tools for placing components inside a constrained volume."""

from .geometry import (
    AxialSegment,
    CompositeAxisymmetricVolume,
    Point3D,
    TruncatedConeVolume,
)
from .primitives import CapsulePrimitive, SpherePrimitive, RegularPrismPrimitive
from .layout import PackagingLayout, PlacedPrimitive, PlacementError
from .routing import GridRouter, RoutingError
from .metrics import (
    components_volume,
    packaged_volume,
    packaging_efficiency,
    piping_volume,
    route_length,
)
from .visualization import plot_layout
from .aft_placement import (
    AftFuselageDimensions,
    AftPlacementResult,
    TankPlacement,
    allowed_radius_at,
    place_tanks_in_aft,
    plot_aft_placement,
)

__all__ = [
    "Point3D",
    "TruncatedConeVolume",
    "AxialSegment",
    "CompositeAxisymmetricVolume",
    "CapsulePrimitive",
    "SpherePrimitive",
    "RegularPrismPrimitive",
    "PackagingLayout",
    "PlacedPrimitive",
    "PlacementError",
    "GridRouter",
    "RoutingError",
    "route_length",
    "components_volume",
    "piping_volume",
    "packaged_volume",
    "packaging_efficiency",
    "plot_layout",
    "AftFuselageDimensions",
    "AftPlacementResult",
    "TankPlacement",
    "allowed_radius_at",
    "place_tanks_in_aft",
    "plot_aft_placement",
]
