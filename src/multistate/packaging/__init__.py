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
from .visualization import plot_layout

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
    "plot_layout",
]
