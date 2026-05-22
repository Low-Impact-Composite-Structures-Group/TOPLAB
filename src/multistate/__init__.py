"""
Multi-Tank System Analysis Framework

A modular framework for analyzing coupled tank systems with inter-tank mass transfer.
Supports various coupling mechanisms and tank configurations with unified physics modeling.

Author: Dante Raso
"""

__all__: list[str] = []

try:
    from .coupling.inter_tank_coupling import InterTankCoupling, PressureTriggeredValve
    from .peripheral_components import Compressor, CryoPumpModel, IdealHeatExchanger
    from .system.tank_system import TankSystem, TankSystemConfig, TankConfig
    from .system.state_management import MultiTankState, MultiTankResults
    from .utilities.tank_geometry import create_tank_from_fuel_mass, create_tank_from_mission

    __all__ = [
        'InterTankCoupling',
        'PressureTriggeredValve',
        'Compressor',
        'CryoPumpModel',
        'IdealHeatExchanger',
        'TankSystem',
        'TankSystemConfig',
        'TankConfig',
        'MultiTankState',
        'MultiTankResults',
        'create_tank_from_fuel_mass',
        'create_tank_from_mission'
    ]
except ModuleNotFoundError:
    # Keep package importable when optional scientific dependencies are absent.
    # Subpackages like src.multistate.packaging can still be used standalone.
    pass