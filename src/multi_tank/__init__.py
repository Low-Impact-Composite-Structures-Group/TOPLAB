"""
Multi-Tank System Analysis Framework

A modular framework for analyzing coupled tank systems with inter-tank mass transfer.
Supports various coupling mechanisms and tank configurations with unified physics modeling.
"""

from .coupling.inter_tank_coupling import InterTankCoupling, PressureTriggeredValve
from .system.tank_system import TankSystem, TankSystemConfig, TankConfig
from .system.state_management import MultiTankState, MultiTankResults
from .utilities.tank_geometry import create_tank_from_fuel_mass, create_tank_from_mission

__all__ = [
    'InterTankCoupling',
    'PressureTriggeredValve',
    'TankSystem',
    'TankSystemConfig',
    'TankConfig',
    'MultiTankState',
    'MultiTankResults',
    'create_tank_from_fuel_mass',
    'create_tank_from_mission'
]