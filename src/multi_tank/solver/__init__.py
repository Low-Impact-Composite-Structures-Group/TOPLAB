"""
Multi-tank system solvers module.

This module provides ODE solvers specifically designed for hydrogen tank systems.
All solvers are based on SciPy's solve_ivp function with optimized configurations
for tank dynamics problems.
"""

from .scipy_solvers import (
    SciPySolver,
    RK45Solver,
    RadauSolver,
    DOP853Solver,
    BDFSolver,
    LSODASolver,
    ScipyMethod  # Backward compatibility
)

__all__ = [
    'SciPySolver',
    'RK45Solver',
    'RadauSolver',
    'DOP853Solver',
    'BDFSolver',
    'LSODASolver',
    'ScipyMethod'
]