"""
Orchestration system for multi-tank hydrogen analysis.

This package provides orchestration classes that expose and coordinate
the core physics components from src/ modules while maintaining clean
configuration-driven interfaces.
"""

from .system_orchestrator import SystemOrchestrator

__all__ = ['SystemOrchestrator']