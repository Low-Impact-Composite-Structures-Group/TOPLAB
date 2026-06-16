"""
Multi-Tank Framework Test Suite

This package contains all tests for the orchestrated multi-tank framework.
Uses pytest for test discovery and execution.

Usage:
    # Run all multi-tank tests
    pytest test/

    # Run specific test module
    pytest test/test_nist_materials.py

    # Run with verbose output
    pytest test/ -v

    # Run with coverage
    pytest test/ --cov=src/materials/materials_for_multi_tank

Structure:
    test_nist_materials.py      - NIST materials framework tests
    test_scenario_config.py     - Configuration parsing tests (future)
    test_system_orchestrator.py - Orchestrator integration tests (future)
    test_integration.py         - End-to-end integration tests (future)
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__author__ = "Multi-Tank Framework"