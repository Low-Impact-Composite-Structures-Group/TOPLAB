#!/usr/bin/env python3
"""
Multi-Tank Framework Test Runner

Comprehensive test runner for the orchestrated multi-tank framework.
Uses pytest for test discovery and execution with coverage reporting.

Usage:
    # Run all tests
    python test/multi_tank_tests/run_tests.py

    # Run with specific options
    python test/multi_tank_tests/run_tests.py --verbose
    python test/multi_tank_tests/run_tests.py --coverage
    python test/multi_tank_tests/run_tests.py --fast
    python test/multi_tank_tests/run_tests.py --include-slow  # Include solver benchmarks

    # Run specific test module
    python test/multi_tank_tests/run_tests.py --module nist_materials

Features:
- Automatic environment detection (micromamba/conda)
- Coverage reporting
- Parallel test execution
- Continuous integration support
- Test result summary
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def detect_python_environment():
    """Detect the appropriate Python environment command."""
    # Check if we're in a micromamba environment
    try:
        result = subprocess.run(['micromamba', 'info'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return ['micromamba', 'run', '-n', 'python-h2-dev', 'python']
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check if we're in a conda environment
    try:
        result = subprocess.run(['conda', 'info'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return ['conda', 'run', '-n', 'python-h2-dev', 'python']
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fall back to system python
    return ['python']


def install_pytest_if_needed(python_cmd):
    """Install pytest if not available."""
    print("🔍 Checking pytest installation...")

    # Check if pytest is available
    test_cmd = python_cmd + ['-c', 'import pytest; print(pytest.__version__)']

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ pytest {result.stdout.strip()} found")
            return True
    except subprocess.TimeoutExpired:
        pass

    # Install pytest
    print("📦 Installing pytest...")
    install_cmd = python_cmd[:-1] + ['pip', 'install', 'pytest', 'pytest-cov', 'pytest-xdist']

    try:
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ pytest installed successfully")
            return True
        else:
            print(f"❌ Failed to install pytest: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ pytest installation timed out")
        return False


def run_tests(args):
    """Run the test suite with specified options."""
    project_root = get_project_root()
    test_dir = project_root / "test" / "multi_tank_tests"

    # Detect Python environment
    python_cmd = detect_python_environment()
    print(f"🐍 Using Python command: {' '.join(python_cmd)}")

    # Install pytest if needed
    if not install_pytest_if_needed(python_cmd):
        print("❌ Cannot proceed without pytest")
        return False

    # Build pytest command
    pytest_cmd = python_cmd + ['-m', 'pytest']

    # Add test directory
    if args.module:
        pytest_cmd.append(str(test_dir / f"test_{args.module}.py"))
    else:
        pytest_cmd.append(str(test_dir))

    # Add options
    if args.verbose:
        pytest_cmd.append('-v')

    if args.coverage:
        pytest_cmd.extend([
            '--cov=src/materials/materials_for_multi_tank',
            '--cov=src/configuration',
            '--cov=src/orchestration',
            '--cov-report=term-missing',
            '--cov-report=html:test/coverage_html'
        ])

    if args.fast:
        pytest_cmd.extend(['-x', '--tb=short'])  # Stop on first failure, short traceback

    # Skip slow tests by default (unless explicitly included)
    if not args.include_slow:
        pytest_cmd.extend(['-m', 'not slow'])

    # Skip parallel execution for now (requires pytest-xdist)

    # Add continuous integration options
    if args.ci:
        pytest_cmd.extend([
            '--junitxml=test/results.xml',
            '--tb=short',
            '-q'
        ])

    print(f"🚀 Running tests...")
    print(f"Command: {' '.join(pytest_cmd)}")
    print("=" * 80)

    # Change to project root for imports
    original_cwd = Path.cwd()

    try:
        # Run tests
        start_time = time.time()
        result = subprocess.run(pytest_cmd, cwd=project_root)
        end_time = time.time()

        # Print summary
        print("=" * 80)
        if result.returncode == 0:
            print(f"✅ All tests passed in {end_time - start_time:.2f} seconds")
            if args.coverage:
                print(f"📊 Coverage report generated: test/coverage_html/index.html")
        else:
            print(f"❌ Tests failed (exit code: {result.returncode})")

        return result.returncode == 0

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return False

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

    finally:
        # Restore original directory
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Tank Framework Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/multi_tank_tests/run_tests.py                    # Run all tests
  python test/multi_tank_tests/run_tests.py --verbose          # Verbose output
  python test/multi_tank_tests/run_tests.py --coverage         # With coverage
  python test/multi_tank_tests/run_tests.py --module nist_materials  # Specific module
  python test/multi_tank_tests/run_tests.py --fast             # Fast mode (stop on first failure)
  python test/multi_tank_tests/run_tests.py --include-slow     # Include solver benchmarks
  python test/multi_tank_tests/run_tests.py --ci               # CI mode
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--fast', '-f', action='store_true',
                       help='Fast mode (stop on first failure)')
    parser.add_argument('--module', '-m', type=str,
                       help='Run specific test module (e.g., nist_materials)')
    parser.add_argument('--ci', action='store_true',
                       help='Continuous integration mode')
    parser.add_argument('--include-slow', action='store_true',
                       help='Include slow tests (solver benchmarks)')

    args = parser.parse_args()

    print("🧪 MULTI-TANK FRAMEWORK TEST RUNNER")
    print("=" * 80)
    print("Orchestrated multi-tank system testing with pytest")
    print("=" * 80)

    success = run_tests(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()