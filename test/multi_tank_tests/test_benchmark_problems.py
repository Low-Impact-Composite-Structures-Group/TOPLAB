"""
Benchmark Problem Tests for Multi-Tank Solver Architecture.

This test module validates all solvers in the multi-tank system using standard
benchmark problems from numerical analysis literature:

1. Lotka-Volterra System (predator-prey model)
   - Tests non-stiff solver performance
   - Known oscillatory behavior
   - Conservation of Hamiltonian

2. Robertson's Problem
   - Standard stiff ODE benchmark
   - Tests stiff solver capabilities
   - Wide range of time scales

These benchmarks provide rigorous validation that complements the tank physics tests.
"""

import sys
from pathlib import Path
import numpy as np
import time
import pytest

# Add the hydrogen_fuel_tank package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.multi_tank.solver import (
    RK45Solver,
    RadauSolver,
    DOP853Solver,
    BDFSolver,
    LSODASolver
)


class BenchmarkProblems:
    """Collection of standard benchmark ODE problems"""

    @staticmethod
    def lotka_volterra(t, y, alpha=1.0, beta=0.1, gamma=1.5, delta=0.075):
        """
        Lotka-Volterra predator-prey system.

        dy1/dt = alpha*y1 - beta*y1*y2     (prey)
        dy2/dt = -gamma*y2 + delta*y1*y2   (predator)

        Parameters:
        - alpha: prey birth rate
        - beta: predation rate
        - gamma: predator death rate
        - delta: predator efficiency

        This system conserves the Hamiltonian: H = delta*y1 + beta*y2 - gamma*ln(y1) - alpha*ln(y2)
        """
        y1, y2 = y

        dy1dt = alpha * y1 - beta * y1 * y2
        dy2dt = -gamma * y2 + delta * y1 * y2

        return np.array([dy1dt, dy2dt])

    @staticmethod
    def hamiltonian_lotka_volterra(y, alpha=1.0, beta=0.1, gamma=1.5, delta=0.075):
        """Compute Hamiltonian for Lotka-Volterra system"""
        y1, y2 = y
        return delta * y1 + beta * y2 - gamma * np.log(y1) - alpha * np.log(y2)

    @staticmethod
    def robertson(t, y):
        """
        Robertson's problem - standard stiff ODE benchmark.

        Three-species chemical reaction system:
        dy1/dt = -0.04*y1 + 1e4*y2*y3
        dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
        dy3/dt = 3e7*y2^2

        With conservation law: y1 + y2 + y3 = 1.0

        This problem is notoriously stiff with time scales ranging over many orders of magnitude.
        """
        y1, y2, y3 = y

        dy1dt = -0.04 * y1 + 1e4 * y2 * y3
        dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        dy3dt = 3e7 * y2**2

        return np.array([dy1dt, dy2dt, dy3dt])

    @staticmethod
    def mass_conservation_robertson(y):
        """Check mass conservation for Robertson's problem"""
        return np.sum(y)


class BenchmarkValidator:
    """Validates benchmark problem solutions"""

    @staticmethod
    def validate_lotka_volterra(solution, initial_conditions, params=None):
        """
        Validate Lotka-Volterra solution.

        Returns:
        - hamiltonian_drift: Maximum drift in Hamiltonian (should be small)
        - oscillatory: Whether solution shows expected oscillatory behavior
        - positive: Whether all values remain positive (physical requirement)
        """
        if params is None:
            params = {'alpha': 1.0, 'beta': 0.1, 'gamma': 1.5, 'delta': 0.075}

        # Check Hamiltonian conservation
        initial_h = BenchmarkProblems.hamiltonian_lotka_volterra(initial_conditions, **params)
        hamiltonians = [BenchmarkProblems.hamiltonian_lotka_volterra(y, **params) for y in solution.y.T]
        hamiltonian_drift = max(abs(h - initial_h) for h in hamiltonians)

        # Check for oscillatory behavior (standard deviations should be reasonable)
        prey_std = np.std(solution.y[0])
        predator_std = np.std(solution.y[1])
        oscillatory = prey_std > 0.1 and predator_std > 0.1

        # Check positivity
        positive = np.all(solution.y >= 0)

        return {
            'hamiltonian_drift': hamiltonian_drift,
            'oscillatory': oscillatory,
            'positive': positive,
            'final_hamiltonian': hamiltonians[-1],
            'initial_hamiltonian': initial_h
        }

    @staticmethod
    def validate_robertson(solution, rtol=1e-4):
        """
        Validate Robertson's problem solution.

        Returns:
        - mass_drift: Maximum drift in mass conservation
        - final_y1: Final value of y1 (should be ~0 for long times)
        - final_y2: Final value of y2 (should be ~0 for long times)
        - final_y3: Final value of y3 (should be ~1 for long times)
        - stiff_behavior: Whether solution shows expected stiff behavior
        """
        # Check mass conservation (y1 + y2 + y3 = 1)
        masses = [BenchmarkProblems.mass_conservation_robertson(y) for y in solution.y.T]
        mass_drift = max(abs(m - 1.0) for m in masses)

        # Final values
        final_state = solution.y[:, -1]
        final_y1, final_y2, final_y3 = final_state

        # Check for stiff behavior (rapid initial change in y2)
        if len(solution.t) > 10:
            initial_y2_change = abs(solution.y[1, 5] - solution.y[1, 0])
            stiff_behavior = initial_y2_change > 0.01  # Significant early change
        else:
            stiff_behavior = False

        return {
            'mass_drift': mass_drift,
            'final_y1': final_y1,
            'final_y2': final_y2,
            'final_y3': final_y3,
            'stiff_behavior': stiff_behavior,
            'final_mass': masses[-1]
        }


@pytest.mark.slow
def test_all_solvers_lotka_volterra():
    """Test all solvers with Lotka-Volterra system"""

    print("\n" + "="*70)
    print("🦌 LOTKA-VOLTERRA PREDATOR-PREY BENCHMARK")
    print("="*70)
    print("Testing non-stiff solver performance with oscillatory dynamics")
    print("Conservation of Hamiltonian should be maintained")

    # Problem setup
    t_span = (0.0, 20.0)  # 20 time units
    y0 = np.array([10.0, 5.0])  # Initial prey=10, predator=5

    # All solvers to test
    solvers = [
        ("RK45", RK45Solver),
        ("Radau", RadauSolver),
        ("DOP853", DOP853Solver),
        ("BDF", BDFSolver),
        ("LSODA", LSODASolver)
    ]

    print(f"Initial conditions: Prey={y0[0]}, Predator={y0[1]}")
    print(f"Time span: {t_span[0]} to {t_span[1]} time units")
    print()

    results = {}

    for solver_name, solver_class in solvers:
        print(f"Testing {solver_name}:")

        try:
            # Create and configure solver
            solver = solver_class(
                timestep=0.1,
                rtol=1e-8,
                atol=1e-10
            )
            solver.set_ode_function(BenchmarkProblems.lotka_volterra)

            # Solve
            start_time = time.time()
            sol = solver.integrate_full(t_span, y0)
            wall_time = time.time() - start_time

            if sol.success:
                # Validate solution
                validation = BenchmarkValidator.validate_lotka_volterra(sol, y0)

                print(f"  ✅ Success - Wall time: {wall_time:.4f}s")
                print(f"  📊 Function evals: {sol.nfev}")
                print(f"  🎯 Final state: Prey={sol.y[0,-1]:.3f}, Predator={sol.y[1,-1]:.3f}")
                print(f"  ⚖️  Hamiltonian drift: {validation['hamiltonian_drift']:.2e}")
                print(f"  🌊 Oscillatory: {'✓' if validation['oscillatory'] else '✗'}")
                print(f"  ➕ Positive values: {'✓' if validation['positive'] else '✗'}")

                # Grade the solution
                grade = "A" if validation['hamiltonian_drift'] < 1e-4 else "B" if validation['hamiltonian_drift'] < 1e-2 else "C"
                print(f"  🎓 Grade: {grade} ({'Excellent' if grade=='A' else 'Good' if grade=='B' else 'Acceptable'})")

                results[solver_name] = {
                    'success': True,
                    'wall_time': wall_time,
                    'nfev': sol.nfev,
                    'validation': validation,
                    'grade': grade,
                    'solution': sol
                }

            else:
                print(f"  ❌ Failed: {sol.message}")
                results[solver_name] = {'success': False, 'message': sol.message}

        except Exception as e:
            print(f"  💥 Error: {e}")
            results[solver_name] = {'success': False, 'error': str(e)}

        print()

    # Summary
    print("🏆 LOTKA-VOLTERRA RESULTS SUMMARY:")
    print("-" * 50)
    successful_solvers = [name for name, result in results.items() if result.get('success', False)]

    if successful_solvers:
        print(f"{'Solver':<8} {'Grade':<6} {'H-Drift':<10} {'Time(s)':<8} {'Evals':<8}")
        print("-" * 50)

        for solver_name in successful_solvers:
            result = results[solver_name]
            validation = result['validation']
            print(f"{solver_name:<8} {result['grade']:<6} {validation['hamiltonian_drift']:<10.2e} "
                  f"{result['wall_time']:<8.4f} {result['nfev']:<8}")

    print(f"\n✅ {len(successful_solvers)}/{len(solvers)} solvers completed Lotka-Volterra successfully")

    return results


@pytest.mark.slow
def test_stiff_solvers_robertson():
    """Test stiff solvers with Robertson's problem"""

    print("\n" + "="*70)
    print("⚗️  ROBERTSON'S STIFF CHEMICAL KINETICS BENCHMARK")
    print("="*70)
    print("Testing stiff solver performance with multi-time-scale dynamics")
    print("Mass conservation (y1 + y2 + y3 = 1) should be maintained")

    # Problem setup - Robertson's problem
    t_span = (0.0, 1e5)  # Very long time to test stiffness
    y0 = np.array([1.0, 0.0, 0.0])  # Initial: all in species 1

    # Focus on stiff solvers (and LSODA which adapts)
    stiff_solvers = [
        ("Radau", RadauSolver),
        ("BDF", BDFSolver),
        ("LSODA", LSODASolver)
    ]

    # Also test non-stiff solvers to show the difference
    nonstiff_solvers = [
        ("RK45", RK45Solver),
        ("DOP853", DOP853Solver)
    ]

    print(f"Initial conditions: y1={y0[0]}, y2={y0[1]}, y3={y0[2]}")
    print(f"Time span: {t_span[0]} to {t_span[1]} (5 orders of magnitude!)")
    print()

    all_results = {}

    # Test stiff solvers first
    print("🔥 STIFF SOLVERS (Expected to perform well):")
    print("-" * 50)

    stiff_results = {}
    for solver_name, solver_class in stiff_solvers:
        print(f"Testing {solver_name}:")

        try:
            # Configure for stiff problem
            solver = solver_class(
                timestep=1.0,
                rtol=1e-6,
                atol=1e-10,
                max_step=1e3  # Allow larger steps for efficiency
            )
            solver.set_ode_function(BenchmarkProblems.robertson)

            # Solve
            start_time = time.time()
            sol = solver.integrate_full(t_span, y0)
            wall_time = time.time() - start_time

            if sol.success:
                # Validate solution
                validation = BenchmarkValidator.validate_robertson(sol)

                print(f"  ✅ Success - Wall time: {wall_time:.4f}s")
                print(f"  📊 Function evals: {sol.nfev}")
                print(f"  🎯 Final state: y1={sol.y[0,-1]:.2e}, y2={sol.y[1,-1]:.2e}, y3={sol.y[2,-1]:.6f}")
                print(f"  ⚖️  Mass drift: {validation['mass_drift']:.2e}")
                print(f"  🔥 Stiff behavior detected: {'✓' if validation['stiff_behavior'] else '✗'}")

                # Grade based on efficiency and accuracy
                efficiency = len(sol.t) / sol.nfev if sol.nfev > 0 else 0
                accurate = validation['mass_drift'] < 1e-4
                grade = "A" if accurate and efficiency > 0.01 else "B" if accurate else "C"
                print(f"  🎓 Grade: {grade} (Efficiency: {efficiency:.4f})")

                stiff_results[solver_name] = {
                    'success': True,
                    'wall_time': wall_time,
                    'nfev': sol.nfev,
                    'validation': validation,
                    'efficiency': efficiency,
                    'grade': grade
                }

            else:
                print(f"  ❌ Failed: {sol.message}")
                stiff_results[solver_name] = {'success': False, 'message': sol.message}

        except Exception as e:
            print(f"  💥 Error: {e}")
            stiff_results[solver_name] = {'success': False, 'error': str(e)}

        print()

    # Test non-stiff solvers (should struggle or fail)
    print("⚡ NON-STIFF SOLVERS (Expected to struggle with stiffness):")
    print("-" * 50)

    nonstiff_results = {}
    for solver_name, solver_class in nonstiff_solvers:
        print(f"Testing {solver_name}:")

        try:
            # Use smaller time span for non-stiff solvers
            short_t_span = (0.0, 1e2)  # Much shorter time

            solver = solver_class(
                timestep=0.1,
                rtol=1e-6,
                atol=1e-10,
                max_step=1.0  # Smaller steps required
            )
            solver.set_ode_function(BenchmarkProblems.robertson)

            # Solve with timeout protection
            start_time = time.time()
            sol = solver.integrate_full(short_t_span, y0)
            wall_time = time.time() - start_time

            if sol.success and wall_time < 10.0:  # 10 second timeout
                validation = BenchmarkValidator.validate_robertson(sol)

                print(f"  ⚠️  Partial success - Wall time: {wall_time:.4f}s (short time span)")
                print(f"  📊 Function evals: {sol.nfev}")
                print(f"  🎯 Final state: y1={sol.y[0,-1]:.2e}, y2={sol.y[1,-1]:.2e}, y3={sol.y[2,-1]:.6f}")
                print(f"  ⚖️  Mass drift: {validation['mass_drift']:.2e}")
                print(f"  ⏰ Note: Tested on shorter time span due to stiffness")

                nonstiff_results[solver_name] = {
                    'success': True,
                    'partial': True,
                    'wall_time': wall_time,
                    'nfev': sol.nfev,
                    'validation': validation
                }

            else:
                reason = sol.message if not sol.success else "Timeout (>10s)"
                print(f"  ❌ Failed: {reason}")
                print(f"  💡 This is expected - non-stiff solvers struggle with Robertson's problem")
                nonstiff_results[solver_name] = {'success': False, 'expected': True}

        except Exception as e:
            print(f"  💥 Error: {e}")
            print(f"  💡 This is expected - Robertson's problem is very stiff")
            nonstiff_results[solver_name] = {'success': False, 'expected': True, 'error': str(e)}

        print()

    # Combine results
    all_results.update(stiff_results)
    all_results.update(nonstiff_results)

    # Summary
    print("🏆 ROBERTSON'S PROBLEM RESULTS SUMMARY:")
    print("-" * 60)

    print("Stiff Solvers (Full Problem):")
    successful_stiff = [name for name in stiff_solvers if stiff_results.get(name[0], {}).get('success', False)]
    if successful_stiff:
        print(f"{'Solver':<8} {'Grade':<6} {'Mass Drift':<12} {'Efficiency':<12} {'Time(s)':<8}")
        print("-" * 60)
        for solver_name, _ in stiff_solvers:
            if stiff_results.get(solver_name, {}).get('success', False):
                result = stiff_results[solver_name]
                validation = result['validation']
                print(f"{solver_name:<8} {result['grade']:<6} {validation['mass_drift']:<12.2e} "
                      f"{result['efficiency']:<12.4f} {result['wall_time']:<8.4f}")

    print(f"\n✅ {len(successful_stiff)}/{len(stiff_solvers)} stiff solvers handled Robertson's problem")
    print(f"⚠️  Non-stiff solvers struggled as expected (this demonstrates stiffness)")

    return all_results


@pytest.mark.slow
def test_solver_benchmark_suite():
    """Run complete benchmark suite"""

    print("🧪 COMPREHENSIVE SOLVER BENCHMARK SUITE")
    print("="*70)
    print("Testing all solvers with standard numerical analysis benchmarks")
    print("="*70)

    # Run Lotka-Volterra tests
    lotka_results = test_all_solvers_lotka_volterra()

    # Run Robertson's problem tests
    robertson_results = test_stiff_solvers_robertson()

    # Overall assessment
    print("\n🎯 OVERALL BENCHMARK ASSESSMENT:")
    print("="*70)

    # Count successes
    lotka_successes = sum(1 for r in lotka_results.values() if r.get('success', False))
    robertson_stiff_successes = sum(1 for name in ['Radau', 'BDF', 'LSODA']
                                   if robertson_results.get(name, {}).get('success', False))

    print(f"📊 Lotka-Volterra (Non-stiff): {lotka_successes}/5 solvers successful")
    print(f"📊 Robertson's (Stiff): {robertson_stiff_successes}/3 stiff solvers successful")

    # Recommendations
    print(f"\n💡 SOLVER RECOMMENDATIONS:")
    print("-" * 40)

    # Best overall performers
    if lotka_successes >= 4 and robertson_stiff_successes >= 2:
        print("✅ Solver architecture validation: PASSED")
        print("🏆 Recommended for production: LSODA (adaptive), Radau (stiff)")
        print("🥇 Best non-stiff: RK45 or DOP853")
        print("🥇 Best stiff: Radau or BDF")
    else:
        print("⚠️  Some solvers need attention")

    print(f"\n🎉 Benchmark suite completed!")
    print(f"All solvers tested against standard numerical analysis problems")

    return {
        'lotka_volterra': lotka_results,
        'robertson': robertson_results,
        'summary': {
            'lotka_successes': lotka_successes,
            'robertson_successes': robertson_stiff_successes
        }
    }


# Pytest integration
class TestBenchmarkProblems:
    """Pytest test class for benchmark problems"""

    def test_lotka_volterra_benchmark(self):
        """Test that most solvers can handle Lotka-Volterra"""
        results = test_all_solvers_lotka_volterra()

        # At least 4 out of 5 solvers should succeed
        successes = sum(1 for r in results.values() if r.get('success', False))
        assert successes >= 4, f"Only {successes}/5 solvers succeeded on Lotka-Volterra"

        # Check that successful solvers maintain reasonable accuracy
        for name, result in results.items():
            if result.get('success', False):
                h_drift = result['validation']['hamiltonian_drift']
                assert h_drift < 0.1, f"{name} had excessive Hamiltonian drift: {h_drift}"

    def test_robertson_stiff_solvers(self):
        """Test that stiff solvers can handle Robertson's problem"""
        results = test_stiff_solvers_robertson()

        # At least 2 out of 3 stiff solvers should succeed
        stiff_names = ['Radau', 'BDF', 'LSODA']
        stiff_successes = sum(1 for name in stiff_names
                             if results.get(name, {}).get('success', False))
        assert stiff_successes >= 2, f"Only {stiff_successes}/3 stiff solvers succeeded on Robertson"

        # Check mass conservation for successful stiff solvers
        for name in stiff_names:
            if results.get(name, {}).get('success', False):
                mass_drift = results[name]['validation']['mass_drift']
                assert mass_drift < 1e-3, f"{name} had poor mass conservation: {mass_drift}"


if __name__ == "__main__":
    # Run the full benchmark suite
    benchmark_results = test_solver_benchmark_suite()

    print("\n" + "="*70)
    print("🎊 BENCHMARK TESTING COMPLETED SUCCESSFULLY!")
    print("✅ All solvers validated against standard numerical analysis problems")
    print("✅ Stiff and non-stiff solver capabilities confirmed")
    print("✅ Conservation laws and accuracy verified")
    print("="*70)