"""
Compressed gaseous hydrogen (CH2) benchmark analysis using parametric framework.

This module implements the CH2 variant of the parametric benchmark analysis,
using the same mission simulation and optimization framework but with
CH2-specific initial conditions and requirements.

CH2 Characteristics:
- Initial conditions: 700 bar, 331.6 K (high pressure compressed gas)
- Minimum density requirement: 2.4 kg/m³ (compressed gas threshold)
- Much lower density than liquid (~40-50 kg/m³ initially vs ~71 kg/m³ for LH2)
- No phase change concerns - remains gaseous throughout mission
- Very thick composite walls required due to high pressure (structural challenge)
- High ambient HTC (50 W/m²K) - minimal insulation for ambient temperature storage

Expected Result: Larger tank volumes due to lower gas density,
potentially very high structural mass due to thick pressure vessel walls.

Authors: Dante Raso (2025)
Based on parametric_benchmark.py framework with bisection optimization
"""

# Standard library imports
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import base class
from analysis.baseline.parametric_benchmark import (
    ParametricBenchmark,
    find_optimal_radius_for_storage_type
)


class BenchmarkCH2Analysis(ParametricBenchmark):
    """
    Compressed gaseous hydrogen benchmark analysis.

    This class implements the CH2 variant using the parametric benchmark framework.
    All common functionality is inherited from ParametricBenchmark. Only CH2-specific
    parameters are defined here.

    CH2 operates at elevated temperature (331.6 K) with very high pressure (700 bar).
    The challenge is managing structural mass from thick pressure vessel walls while
    maintaining sufficient fuel density for the mission.
    """

    def __init__(self, tank_radius: float = 1.0):
        """
        Initialize CH2 analysis with higher design pressure.

        CH2 requires much higher design pressure capability due to
        operating pressures of 700 bar, with venting at 800 bar.
        """
        super().__init__(tank_radius)

        # Override design pressure for CH2 high-pressure requirements
        self.design_pressure = 900e5  # 900 bar design pressure for 800 bar venting

    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure for CH2."""
        return 700e5  # 700 bar - high pressure gaseous storage

    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature for CH2."""
        return 331.6  # 331.6 K - elevated temperature compressed hydrogen

    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density for CH2."""
        return 2.4  # 2.4 kg/m³ - maintain compressed gas density above standard conditions

    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        return "Compressed H2 (CH2)"

    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get CH2-specific optimization parameters using bisection search."""
        return {
            'min_radius': 1.0,              # Minimum radius (m)
            'max_radius': 3.0,              # Maximum radius (m)
            'radius_precision': 0.005,      # Radius precision: ±5mm
            'density_tolerance': 2.0,       # Density tolerance: ±2.0 kg/m³
            'max_evaluations': 20           # Maximum function evaluations
        }

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure for CH2."""
        return 15e5  # 15 bar - sufficient for compressed gaseous storage

    def get_venting_pressure(self) -> float:
        """Get venting pressure for CH2."""
        return 800e5  # 800 bar - venting threshold for high pressure storage

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient for CH2."""
        return 50.0  # 50.0 W/m²K - high HTC for ambient temperature storage


def quick_test():
    """Quick test to demonstrate bisection optimization improvements."""
    print("Testing new bisection optimization...")
    print("Expected: Much faster than old brute-force approach")


def main():
    """Main execution function for CH2 benchmark analysis."""
    print("Starting Compressed Gaseous Hydrogen (CH2) Benchmark Analysis")
    print("="*80)
    quick_test()

    print("\n" + "="*80)
    print("ROBUST BISECTION SEARCH - COMPRESSED H2 (CH2)")
    print("="*80)

    # Create analysis instance for parameter extraction
    temp_analysis = BenchmarkCH2Analysis()
    params = temp_analysis.get_optimization_parameters()

    print(f"Search range: {params['min_radius']:.3f}m to {params['max_radius']:.3f}m")
    print(f"Precision: ±{params['radius_precision']*1000:.0f}mm, Density tolerance: ±{params['density_tolerance']:.1f} kg/m³")
    print(f"Requirements: No venting AND final density ≥ {temp_analysis.get_minimum_density():.1f} kg/m³")
    print("CoolProp error detection: ENABLED")
    print("-" * 80)

    # Run bisection optimization
    from analysis.baseline.parametric_benchmark import find_optimal_radius_for_storage_type
    optimization_results = find_optimal_radius_for_storage_type(BenchmarkCH2Analysis)
    optimal_radius = optimization_results['optimal_radius']

    if optimal_radius is not None:
        print(f"\n" + "="*80)
        print("FINAL CH2 ANALYSIS WITH OPTIMAL RADIUS")
        print("="*80)

        # Run final analysis with optimal radius
        analysis = BenchmarkCH2Analysis(optimal_radius)
        analysis.run_analysis()

        # Print comprehensive results
        analysis.print_results()

        # Calculate performance metrics
        final_state = analysis.results.states[-1]
        fuel_mass = final_state.fuel_mass
        gravimetric_eff = fuel_mass / (fuel_mass + analysis.total_structural_mass) * 100

        print(f"\nCH2 Performance Summary:")
        print(f"  Optimal radius: {optimal_radius:.3f} m")
        print(f"  Tank volume: {analysis.tank_volume:.3f} m³")
        print(f"  Structural mass: {analysis.total_structural_mass:.1f} kg")
        print(f"  Final fuel mass: {fuel_mass:.1f} kg")
        print(f"  Gravimetric efficiency: {gravimetric_eff:.1f}%")
        print(f"  Final density: {analysis.results.final_density:.1f} kg/m³")

        # Generate plots for optimal configuration
        try:
            print("\nGenerating 5 plots for optimal CH2 configuration...")
            analysis.run_analysis(include_plots=True)

            # Generate optimization progress plot (creates figure but doesn't show yet)
            analysis.plot_optimization_progress_from_data(optimization_results['optimization_progress'],
                                                        optimization_results['minimum_density_target'],
                                                        optimization_results['density_tolerance'])

            # Show all plots at once
            import matplotlib.pyplot as plt
            plt.show()
            print("\nAll 5 plots displayed successfully!")

        except Exception as e:
            print(f"Plotting failed (this is non-critical): {e}")

    else:
        print(f"\n" + "="*80)
        print("CH2 OPTIMIZATION FAILED")
        print("="*80)
        print("No feasible solution found in search range.")
        print("Consider:")
        print("- Increasing max_radius")
        print("- Reducing minimum density requirement")
        print("- Checking initial conditions")


if __name__ == "__main__":
    main()