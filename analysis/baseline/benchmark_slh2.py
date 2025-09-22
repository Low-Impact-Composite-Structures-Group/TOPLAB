"""
Subcooled liquid hydrogen (sLH2) benchmark analysis using parametric framework.

This module implements the sLH2 variant of the parametric benchmark analysis,
using the same mission simulation and optimization framework but with
sLH2-specific initial conditions and requirements.

sLH2 Characteristics:
- Initial conditions: ~16 bar, 30 K (subcooled liquid hydrogen)
- Minimum density requirement: 14.0 kg/m³ (higher than regular LH2)
- Higher pressure and slightly higher temperature than regular LH2
- More stable liquid state due to subcooling
- Better performance characteristics but higher storage pressure requirements

Expected Result: Similar or slightly larger tank volumes than LH2 due to higher
temperature, but potentially better mission performance due to subcooled stability.

Authors: Dante Raso (2025)
Based on parametric_benchmark.py framework
"""

# Standard library imports
import sys
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add parent directories for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import base class
from analysis.baseline.parametric_benchmark import (
    ParametricBenchmark,
    find_optimal_radius_for_storage_type
)


class BenchmarkSLH2Analysis(ParametricBenchmark):
    """
    Subcooled liquid hydrogen benchmark analysis.

    This class implements the sLH2 variant using the parametric benchmark framework.
    All common functionality is inherited from ParametricBenchmark. Only sLH2-specific
    parameters are defined here.

    sLH2 operates at subcooled conditions (16 bar, 30 K) providing more stable liquid
    hydrogen storage with better performance characteristics than regular LH2.
    """

    def __init__(self, tank_radius: float = 0.69):
        """
        Initialize sLH2 analysis with modified venting pressure.

        sLH2 uses lower venting pressure (10 bar) similar to LH2, but operates
        at higher initial pressure to maintain subcooled liquid state.
        """
        super().__init__(tank_radius)

    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure for sLH2."""
        return 16e5  # 16 bar - higher pressure for subcooled liquid state

    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature for sLH2."""
        return 28.20  # 30 K - subcooled liquid hydrogen temperature

    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density for sLH2."""
        return 2.4  # 14.0 kg/m³ - maintain subcooled liquid density, higher than regular LH2

    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        return "Subcooled Liquid H2 (sLH2)"

    def get_venting_pressure(self) -> float:
        """Get venting pressure for sLH2."""
        return 20e5  # 10 bar venting pressure for sLH2 (same as LH2)

    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get sLH2-specific optimization parameters."""
        return {
            'min_radius': 0.5,              # Minimum search radius [m]
            'max_radius': 1.5,              # Maximum search radius [m]
            'radius_precision': 0.005,      # Precision: ±5mm
            'density_tolerance': 2.0,       # Within 2 kg/m³ of minimum is acceptable
            'max_evaluations': 20           # Maximum function evaluations
        }

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure for sLH2."""
        return 6e5

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient for sLH2."""
        return 0.005  # 0.015 W/m²K - moderate insulation for subcooled liquid hydrogen


def quick_test(radius=1.0, include_plots=False):
    """Quick test function to run analysis with a specific radius without optimization."""
    print(f"Quick sLH2 Test - Radius: {radius:.3f}m")
    print("="*50)

    analysis = BenchmarkSLH2Analysis(tank_radius=radius)
    success = analysis.run_single_analysis()

    if success:
        analysis.print_results()

        if include_plots:
            try:
                print("\nGenerating plots...")
                analysis.run_analysis(include_plots=True)
            except Exception as e:
                print(f"Plotting failed: {e}")
    else:
        print("Analysis failed")

    return analysis if success else None


def main():
    """Main execution function for sLH2 benchmark analysis."""
    print("Starting Subcooled Liquid Hydrogen (sLH2) Benchmark Analysis")
    print("="*80)

    # Test optimization directly
    print("Testing new bisection optimization...")
    print("Expected: Much faster than old brute-force approach")

    # Run optimal radius search for sLH2 (commented out for quick testing)
    search_results = find_optimal_radius_for_storage_type(BenchmarkSLH2Analysis)

    if search_results['search_successful']:
        print(f"\n" + "="*80)
        print("FINAL sLH2 ANALYSIS WITH OPTIMAL RADIUS")
        print("="*80)

        # Use the optimal configuration
        optimal_data = search_results['optimal_results']
        analysis = optimal_data['analysis']

        # Print comprehensive results
        analysis.print_results()

        # Calculate gravimetric efficiency comparison
        final_state = analysis.results.states[-1]
        fuel_mass = final_state.fuel_mass
        gravimetric_eff = fuel_mass / (fuel_mass + analysis.total_structural_mass) * 100

        print(f"\nsLH2 Performance Summary:")
        print(f"  Optimal radius: {optimal_data['radius']:.3f} m")
        print(f"  Tank volume: {optimal_data['volume']:.3f} m³")
        print(f"  Structural mass: {optimal_data['structural_mass']:.1f} kg")
        print(f"  Final fuel mass: {fuel_mass:.1f} kg")
        print(f"  Gravimetric efficiency: {gravimetric_eff:.1f}%")
        print(f"  Final density: {optimal_data['final_density']:.1f} kg/m³")

        # Generate plots for optimal configuration
        print("\nGenerating 5 plots for optimal sLH2 configuration...")
        analysis.run_analysis(include_plots=True)

        # Generate optimization progress plot (creates figure but doesn't show yet)
        analysis.plot_optimization_progress_from_data(search_results['optimization_progress'],
                                                    search_results['minimum_density_target'],
                                                    search_results['density_tolerance'])

        plt.show()

    else:
        print(f"\n" + "="*80)
        print("sLH2 SEARCH FAILED - ANALYZING BEST ATTEMPT")
        print("="*80)

        if search_results['search_results']:
            # Use the best attempt
            best_result = min(search_results['search_results'], key=lambda x: x['max_pressure'])
            analysis = best_result['analysis']

            print(f"Using radius {best_result['radius']:.3f}m (max pressure: {best_result['max_pressure_bar']:.1f} bar, final density: {best_result['final_density']:.2f} kg/m³)")
            analysis.print_results()

            try:
                print("\nGenerating 5 plots for best sLH2 attempt...")
                analysis.run_analysis(include_plots=True)

                # Generate optimization progress plot (creates figure but doesn't show yet)
                analysis.plot_optimization_progress_from_data(search_results['optimization_progress'],
                                                            search_results['minimum_density_target'],
                                                            search_results['density_tolerance'])

                plt.show()

            except Exception as e:
                print(f"Plotting failed (this is non-critical): {e}")
        else:
            print("No valid sLH2 results to analyze.")


if __name__ == "__main__":
    main()