"""
Liquid hydrogen (LH2) benchmark analysis using parametric framework.

This module implements the LH2 variant of the parametric benchmark analysis,
using the same mission simulation and optimization framework but with
LH2-specific initial conditions and requirements.

LH2 Characteristics:
- Initial conditions: ~5 bar, 20.3 K (saturated liquid hydrogen)
- Minimum density requirement: 12.0 kg/m³ (liquid threshold)
- Much higher density than gas phases (~71 kg/m³ initially)
- Extremely sensitive to thermal losses due to very low temperature
- Boil-off management critical for mission success

Expected Result: Smaller tank volumes due to high liquid density,
but potentially higher structural mass due to cryogenic insulation requirements.

Authors: Dante Raso (2025)
Based on parametric_benchmark.py framework
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


class BenchmarkLH2Analysis(ParametricBenchmark):
    """
    Liquid hydrogen benchmark analysis.

    This class implements the LH2 variant using the parametric benchmark framework.
    All common functionality is inherited from ParametricBenchmark. Only LH2-specific
    parameters are defined here.

    LH2 operates at very low temperature (20.3 K) with high liquid density (~71 kg/m³).
    The challenge is maintaining liquid state during discharge while minimizing boil-off.
    """

    def __init__(self, tank_radius: float = 0.69):
        """
        Initialize LH2 analysis with modified venting pressure.

        LH2 uses lower venting pressure (10 bar) compared to other storage types
        due to the challenges of maintaining liquid state at higher pressures.
        """
        super().__init__(tank_radius)

    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure for LH2."""
        return 10e5  # 5 bar - slightly above atmospheric to maintain liquid state

    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature for LH2."""
        return 20.3  # 20.3 K - normal boiling point of hydrogen at ~1 atm

    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density for LH2."""
        return 5.0  # 12.0 kg/m³ - maintain liquid-like density, well above gas density

    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        return "Liquid H2 (LH2)"

    def get_venting_pressure(self) -> float:
        """Get venting pressure for LH2."""
        return 15e5

    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get LH2-specific optimization parameters."""
        return {
            'initial_radius': 0.2,
            'max_radius': 1.0,          # Likely won't need very large tanks
            'radius_increment': 0.03,   # 30 mm steps for coarse search (finer due to smaller range)
            'max_iterations': 30,       # Maximum iterations
            'target_density_margin': 1.0  # Target 1.0 kg/m³ above minimum (larger margin for liquid)
        }

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure for LH2."""
        return 1.0e5  # 8 bar - safely above hydrogen triple point (7.4 bar)

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient for LH2."""
        return 0.010  # 0.010 W/m²K - superior insulation required for cryogenic liquid


def quick_test(radius=1.0, include_plots=False):
    """Quick test function to run analysis with a specific radius without optimization."""
    print(f"Quick LH2 Test - Radius: {radius:.3f}m")
    print("="*50)

    analysis = BenchmarkLH2Analysis(tank_radius=radius)
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
    """Main execution function for LH2 benchmark analysis."""
    print("Starting Liquid Hydrogen (LH2) Benchmark Analysis")
    print("="*80)

    # Quick test mode - run single analysis without optimization
    # test_radius = 0.62  # Change this to test different radii
    # analysis = BenchmarkLH2Analysis(tank_radius=test_radius)

    # print(f"Running single analysis with radius {test_radius:.3f}m (no optimization)")
    # success = analysis.run_single_analysis()

    # if success:
        # analysis.print_results()

        # Generate plots
        # try:
            # print("\nGenerating 4 separate plots...")
            # analysis.run_analysis(include_plots=True)
        # except Exception as e:
            # print(f"Plotting failed (this is non-critical): {e}")
    # else:
        # print("Analysis failed")

    # return  # Exit early to skip optimization

    # Run optimal radius search for LH2 (commented out for quick testing)
    search_results = find_optimal_radius_for_storage_type(BenchmarkLH2Analysis)

    if search_results['search_successful']:
        print(f"\n" + "="*80)
        print("FINAL LH2 ANALYSIS WITH OPTIMAL RADIUS")
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

        print(f"\nLH2 Performance Summary:")
        print(f"  Optimal radius: {optimal_data['radius']:.3f} m")
        print(f"  Tank volume: {optimal_data['volume']:.3f} m³")
        print(f"  Structural mass: {optimal_data['structural_mass']:.1f} kg")
        print(f"  Final fuel mass: {fuel_mass:.1f} kg")
        print(f"  Gravimetric efficiency: {gravimetric_eff:.1f}%")
        print(f"  Final density: {optimal_data['final_density']:.1f} kg/m³")

        # Generate plots for optimal configuration
        try:
            print("\nGenerating 4 separate plots for optimal LH2 configuration...")
            analysis.run_analysis(include_plots=True)

        except Exception as e:
            print(f"Plotting failed (this is non-critical): {e}")

    else:
        print(f"\n" + "="*80)
        print("LH2 SEARCH FAILED - ANALYZING BEST ATTEMPT")
        print("="*80)

        if search_results['search_results']:
            # Use the best attempt
            best_result = min(search_results['search_results'], key=lambda x: x['max_pressure'])
            analysis = best_result['analysis']

            print(f"Using radius {best_result['radius']:.3f}m (max pressure: {best_result['max_pressure_bar']:.1f} bar, final density: {best_result['final_density']:.2f} kg/m³)")
            analysis.print_results()

            try:
                print("\nGenerating 4 separate plots for best LH2 attempt...")
                analysis.run_analysis(include_plots=True)

            except Exception as e:
                print(f"Plotting failed (this is non-critical): {e}")
        else:
            print("No valid LH2 results to analyze.")


if __name__ == "__main__":
    main()