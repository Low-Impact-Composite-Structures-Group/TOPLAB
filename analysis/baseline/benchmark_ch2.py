"""
Compressed gaseous hydrogen (CH2) benchmark analysis using parametric framework.

This module implements the CH2 variant of the parametric benchmark analysis,
using the same mission simulation and optimization framework but with
CH2-specific initial conditions and requirements.

CH2 Characteristics:
- Initial conditions: ~700-900 bar, 288 K (high pressure ambient temperature gas)
- Minimum density requirement: 4.0 kg/m³ (compressed gas threshold)
- Much lower density than liquid (~40-50 kg/m³ initially vs ~71 kg/m³ for LH2)
- No phase change concerns - remains gaseous throughout mission
- Very thick composite walls required due to high pressure (structural challenge)

Expected Result: Larger tank volumes due to lower gas density,
potentially very high structural mass due to thick pressure vessel walls.

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


class BenchmarkCH2Analysis(ParametricBenchmark):
    """
    Compressed gaseous hydrogen benchmark analysis.

    This class implements the CH2 variant using the parametric benchmark framework.
    All common functionality is inherited from ParametricBenchmark. Only CH2-specific
    parameters are defined here.

    CH2 operates at ambient temperature (288 K) with very high pressure (~700-900 bar).
    The challenge is managing structural mass from thick pressure vessel walls while
    maintaining sufficient fuel density for the mission.
    """

    def __init__(self, tank_radius: float = 0.69):
        """
        Initialize CH2 analysis with higher design pressure.

        CH2 requires much higher design pressure capability due to
        operating pressures of 700-900 bar vs 400 bar for CCH2.
        """
        super().__init__(tank_radius)

        # Override design pressure for CH2 high-pressure requirements
        self.design_pressure = 1000e5  # 1000 bar design pressure for 900 bar operating

    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure for CH2."""
        return 800e5  # 800 bar - high pressure gaseous storage

    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature for CH2."""
        return 288.15  # 288.15 K - ambient temperature (15°C)

    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density for CH2."""
        return 4.0  # 4.0 kg/m³ - maintain compressed gas density above standard conditions

    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        return "Compressed H2 (CH2)"

    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get CH2-specific optimization parameters."""
        return {
            'initial_radius': 0.5,      # Start larger due to lower gas density
            'max_radius': 1.5,          # May need very large tanks due to low density
            'radius_increment': 0.08,   # 80 mm steps for coarse search (larger steps due to bigger range)
            'max_iterations': 25,       # Maximum iterations
            'target_density_margin': 0.5  # Target 0.5 kg/m³ above minimum
        }

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure for CH2."""
        return 15e5  # 15 bar - sufficient for compressed gaseous storage

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient for CH2."""
        return 0.050  # 0.050 W/m²K - minimal insulation needed for ambient temperature storage


def main():
    """Main execution function for CH2 benchmark analysis."""
    print("Starting Compressed Gaseous Hydrogen (CH2) Benchmark Analysis")
    print("="*80)

    # Run optimal radius search for CH2
    search_results = find_optimal_radius_for_storage_type(BenchmarkCH2Analysis)

    if search_results['search_successful']:
        print(f"\n" + "="*80)
        print("FINAL CH2 ANALYSIS WITH OPTIMAL RADIUS")
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

        print(f"\nCH2 Performance Summary:")
        print(f"  Optimal radius: {optimal_data['radius']:.3f} m")
        print(f"  Tank volume: {optimal_data['volume']:.3f} m³")
        print(f"  Structural mass: {optimal_data['structural_mass']:.1f} kg")
        print(f"  Final fuel mass: {fuel_mass:.1f} kg")
        print(f"  Gravimetric efficiency: {gravimetric_eff:.1f}%")
        print(f"  Final density: {optimal_data['final_density']:.1f} kg/m³")
        print(f"  Composite thickness: {analysis.composite_thickness*1000:.1f} mm (high pressure design)")

        # Generate plots for optimal configuration
        try:
            print("\nGenerating 4 separate plots for optimal CH2 configuration...")
            analysis.run_analysis(include_plots=True)

        except Exception as e:
            print(f"Plotting failed (this is non-critical): {e}")

    else:
        print(f"\n" + "="*80)
        print("CH2 SEARCH FAILED - ANALYZING BEST ATTEMPT")
        print("="*80)

        if search_results['search_results']:
            # Use the best attempt
            best_result = min(search_results['search_results'], key=lambda x: x['max_pressure'])
            analysis = best_result['analysis']

            print(f"Using radius {best_result['radius']:.3f}m (max pressure: {best_result['max_pressure_bar']:.1f} bar, final density: {best_result['final_density']:.2f} kg/m³)")
            analysis.print_results()

            try:
                print("\nGenerating 4 separate plots for best CH2 attempt...")
                analysis.run_analysis(include_plots=True)

            except Exception as e:
                print(f"Plotting failed (this is non-critical): {e}")
        else:
            print("No valid CH2 results to analyze.")


if __name__ == "__main__":
    main()