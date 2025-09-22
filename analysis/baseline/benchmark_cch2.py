"""
Cryocompressed hydrogen (CCH2) benchmark analysis using parametric framework.

This module implements the CCH2 variant of the parametric benchmark analysis,
using the same advanced radius optimization and mission simulation framework
but with CCH2-specific initial conditions and requirements.

CCH2 Characteristics:
- Initial conditions: 400 bar, 55 K (cryocompressed state)
- Minimum density requirement: 5.0 kg/m³ (configuration B threshold)
- Two-phase storage with potential phase transitions
- Configuration B/C logic for heat exchanger operation

Authors: Dante Raso (2025)
Based on baseline_cch2.py framework using parametric_benchmark.py base class
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


class BenchmarkCCH2Analysis(ParametricBenchmark):
    """
    Cryocompressed hydrogen benchmark analysis.

    This class implements the CCH2 variant using the parametric benchmark framework.
    All common functionality (geometry, structural, thermal, mission, optimization)
    is inherited from ParametricBenchmark. Only CCH2-specific parameters are defined here.
    """

    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure for CCH2."""
        return 400e5  # 400 bar - typical CCH2 storage pressure

    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature for CCH2."""
        return 55.0  # 55 K - cryogenic temperature for CCH2

    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density for CCH2."""
        return 5.0  # 5.0 kg/m³ - Configuration B threshold

    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        return "Cryocompressed H2 (CCH2)"

    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get CCH2-specific optimization parameters."""
        return {
            'initial_radius': 0.4,      # Start searching from 0.4 m
            'max_radius': 1.0,          # Search up to 1.0 m
            'radius_increment': 0.05,   # 50 mm steps for coarse search
            'max_iterations': 30,       # Maximum iterations
            'target_density_margin': 0.3  # Target 0.3 kg/m³ above minimum
        }

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure for CCH2."""
        return 15e5  # 15 bar - safe for cryocompressed operation

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient for CCH2."""
        return 0.025  # 0.025 W/m²K - vacuum insulation typical for cryocompressed


def main():
    """Main execution function for CCH2 benchmark analysis."""
    print("Starting Cryocompressed Hydrogen (CCH2) Benchmark Analysis")
    print("="*80)

    # Run optimal radius search for CCH2
    search_results = find_optimal_radius_for_storage_type(BenchmarkCCH2Analysis)

    if search_results['search_successful']:
        print(f"\n" + "="*80)
        print("FINAL CCH2 ANALYSIS WITH OPTIMAL RADIUS")
        print("="*80)

        # Use the optimal configuration
        optimal_data = search_results['optimal_results']
        analysis = optimal_data['analysis']

        # Print comprehensive results
        analysis.print_results()

        # Generate plots for optimal configuration
        try:
            print("\nGenerating 4 separate plots for optimal CCH2 configuration...")
            analysis.run_analysis(include_plots=True)

        except Exception as e:
            print(f"Plotting failed (this is non-critical): {e}")

    else:
        print(f"\n" + "="*80)
        print("CCH2 SEARCH FAILED - ANALYZING BEST ATTEMPT")
        print("="*80)

        if search_results['search_results']:
            # Use the best attempt
            best_result = min(search_results['search_results'], key=lambda x: x['max_pressure'])
            analysis = best_result['analysis']

            print(f"Using radius {best_result['radius']:.3f}m (max pressure: {best_result['max_pressure_bar']:.1f} bar, final density: {best_result['final_density']:.2f} kg/m³)")
            analysis.print_results()

            try:
                print("\nGenerating 4 separate plots for best CCH2 attempt...")
                analysis.run_analysis(include_plots=True)

            except Exception as e:
                print(f"Plotting failed (this is non-critical): {e}")
        else:
            print("No valid CCH2 results to analyze.")


if __name__ == "__main__":
    main()