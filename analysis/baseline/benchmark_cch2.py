"""
Cryocompressed hydrogen (CCH2) benchmark analysis using parametric framework.

This module implements the CCH2 variant of the parametric benchmark analysis,
using the same mission simulation and optimization framework but with
CCH2-specific initial conditions and requirements.

CCH2 Characteristics:
- Initial conditions: 400 bar, 53.25 K (cryocompressed state)
- Minimum density requirement: 5.8 kg/m³
- Pressure maintenance threshold: 15 bar (p_min)
- Venting pressure: 500 bar (p_max)
- Vacuum insulation: 0.025 W/m²K ambient HTC
- Search range: 0.2m to 1.5m radius

Expected Result: Efficient high-pressure storage with good gravimetric efficiency
due to high initial density and pressure maintenance capabilities.

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


class BenchmarkCCH2Analysis(ParametricBenchmark):
    """
    Cryocompressed hydrogen benchmark analysis.

    This class implements the CCH2 variant using the parametric benchmark framework.
    All common functionality (geometry, structural, thermal, mission, optimization)
    is inherited from ParametricBenchmark. Only CCH2-specific parameters are defined here.
    """

    def get_initial_pressure(self) -> float:
        """Get initial hydrogen pressure for CCH2."""
        return 400e5  # 400 bar - cryocompressed storage pressure

    def get_initial_temperature(self) -> float:
        """Get initial hydrogen temperature for CCH2."""
        return 53.25  # 53.25 K - cryogenic temperature for CCH2

    def get_minimum_density(self) -> float:
        """Get minimum acceptable final density for CCH2."""
        return 5.8  # 5.8 kg/m³ - minimum density requirement

    def get_storage_type_name(self) -> str:
        """Get storage type name for displays."""
        return "Cryocompressed H2 (CCH2)"

    def get_optimization_parameters(self) -> Dict[str, Any]:
        """Get CCH2-specific optimization parameters."""
        return {
            'min_radius': 0.2,              # Minimum search radius [m]
            'max_radius': 1.5,              # Maximum search radius [m]
            'radius_precision': 0.005,      # Precision: ±5mm
            'density_tolerance': 2.0,       # Within 2 kg/m³ of minimum is acceptable
            'max_evaluations': 20           # Maximum function evaluations
        }

    def get_minimum_pressure(self) -> float:
        """Get minimum allowable pressure for CCH2."""
        return 15e5  # 15 bar - Configuration B threshold

    def get_venting_pressure(self) -> float:
        """Get venting pressure for CCH2."""
        return 500e5

    def get_ambient_htc(self) -> float:
        """Get ambient heat transfer coefficient for CCH2."""
        return 0.025  # 0.025 W/m²K - vacuum insulation for cryocompressed


def quick_test(radius=1.0, include_plots=False):
    """Quick test function to run analysis with a specific radius without optimization."""
    print(f"Quick CCH2 Test - Radius: {radius:.3f}m")
    print("="*50)

    analysis = BenchmarkCCH2Analysis(tank_radius=radius)
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
    """Main execution function for CCH2 benchmark analysis."""
    print("Starting Cryocompressed Hydrogen (CCH2) Benchmark Analysis")
    print("="*80)

    # Test optimization directly
    print("Testing new bisection optimization...")
    print("Expected: Much faster than old brute-force approach")

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

        # Calculate gravimetric efficiency comparison
        final_state = analysis.results.states[-1]
        fuel_mass = final_state.fuel_mass
        gravimetric_eff = fuel_mass / (fuel_mass + analysis.total_structural_mass) * 100

        print(f"\nCCH2 Performance Summary:")
        print(f"  Optimal radius: {optimal_data['radius']:.3f} m")
        print(f"  Tank volume: {optimal_data['volume']:.3f} m³")
        print(f"  Structural mass: {optimal_data['structural_mass']:.1f} kg")
        print(f"  Final fuel mass: {fuel_mass:.1f} kg")
        print(f"  Gravimetric efficiency: {gravimetric_eff:.1f}%")
        print(f"  Final density: {optimal_data['final_density']:.1f} kg/m³")

        # Generate plots for optimal configuration and save them
        try:
            # Setup save paths
            import os
            from pathlib import Path
            results_dir = Path("../../data/results/benchmark_cch2_results")
            results_dir.mkdir(parents=True, exist_ok=True)

            print("\nGenerating and saving 5 plots for optimal CCH2 configuration...")

            # Generate and save individual plots
            analysis.plot_results(save_path=results_dir / "01_tank_states.png")
            analysis.plot_fuel_flow_profile(save_path=results_dir / "02_fuel_flow_profile.png")
            analysis.plot_density_temperature(save_path=results_dir / "03_density_temperature.png")
            analysis.plot_heat_exchanger_requirements(save_path=results_dir / "04_heat_exchanger_requirements.png")

            # Generate optimization progress plot and save it
            analysis.plot_optimization_progress_from_data(search_results['optimization_progress'],
                                                        search_results['minimum_density_target'],
                                                        search_results['density_tolerance'],
                                                        save_path=results_dir / "05_optimization_progress.png")

            # Generate and save analysis summary
            analysis.generate_analysis_summary(save_path=results_dir / "analysis_summary.md",
                                             optimization_results=search_results)

            # Show all plots at once
            import matplotlib.pyplot as plt
            print(f"All plots and summary saved to: {results_dir}")
            plt.show()

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
                # Setup save paths
                import os
                from pathlib import Path
                results_dir = Path("../../data/results/benchmark_cch2_results")
                results_dir.mkdir(parents=True, exist_ok=True)

                print("\nGenerating and saving 5 plots for best CCH2 attempt...")

                # Generate and save individual plots
                analysis.plot_results(save_path=results_dir / "01_tank_states_best_attempt.png")
                analysis.plot_fuel_flow_profile(save_path=results_dir / "02_fuel_flow_profile_best_attempt.png")
                analysis.plot_density_temperature(save_path=results_dir / "03_density_temperature_best_attempt.png")
                analysis.plot_heat_exchanger_requirements(save_path=results_dir / "04_heat_exchanger_requirements_best_attempt.png")

                # Generate optimization progress plot and save it
                analysis.plot_optimization_progress_from_data(search_results['optimization_progress'],
                                                            search_results['minimum_density_target'],
                                                            search_results['density_tolerance'],
                                                            save_path=results_dir / "05_optimization_progress_best_attempt.png")

                # Generate and save analysis summary
                analysis.generate_analysis_summary(save_path=results_dir / "analysis_summary_best_attempt.md",
                                                 optimization_results=search_results)

                # Show all plots at once
                import matplotlib.pyplot as plt
                print(f"All plots and summary saved to: {results_dir}")
                plt.show()

            except Exception as e:
                print(f"Plotting failed (this is non-critical): {e}")
        else:
            print("No valid CCH2 results to analyze.")


if __name__ == "__main__":
    main()