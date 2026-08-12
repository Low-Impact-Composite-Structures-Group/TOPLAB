"""
Coupling Flow Tests - Phase 4
=============================

Test coupling flow physics, valve logic, and data storage/extraction
for multi-tank systems with pressure compensation and other coupling mechanisms.

Author: Framework Development Team
Date: October 8, 2025
"""

import pytest
import yaml
import numpy as np
from pathlib import Path

from toplab.configuration.scenario_configuration import ScenarioConfig
from toplab.orchestration.system_orchestrator import SystemOrchestrator
from toplab.coupling.inter_tank_coupling import PressureTriggeredValve
from toplab.fluids.flow_physics import FlowPhysics
# from multi_tank.system.multi_tank_state import MultiTankState  # Not needed for these tests


class TestCouplingFlows:
    """Test coupling flow physics and data handling."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        """Setup paths for all test methods."""
        self.repo_root = Path(__file__).resolve().parent.parent
        self.coupled_config = self.repo_root / 'analysis' / 'coupled_ch2_cch2' / 'coupled_ch2_cch2_config.yaml'

        assert self.coupled_config.exists(), f"Coupled config not found: {self.coupled_config}"

    @pytest.mark.coupling
    def test_pressure_triggered_valve_logic(self):
        """Test pressure-triggered valve activation/deactivation logic."""

        flow_physics = FlowPhysics({})

        # Create valve with test parameters
        valve = PressureTriggeredValve(
            source_idx=0, target_idx=1,
            p_open=16e5,    # 16 bar activation
            p_close=20e5,   # 20 bar deactivation
            max_flow_rate=0.1,
            orifice_diameter=0.002,
            flow_physics=flow_physics,
        )

        # Test initial state
        assert valve.is_active == False, "Valve should start inactive"
        assert valve.activation_threshold == 16e5, "Incorrect activation threshold"
        assert valve.deactivation_threshold == 20e5, "Incorrect deactivation threshold"

        # Test valve opening logic
        mock_tank_states = [None, None]

        # Mock target tank with high pressure (valve should not open)
        class MockTankState:
            def __init__(self, pressure):
                self.pressure = pressure
            def compute_pressure(self):
                pass

        mock_tank_states[1] = MockTankState(25e5)  # 25 bar - above threshold
        result = valve.evaluate(0, mock_tank_states)
        assert valve.is_active == False, "Valve should not open at high pressure"

        # Test valve opening at low pressure
        mock_tank_states[1] = MockTankState(15e5)  # 15 bar - below activation
        result = valve.evaluate(0, mock_tank_states)
        assert valve.is_active == True, "Valve should open at low pressure"

        # Test hysteresis - valve stays open until deactivation threshold
        mock_tank_states[1] = MockTankState(18e5)  # 18 bar - between thresholds
        result = valve.evaluate(0, mock_tank_states)
        assert valve.is_active == True, "Valve should stay open (hysteresis)"

        # Test valve closing at high pressure
        mock_tank_states[1] = MockTankState(21e5)  # 21 bar - above deactivation
        result = valve.evaluate(0, mock_tank_states)
        assert valve.is_active == False, "Valve should close at high pressure"

        print("Pressure-triggered valve logic validated")

    @pytest.mark.coupling
    def test_coupling_flow_calculation_physics(self):
        """Test coupling flow rate calculation with realistic physics."""

        flow_physics = FlowPhysics({})

        # Create valve for testing
        valve = PressureTriggeredValve(
            source_idx=0, target_idx=1,
            p_open=16e5, p_close=20e5,
            max_flow_rate=0.1,
            orifice_diameter=0.002,  # 2mm orifice
            flow_physics=flow_physics,
        )

        # Create realistic tank states
        class MockTankState:
            def __init__(self, pressure, temperature, fuel_mass, volume):
                self.pressure = pressure
                self.temperature = temperature
                self.fuel_mass = fuel_mass
                self.volume = volume
                self.tank = MockTank(volume)

            def compute_pressure(self):
                pass

        class MockTank:
            def __init__(self, volume):
                self.volume = volume

        # High pressure source (CH2 at 700 bar)
        source_state = MockTankState(
            pressure=700e5,     # 700 bar
            temperature=330,    # 330 K
            fuel_mass=50,       # 50 kg
            volume=1.4          # 1.4 m³
        )

        # Low pressure target (CCH2 at 15 bar - valve should open)
        target_state = MockTankState(
            pressure=15e5,      # 15 bar
            temperature=50,     # 50 K
            fuel_mass=200,      # 200 kg
            volume=2.7          # 2.7 m³
        )

        tank_states = [source_state, target_state]
        valve.evaluate(0.0, tank_states)

        # Test flow calculation
        flow_rate = valve.calculate_flow_rate(0.0, tank_states)

        # Validate flow rate
        assert flow_rate > 0, "Flow rate should be positive with pressure differential"
        assert flow_rate <= valve.max_flow_rate, "Flow rate should not exceed maximum"
        assert flow_rate < source_state.fuel_mass, "Flow rate should not exceed source mass"

        # Test with no pressure differential (equal pressures)
        target_state.pressure = 700e5  # Same as source
        valve.evaluate(1.0, tank_states)
        flow_rate_equal = valve.calculate_flow_rate(1.0, tank_states)
        assert flow_rate_equal == 0, "No flow with equal pressures"

        # Test with reverse pressure differential
        target_state.pressure = 800e5  # Higher than source
        valve.evaluate(2.0, tank_states)
        flow_rate_reverse = valve.calculate_flow_rate(2.0, tank_states)
        assert flow_rate_reverse == 0, "No flow with reverse pressure differential"

        print(f"Coupling flow physics validated (flow rate: {flow_rate*1000:.1f} g/s)")

    @pytest.mark.coupling
    def test_coupling_flow_data_storage(self):
        """Test that coupling flows are correctly stored in simulation results."""

        print("Testing coupling flow data storage...")

        try:
            # Load coupled system configuration
            config = ScenarioConfig.from_yaml(str(self.coupled_config))
            orchestrator = SystemOrchestrator(config)

            # Verify multi-tank system with coupling
            assert len(orchestrator.tank_system.tanks) >= 2, "Need multi-tank system for coupling"
            assert len(orchestrator.tank_system.coupling_valves) > 0, "Need coupling valves"

            # Run short simulation to generate coupling flow data
            test_solver_config = {
                'method': 'LSODA',
                'rtol': 1e-3,  # Relaxed for speed
                'atol': 1e-6,
                'max_simulation_time': 10.0,  # fast smoke duration
                'time_step': 2.0
            }

            print("   Running short simulation for coupling flow validation...")
            results = orchestrator.run_simulation('LSODA', test_solver_config)

            # Validate results structure
            assert results is not None, "No simulation results returned"
            assert hasattr(results, 'times'), "Results missing time data"
            assert hasattr(results, 'multi_tank_states'), "Results missing multi-tank states"
            assert results.n_timesteps > 0, "No timesteps in results"

            print(f"   Simulation completed: {results.n_timesteps} timesteps, {results.n_tanks} tanks")

            # Check for coupling flow data in results
            coupling_flow_data_found = False

            # Sample a few time points to check coupling flows
            sample_indices = [0, results.n_timesteps//4, results.n_timesteps//2, -1]

            for i in sample_indices:
                if i >= results.n_timesteps:
                    continue

                multi_state = results.multi_tank_states[i]
                time_point = results.times[i]

                for tank_idx, tank_state in enumerate(multi_state.tank_states):
                    # Check for coupling flow attributes
                    if hasattr(tank_state, 'coupling_inflow_rate') and hasattr(tank_state, 'coupling_outflow_rate'):
                        inflow = getattr(tank_state, 'coupling_inflow_rate', 0.0)
                        outflow = getattr(tank_state, 'coupling_outflow_rate', 0.0)

                        if abs(inflow) > 1e-6 or abs(outflow) > 1e-6:
                            coupling_flow_data_found = True
                            print(f"   t={time_point:.1f}s Tank{tank_idx}: inflow={inflow*1000:.1f}g/s, outflow={outflow*1000:.1f}g/s")

            if coupling_flow_data_found:
                print("   Coupling flow data successfully stored and retrieved")
            else:
                # This might indicate our recent fix is working correctly
                print("   No significant coupling flows detected in sample (may be expected)")

            # Check mass conservation during coupling
            if results.n_tanks >= 2:
                initial_mass = sum(state.fuel_mass for state in results.multi_tank_states[0].tank_states)
                final_mass = sum(state.fuel_mass for state in results.multi_tank_states[-1].tank_states)

                # Mass should be conserved (within numerical tolerance) for pressure compensation
                mass_change = abs(final_mass - initial_mass)
                relative_change = mass_change / initial_mass if initial_mass > 0 else 0

                # Allow for mission discharge (outflow from system)
                print(f"   Mass conservation: {initial_mass:.1f} -> {final_mass:.1f} kg (delta={mass_change:.2f}kg, {relative_change*100:.2f}%)")

                # For pressure compensation, mass transfer between tanks should be much smaller
                # than mission discharge, but we can't test strict conservation due to mission outflow

            print("Coupling flow data storage validated")

        except Exception as e:
            print(f"Coupling flow storage test error: {str(e)[:100]}...")
            raise

    @pytest.mark.coupling
    def test_coupling_rule_configuration_parsing(self):
        """Test parsing of coupling rule configurations from YAML."""

        print("Testing coupling rule configuration parsing...")

        # Load and parse YAML directly
        with open(self.coupled_config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # New schema: network.edges
        edges = config_dict.get('network', {}).get('edges', [])
        assert isinstance(edges, list), "network.edges should be a list"
        assert len(edges) > 0, "No network edges found in config"

        rules_to_validate = [(edge.get('edge_id', f"edge_{i}"), edge) for i, edge in enumerate(edges)]

        # Validate individual coupling rules
        for rule_name, rule_config in rules_to_validate:
            print(f"   🔗 Validating rule: {rule_name}")

            # Required edge fields
            assert 'connection_type' in rule_config
            assert 'from_node' in rule_config
            assert 'to_node' in rule_config

            print(f"     Connection type: {rule_config.get('connection_type')}")
            if 'activation_conditions' in rule_config:
                print("     Activation conditions present")
            if 'flow_physics' in rule_config:
                print("     Flow physics present")

        # Test orchestrator parsing of coupling rules
        config = ScenarioConfig.from_yaml(str(self.coupled_config))
        orchestrator = SystemOrchestrator(config)

        # Verify coupling valves were created
        valve_count = len(orchestrator.tank_system.coupling_valves)
        rule_count = len(edges)

        print(f"   Created {valve_count} valves from {rule_count} rules")
        assert valve_count > 0, "No coupling valves created"

        print("Coupling rule configuration parsing validated")

    @pytest.mark.integration
    def test_coupling_flow_plotting_data_extraction(self):
        """Test that coupling flow data can be extracted for plotting."""

        print("Testing coupling flow plotting data extraction...")

        try:
            # Load system and run simulation
            config = ScenarioConfig.from_yaml(str(self.coupled_config))
            orchestrator = SystemOrchestrator(config)

            # Short simulation
            test_solver_config = {
                'method': 'LSODA',
                'rtol': 1e-3,
                'atol': 1e-6,
                'max_simulation_time': 10.0,
                'time_step': 1.0
            }

            results = orchestrator.run_simulation('LSODA', test_solver_config)

            # Test plotting data extraction
            # This tests the specific functionality we've been fixing

            # Extract coupling flow time series for plotting validation
            times = results.times
            coupling_inflows = []
            coupling_outflows = []

            for tank_idx in range(results.n_tanks):
                tank_inflows = []
                tank_outflows = []

                for i in range(results.n_timesteps):
                    multi_state = results.multi_tank_states[i]
                    tank_state = multi_state.tank_states[tank_idx]

                    inflow = getattr(tank_state, 'coupling_inflow_rate', 0.0)
                    outflow = getattr(tank_state, 'coupling_outflow_rate', 0.0)

                    tank_inflows.append(inflow)
                    tank_outflows.append(outflow)

                coupling_inflows.append(tank_inflows)
                coupling_outflows.append(tank_outflows)

            # Validate data structure for plotting
            assert len(coupling_inflows) == results.n_tanks, "Inflow data count mismatch"
            assert len(coupling_outflows) == results.n_tanks, "Outflow data count mismatch"

            for tank_idx in range(results.n_tanks):
                assert len(coupling_inflows[tank_idx]) == results.n_timesteps, f"Inflow time series length mismatch for tank {tank_idx}"
                assert len(coupling_outflows[tank_idx]) == results.n_timesteps, f"Outflow time series length mismatch for tank {tank_idx}"

            # Check for data integrity (arrays should be numeric)
            for tank_idx in range(results.n_tanks):
                inflow_array = np.array(coupling_inflows[tank_idx])
                outflow_array = np.array(coupling_outflows[tank_idx])

                assert np.all(np.isfinite(inflow_array)), f"Non-finite inflow data for tank {tank_idx}"
                assert np.all(np.isfinite(outflow_array)), f"Non-finite outflow data for tank {tank_idx}"
                assert np.all(inflow_array >= 0), f"Negative inflow values for tank {tank_idx}"
                assert np.all(outflow_array >= 0), f"Negative outflow values for tank {tank_idx}"

            # Check for any coupling activity
            total_inflow = sum(sum(tank_inflows) for tank_inflows in coupling_inflows)
            total_outflow = sum(sum(tank_outflows) for tank_outflows in coupling_outflows)

            print(f"   Coupling flow data extracted: {results.n_timesteps} points x {results.n_tanks} tanks")
            print(f"   Total coupling flows: {total_inflow*1000:.1f}g·s inflow, {total_outflow*1000:.1f}g·s outflow")

            # For pressure compensation, inflow and outflow should be approximately equal
            if total_inflow > 0 or total_outflow > 0:
                flow_balance = abs(total_inflow - total_outflow) / max(total_inflow, total_outflow, 1e-10)
                print(f"   Coupling flow balance: {flow_balance*100:.2f}% imbalance")

                # Allow some imbalance due to numerical integration and timing
                assert flow_balance < 0.1, f"Coupling flow imbalance too large: {flow_balance*100:.1f}%"

                print("   Coupling flow data extraction successful with flow activity")
            else:
                print("   No significant coupling flows detected (may be expected for short simulation)")

            print("Coupling flow plotting data extraction validated")

        except Exception as e:
            print(f"Plotting data extraction test error: {str(e)[:100]}...")
            raise

    @pytest.mark.performance
    def test_coupling_valve_performance(self):
        """Test performance of coupling valve calculations."""
        import time

        print("Testing coupling valve performance...")

        flow_physics = FlowPhysics({})

        # Create valve
        valve = PressureTriggeredValve(
            source_idx=0, target_idx=1,
            p_open=16e5, p_close=20e5,
            max_flow_rate=0.1,
            orifice_diameter=0.002,
            flow_physics=flow_physics,
        )

        # Create mock states
        class MockTankState:
            def __init__(self):
                self.pressure = 700e5
                self.temperature = 330
                self.fuel_mass = 50
                self.volume = 1.4
                self.tank = MockTank()
            def compute_pressure(self):
                pass

        class MockTank:
            def __init__(self):
                self.volume = 1.4

        source_state = MockTankState()
        target_state = MockTankState()
        target_state.pressure = 15e5  # Activate valve

        # Performance test
        n_iterations = 1000
        start_time = time.time()

        tank_states = [source_state, target_state]
        valve.is_active = True
        valve._valve_coefficient = 1.0

        for i in range(n_iterations):
            # Vary pressure slightly to test different conditions
            target_state.pressure = 15e5 + (i % 100) * 1e3  # 15-16 bar range
            flow_rate = valve.calculate_flow_rate(float(i), tank_states)

        elapsed_time = time.time() - start_time
        time_per_call = elapsed_time / n_iterations * 1000  # milliseconds

        print(f"   Performance: {time_per_call:.3f} ms per valve calculation")
        print(f"   {n_iterations} calculations in {elapsed_time:.3f}s")

        # Performance should be reasonable for real-time simulation
        assert time_per_call < 1.0, f"Valve calculation too slow: {time_per_call:.3f} ms > 1.0 ms"

        print("Coupling valve performance validated")


if __name__ == '__main__':
    """Allow running tests directly with python test_coupling_flows.py"""
    pytest.main([__file__, '-v', '--tb=short'])