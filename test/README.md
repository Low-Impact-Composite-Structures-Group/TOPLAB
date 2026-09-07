# Toplab Test Suite

Unit, integration, and regression tests for the multi-tank hydrogen storage framework, run with [pytest](https://docs.pytest.org).

## Running tests

From the repository root (with the `toplab` package installed or on `PYTHONPATH`):

```bash
# All tests
pytest test/

# Stop on first failure
pytest test/ -x

# Only fast unit tests
pytest test/ -m unit

# Only integration / simulation tests (slower)
pytest test/ -m integration

# Exclude slow benchmarks
pytest test/ -m "not slow"

# Verbose with live output
pytest test/ -v -s
```

## Test categories (markers)

| Marker | Scope |
|---|---|
| `unit` | Pure-logic tests, no I/O, no simulation |
| `integration` | Runs the ODE solver for a short simulation |
| `regression` | Guards against specific previously-fixed bugs |
| `coupling` | Coupling valves and inter-tank flow physics |
| `physics` | Edge-flow sign conventions, thermodynamic identities |
| `plotting` | Report and plot generation smoke tests |
| `slow` | Full-length numerical benchmarks (Lotka-Volterra, Robertson) |

## What is tested

**Thermal model and state** (`test_tank_thermal_model.py`)
Validates the four-layer insulation model introduced with the 4-state DAE:
Rohacell material properties, `InsulatedTankThermalModel` heat-flux sign
conventions and scaling, `IsochoricTankState` 4-element state vector, and
`MultiTankState` stride-4 assembly/disassembly.

**NIST materials** (`test_nist_materials.py`)
Temperature-dependent specific heat and thermal capacity for aluminium and
composite materials against the raw NIST interpolation functions.

**Edge-flow physics** (`test_edge_flow.py`)
Sign conventions and enthalpy contributions for coupling, discharge, refuel,
and vent edge types.

**Coupling flows** (`test_coupling_flows.py`)
Pressure-triggered valve hysteresis logic, orifice flow-rate calculation, and
coupling-flow data storage across a short two-tank simulation.

**Peripheral component benchmarks** (`test_peripheral_component_benchmarks.py`)
Compressor, ideal heat exchanger, and cryopump components against
analytically-derived steady-state enthalpy and temperature targets.

**Orchestrator smoke** (`test_orchestrator_smoke.py`)
Loads every config under `examples/`, wires the full orchestrator stack, and
runs a 10-second simulation to verify that the integration pipeline executes
without error for both single-tank and coupled configurations.

**Report and plot smoke** (`test_report_and_plot_smoke.py`)
Verifies that the comprehensive results report can be generated after a short
simulation and that `generate_plots` is callable on all orchestrators.

**Scenario configuration** (`test_scenario_config.py`)
YAML parsing, material look-up, mission extraction, and validation for the
`ScenarioConfig` class.

**Solver benchmarks** (`test_benchmark_problems.py`)
Solver-architecture validation using standard numerical benchmarks
(Lotka-Volterra and Robertson's stiff problem) independent of tank physics.
Marked `slow`; excluded from the default CI run.

## Configuration

pytest settings (markers, default paths, etc.) are in `pytest.ini`.
Session-scoped fixtures (repo root, example config paths) are in `conftest.py`.


## 🚀 Quick Start

### Run All Tests
```bash
# Simple run
python test/run_tests.py

# Verbose output
python test/run_tests.py --verbose

# With coverage reporting
python test/run_tests.py --coverage
```

### Run Specific Tests
```bash
# Run only NIST materials tests
python test/run_tests.py --module nist_materials

# Fast mode (stop on first failure)
python test/run_tests.py --fast

# CI mode (quiet output, XML results)
python test/run_tests.py --ci
```

## 📁 Test Structure

```
test/
├── run_tests.py              # Main test runner
├── pytest.ini               # pytest configuration
├── __init__.py              # Package initialization
├── test_nist_materials.py   # NIST materials framework tests
├── test_scenario_config.py  # Configuration parsing tests (future)
├── test_system_orchestrator.py  # Orchestrator tests (future)
└── test_integration.py      # End-to-end integration tests (future)
```

## 🧪 Current Test Coverage

### ✅ NIST Materials Framework (`test_nist_materials.py`)
- **30 tests** covering all aspects of the NIST materials system
- **Test Classes:**
  - `TestNISTMaterialBasics` - Basic material properties and creation
  - `TestTemperatureDependentProperties` - Temperature-dependent specific heat
  - `TestThermalCapacityCalculations` - Thermal capacity calculations
  - `TestMaterialRegistry` - Material lookup by NIST path
  - `TestModelCompatibility` - Compatibility with existing thermal models
  - `TestTemperatureRangeValidation` - Temperature clamping behavior
  - `TestOriginalNISTComparison` - Accuracy vs original NIST implementation
  - `TestIntegrationScenarios` - Realistic tank configuration scenarios

### ✅ Mission Configuration Framework (`test_mission_config.py`)
- **24 tests** covering all mission configuration parsing and validation
- **Test Classes:**
  - `TestMissionConfigBasics` - Basic mission parsing (ATR72, constant flow, custom)
  - `TestMissionValidation` - Configuration validation and error handling
  - `TestRefuelMissionParsing` - Refuel mission types (cryogenic, ambient)
  - `TestDormancyMissionParsing` - Dormancy mission types (storage)
  - `TestYAMLConfigurationFiles` - Real YAML file parsing validation
  - `TestMissionParameterExtraction` - Parameter extraction for different profiles
  - `TestEdgeCases` - Edge cases and error conditions

### ✅ Solver Benchmark Problems (`test_benchmark_problems.py`)
- **Standard numerical analysis benchmarks** for solver architecture validation
- **Test Problems:**
  - `Lotka-Volterra System` - Predator-prey oscillatory dynamics (non-stiff)
  - `Robertson's Problem` - Chemical kinetics with extreme stiffness (stiff)
- **Solver Coverage:** All 5 solvers (RK45, Radau, DOP853, BDF, LSODA)
- **Validation:** Conservation laws, accuracy, performance metrics

### ✅ Coupling Flow Framework (`test_coupling_flows.py`)
- **6 tests** covering coupling flow physics and data handling
- **Test Classes:**
  - `TestCouplingFlows` - Pressure-triggered valve logic, flow calculations, data storage
- **Coverage:**
  - Valve activation/deactivation hysteresis logic
  - Coupling flow rate calculation with realistic physics
  - Data storage and extraction for plotting
  - Configuration parsing for different coupling rule formats
  - Performance validation of valve calculations
  - Integration testing with simulation results

### ✅ Multi-Tank Plotting Enhancements (`test_multi_tank_plotting_enhancements.py`)
- **6 tests** covering enhanced plotting features for multi-tank systems
- **Test Classes:**
  - `TestMultiTankPlottingEnhancements` - Auto temperature ranges, legend improvements, coupling flow visualization
- **Coverage:**
  - Auto-computed temperature range calculation for density-temperature plots
  - Legend formatting improvements (title, isobars entry, larger arrows)
  - Coupling flow visualization data extraction
  - Density-temperature plot enhancements for multi-tank systems
  - Plot file generation capability validation
  - Integration of all plotting enhancements

## 🎯 Test Categories

Tests are organized using pytest markers:

```bash
# Run only unit tests
pytest test/ -m unit

# Run only integration tests
pytest test/ -m integration

# Run coupling-related tests
pytest test/ -m coupling

# Run plotting framework tests
pytest test/ -m plotting

# Run NIST-related tests
pytest test/ -m nist

# Run performance tests
pytest test/ -m performance

# Skip slow tests
pytest test/ -m "not slow"
```

## 🔍 Environment Detection

The test runner automatically detects your Python environment:

1. **Micromamba** - `micromamba run -n python-h2-dev python`
2. **Conda** - `conda run -n python-h2-dev python`
3. **System Python** - `python` (fallback)

## 📊 Coverage Reporting

Generate coverage reports to ensure comprehensive testing:

```bash
# Generate HTML coverage report
python test/run_tests.py --coverage

# View coverage report
open test/coverage_html/index.html
```

Coverage targets:
- **Minimum**: 80% overall coverage
- **Components**: Each major component should have >90% coverage
- **Integration**: End-to-end scenarios should cover all critical paths

## 🚥 Continuous Integration

For CI/CD pipelines:

```bash
# CI mode - quiet output, XML results, no coverage HTML
python test/run_tests.py --ci

# Results saved to: test/results.xml
```

## 🛠️ Test Development Guidelines

### Adding New Tests

1. **File naming**: `test_<component_name>.py`
2. **Class naming**: `Test<ComponentName><Aspect>`
3. **Method naming**: `test_<specific_behavior>`
4. **Markers**: Add appropriate pytest markers

Example:
```python
import pytest

class TestScenarioConfig:
    """Test scenario configuration parsing."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_yaml_parsing(self):
        """Test YAML configuration file parsing."""
        # Test implementation
        pass
```

### Test Organization

- **Unit tests**: Individual component testing
- **Integration tests**: Component interaction testing
- **End-to-end tests**: Full system scenarios
- **Performance tests**: Benchmark critical operations

### Best Practices

1. **Isolation**: Each test should be independent
2. **Descriptive names**: Test names should describe behavior
3. **Fixtures**: Use pytest fixtures for common setup
4. **Parametrization**: Use `@pytest.mark.parametrize` for multiple scenarios
5. **Assertions**: Clear, descriptive assertion messages

## 📈 Future Tests

Planned test modules:

- **`test_scenario_config.py`** - YAML configuration parsing and validation
- **`test_system_orchestrator.py`** - System orchestration and coordination
- **`test_tank_design.py`** - Tank geometry and structural calculations
- **`test_mission_profiles.py`** - Mission profile parsing and execution
- **`test_integration.py`** - Full end-to-end system tests
- **`test_performance.py`** - Performance benchmarks and profiling

## 🎉 Success Metrics

Current status: **✅ 111/111 tests passing (100%)**

Target metrics:
- **Test coverage**: >90% for all core components
- **Test execution time**: <30 seconds for full suite
- **Test reliability**: 99.9% pass rate in CI
- **Regression detection**: All breaking changes caught by tests

---

*This testing framework ensures the multi-tank orchestrator system remains robust and reliable as development progresses.*