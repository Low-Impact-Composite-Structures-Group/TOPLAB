# Multi-Tank Framework Test Suite

Comprehensive testing framework for the orchestrated multi-tank hydrogen storage system using pytest.

## 🚀 Quick Start

### Run All Tests
```bash
# Simple run
python test/multi_tank_tests/run_tests.py

# Verbose output
python test/multi_tank_tests/run_tests.py --verbose

# With coverage reporting
python test/multi_tank_tests/run_tests.py --coverage
```

### Run Specific Tests
```bash
# Run only NIST materials tests
python test/multi_tank_tests/run_tests.py --module nist_materials

# Fast mode (stop on first failure)
python test/multi_tank_tests/run_tests.py --fast

# CI mode (quiet output, XML results)
python test/multi_tank_tests/run_tests.py --ci
```

## 📁 Test Structure

```
test/multi_tank_tests/
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

## 🎯 Test Categories

Tests are organized using pytest markers:

```bash
# Run only unit tests
pytest test/multi_tank_tests/ -m unit

# Run only integration tests
pytest test/multi_tank_tests/ -m integration

# Run NIST-related tests
pytest test/multi_tank_tests/ -m nist

# Skip slow tests
pytest test/multi_tank_tests/ -m "not slow"
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
python test/multi_tank_tests/run_tests.py --coverage

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
python test/multi_tank_tests/run_tests.py --ci

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

Current status: **✅ 54/54 tests passing (100%)**

Target metrics:
- **Test coverage**: >90% for all core components
- **Test execution time**: <30 seconds for full suite
- **Test reliability**: 99.9% pass rate in CI
- **Regression detection**: All breaking changes caught by tests

---

*This testing framework ensures the multi-tank orchestrator system remains robust and reliable as development progresses.*