# Configuration Schema Migration Strategy

**Date:** October 13, 2025
**Purpose:** Strategy for migrating from current flat YAML structure to intuitive network-based schema

## Current State vs Ideal State

### Current Problems with Existing Schema
```yaml
# CURRENT (PROBLEMATIC):
geometry:
  1: {...}      # Tank-specific but in separate section
  2: {...}

tank_materials:
  1: {...}      # Tank-specific but in separate section
  2: {...}

coupling_rules:  # Connection info but not clearly linked to network
  - coupling_id: "..."
    participants: {source: 1, target: 2}

flow_physics: {...}  # Global but should be per-connection

stopping_criteria: {...}  # Global but should be per-tank

output:
  plots:
    show_reference_pressures: false  # Global but should be per-tank
```

### Ideal (Intuitive) Schema
```yaml
# IDEAL (INTUITIVE):
network:
  nodes:
    - node_id: 1
      geometry: {...}        # All tank properties together
      materials: {...}       # Where they logically belong
      stopping_criteria: {...}
      plotting: {...}        # Per-tank plot settings
    - node_id: 2
      # Same structure

  edges:
    - edge_id: "connection_1"
      from_node: 1
      to_node: 2
      flow_physics: {...}    # Physics specific to this connection
      control_parameters: {...}
```

## Migration Phases

### Phase 1: Create Validation Infrastructure ✅ DONE
- [x] Created `StrictConfigValidator` class
- [x] Created ideal schema example
- [x] Defined error handling philosophy

### Phase 2: Create Configuration Adapter (NEXT STEP)
Create a compatibility layer that can read both old and new formats:

```python
class ConfigurationAdapter:
    """
    Adapter that converts between old flat format and new network format.
    Allows gradual migration without breaking existing analyses.
    """

    def migrate_old_to_new(self, old_config: Dict) -> Dict:
        """Convert old flat format to new network format."""
        pass

    def migrate_new_to_old(self, new_config: Dict) -> Dict:
        """Convert new format to old format for legacy code."""
        pass
```

### Phase 3: Update Source Code Incrementally
Update source modules one by one to handle new format:

1. **ScenarioConfig** - Add new format support while maintaining old format
2. **SystemOrchestrator** - Update to use new network structure
3. **Plotting system** - Use per-tank plotting settings
4. **Coupling rules** - Use edge-based configuration

### Phase 4: Migrate Existing Analyses
Convert each analysis to new format:

1. `coupled_ch2_lh2` (template)
2. `single_tank_ch2`
3. `single_tank_cch2`
4. `single_tank_slh2`
5. `coupled_ch2_cch2`

### Phase 5: Remove Legacy Support
Remove old format support and adapter code.

## Implementation Strategy

### Safeguard Requirements

1. **Zero Fallback Values**
   ```python
   # NEVER DO THIS:
   pressure = config.get('pressure', 101325)  # NO FALLBACKS!

   # ALWAYS DO THIS:
   if 'pressure' not in config:
       raise ConfigurationError("Missing required parameter: pressure")
   pressure = config['pressure']
   ```

2. **Comprehensive Validation**
   ```python
   # Check EVERY required parameter exists
   # Validate data types and ranges
   # Check parameter consistency (min < max, etc.)
   # Validate node/edge references
   ```

3. **Clear Error Messages**
   ```
   ERROR: Simulation halted due to missing parameter.
   You must declare an initial pressure for tank 1 in:
   network.nodes[0].initial_conditions.pressure

   File: /path/to/config.yaml
   Line: Expected around line 45
   ```

### Backward Compatibility Strategy

```python
class ScenarioConfig:
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration with automatic format detection."""
        raw_config = yaml.safe_load(open(config_path))

        # Detect format
        if 'network' in raw_config and 'nodes' in raw_config['network']:
            # New format - validate strictly
            validator = StrictConfigValidator(config_path)
            config = validator.load_and_validate()
            return cls._from_new_format(config)
        else:
            # Old format - migrate then validate
            adapter = ConfigurationAdapter()
            new_config = adapter.migrate_old_to_new(raw_config)
            validator = StrictConfigValidator(config_path)
            config = validator.load_and_validate(new_config)
            return cls._from_new_format(config)
```

## Testing Strategy

### 1. Validation Testing
```python
def test_missing_required_parameter():
    """Test that missing parameters cause immediate failure."""
    config = create_incomplete_config()  # Missing initial pressure

    with pytest.raises(ConfigurationError) as exc_info:
        validate_config_file(config)

    assert "initial pressure for tank 1" in str(exc_info.value)
```

### 2. Migration Testing
```python
def test_old_to_new_migration():
    """Test that old format converts correctly to new format."""
    old_config = load_old_format_config()
    adapter = ConfigurationAdapter()
    new_config = adapter.migrate_old_to_new(old_config)

    # Validate conversion is correct
    assert new_config['network']['nodes'][0]['initial_conditions']['pressure'] == old_config['geometry'][1]['initial_pressure']
```

### 3. End-to-End Testing
```python
def test_full_analysis_pipeline():
    """Test complete analysis with new format."""
    config_path = "test_configs/new_format_test.yaml"

    # Should not raise any errors
    orchestrator = SystemOrchestrator.from_config(config_path)
    results = orchestrator.run_simulation()

    # Validate results are reasonable
    assert results is not None
    assert len(results.tank_states) > 0
```

## Example Migration

### Before (Current flat format):
```yaml
geometry:
  1:
    phi: 3.0
    radius: 0.8
    initial_pressure: 100000000

tank_materials:
  1:
    liner:
      nist_path: "aluminum_6061T6_nist"

coupling_rules:
  - coupling_id: "ch2_lh2"
    participants:
      source: 1
      target: 2

output:
  plots:
    show_reference_pressures: false
```

### After (New network format):
```yaml
network:
  nodes:
    - node_id: 1
      geometry:
        phi: 3.0
        radius: 0.8
      initial_conditions:
        pressure: 100000000
      materials:
        liner:
          nist_path: "aluminum_6061T6_nist"
      plotting:
        show_reference_pressures: true  # Per-tank setting

  edges:
    - edge_id: "ch2_lh2_coupling"
      from_node: 1
      to_node: 2
      connection_type: "mission_adaptive_pressurization"
```

## Risk Mitigation

### 1. Gradual Migration
- Keep both formats working during transition
- Migrate one analysis at a time
- Maintain full test coverage

### 2. Validation at Every Step
- Strict validation prevents silent failures
- Clear error messages guide users
- Pre-flight checks catch issues early

### 3. Rollback Capability
- Keep old format support until migration complete
- Automated conversion between formats
- Easy to revert if issues discovered

## Success Criteria

### ✅ Configuration Quality
- Zero fallback values used anywhere in codebase
- Every required parameter explicitly validated
- Clear error messages for all failure modes
- No silent parameter misses

### ✅ User Experience
- Intuitive YAML structure matches system architecture
- Clear separation of concerns (tank properties vs connections)
- Easy to understand and modify configurations
- Self-documenting parameter organization

### ✅ Maintainability
- Single source of truth for each parameter
- No duplicate parameter definitions
- Easy to add new tank types or connection types
- Consistent parameter naming and organization

## Next Steps

1. **Create Configuration Adapter** - Handle format conversion
2. **Update StrictConfigValidator** - Validate new format completely
3. **Test Migration on Simple Case** - Start with single tank analysis
4. **Update Source Code Incrementally** - One module at a time
5. **Migrate Template Analysis** - Use coupled_ch2_lh2 as template
6. **Document New Schema** - Complete parameter reference guide

Would you like me to proceed with creating the Configuration Adapter next?