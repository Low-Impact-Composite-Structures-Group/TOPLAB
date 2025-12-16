# Repository Architecture

**Last Updated**: December 16, 2025
**Multi-State Author**: Dante Raso
**Original Author**: Victor Poorte

---

## Overview

This repository contains **two complementary hydrogen fuel tank analysis systems**:

1. **Original System** (`src/`, `analysis/`) - Victor Poorte's single-tank transient analysis framework
2. **Multi-State System** (`src/multi_tank/`, `analysis/multi_tank_systems/`) - Dante Raso's multi-tank coupled system framework

Both systems coexist in the same repository with **minimal coupling** through a well-defined API contract.

---

## Directory Structure

```
hydrogen_fuel_tank/
├── src/
│   ├── dynamics/              # Original: Single-tank dynamics
│   ├── fluids/                # Original: Hydrogen properties, flow physics
│   ├── materials/             # Original: Material definitions
│   │   └── materials_for_multi_tank/  # Multi-state: Table-based materials
│   ├── mission/               # Original: Mission framework base classes
│   ├── multistep_methods/     # Original: ODE solver wrappers
│   ├── tank_design/           # Original: Tank geometry and shapes
│   ├── thermodynamics/        # Original: Thermal models and tank states
│   │
│   └── multi_tank/            # Multi-state: Coupled tank system (NEW)
│       ├── configuration/     # YAML scenario configuration system
│       ├── coupling/          # Inter-tank coupling (valves, heat exchangers)
│       ├── dynamics/          # Isochoric dynamic models
│       ├── fluids/            # Flow physics for multi-tank
│       ├── materials/         # NIST material properties (polynomial-based)
│       ├── missions/          # Mission profiles with CSV loading
│       ├── orchestration/     # System-level orchestrator
│       ├── plotting/          # Visualization tools (Seaborn styling)
│       ├── solver/            # Scipy solver wrappers
│       ├── system/            # TankSystem and state management core
│       ├── thermodynamics/    # Isochoric thermal models
│       └── utilities/         # Tank geometry utilities
│
├── analysis/
│   ├── *.py                   # Original: Single-tank analysis scripts
│   │
│   └── multi_tank_systems/    # Multi-state: Coupled system analyses (NEW)
│       ├── coupled_ch2_cch2/  # Compressed H2 → Cold compressed H2
│       ├── coupled_ch2_lh2/   # Compressed H2 → Liquid H2 (feedforward)
│       ├── single_tank_cch2/  # Single cold compressed H2
│       ├── single_tank_ch2/   # Single compressed H2
│       ├── single_tank_slh2/  # Single subcooled liquid H2
│       └── verification/      # Reference data (Stops et al. 2024)
│
├── test/
│   ├── dynamics/, fluids/, ... # Original: Unit tests for original system
│   └── multi_tank_tests/       # Multi-state: Unit tests for multi_tank
│
├── facades/                   # Analysis facades (legacy, not used)
├── optimization/              # Optimization studies (original system)
└── plotting/                  # Original plotting utilities
```

---

## System Separation

### What Belongs Where?

| Component | Original System | Multi-State System |
|-----------|----------------|-------------------|
| **Purpose** | Single-tank transient thermal analysis | Multi-tank coupled system with inter-tank flows |
| **Source Code** | `src/*` (excluding `src/multi_tank/`) | `src/multi_tank/*` |
| **Analysis Scripts** | `analysis/*.py` | `analysis/multi_tank_systems/*` |
| **Tests** | `test/dynamics/`, `test/fluids/`, etc. | `test/multi_tank_tests/` |
| **Material System** | `src.materials.materials` (base classes) | `src.multi_tank.materials.nist_materials` (polynomial)<br>`src.materials.materials_for_multi_tank` (table-based) |
| **Dynamic Models** | `src.dynamics.dynamic_models` | `src.multi_tank.dynamics.isochoric_dynamic_models` |
| **Missions** | `src.mission.mission` | `src.multi_tank.missions.isochoric_missions` |
| **Configuration** | Code-based | YAML-based (`ScenarioConfig`) |

---

## API Contract: Multi-State → Original Dependencies

The multi-state system depends on **8 original modules** for fundamental shared infrastructure. These constitute the **stable API contract** between the two systems.

### Core Dependencies

| Original Module | What Multi-State Uses | Why Shared |
|----------------|----------------------|------------|
| **`src.thermodynamics.tank_states`** | `IsochoricTankState`, `IsochoricTankStates` | Core state representation used throughout codebase |
| **`src.tank_design.tank_shapes`** | `SphericalTank` | Tank geometry and volume calculations |
| **`src.materials.materials`** | `Material`, `Metal`, `Composite` | Base classes for material property definitions |
| **`src.fluids.convective_mediums`** | `IsochoricHydrogen` | Hydrogen convective heat transfer |
| **`src.fluids.hydrogen_retrievers`** | `IsochoricHydrogenRequester` | CoolProp wrapper for hydrogen properties |
| **`src.mission.mission`** | `Mission` base class | Mission framework foundation |
| **`src.mission.mission_sections`** | `OutFlow`, `InFlow` | Mission section definitions |
| **`src.multistep_methods.linear_multistep_methods`** | `ScipyMethod` | ODE solver interface |

### Additional Utilities

| Original Module | What Multi-State Uses | Purpose |
|----------------|----------------------|---------|
| **`src.fluids.coolprop_safe`** | `safe_pressure_from_T_rho`, `safe_enthalpy` | Safe CoolProp calls with error handling |
| **`src.dynamics.cryopump_model`** | `CryoPumpModel` | Cryogenic pump modeling (used by `dynamic_analysis.py`) |

---

## API Stability Contract

### ✅ Changes to These Modules Are SAFE (Multi-State Will NOT Break):
- Adding new classes/functions (backwards compatible)
- Internal implementation changes that preserve public API
- Performance improvements
- Bug fixes that don't change behavior

### ⚠️ Changes Requiring Coordination:
- Renaming public classes/functions used by multi_tank
- Changing function signatures (parameters, return types)
- Removing classes/methods
- Changing behavior of `IsochoricTankState`, `SphericalTank`, material base classes

### 🔴 Changes That Would Break Multi-State:
- Deleting `IsochoricTankState`, `IsochoricTankStates`
- Removing `SphericalTank.compute_volume()` or geometry methods
- Changing `Material`, `Metal`, `Composite` base class interface
- Modifying `Mission` or `MissionSection` core behavior

---

## Material System Architecture

Two material systems coexist:

### 1. Original Material System (Victor's)
**Location**: `src.materials.materials`
```python
from src.materials.materials import Material, Metal, Composite
```
- **Purpose**: Base classes for material definitions
- **Used by**: Original system + Multi-state (base classes only)
- **Properties**: Abstract interface for cp, k, rho

### 2. Multi-State Material System - Polynomial (Dante's)
**Location**: `src.multi_tank.materials.nist_materials`
```python
from src.multi_tank.materials.nist_materials import NISTMetal, NISTComposite
```
- **Purpose**: NIST-based materials with polynomial temperature-dependent properties
- **Used by**: Multi-state thermal models
- **Materials**: Aluminum 5083, Aluminum 3003F, Aluminum 6061-T6, G10

### 3. Multi-State Material System - Table-Based (Dante's)
**Location**: `src.materials.materials_for_multi_tank.nist_material`
```python
from src.materials.materials_for_multi_tank.nist_material import NISTMaterial
```
- **Purpose**: CSV table-based materials for configuration system
- **Used by**: `ScenarioConfig` YAML loader
- **Materials**: Aluminum 6061-T6 (tabulated), Carbon/Epoxy, G10

**Why Two Multi-State Material Systems?**
- **Polynomial**: Used directly in thermal models for fast evaluation
- **Table-based**: Used by configuration system for flexibility and easy material swapping via YAML

---

## Development Guidelines

### For Original System (Victor's Code)
- Files in `src/*` (excluding `src/multi_tank/`)
- Analysis scripts in `analysis/*.py` (excluding `multi_tank_systems/`)
- Tests in `test/*` (excluding `test/multi_tank_tests/`)
- **Do not** modify files in `src/multi_tank/`
- **Coordinate** changes to shared API modules (see API Contract)

### For Multi-State System (Dante's Code)
- Files in `src/multi_tank/`
- Analysis scripts in `analysis/multi_tank_systems/`
- Tests in `test/multi_tank_tests/`
- **Do not** modify original system files without coordination
- **Rely on** stable API from original modules

### Adding New Features

**To Original System:**
1. Add code to `src/` subdirectories
2. Add analysis scripts to `analysis/`
3. Add tests to `test/`
4. Update original documentation

**To Multi-State System:**
1. Add code to `src/multi_tank/` subdirectories
2. Add analysis scripts to `analysis/multi_tank_systems/`
3. Add tests to `test/multi_tank_tests/`
4. Update this ARCHITECTURE.md if API contract changes

---

## Testing Strategy

### Original System Tests
**Location**: `test/dynamics/`, `test/fluids/`, `test/materials/`, etc.
- Run: `pytest test/ -k "not multi_tank"`
- Must pass before merging to main
- Ensures Victor's system remains functional

### Multi-State System Tests
**Location**: `test/multi_tank_tests/`
- Run: `pytest test/multi_tank_tests/`
- Includes benchmark problems, coupling tests, physics validation
- Tests multi-tank-specific functionality

### Integration Testing
- Run multi_tank analyses: `analysis/multi_tank_systems/*/driver_*.py`
- Verify outputs match verification data
- Check plots are generated correctly

---

## Merge Strategy: multistate → main

### Pre-Merge Checklist
- [ ] All original tests pass
- [ ] Multi-state tests pass
- [ ] Multi-state analyses run successfully
- [ ] Documentation updated (README, ARCHITECTURE)
- [ ] No duplicate files between `src/` and `src/multi_tank/`
- [ ] API contract documented and stable
- [ ] Code reviewed

### Post-Merge Workflow
1. Both systems coexist in `main` branch
2. Original system continues independent development
3. Multi-state system continues independent development
4. Changes to shared API require coordination
5. Both test suites must pass

---

## Key Design Decisions

### Why Not Refactor Into Single System?
- **Different use cases**: Single-tank vs coupled multi-tank
- **Minimal overlap**: ~90% of code is system-specific
- **Preservation**: Victor's system must remain functional
- **Maintainability**: Clear separation easier to maintain than merged monolith

### Why Share Tank States and Geometry?
- **Fundamental**: These represent physical reality, not implementation
- **Duplication waste**: Would require maintaining two identical implementations
- **Small interface**: Limited API surface reduces coupling

### Why Two Material Systems in Multi-State?
- **Configuration flexibility**: Table-based allows YAML material swapping
- **Performance**: Polynomial-based faster for repeated thermal model calls
- **Future-proofing**: Can switch between them transparently

---

## Future Considerations

### If Original System Evolves
- New features can be added to `src/*` without affecting multi_tank
- API changes to shared modules require coordination
- Consider creating `src/common/` for truly shared infrastructure if coupling increases

### If Multi-State System Expands
- Keep all new code in `src/multi_tank/`
- Add new analyses to `analysis/multi_tank_systems/`
- Consider creating sub-packages if modules grow large

### If Systems Need to Converge
- Move shared code to `src/core/` or `src/common/`
- Establish clear API versioning
- Gradual refactoring with comprehensive tests

---

## References

### Multi-State System Publications
- Stops et al. (2024), "Experimental investigation of liquid hydrogen tank filling", *Cryogenics*, DOI in `analysis/multi_tank_systems/verification/README.md`

### Original System
- Victor Poorte's thesis and publications (reference TBD)

---

## Contact

- **Original System**: Victor Poorte
- **Multi-State System**: Dante Raso
- **Questions about API Contract**: Coordinate between both authors
