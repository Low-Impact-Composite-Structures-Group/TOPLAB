Audit Checklist: Changes vs origin/main

Scope: multistate branch compared to origin/main at time of audit.

Purpose: Manual review for formatting, documentation, unused code, and potential bugs.

Edited Original Files (authored by victorpoorte, modified here)
- `.gitattributes` — add git-crypt rule for triathlon CSV
- `.gitignore` — ignore plot outputs (`*.png`, `*.pgf`, `*.pdf`), `results/`, `check_results/`
- `README.md` — add multi_tank quick start, outputs, verification notes
- `analysis/__init__.py` — rename from `analysis_deprecated/__init__.py`
- `analysis/compare_dynamic_models.py` — update imports to consolidated dynamic models
- `analysis/compare_thermal_models.py` — update typing, imports, and initial/target state interfaces
- `test/tank_design/test_tank_shapes.py` — whitespace/formatting cleanups

Added Files (new in multistate vs origin/main)
- `MISSION_PARSING_SURVEY.md` — design survey for CSV mission parsing
- `analysis/check_optimized_model.py` — single-run geometry/efficiency check script
- `test/multi_tank_tests/test_nist_materials.py` — NIST materials framework tests
- `test/multi_tank_tests/test_orchestrated_framework.py` — orchestrator integration tests
- `test/multi_tank_tests/test_physics_validation.py` — physics consistency validations
- `test/multi_tank_tests/test_plotting_framework.py` — plotting configuration integration tests
- `test/multi_tank_tests/test_scenario_config.py` — ScenarioConfig parser tests
- `test/plotting/__init__.py` — new test package init
- `test/plotting/test_tank_render.py` — tank render test

Renames and Deletions (for reference; originals preserved elsewhere)
- Renamed `analysis_deprecated/...` → `analysis/...` for `compare_*` modules
- Deleted legacy `analysis/geometry/*`, `analysis/initial_conditions/*`, `analysis/insulation_thickness/*` files (migrated/obsolete)

Notes
- Multi-tank modules and verification data were added previously under `src/multi_tank/...` and `analysis/multi_tank_systems/verification/` and are already committed.
- Use this checklist to audit style, docs, and dead code; prioritize edited originals, then newly added tests/scripts.

Next Actions
- Run `pytest -q` in `python-dev` env to validate new tests.
- Review `README.md` sections for clarity and consistency.
- Confirm `.gitattributes` git-crypt rule applies only to proprietary CSV.