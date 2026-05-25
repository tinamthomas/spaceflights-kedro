<!-- SYNC IMPACT REPORT
Version Change: 0.0.0 (template) → 1.0.0 (initial ratification)
Modified Principles: N/A (initial creation)
Added Sections: Kedro Data Engineering Standards, Quality Assurance Standards
Removed Sections: N/A
Templates Updated: ✅ plan-template.md, ✅ spec-template.md, ✅ tasks-template.md
Pending Follow-ups: None
-->

# Spaceflights Constitution
## Code Quality & Data Excellence

## Core Principles

### I. Test-First Development (NON-NEGOTIABLE)
Every feature, bugfix, and refactor MUST follow Test-Driven Development (TDD):
- Tests written and reviewed first
- User acceptance criteria explicitly encoded in test cases
- Tests must FAIL before implementation begins
- Red-Green-Refactor cycle strictly enforced
- Minimum 80% code coverage required for all pipelines and nodes
- Integration tests mandatory for inter-pipeline contracts and data flows

**Rationale**: Tests document expected behavior; they prevent regressions and enable confident refactoring. In data pipelines, test-first ensures data contracts are explicit before transformation logic is written.

### II. Kedro Pipeline Architecture
All data processing MUST use Kedro's declarative pipeline pattern:
- Each transformation is a pure function (node) with explicit inputs and outputs
- Pipelines defined in YAML or Python; no ad-hoc scripts
- Data flows explicitly declared via `pipeline.py` and `catalog.yml`
- Nodes must be independently testable without external state
- No side effects: all outputs materialized through the Catalog

**Rationale**: Kedro's structure ensures reproducibility, auditability, and enables parallel execution. Pure functions simplify testing and debugging.

### III. Data Quality & Lineage Standards
All data transformations MUST maintain data integrity and traceability:
- Data validation occurs at pipeline entry points and after each transformation
- Data schema versioning required in `parameters.yml` with explicit `data_version`
- CSV/Parquet metadata (row count, column types, checksums) logged to `07_model_output/`
- Catalog entries MUST document expected schema, purpose, and data stewardship owner
- Column naming: lowercase with underscores; no spaces or special characters
- Missing value handling explicitly defined per dataset and field

**Rationale**: Data quality is non-negotiable; lineage enables reproducibility and incident investigation. Clear ownership prevents orphaned datasets.

## Code Quality Standards

### Code Style & Linting
- All Python code MUST pass `ruff` linting (configured in `pyproject.toml`)
- Type hints MANDATORY for all public functions (core pipelines, APIs)
- Docstrings for all nodes: purpose, input/output schema, example usage
- Max line length: 100 characters (enforced by ruff)
- Imports alphabetically sorted; unused imports forbidden

### Documentation
- Every pipeline MUST have a `README.md` in its folder explaining:
  - Purpose and high-level data flow
  - Input datasets and their schemas
  - Output datasets and their transformation logic
  - Known limitations or data quality caveats
- Node docstrings include parameter descriptions and return types

## Kedro Data Engineering Conventions

### Directory Structure & Naming
- Follow Kedro data layers strictly:
  - `01_raw/`: Untouched source data (immutable)
  - `02_intermediate/`: Cleaned, validated intermediate outputs
  - `03_primary/`: Aggregated, deduplicated, business-ready tables
  - `04_feature/`: Feature-engineered data for modeling
  - `05_model_input/`: Final training/testing datasets
  - `06_models/`: Trained model artifacts and metadata
  - `07_model_output/`: Predictions, metrics, reports
  - `08_reporting/`: Dashboard and BI outputs
- Node names: `verb_noun` (e.g., `clean_reviews`, `merge_companies`)
- Pipeline files: lowercase with underscores (e.g., `data_processing/pipeline.py`)

### Configuration Management
- All parameters in `conf/base/parameters.yml` with explicit types and defaults
- Environment-specific overrides in `conf/local/credentials.yml` (git-ignored)
- No hardcoded values in node functions; parameterize via `parameters_[layer_name].yml`
- Configuration schema documented with comments explaining each parameter

### Dataset Catalog
- All datasets declared in `conf/base/catalog.yml`
- Each dataset MUST include:
  - `type`: dataset format (CSVDataSet, ParquetDataSet, etc.)
  - `filepath`: location relative to data directory
  - `metadata`: JSON object with `schema` (column names/types), `owner`, `refresh_frequency`
  - `versioned: true` for outputs to enable point-in-time reproducibility
- Intermediate outputs MUST be partitioned by date if > 100MB

## Testing & Quality Assurance

### Test Organization
- Unit tests in `tests/pipelines/[layer]/test_nodes.py`
- Integration tests in `tests/pipelines/[layer]/test_pipeline.py`
- Fixture data in `tests/fixtures/` with small, representative samples
- Test naming: `test_[node_name]_[scenario]` (e.g., `test_clean_reviews_handles_missing_email`)

### Test Coverage
- New nodes require ≥ 2 test cases (happy path + edge case)
- Pipeline-level tests verify data flows between nodes (e.g., output of node A becomes input to node B)
- Data validation tests verify schema contracts (column types, null handling, uniqueness constraints)
- Run `pytest --cov=src/spaceflights` before submitting; coverage report must show ≥ 80%

### Continuous Quality Checks
- Pre-commit: ruff lint and format checks
- GitHub Actions: run full test suite on every PR
- Weekly: catalog consistency audit (all declared datasets exist and match schema)

## Development Workflow

### Pipeline Development
1. **Design**: Update `conf/base/parameters.yml` with input/output specs
2. **Catalog**: Add entries to `conf/base/catalog.yml` for new datasets
3. **Tests First**: Write failing tests in `tests/pipelines/[layer]/`
4. **Implement**: Write pure functions in `pipelines/[layer]/nodes.py`
5. **Wire**: Add nodes to `pipelines/[layer]/pipeline.py`
6. **Validate**: Ensure `kedro run [pipeline-name]` passes; verify output data quality
7. **Document**: Update pipeline `README.md` and node docstrings

### Code Review Requirements
- All changes require peer review of nodes and catalog updates
- Reviewer checklist:
  - Tests demonstrate expected behavior
  - Data schema matches catalog specification
  - Node function is pure (no external state, no side effects)
  - Docstrings complete and accurate
  - No hardcoded values; parameters properly externalized

## Governance

### Constitution Authority
This constitution is the authoritative guide for data pipeline development in Spaceflights. All code, configurations, and practices MUST comply with these principles. The constitution supersedes informal conventions, Slack discussions, and ad-hoc standards.

### Amendment Process
1. Propose amendment via pull request to `.specify/memory/constitution.md`
2. Amendment must document: rationale, affected principles, migration plan
3. Require approval from at least 2 data team members
4. Version MUST increment (MAJOR for principle removals; MINOR for additions/expansions; PATCH for clarifications)
5. Update `LAST_AMENDED_DATE` to current date and document changes in sync report

### Compliance Verification
- All PRs MUST verify compliance via GitHub Actions checks:
  - Ruff linting passes
  - Test coverage ≥ 80%
  - Catalog consistency valid
  - All nodes documented
- Code review MUST explicitly confirm constitutional compliance before merge

### Runtime Guidance
For day-to-day development questions, consult:
- Kedro official docs: https://docs.kedro.org/
- Project README for setup and execution
- Individual pipeline READMEs for specific layer documentation
- Test examples in `tests/` for implementation patterns

**Version**: 1.0.0 | **Ratified**: 2026-05-25 | **Last Amended**: 2026-05-25
