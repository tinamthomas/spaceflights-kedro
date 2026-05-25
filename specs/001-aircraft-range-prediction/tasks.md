---
description: "Task list for Aircraft Range Prediction Neural Network Pipeline"
---

# Tasks: Aircraft Range Prediction Neural Network Pipeline

**Input**: Design documents from `/specs/001-aircraft-range-prediction/`

**Prerequisites**: ✅ plan.md, ✅ spec.md, ✅ research.md, ✅ data-model.md, ✅ contracts/

**Organization**: Tasks organized by user story (US1→US3) to enable independent implementation, testing, and delivery of each story as incremental MVP.

**Constitution Compliance**: All tasks follow Spaceflights principles:
- ✅ **Test-First Development**: Tests written and reviewed BEFORE implementation (Red-Green-Refactor)
- ✅ **Kedro Pipeline Architecture**: Pure functions as nodes; declarative pipeline.py
- ✅ **Data Quality & Lineage**: Schema versioning; catalog.yml metadata
- ✅ **Code Style**: ruff linting (100 char lines); type hints; docstrings
- ✅ **Parameters**: All values externalized to conf/base/parameters_data_science.yml

## Format: `[ID] [P?] [Story] Description [TASK_TYPE]`

- **[P]**: Parallelizable (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (US1, US2, US3)
- **[TASK_TYPE]**: `[SETUP]`, `[TEST]`, `[NODE]`, `[PIPELINE]`, `[QUALITY]`
- **Exact file paths**: All deliverables include repository-relative paths

---

## Phase 1: Setup & Infrastructure

**Purpose**: Project initialization, dependencies, configuration, and foundation

### Dependency & Configuration Setup

- [ ] T001 Add PyTorch to `pyproject.toml` dependencies: `torch>=2.0,<3.0` and update `pip install -e ".[dev]"`
- [ ] T002 [P] [SETUP] Create `conf/base/parameters_data_science.yml` with training hyperparameters (learning_rate, batch_size, epochs, architecture, optimizer, loss) and uncertainty settings (n_mc_samples, ci_percentile)
- [ ] T003 [P] [SETUP] Update `conf/base/catalog.yml` with aircraft datasets: aircraft_raw, aircraft_cleaned, aircraft_train_test_split, trained_model, predictions, evaluation_metrics (includes schema metadata, owner, refresh_frequency, versioning)

### Directory & File Structure

- [ ] T004 [P] [SETUP] Create `src/spaceflights/pipelines/data_science/` directory structure with `__init__.py`, `nodes.py`, `pipeline.py`, `utils/` subdirectory
- [ ] T005 [P] [SETUP] Create `tests/pipelines/data_science/` directory with `__init__.py`, `test_nodes.py`, `test_pipeline.py`, and `tests/fixtures/sample_aircraft_data.py`
- [ ] T006 [SETUP] Create `src/spaceflights/pipelines/data_science/README.md` explaining pipeline architecture, node descriptions, data flow, input/output schemas, and references to data-model.md & contracts/

### Data Preparation

- [ ] T007 [SETUP] Prepare raw aircraft dataset in `data/01_raw/companies.csv` with required schema: aircraft_id, passenger_capacity, num_engines, crew_size, wingspace_m, length_m, range_km (100–1000 rows; validate against spec constraints)
- [ ] T008 [P] [SETUP] Create `tests/fixtures/sample_aircraft_data.py` with small representative dataset (10–20 aircraft) for unit/integration testing

---

## Phase 2: Foundational Infrastructure (Blocking Prerequisites)

**Purpose**: Core infrastructure required before ANY user story work can begin

### Utility Functions & Model Architecture

- [ ] T009 [P] [NODE] Implement model architecture utility in `src/spaceflights/pipelines/data_science/utils/model_utils.py`: `build_pytorch_model(input_features, hidden_layers, activation, batch_norm, dropout)` returns torch.nn.Module per architecture spec (5→64→32→1)
- [ ] T010 [P] [NODE] Implement `src/spaceflights/pipelines/data_science/utils/model_utils.py`: `save_model(model, metadata, path)` and `load_model(path)` for PyTorch state_dict serialization with JSON metadata

### Data Validation & Logging Infrastructure

- [ ] T011 [P] [NODE] Implement data validation utility in `src/spaceflights/pipelines/data_science/nodes.py`: `validate_input_schema(data, expected_columns, data_types)` to check column names, types, value ranges against spec (positive numerics, range 500–15000 km, etc.)
- [ ] T012 [P] [NODE] Implement logging infrastructure in nodes.py: configure logger with structured logging (model_id, batch_size, row counts, duration) for data lineage tracking

### Preprocessing Tests (TDD: RED phase)

- [ ] T013 [P] [TEST] Write failing tests in `tests/pipelines/data_science/test_nodes.py`: `test_preprocess_features_normalizes_to_unit_variance()` — verify StandardScaler centering (μ≈0) and scaling (σ≈1)
- [ ] T014 [P] [TEST] Write failing test: `test_preprocess_features_handles_missing_values()` — impute with median; check no NaN in output
- [ ] T015 [P] [TEST] Write failing test: `test_preprocess_features_split_assignment()` — verify train/val/test split: 70%/15%/15% with correct split column assignment
- [ ] T016 [TEST] Write failing test: `test_preprocess_features_deterministic_with_seed()` — same random seed → identical splits (reproducibility)

**Checkpoint**: Foundation infrastructure ready; can begin user story implementation in parallel

---

## Phase 3: User Story 1 — Train Range Prediction Model (P1) 🎯 MVP

**Goal**: Build and train a PyTorch neural network to learn aircraft range from features; save versioned model artifact with performance metadata

**Independent Test**: `kedro run --pipeline=data_science_train` produces model in `06_models/`, achieves RMSE < 15% of mean range and R² > 0.85 on held-out test set

### Tests for User Story 1 (TDD: RED phase — write failing tests FIRST)

- [ ] T017 [P] [US1] [TEST] Write failing test: `test_preprocess_features_returns_correct_schema()` in `tests/pipelines/data_science/test_nodes.py` — output has columns: aircraft_id, passenger_capacity_normalized, num_engines_normalized, crew_size_normalized, wingspace_m_normalized, length_m_normalized, split, range_km
- [ ] T018 [P] [US1] [TEST] Write failing test: `test_train_model_produces_model_artifact()` — training returns (model: torch.nn.Module, metadata: dict) with metadata keys: training_loss_history, validation_loss_history, rmse, r2, epochs, learning_rate, timestamp
- [ ] T019 [P] [US1] [TEST] Write failing test: `test_train_model_metadata_contains_normalization_params()` — metadata includes standardscaler_means, standardscaler_stds, feature_names for inference-time denormalization
- [ ] T020 [US1] [TEST] Write failing test: `test_train_model_achieves_minimum_performance()` — model R² > 0.70 on test set (loose threshold for dev; spec requires > 0.85 in production)

### Implementation for User Story 1 (TDD: GREEN phase — implement to make tests pass)

- [ ] T021 [P] [US1] [NODE] Implement `preprocess_features(data: pd.DataFrame, random_seed: int) → Tuple[pd.DataFrame, StandardScaler]` in `src/spaceflights/pipelines/data_science/nodes.py`: load aircraft data, validate schema (FR-001), handle missing values via median imputation, remove outliers (IQR method), normalize features (StandardScaler fit on data), split train/val/test (70%/15%/15%, FR-003), assign split column, return processed data + scaler
- [ ] T022 [P] [US1] [NODE] Implement `train_model(train_data: pd.DataFrame, val_data: pd.DataFrame, learning_rate: float, batch_size: int, epochs: int, random_seed: int) → Tuple[torch.nn.Module, Dict]` in nodes.py: build model architecture (FR-004: 2+ hidden layers, ReLU), convert data to tensors, train loop with Adam optimizer (FR-005: track loss history), validate each epoch, early stopping on val loss plateau, compute final RMSE/R² on val set, return model + metadata JSON (architecture, hyperparams, training history, FR-006)
- [ ] T023 [P] [US1] [NODE] Implement `save_model_with_metadata(model: torch.nn.Module, metadata: Dict, model_dir: Path) → None` in nodes.py: serialize model state_dict via torch.save(), save metadata JSON with timestamp, feature names, normalization params, training history; log versioning info
- [ ] T024 [US1] [PIPELINE] Wire nodes into pipeline in `src/spaceflights/pipelines/data_science/pipeline.py`: create `data_science_train` pipeline connecting: preprocess_features → train_model → save_model_with_metadata (depends on T021, T022, T023)
- [ ] T025 [US1] [QUALITY] Verify all nodes pass `ruff check` (100 char lines, type hints, docstrings); update docstrings with input/output schema per Constitution

### Integration Test for User Story 1

- [ ] T026 [US1] [TEST] Write integration test: `test_data_science_train_pipeline_end_to_end()` in `tests/pipelines/data_science/test_pipeline.py` — run full training pipeline on sample data; verify model artifact created at `06_models/`, model loads without error, achieves non-zero RMSE/R²
- [ ] T027 [US1] [QUALITY] Run `pytest tests/pipelines/data_science/test_nodes.py tests/pipelines/data_science/test_pipeline.py --cov=src/spaceflights/pipelines/data_science --cov-report=term-missing` — ensure ≥80% coverage for T021–T024

**Checkpoint**: User Story 1 complete and independently testable; training pipeline ready for MVP deployment

---

## Phase 4: User Story 2 — Generate Range Predictions (P1) 🎯 MVP

**Goal**: Load trained model and generate range predictions with 95% confidence intervals for new aircraft features

**Independent Test**: `kedro run --pipeline=data_science_predict` generates predictions in `07_model_output/predictions.parquet` with correct schema; CI bounds satisfy: lower < predicted < upper

### Tests for User Story 2 (TDD: RED phase)

- [ ] T028 [P] [US2] [TEST] Write failing test: `test_predict_with_confidence_output_schema()` in `tests/pipelines/data_science/test_nodes.py` — predictions DataFrame has columns: aircraft_id, predicted_range_km, range_lower_ci_km, range_upper_ci_km, prediction_confidence
- [ ] T029 [P] [US2] [TEST] Write failing test: `test_predict_with_confidence_ci_bounds_valid()` — all rows satisfy: range_lower_ci_km < predicted_range_km < range_upper_ci_km
- [ ] T030 [P] [US2] [TEST] Write failing test: `test_predict_with_confidence_handles_extrapolation()` — features outside training range are clipped with warning logged; predictions still generated (FR-010)
- [ ] T031 [US2] [TEST] Write failing test: `test_predict_on_known_aircraft()` — sample Boeing 747 data → predicted range within ±15% of actual ~13,450 km (per acceptance scenario)

### Implementation for User Story 2 (TDD: GREEN phase)

- [ ] T032 [P] [US2] [NODE] Implement `load_model_and_scaler(model_path: Path, metadata_path: Path) → Tuple[torch.nn.Module, StandardScaler, Dict]` in `src/spaceflights/pipelines/data_science/nodes.py`: load PyTorch state_dict, recreate architecture from metadata, initialize StandardScaler with saved mean/std, return model + scaler + metadata
- [ ] T033 [P] [US2] [NODE] Implement `predict_with_confidence(model: torch.nn.Module, scaler: StandardScaler, features: pd.DataFrame, n_mc_samples: int = 10, ci_percentile: float = 95) → pd.DataFrame` in nodes.py: denormalize input features using scaler, implement MC Dropout inference (enable dropout, run n_mc_samples forward passes, compute mean/std of predictions per FR-008), compute confidence interval as mean ± 1.96*std (95% CI), compute confidence score, return DataFrame with aircraft_id, predicted_range_km, lower/upper CI, confidence (FR-007)
- [ ] T034 [P] [US2] [NODE] Implement `validate_predictions(predictions: pd.DataFrame) → pd.DataFrame` in nodes.py: check schema (no missing columns), verify CI bounds (lower < predicted < upper), flag extrapolation (predicted range ±20% outside training range), log data quality issues, return flagged predictions
- [ ] T035 [US2] [PIPELINE] Wire nodes into pipeline in `src/spaceflights/pipelines/data_science/pipeline.py`: create `data_science_predict` pipeline: load_model_and_scaler + predict_with_confidence + validate_predictions → save to `07_model_output/predictions.parquet` (depends on T032, T033, T034)
- [ ] T036 [US2] [QUALITY] Verify nodes pass ruff linting; add docstrings with MC Dropout explanation, CI computation details

### Integration Test for User Story 2

- [ ] T037 [US2] [TEST] Write integration test: `test_data_science_predict_pipeline_end_to_end()` in `tests/pipelines/data_science/test_pipeline.py` — load trained model from Phase 3, run prediction on sample aircraft, verify predictions.parquet created with correct row count and schema
- [ ] T038 [US2] [QUALITY] Run pytest with coverage; ensure ≥80% for prediction nodes (T032–T034)

**Checkpoint**: User Story 2 complete; prediction pipeline independently deployable; together with US1 forms MVP (train + predict)

---

## Phase 5: User Story 3 — Evaluate & Monitor Model Performance (P2)

**Goal**: Compute performance metrics (RMSE, R², MAE) overall and segmented by aircraft type; generate evaluation reports

**Independent Test**: `kedro run --pipeline=data_science_evaluate` generates `07_model_output/evaluation_metrics.json` with RMSE, R², MAE per aircraft type

### Tests for User Story 3 (TDD: RED phase)

- [ ] T039 [P] [US3] [TEST] Write failing test: `test_evaluate_model_output_schema()` in `tests/pipelines/data_science/test_nodes.py` — metrics JSON has keys: rmse_overall, r2_overall, mae_overall, rmse_by_type, r2_by_type (dicts keyed by aircraft_type)
- [ ] T040 [P] [US3] [TEST] Write failing test: `test_evaluate_model_ci_calibration()` — confidence intervals contain actual ranges 90–95% of the time (calibration check per FR-008)
- [ ] T041 [P] [US3] [TEST] Write failing test: `test_evaluate_model_detects_poor_performance()` — identifies aircraft types with >30% higher RMSE than overall model (per acceptance scenario)
- [ ] T042 [US3] [TEST] Write failing test: `test_evaluate_model_compares_versions()` — compares current model performance to previous version (model drift detection)

### Implementation for User Story 3 (TDD: GREEN phase)

- [ ] T043 [P] [US3] [NODE] Implement `evaluate_model(predictions: pd.DataFrame, test_data: pd.DataFrame) → Dict` in `src/spaceflights/pipelines/data_science/nodes.py`: compute RMSE, R², MAE overall (FR-009), segment by aircraft_type if available, check CI calibration (% of actual ranges in CI), detect model drift vs previous version, return metrics dict with keys: rmse_overall, r2_overall, mae_overall, rmse_by_type, r2_by_type, mae_by_type, ci_calibration, drift_flag
- [ ] T044 [P] [US3] [NODE] Implement `generate_evaluation_report(metrics: Dict, predictions: pd.DataFrame) → str` in nodes.py: format metrics as markdown/HTML report highlighting: per-aircraft-type performance, poor-performing segments (>30% higher RMSE), CI calibration status, model drift alerts, save to `07_model_output/evaluation_report.md`
- [ ] T045 [US3] [PIPELINE] Wire nodes into pipeline in `src/spaceflights/pipelines/data_science/pipeline.py`: create `data_science_evaluate` pipeline: predict_with_confidence (from US2) + evaluate_model + generate_evaluation_report → save metrics.json + report (depends on T043, T044)
- [ ] T046 [US3] [QUALITY] Verify nodes pass ruff linting; add docstrings explaining per-type segmentation logic, CI calibration formula, drift detection

### Integration Test for User Story 3

- [ ] T047 [US3] [TEST] Write integration test: `test_data_science_evaluate_pipeline_end_to_end()` in `tests/pipelines/data_science/test_pipeline.py` — run full evaluation on test data from Phase 3; verify metrics.json and report generated
- [ ] T048 [US3] [QUALITY] Run pytest with coverage; ensure ≥80% for evaluation nodes (T043–T044)

**Checkpoint**: User Story 3 complete; evaluation pipeline ready for model monitoring & governance

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Code quality, documentation, CI/CD, and final validation

### Code Quality & Linting

- [ ] T049 [P] [QUALITY] Run full linting pass: `ruff check src/spaceflights/pipelines/data_science/ --line-length=100`; fix all violations (import sorting, unused imports, type hints)
- [ ] T050 [P] [QUALITY] Run pytest coverage: `pytest tests/pipelines/data_science/ --cov=src/spaceflights/pipelines/data_science --cov-report=term-missing` — ensure ≥80% coverage across all nodes and tests; add missing tests if coverage < 80%

### Documentation

- [ ] T051 [QUALITY] Ensure all nodes have complete docstrings with: purpose, input schema (types, expected values), output schema, references to data-model.md, examples of usage (in src/spaceflights/pipelines/data_science/nodes.py)
- [ ] T052 [QUALITY] Update `src/spaceflights/pipelines/data_science/README.md` with: pipeline architecture diagram (ASCII or link to data-model.md Fig), node descriptions, data flow, known limitations, references to contracts/model_io_contract.md

### Pipeline Integration & Testing

- [ ] T053 [P] [QUALITY] Verify full pipeline runs end-to-end: `kedro run --pipeline=data_science` loads data, trains model, generates predictions, evaluates; all outputs created at expected paths
- [ ] T054 [QUALITY] Test error handling: run pipeline with invalid input data (negative values, missing columns, NaN); verify graceful error messages and no crashes

### CI/CD Setup (Optional but Recommended)

- [ ] T055 [QUALITY] Create GitHub Actions workflow `.github/workflows/tests-data-science.yml` to run on PR: lint (`ruff check`), test (`pytest --cov`), coverage check (fail if <80%)
- [ ] T056 [QUALITY] Document deployment checklist in `specs/001-aircraft-range-prediction/DEPLOYMENT.md`: version model artifact, validate performance metrics, backup previous model, update catalog versioning, trigger GitHub Actions

### Final Validation

- [ ] T057 [QUALITY] Constitution compliance audit: verify all code meets Constitution requirements (test coverage, Kedro patterns, linting, documentation, parameter externalization)
- [ ] T058 [QUALITY] Success criteria verification (from spec): 
  - SC-001: RMSE < 15% ✅ [defer to production data]
  - SC-002: R² > 0.85 ✅ [defer to production data]
  - SC-003: Latency < 100ms ✅ [verify on batch]
  - SC-004: 95% success rate ✅ [error handling validation]
  - SC-007: ≥80% coverage ✅ [pytest report]

**Checkpoint**: All user stories complete; codebase production-ready; Constitution compliant; CI/CD configured

---

## Dependency Completion Order

### Critical Path (Blocking Dependencies)

```
T001 (PyTorch deps) 
  ↓
T002 (parameters) + T003 (catalog)
  ↓
T004 (directories) + T007 (data prep)
  ↓
T009–T012 (foundational utilities)
  ↓
US1: T013–T027 (training pipeline)
  ↓
US2: T028–T038 (prediction pipeline)
  ↓
US3: T039–T048 (evaluation pipeline)
  ↓
T049–T058 (polish & validation)
```

### Parallelizable Task Groups

**Setup Phase** (T001–T008): All can run in parallel
**Foundational Tests** (T013–T016): All can run in parallel
**US1 Implementation** (T021–T023): Can run in parallel after T013–T016 pass
**US2 Implementation** (T032–T034): Can run in parallel after T028–T031 pass (independent of US1)
**US3 Implementation** (T043–T044): Can run in parallel after T039–T042 pass (independent of US1, US2)
**Quality** (T049–T050): Can run in parallel after all implementation complete

---

## Expected Timeline & MVP Delivery

### Phase 1: Setup (1–2 days)
- T001–T008: Dependencies, configuration, raw data preparation
- Prerequisite for all downstream work

### Phase 2: Foundational (1–2 days)
- T009–T016: Utilities, validation, tests
- Unblocks all user story work

### Phase 3: US1 Training (2–3 days)
- T017–T027: Tests + implementation + integration
- **MVP Milestone 1**: Can train models ✅

### Phase 4: US2 Prediction (2–3 days)
- T028–T038: Tests + implementation + integration
- **MVP Milestone 2**: Can generate predictions ✅

### Phase 5: US3 Evaluation (1–2 days)
- T039–T048: Tests + implementation + integration
- **Milestone 3**: Model governance ready ✅

### Phase 6: Polish (1–2 days)
- T049–T058: Linting, docs, CI/CD, validation
- **Production Ready**: Constitution compliant ✅

**Total: ~8–12 days** for full implementation with 80%+ code coverage

---

## Task Status Tracking

**Legend**:
- [ ] Not started
- [P] Parallelizable
- [SETUP] / [TEST] / [NODE] / [PIPELINE] / [QUALITY]: Task type

**To use**: Copy this file locally; check boxes as tasks complete; maintain git history of task progression

---

## References

- **Spec**: specs/001-aircraft-range-prediction/spec.md
- **Plan**: specs/001-aircraft-range-prediction/plan.md
- **Research**: specs/001-aircraft-range-prediction/research.md (PyTorch + Kedro integration details)
- **Data Model**: specs/001-aircraft-range-prediction/data-model.md (entities, validation rules)
- **Contract**: specs/001-aircraft-range-prediction/contracts/model_io_contract.md (input/output schema)
- **Quickstart**: specs/001-aircraft-range-prediction/quickstart.md (setup guide)
- **Constitution**: .specify/memory/constitution.md (code quality standards)

