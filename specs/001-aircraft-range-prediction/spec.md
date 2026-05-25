# Feature Specification: Aircraft Range Prediction Neural Network Pipeline

**Feature Branch**: `001-aircraft-range-prediction`

**Created**: 2026-05-25

**Status**: Draft

**Input**: User description: "I would like to build a pipeline that uses neural networks to predict aircraft range based on features such as passenger capacity, engines, crew, wingspace, length."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Range Prediction Model (Priority: P1)

A data scientist needs to train a neural network model that learns the relationship between aircraft characteristics (passenger capacity, number of engines, crew size, wingspace, length) and maximum range. The trained model should be saved and versioned for reproducible predictions.

**Why this priority**: Core MVP requirement - model training is the foundation for all downstream predictions. Must be completed before any inference pipeline can be deployed.

**Independent Test**: Can be fully tested by running the training pipeline in isolation (`kedro run --pipeline=data_science` in the training phase), verifying that a trained model artifact is produced, and ensuring model achieves acceptable accuracy on validation data.

**Acceptance Scenarios**:

1. **Given** cleaned aircraft feature data with known range values, **When** training pipeline executes, **Then** a trained neural network model is saved to `06_models/` with model metadata (architecture, training date, performance metrics)
2. **Given** training completes successfully, **When** model is evaluated on held-out test set, **Then** predictions have RMSE < 15% of range values and R² > 0.85
3. **Given** model training completes, **When** model is saved, **Then** version is automatically incremented in `06_models/model_metadata.json` with training timestamp and hyperparameters

---

### User Story 2 - Generate Range Predictions (Priority: P1)

An analyst or system needs to use the trained model to predict aircraft range for new aircraft specifications. The pipeline takes aircraft features (capacity, engines, crew, wingspace, length) and outputs predicted range values with confidence intervals.

**Why this priority**: Core MVP - directly delivers user value. Enables downstream use cases (fleet planning, performance reporting). Can run independently after model is trained.

**Independent Test**: Can be fully tested by invoking the prediction pipeline (`kedro run --pipeline=data_science_predict`) on sample aircraft features, verifying output contains predicted ranges with associated confidence estimates, and spot-checking predictions against known aircraft data.

**Acceptance Scenarios**:

1. **Given** a trained neural network model and new aircraft feature data, **When** prediction pipeline executes, **Then** output contains predicted range values for each aircraft in `07_model_output/predictions.parquet`
2. **Given** predictions are generated, **When** output is examined, **Then** each prediction includes a confidence interval (e.g., 95% CI) reflecting model uncertainty
3. **Given** sample aircraft features for known aircraft (e.g., Boeing 747), **When** predictions are generated, **Then** predicted range falls within ±10% of actual specifications

---

### User Story 3 - Evaluate & Monitor Model Performance (Priority: P2)

A data scientist needs to assess model performance across different aircraft types and feature ranges, identify potential prediction errors, and generate reports for model monitoring and governance.

**Why this priority**: Enables operational confidence and model maintenance. Important for production deployment but not blocking initial model delivery.

**Independent Test**: Can be fully tested by running evaluation pipeline on test data (`kedro run --pipeline=data_science_evaluate`), verifying that performance metrics by aircraft type are generated, and checking that evaluation reports are created in `07_model_output/`.

**Acceptance Scenarios**:

1. **Given** trained model and test dataset, **When** evaluation pipeline executes, **Then** generates performance metrics (RMSE, R², MAE) overall and segmented by aircraft type in `07_model_output/evaluation_metrics.json`
2. **Given** evaluation completes, **When** performance report is reviewed, **Then** identifies aircraft features or ranges where model performs poorly (e.g., very large aircraft)
3. **Given** trained model is evaluated, **When** model drift is checked, **Then** previous model versions' performance is compared to detect degradation

---

### Edge Cases

- What happens when prediction is requested for aircraft with feature values outside training data range (extrapolation)?
- How does model handle missing or invalid feature values (e.g., negative engines or zero crew)?
- What is the performance impact of predictions on very large aircraft (e.g., Airbus A380) vs. small aircraft?
- How does model perform on aircraft from different manufacturers or design eras not well-represented in training data?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Pipeline MUST load aircraft feature data (passenger capacity, engines, crew, wingspace, length) from `01_raw/companies.csv` or dedicated aircraft dataset
- **FR-002**: Pipeline MUST preprocess features (handle missing values, normalize/scale numeric ranges, remove outliers)
- **FR-003**: Pipeline MUST split data into training (70%), validation (15%), and test (15%) sets with reproducible random seed
- **FR-004**: Neural network model MUST have at least 2 hidden layers with ReLU activation and input features: [passenger_capacity, num_engines, crew_size, wingspace, length]
- **FR-005**: Training pipeline MUST track loss and validation metrics during training and save loss history to model artifacts
- **FR-006**: Trained model MUST be saved as versioned artifact in `06_models/` with serialized weights, architecture, and metadata
- **FR-007**: Prediction pipeline MUST load trained model and generate range predictions for input aircraft features
- **FR-008**: Predictions MUST include confidence intervals (95% CI) reflecting model uncertainty
- **FR-009**: Evaluation pipeline MUST compute RMSE, R², MAE, and segmented performance metrics (by aircraft type if available)
- **FR-010**: Model MUST handle prediction requests for feature values outside training range gracefully (warn but predict; no crashes)
- **FR-011**: All pipelines MUST log data lineage: input dataset versions, model versions, feature statistics, and data quality checks

### Key Entities

- **Aircraft Features**: Numeric properties describing aircraft (passenger_capacity, num_engines, crew_size, wingspace_m, length_m)
- **Range (Target)**: Maximum range in kilometers or miles; continuous numeric value
- **Trained Neural Network Model**: Serialized model with weights, architecture (layers, units, activations), hyperparameters, training metadata
- **Predictions**: Output dataset with aircraft features + predicted range + confidence intervals + residuals (for evaluation)
- **Model Metadata**: Version, training date, hyperparameters, training/validation loss, test performance metrics

### Data Quality & Schema *(mandatory for pipeline features)*

**Input Data Requirements**:
- **Source Dataset**: `01_raw/companies.csv` (or dedicated aircraft specifications dataset to be sourced)
  - **Schema**: `aircraft_id (string), passenger_capacity (int), num_engines (int), crew_size (int), wingspace_m (float), length_m (float), range_km (float)`
  - **Validation**: 
    - All numeric fields must be positive (no negative engines/crew/dimensions)
    - Passenger capacity 10–1000 (reasonable for commercial aircraft)
    - Range 500–15,000 km (reasonable for commercial/cargo aircraft)
    - Missing values in features: < 5%; missing values in range (target): none for training set
    - No duplicates on aircraft_id
  - **Expected Volume**: 100–1000 unique aircraft

**Processing Stages**:
- **02_intermediate/aircraft_cleaned.csv**: Cleaned features, missing values imputed, outliers flagged
- **03_primary/aircraft_features_normalized.parquet**: Normalized features (0–1 scale), train/val/test split metadata
- **04_feature/aircraft_training_set.parquet**: Features + target (range) for model training
- **05_model_input/aircraft_train_test_split.parquet**: Final train/val/test data with fold assignments

**Output Data Specification**:
- **Target Layer**: `06_models/` (trained model artifact) + `07_model_output/` (predictions & metrics)
- **Model Output Schema**: 
  - `aircraft_id (string), predicted_range_km (float), range_lower_ci_km (float), range_upper_ci_km (float), feature_passenger_capacity (int), feature_num_engines (int), ...`
  - Confidence intervals: 95% CI based on model ensemble uncertainty or quantile regression
  - All predictions >= 0
- **Metrics Output Schema**:
  - `metric_name (string), value (float), split (string), aircraft_type (string)` for segmented metrics
  - Metrics include: RMSE, R², MAE, MAPE; computed for test set overall and by aircraft type (if available)
- **Data Quality Checks Post-Prediction**:
  - No null predicted ranges
  - All predicted ranges within ±20% of training data range (flag extrapolations)
  - Confidence interval lower bound < predicted range < upper bound
- **Versioning**: Model versioned by training timestamp; predictions versioned by run ID

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Trained neural network achieves RMSE < 15% of mean training range (e.g., if mean range is 5,000 km, RMSE < 750 km) on held-out test set
- **SC-002**: Model achieves R² > 0.85 on test set, indicating >85% of variance in aircraft range explained by features
- **SC-003**: Prediction latency < 100ms per aircraft on standard hardware (< 1000 aircraft/second throughput)
- **SC-004**: Prediction pipeline successfully generates predictions for ≥95% of input aircraft without errors
- **SC-005**: Model performs consistently across aircraft types: no aircraft type has >30% higher RMSE than overall model
- **SC-006**: Confidence intervals contain actual range values 90–95% of the time (calibration check)
- **SC-007**: Pipeline code coverage ≥80% with unit tests for preprocessing, model training, and prediction steps
- **SC-008**: Model training completes in <5 minutes on sample dataset (1000 aircraft); full training on production data completes in <1 hour

## Assumptions

- **Data Availability**: Aircraft feature data (passenger capacity, engines, crew, wingspace, length, range) is available in `01_raw/` or will be sourced externally before implementation starts
- **Feature Relationships**: Assumption that aircraft range is a continuous function of the provided features (no significant discrete jumps or non-monotonic relationships)
- **Model Scope**: Neural network focuses on aircraft specifications; external factors (fuel type, aerodynamics standards, era) are not explicitly modeled but assumed reflected in training data
- **Training Data Representativeness**: Training data represents diverse aircraft types and design eras; model not expected to generalize far beyond known aircraft families
- **Reuse of Existing Pipeline**: Data processing pipeline (01_raw → 03_primary) will reuse existing Kedro structure; only new nodes for feature engineering and model training are added
- **Scikit-learn/TensorFlow**: Implementation will use TensorFlow/Keras or scikit-learn MLPRegressor (already in dependencies or will be added); no custom neural network implementation from scratch
- **Hyperparameter Tuning**: Initial model uses reasonable defaults (2 hidden layers, 64–128 units, Adam optimizer, MSE loss); hyperparameter optimization is out of scope for v1
- **Deployment Scope**: v1 focuses on model training and batch prediction; real-time API serving is out of scope
