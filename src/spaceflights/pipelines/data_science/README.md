# Data Science Pipeline: Aircraft Range Prediction

## Overview

This pipeline implements a PyTorch-based neural network to predict aircraft range based on aircraft characteristics. The pipeline comprises three independent phases: (1) data preprocessing and normalization, (2) model training with PyTorch, and (3) batch prediction and evaluation.

**Related Documents**:
- [Feature Specification](../../../../specs/001-aircraft-range-prediction/spec.md)
- [Implementation Plan](../../../../specs/001-aircraft-range-prediction/plan.md)
- [Data Model](../../../../specs/001-aircraft-range-prediction/data-model.md)
- [Model I/O Contract](../../../../specs/001-aircraft-range-prediction/contracts/model_io_contract.md)
- [Quickstart Guide](../../../../specs/001-aircraft-range-prediction/quickstart.md)

## Architecture

### Data Flow (8-Layer Structure)

```
01_raw
  ↓ aircraft_raw (raw input: aircraft_id, features, range_km)
  ├─→ [preprocess_features node]
  
02_intermediate
  ↓ aircraft_cleaned (after: missing value imputation, outlier removal)
  
03_primary
  ↓ aircraft_features_normalized (after: StandardScaler normalization)
  
05_model_input
  ↓ aircraft_train_test_split (train/val/test split: 70/15/15)
  ├─→ [train_model node] (training phase)
  ├─→ [predict_with_confidence node] (prediction phase)
  
06_models
  ↓ trained_model (trained PyTorch model + metadata)
  
07_model_output
  ├─ predictions (predicted_range_km + 95% CI + confidence)
  ├─ evaluation_metrics (RMSE, R², MAE by aircraft type)
  └─ evaluation_report (markdown report with insights)

08_reporting
  (downstream visualization/dashboards)
```

### Pipelines

#### 1. **data_science_train** (User Story 1)
Trains a PyTorch neural network on training data and saves the model.

```
preprocess_features
    ↓
train_model
    ↓
save_model_with_metadata
```

**Inputs**:
- `aircraft_raw`: Raw aircraft dataset (CSV)
- `parameters.training`: Hyperparameters (learning_rate, batch_size, epochs, etc.)
- `parameters.preprocessing`: Preprocessing config (normalization, outlier removal)

**Outputs**:
- `trained_model`: Serialized PyTorch model + metadata JSON
- Dataset progression: 01_raw → 02_intermediate → 03_primary → 05_model_input → 06_models

**Command**: `kedro run --pipeline=data_science_train`

#### 2. **data_science_predict** (User Story 2)
Loads the trained model and generates predictions with 95% confidence intervals for new aircraft.

```
load_model_and_scaler
    ↓
predict_with_confidence
    ↓
validate_predictions
```

**Inputs**:
- `trained_model`: Serialized PyTorch model + metadata
- `aircraft_train_test_split`: Test set (contains test split rows)
- `parameters.prediction`: MC Dropout config (n_mc_samples, ci_percentile)

**Outputs**:
- `predictions`: Predictions with lower/upper 95% CI, confidence score

**Command**: `kedro run --pipeline=data_science_predict`

#### 3. **data_science_evaluate** (User Story 3)
Computes performance metrics and generates evaluation reports.

```
evaluate_model
    ↓
generate_evaluation_report
```

**Inputs**:
- `predictions`: Predictions from prediction pipeline
- `aircraft_train_test_split`: Test set labels
- `parameters.evaluation`: Metrics config

**Outputs**:
- `evaluation_metrics`: JSON with RMSE, R², MAE (overall + by aircraft type)
- `evaluation_report`: Markdown report with insights

**Command**: `kedro run --pipeline=data_science_evaluate`

#### Full Pipeline
Run all three pipelines sequentially:

```
kedro run --pipeline=data_science
```

This executes: train → predict → evaluate.

## Node Reference

### Preprocessing Nodes

#### `preprocess_features(data, random_seed)`
**Purpose**: Load, clean, normalize, and split aircraft data.

**Inputs**:
- `data` (DataFrame): Raw aircraft data with columns: aircraft_id, passenger_capacity, num_engines, crew_size, wingspace_m, length_m, range_km
- `random_seed` (int): Random seed for reproducibility

**Processing**:
1. Validate schema (expected columns, numeric types)
2. Impute missing values (median strategy)
3. Remove outliers (IQR method: Q3 + 1.5*IQR)
4. Normalize features (StandardScaler: μ≈0, σ≈1)
5. Split data (70% train, 15% val, 15% test)
6. Assign split labels to each row

**Outputs**:
- DataFrame with columns: aircraft_id, [5 normalized features], range_km, split
- StandardScaler object (for inference-time denormalization)

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:preprocess_features()`

### Training Nodes

#### `train_model(train_data, val_data, learning_rate, batch_size, epochs, random_seed)`
**Purpose**: Train PyTorch neural network to predict aircraft range.

**Inputs**:
- `train_data` (DataFrame): Training set
- `val_data` (DataFrame): Validation set (for early stopping)
- `learning_rate` (float): Adam optimizer learning rate
- `batch_size` (int): Training batch size
- `epochs` (int): Maximum number of training epochs
- `random_seed` (int): For reproducibility

**Architecture**:
- Input layer: 5 features (normalized)
- Hidden layer 1: 64 units + ReLU + BatchNorm + Dropout(0.2)
- Hidden layer 2: 32 units + ReLU + BatchNorm
- Output layer: 1 unit (range_km prediction)
- Loss: MSE
- Optimizer: Adam

**Outputs**:
- Trained `torch.nn.Module` model
- Metadata dict: training_loss_history, validation_loss_history, rmse, r2, epochs_trained, learning_rate, timestamp, feature_names, normalization_params

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:train_model()`

#### `save_model_with_metadata(model, metadata, model_dir)`
**Purpose**: Serialize trained PyTorch model and metadata for reproducibility and versioning.

**Inputs**:
- `model` (torch.nn.Module): Trained model
- `metadata` (dict): Training metadata
- `model_dir` (Path): Output directory

**Processing**:
1. Generate model ID: `aircraft_range_v1_[TIMESTAMP]`
2. Save model state_dict: `model_dir/aircraft_range_model.pt`
3. Save metadata JSON: `model_dir/aircraft_range_model_metadata.json`
4. Backup previous model version (if exists)
5. Log versioning info

**Outputs**:
- PyTorch model file (.pt)
- Metadata JSON file

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:save_model_with_metadata()`

### Prediction Nodes

#### `load_model_and_scaler(model_path, metadata_path)`
**Purpose**: Load trained model and scaler for inference.

**Inputs**:
- `model_path` (Path): Path to saved model file
- `metadata_path` (Path): Path to saved metadata JSON

**Processing**:
1. Load PyTorch state_dict
2. Recreate architecture from metadata
3. Load StandardScaler (fitted mean/std from training)

**Outputs**:
- torch.nn.Module model
- StandardScaler object
- Metadata dict

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:load_model_and_scaler()`

#### `predict_with_confidence(model, scaler, features, n_mc_samples=10, ci_percentile=95)`
**Purpose**: Generate predictions with 95% confidence intervals using Monte Carlo Dropout.

**Inputs**:
- `model` (torch.nn.Module): Trained model with Dropout layers
- `scaler` (StandardScaler): Fitted scaler for denormalization
- `features` (DataFrame): Test features (normalized)
- `n_mc_samples` (int): Number of MC Dropout forward passes
- `ci_percentile` (float): Confidence level (95 = 95% CI)

**Processing** (MC Dropout Uncertainty):
1. For each aircraft:
   - Run model forward pass n_mc_samples times (with dropout enabled)
   - Collect predictions: [pred_1, pred_2, ..., pred_n]
   - Compute mean: predicted_range_km = mean(predictions)
   - Compute std: uncertainty = std(predictions)
   - Compute CI: lower = mean - 1.96*std, upper = mean + 1.96*std
   - Compute confidence: 1 / (1 + uncertainty)

2. Denormalize predictions using scaler

**Outputs**:
- DataFrame with columns: aircraft_id, predicted_range_km, range_lower_ci_km, range_upper_ci_km, prediction_confidence

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:predict_with_confidence()`

#### `validate_predictions(predictions)`
**Purpose**: Validate prediction schema and data quality.

**Inputs**:
- `predictions` (DataFrame): Predictions from predict_with_confidence

**Validation**:
1. Check schema (all required columns present, correct types)
2. Verify CI bounds: lower < predicted < upper (for all rows)
3. Flag extrapolation (predicted outside training range ±20%)
4. Log quality issues
5. Return flagged predictions

**Outputs**:
- DataFrame with validation flags

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:validate_predictions()`

### Evaluation Nodes

#### `evaluate_model(predictions, test_data)`
**Purpose**: Compute performance metrics (RMSE, R², MAE) overall and segmented by aircraft type.

**Inputs**:
- `predictions` (DataFrame): Predictions with confidence intervals
- `test_data` (DataFrame): Ground truth labels

**Metrics**:
- **Overall**: RMSE, R², MAE on full test set
- **Per Aircraft Type**: RMSE, R², MAE segmented by aircraft_type (if available)
- **CI Calibration**: % of actual values within predicted CI (target: 90–95%)
- **Model Drift**: Compare to previous model (if available)

**Outputs**:
- Metrics dict: rmse_overall, r2_overall, mae_overall, rmse_by_type, r2_by_type, mae_by_type, ci_calibration, drift_flag

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:evaluate_model()`

#### `generate_evaluation_report(metrics, predictions)`
**Purpose**: Format metrics as markdown report with insights and alerts.

**Inputs**:
- `metrics` (dict): Evaluation metrics
- `predictions` (DataFrame): Predictions with actual values

**Report Contents**:
1. Overall performance (RMSE, R², MAE)
2. Per-aircraft-type breakdown
3. Poor-performing segments (>30% higher RMSE)
4. CI calibration status
5. Model drift alerts
6. Recommendations

**Outputs**:
- Markdown report saved to `data/07_model_output/evaluation_report.md`

**Implementation**: `src/spaceflights/pipelines/data_science/nodes.py:generate_evaluation_report()`

## Testing

All nodes have corresponding unit and integration tests following TDD (Test-Driven Development):

**Unit Tests** (`tests/pipelines/data_science/test_nodes.py`):
- `test_preprocess_features_*`: Data preprocessing validation
- `test_train_model_*`: Model training validation
- `test_predict_with_confidence_*`: Prediction validation
- `test_evaluate_model_*`: Evaluation validation

**Integration Tests** (`tests/pipelines/data_science/test_pipeline.py`):
- `test_data_science_train_pipeline_end_to_end`: Full training pipeline
- `test_data_science_predict_pipeline_end_to_end`: Full prediction pipeline
- `test_data_science_evaluate_pipeline_end_to_end`: Full evaluation pipeline

**Run Tests**:
```bash
# Unit tests
pytest tests/pipelines/data_science/test_nodes.py -v

# Integration tests
pytest tests/pipelines/data_science/test_pipeline.py -v

# With coverage report
pytest tests/pipelines/data_science/ --cov=src/spaceflights/pipelines/data_science --cov-report=html
```

**Code Quality**:
```bash
# Linting
ruff check src/spaceflights/pipelines/data_science/ --line-length=100

# Type checking (optional)
mypy src/spaceflights/pipelines/data_science/
```

## Known Limitations

1. **Extrapolation**: Predictions for aircraft outside training feature ranges are clipped with a warning logged
2. **MC Dropout Uncertainty**: Assumes dropout calibration is adequate for 95% CI; validation by CI calibration check
3. **Aircraft Type Segmentation**: Requires aircraft_type column; optional feature
4. **GPU Support**: Currently CPU-only; GPU support in future versions

## Configuration

All hyperparameters are externalized to `conf/base/parameters_data_science.yml`:

```yaml
model:
  architecture:
    hidden_layers: [64, 32]
    dropout_rate: 0.2

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

prediction:
  n_mc_samples: 10
  confidence_level: 0.95
```

## Troubleshooting

**Issue**: "PyTorch not installed"  
**Solution**: `pip install torch>=2.0`

**Issue**: "Model file not found"  
**Solution**: Run training pipeline first: `kedro run --pipeline=data_science_train`

**Issue**: "Predictions have NaN values"  
**Solution**: Check for NaN in input features; see `validate_predictions()` logs

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Kedro Documentation](https://docs.kedro.org/)
- [Data Catalog](../../../../conf/base/catalog.yml)
- [Parameters](../../../../conf/base/parameters_data_science.yml)
- [Model I/O Contract](../../../../specs/001-aircraft-range-prediction/contracts/model_io_contract.md)
