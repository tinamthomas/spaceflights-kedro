# Contract: PyTorch Neural Network Model I/O Specification

**Version**: 1.0 | **Phase**: 1 Design | **Created**: 2026-05-25

**Scope**: Defines the interface contract for aircraft range prediction model—inputs, outputs, error handling, versioning, and compatibility guarantees.

---

## Model Input Contract

### Input Dataset Schema

**Name**: `aircraft_train_test_split` (or any aircraft feature dataset)

**Format**: Pandas DataFrame or PyArrow Table (via Parquet)

**Columns**:

| Column Name | Data Type | Domain | Constraints | Notes |
|-------------|-----------|--------|-------------|-------|
| `aircraft_id` | `str` | Text identifier | Unique per batch; non-null | Optional at inference time (may use row index) |
| `passenger_capacity_normalized` | `float64` | [-2, +5] (normalized) | Finite; no NaN | Or raw values if model handles denormalization |
| `num_engines_normalized` | `float64` | [-2, +3] (normalized) | Finite; no NaN | Must be normalized using training-set StandardScaler |
| `crew_size_normalized` | `float64` | [-2, +5] (normalized) | Finite; no NaN | — |
| `wingspace_m_normalized` | `float64` | [-2, +4] (normalized) | Finite; no NaN | — |
| `length_m_normalized` | `float64` | [-2, +4] (normalized) | Finite; no NaN | — |

**Alternative: Raw Input** (if preprocessing is handled in pipeline):
- Accept non-normalized features; apply StandardScaler in predict node before passing to model

**Minimum Row Count**: 1 (single aircraft) to 1M+ (batch inference)

**Example**:
```python
# Valid input
features = pd.DataFrame({
    'aircraft_id': ['A380', 'B747', '737'],
    'passenger_capacity_normalized': [1.8, 0.5, -0.3],
    'num_engines_normalized': [0.2, 0.2, -0.8],
    'crew_size_normalized': [0.1, -0.1, -0.5],
    'wingspace_m_normalized': [2.1, 1.5, -0.9],
    'length_m_normalized': [1.8, 1.2, -0.7],
})
# Shape: (3, 6)
# Ready for inference
```

### Input Validation Rules

**Pre-inference Checks**:
1. ✅ DataFrame shape: (N, 5) — exactly 5 normalized features
2. ✅ Column order irrelevant; matching by name
3. ✅ No null values in feature columns
4. ✅ All values finite (no inf, no NaN)
5. ✅ Feature value ranges within training envelope (soft check; log warning if extrapolated)

**Error Handling**:
- ❌ If validation fails: raise `ValueError` with descriptive message (e.g., "Expected 5 features; got 6")
- ❌ If NaN detected: raise `ValueError` with row/column info
- ⚠️ If extrapolation detected: log warning but proceed (graceful degradation per spec)

---

## Model Output Contract

### Output Format

**Type**: Tuple of (predictions, lower_ci, upper_ci)

```python
predictions, lower_ci, upper_ci = model(features)
# predictions: Tensor[N] — predicted range in km for N aircraft
# lower_ci: Tensor[N] — 95% CI lower bound
# upper_ci: Tensor[N] — 95% CI upper bound
```

**Alternative: DataFrame Output** (recommended for Kedro integration):
```python
output_df = pd.DataFrame({
    'aircraft_id': features['aircraft_id'],
    'predicted_range_km': predictions.numpy(),
    'range_lower_ci_km': lower_ci.numpy(),
    'range_upper_ci_km': upper_ci.numpy(),
    'prediction_confidence': confidence.numpy(),
})
# Shape: (N, 4 or 5)
```

### Output Schema

| Column | Type | Domain | Constraints | Notes |
|--------|------|--------|-------------|-------|
| `aircraft_id` | `str` | Identifier | Non-null | Propagated from input |
| `predicted_range_km` | `float64` | [500, 15000] | Positive; finite | Point estimate in kilometers |
| `range_lower_ci_km` | `float64` | [≥0, < predicted] | Positive; finite | 95% CI lower bound |
| `range_upper_ci_km` | `float64` | [> predicted, < ∞] | Positive; finite | 95% CI upper bound |
| `prediction_confidence` | `float64` | [0.0, 1.0] | Finite; typically [0.2, 0.9] | Inverse of CI width relative to prediction |

**Example**:
```python
aircraft_id  predicted_range_km  range_lower_ci_km  range_upper_ci_km  prediction_confidence
"A380"       11500.0             10200.0             12800.0            0.75
"B747"       9200.0              7800.0              10600.0            0.68
"737"        4100.0              3200.0              5000.0             0.72
```

### Output Validation Rules

**Post-inference Checks** (in Kedro node):
1. ✅ Output shape matches input shape: (N, 5)
2. ✅ All predicted ranges > 0 and < 20000 (sanity bounds)
3. ✅ CI bounds satisfy: lower_ci < predicted < upper_ci (monotonicity)
4. ✅ CI width ≤ 50% of predicted range (sanity check on uncertainty)
5. ✅ Confidence score in [0, 1] (normalized)
6. ✅ No NaN or inf values

**Failure Mode**:
- ❌ If output validation fails: log ERROR; optionally raise exception for batch jobs; return flagged rows for retry

---

## Model Versioning & Compatibility

### Model Artifact Structure

```
06_models/
├── aircraft_range_model_20260525_163000.pt       # PyTorch state_dict
├── aircraft_range_model_20260525_163000.json    # Metadata + schema
└── model_registry.csv                            # Version log
```

### Metadata JSON Structure

```json
{
  "model_id": "aircraft_range_v1_20260525_163000",
  "version": "1.0",
  "timestamp": "2026-05-25T16:30:00Z",
  "pytorch_version": "2.0.1",
  "input_contract": {
    "feature_names": ["passenger_capacity_normalized", "num_engines_normalized", "crew_size_normalized", "wingspace_m_normalized", "length_m_normalized"],
    "feature_count": 5,
    "normalization": {
      "mean": [100.5, 2.3, 8.1, 35.2, 50.1],
      "std": [150.3, 1.2, 3.5, 12.4, 15.8],
      "method": "StandardScaler"
    }
  },
  "output_contract": {
    "output_columns": ["predicted_range_km", "range_lower_ci_km", "range_upper_ci_km"],
    "ci_percentile": 95,
    "uncertainty_method": "mc_dropout"
  },
  "performance_metrics": {
    "test_rmse": 720.1,
    "test_r2": 0.86,
    "test_mae": 580.3
  }
}
```

### Backward Compatibility

**v1 → v2 Migration Strategy** (future planning):
- If input feature count changes: require explicit pipeline update (fail-fast)
- If output schema changes: add new columns; retain old columns for compatibility
- If normalization params change: store old scaler; support inference on both versions

**Breaking Changes** (require redeployment):
- ❌ Reduction in feature count (e.g., 5 → 4 features)
- ❌ Change in feature order or names
- ❌ Change in output units (e.g., km → miles)

**Non-Breaking Changes** (backward-compatible):
- ✅ Addition of new features
- ✅ Addition of new output columns (e.g., model_id, timestamp)
- ✅ Improvement in performance (same schema, better metrics)

---

## Error Handling & Fallback

### Graceful Degradation

**Scenario 1**: Single Aircraft with Extrapolated Features
```
Input: aircraft_id="experimental", features=[outlier, outlier, 5.0, 5.0, 5.0]
Action: Clip to training bounds; log warning
Output: {predicted_range: 6500, ci: [5000, 8000], warning: "Features extrapolated"}
```

**Scenario 2**: Batch Inference with Some NaN Values
```
Input: N=100 aircraft; 2 rows have NaN in crew_size_normalized
Action: 
  - Option A: Skip NaN rows; output N-2 rows (report skipped rows)
  - Option B: Impute with mean; proceed (log imputation event)
Selected: Option A (fail-safe for production)
Output: (98, 4) DataFrame with metadata {"rows_skipped": 2, "reason": "NaN values"}
```

**Scenario 3**: Model Inference Fails (e.g., OOM)
```
Action: 
  1. Log detailed error (model_id, batch_size, feature summary)
  2. Return empty DataFrame with error metadata
  3. Alert monitoring system
  4. Suggest: Reduce batch size; retry
```

### Logging Contract

**Event**: Model inference begins
```
logger.info(f"[MODEL] inference_start | model_id={model_id} | batch_size={N} | timestamp={ts}")
```

**Event**: Extrapolation detected
```
logger.warning(f"[MODEL] extrapolation | aircraft_ids={ids} | features_clipped=True")
```

**Event**: Inference completes
```
logger.info(f"[MODEL] inference_end | model_id={model_id} | rows_processed={N} | duration_ms={t} | success_rate={ok_pct}%")
```

---

## Usage Examples

### Example 1: Training Phase

```python
from pathlib import Path
import torch
from kedro.io import DataCatalog
from spaceflights.pipelines.data_science.nodes import train_model

# Load training data via Kedro catalog
catalog = DataCatalog()
train_data = catalog.load("aircraft_train_test_split")

# Train model (satisfies contract)
model, metadata = train_model(train_data, learning_rate=0.001, epochs=50)

# Save model & metadata (contract preserved)
torch.save(model.state_dict(), Path("06_models/aircraft_range_model_20260525_163000.pt"))
with open("06_models/aircraft_range_model_20260525_163000.json", "w") as f:
    json.dump(metadata, f)
```

### Example 2: Inference Phase

```python
from spaceflights.pipelines.data_science.nodes import predict_with_confidence

# Load model
model = load_model("06_models/aircraft_range_model_20260525_163000.pt")

# Load new aircraft features (already normalized)
new_features = catalog.load("aircraft_features_for_prediction")

# Predict (contract input/output satisfied)
predictions_df = predict_with_confidence(model, new_features, n_mc_samples=10)

# Validate output (contract verified)
assert predictions_df.shape[0] == len(new_features)
assert all(col in predictions_df.columns for col in ["predicted_range_km", "range_lower_ci_km", "range_upper_ci_km"])
assert (predictions_df["range_lower_ci_km"] < predictions_df["predicted_range_km"]).all()
```

### Example 3: Batch Evaluation

```python
from spaceflights.pipelines.data_science.nodes import evaluate_model

# Load test data with ground truth
test_data = catalog.load("aircraft_test_data")

# Get predictions
predictions = predict_with_confidence(model, test_data)

# Evaluate (compare predictions vs ground truth)
metrics = evaluate_model(predictions, test_data["range_km"])
# Returns: {"rmse": 720.1, "r2": 0.86, "mae": 580.3, ...}

# Save metrics (contract preserved for downstream)
catalog.save("model_evaluation_metrics", metrics)
```

---

## Contract Compliance Checklist

**For Model Developers**:
- [ ] Model accepts exactly 5 normalized features per input contract
- [ ] All outputs are finite (no NaN, no inf)
- [ ] CI bounds satisfy: lower < predicted < upper
- [ ] Predictions stay within training range envelope ±20% (or flag extrapolation)
- [ ] Model metadata JSON matches specified schema
- [ ] Error handling includes logging for observability

**For Pipeline Integrators**:
- [ ] Input data normalized using training-set StandardScaler before passing to model
- [ ] Output DataFrame validated against schema (column names, types, value ranges)
- [ ] Model versioning tracked (model_id + timestamp in outputs)
- [ ] Inference performance monitored (latency < 100ms per 1000 aircraft)
- [ ] Batch failure modes handled gracefully (partial results + error metadata)

**For Code Reviewers**:
- [ ] Contract assumptions documented in node docstrings
- [ ] Input/output validation code tested (unit tests for validation logic)
- [ ] Error messages include sufficient context for debugging
- [ ] No hardcoded feature names; all externalized to configuration/metadata

