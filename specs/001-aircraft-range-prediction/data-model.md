# Data Model: Aircraft Range Prediction Pipeline

**Phase**: 1 (Design) | **Created**: 2026-05-25

**Purpose**: Define entities, relationships, and data structures for aircraft range prediction pipeline

---

## Core Entities

### Aircraft Features Entity

**Role**: Input data entity; represents numeric characteristics of an aircraft.

**Fields**:
| Field | Type | Domain | Constraints | Purpose |
|-------|------|--------|-------------|---------|
| `aircraft_id` | String | Identifier | Unique, non-null | Primary key; enables tracing predictions back to source |
| `passenger_capacity` | Integer | [10, 1000] | Positive; non-null | Core predictor; proxy for aircraft size |
| `num_engines` | Integer | [1, 4] | Positive; non-null | Core predictor; affects fuel consumption & range |
| `crew_size` | Integer | [2, 20] | Positive; non-null | Core predictor; aircraft complexity indicator |
| `wingspace_m` | Float | [10.0, 80.0] | Positive; non-null | Core predictor; wing surface area (meters) |
| `length_m` | Float | [10.0, 75.0] | Positive; non-null | Core predictor; fuselage length (meters) |
| `aircraft_type` | String (enum) | See note | Optional; non-null if available | Grouping for segmented evaluation (e.g., "commercial", "cargo", "regional") |

**Notes**:
- All numeric fields must be strictly positive (no zero crew, no negative engines)
- If `aircraft_type` available in source data, use for segmented evaluation metrics
- Missing values: any missing in numeric fields → impute via median; missing type → flag for single-type analysis
- Data Quality Validation: run before preprocessing; drop rows with critical nulls

**Data Lineage Path**: `01_raw/companies.csv` → (raw features extracted/validated here)

---

### Aircraft Range Target Entity

**Role**: Output target; what model predicts.

**Fields**:
| Field | Type | Domain | Constraints | Purpose |
|-------|------|--------|-------------|---------|
| `aircraft_id` | String | Identifier | Matches Aircraft Features | Link to aircraft |
| `range_km` | Float | [500, 15000] | Positive; non-null for training | Ground truth; model trains to predict this |

**Notes**:
- Target must be non-null for all training data
- No missing values tolerated in training set; validation/test sets must be complete or dropped
- Strongly right-skewed distribution expected (most aircraft 2000–8000 km; few ultra-long-range ~15000 km)
- Outliers (e.g., 100 km) are data errors, not aircraft types; remove in preprocessing

---

### Preprocessed Features Entity

**Role**: Intermediate entity after cleaning and normalization; ready for model input.

**Fields**:
| Field | Type | Source | Constraints | Purpose |
|-------|------|--------|-------------|---------|
| `aircraft_id` | String | Aircraft Features | Unique | Traceability |
| `passenger_capacity_normalized` | Float | Standardized | [μ≈0, σ≈1] | Neural network input; z-score normalized |
| `num_engines_normalized` | Float | Standardized | [μ≈0, σ≈1] | Neural network input |
| `crew_size_normalized` | Float | Standardized | [μ≈0, σ≈1] | Neural network input |
| `wingspace_m_normalized` | Float | Standardized | [μ≈0, σ≈1] | Neural network input |
| `length_m_normalized` | Float | Standardized | [μ≈0, σ≈1] | Neural network input |
| `split` | String (enum) | Assigned | {train, validation, test} | Fold assignment for reproducible splits |
| `range_km` | Float | Target | [500, 15000] | Ground truth (for training/eval sets) |

**Normalization Applied**:
- StandardScaler fit on training set only
- Fitted mean & std saved in model metadata (for inference-time denormalization)
- All splits (train/val/test) normalized using training-set statistics

**Data Lineage Path**: Aircraft Features → [outlier removal] → [imputation] → [split] → [standardization] → Preprocessed Features

---

### Trained Neural Network Model Entity

**Role**: Learned model artifact; reproducible and versioned.

**Structure**:

**Model Weights File** (`06_models/aircraft_range_model_[TIMESTAMP].pt`):
```
PyTorch state_dict containing:
  - layer_1.weight: Tensor[64, 5]        # Input layer → hidden layer 1
  - layer_1.bias: Tensor[64]
  - layer_2.weight: Tensor[32, 64]       # Hidden layer 1 → hidden layer 2
  - layer_2.bias: Tensor[32]
  - layer_3.weight: Tensor[1, 32]        # Hidden layer 2 → output (range prediction)
  - layer_3.bias: Tensor[1]
```

**Model Metadata File** (`06_models/aircraft_range_model_[TIMESTAMP].json`):
```json
{
  "model_version": "1.0",
  "timestamp": "2026-05-25T16:30:00Z",
  "architecture": {
    "input_features": 5,
    "hidden_layers": [64, 32],
    "activation": "relu",
    "output_units": 1,
    "batch_norm": true,
    "dropout": 0.2
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "adam",
    "loss_fn": "mse"
  },
  "normalization": {
    "standardscaler_means": [100.5, 2.3, 8.1, 35.2, 50.1],
    "standardscaler_stds": [150.3, 1.2, 3.5, 12.4, 15.8],
    "feature_names": ["passenger_capacity", "num_engines", "crew_size", "wingspace_m", "length_m"]
  },
  "training_statistics": {
    "training_loss_history": [...],
    "validation_loss_history": [...],
    "train_rmse": 620.5,
    "train_r2": 0.89,
    "validation_rmse": 685.3,
    "validation_r2": 0.87,
    "test_rmse": 720.1,
    "test_r2": 0.86
  },
  "data_statistics": {
    "training_set_size": 700,
    "validation_set_size": 150,
    "test_set_size": 150,
    "target_mean": 5000.0,
    "target_std": 3500.0,
    "random_seed": 42
  },
  "pytorch_version": "2.0.1",
  "python_version": "3.10"
}
```

**Key Relationships**:
- Metadata MUST always accompany `.pt` file (enables reproducibility & auditability)
- Training statistics inform deployment decisions (is R² > 0.85? proceed; else retrain)
- Normalization params are CRITICAL for inference (predictions are meaningless without correct denormalization)

---

### Predictions Entity

**Role**: Output entity; model predictions on new/test aircraft.

**Fields**:
| Field | Type | Source | Constraints | Purpose |
|-------|------|--------|-------------|---------|
| `aircraft_id` | String | Input | Non-null | Link to aircraft |
| `predicted_range_km` | Float | Model | [500, 15000] | Point estimate of range |
| `predicted_range_lower_ci_km` | Float | Model uncertainty | ≤ predicted_range_km | 95% CI lower bound |
| `predicted_range_upper_ci_km` | Float | Model uncertainty | ≥ predicted_range_km | 95% CI upper bound |
| `prediction_confidence` | Float | Calculation | [0, 1] | 1 - (CI_width / mean_prediction); proxy for certainty |
| `actual_range_km` | Float | Ground truth (if available) | Optional | For evaluation; null if no ground truth |
| `residual_km` | Float | Calculation | predicted - actual | For error analysis (null if actual_range_km is null) |

**Data Quality Constraints**:
- All predicted ranges must be positive
- CI bounds must satisfy: lower < predicted < upper
- CI width ≤ 20% of predicted range (otherwise flagged as high-uncertainty)
- No null predicted ranges (model never fails; see error handling in research.md)

**Data Lineage Path**: Preprocessed Features + Trained Model → [inference] → Predictions

---

### Evaluation Metrics Entity

**Role**: Summary statistics for model governance and monitoring.

**Fields**:
| Field | Type | Aggregation | Purpose |
|-------|------|-------------|---------|
| `metric_name` | String | — | RMSE, R², MAE, MAPE |
| `split` | String | {train, validation, test} | Which data partition |
| `value` | Float | Computed | Metric value (e.g., 0.86 for R²) |
| `aircraft_type` | String (optional) | Segmented | If available, compute per-type metrics |
| `sample_count` | Integer | — | N records in segment |

**Example Records**:
```
{metric_name: "rmse", split: "test", value: 720.1, aircraft_type: null, sample_count: 150}
{metric_name: "r2", split: "test", value: 0.86, aircraft_type: null, sample_count: 150}
{metric_name: "rmse", split: "test", value: 650.0, aircraft_type: "commercial", sample_count: 100}
{metric_name: "rmse", split: "test", value: 850.0, aircraft_type: "cargo", sample_count: 50}
```

**Compliance Gate** (from Spec):
- Overall RMSE < 15% of mean range (5000 km → RMSE < 750 km) ✅
- Overall R² > 0.85 ✅
- Per-type RMSE variance < 30% relative difference ✅

---

## Entity Relationships

```
┌─────────────────────┐
│ Aircraft Features   │ (01_raw)
│ (raw, untouched)    │
└──────────┬──────────┘
           │
           ├─ [clean, validate, impute]
           ↓
┌─────────────────────┐
│ Preprocessed        │ (02_intermediate/03_primary)
│ Aircraft Data       │
└──────────┬──────────┘
           │
           ├─ [split train/val/test]
           ├─ [standardize features]
           ↓
┌─────────────────────┐
│ Training Dataset    │ (04_feature/05_model_input)
│ (normalized, split) │
└────────────┬────────┘
             │
             ├─ [train_test_split] → train (70%)
             ├─              → val (15%)
             └─              → test (15%)
                              |
                              ├─ [PyTorch training]
                              ↓
                    ┌──────────────────────┐
                    │ Trained NN Model     │ (06_models)
                    │ + Metadata (JSON)    │
                    └──────────┬───────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
           [train]                      [predict+eval]
               │                                │
               ├─ predictions on train  │ new aircraft
               ├─ evaluation metrics    ↓
               └──→ ┌──────────────────────┐
                    │ Predictions + CIs    │ (07_model_output)
                    │ Evaluation Metrics   │
                    └──────────────────────┘
```

---

## Data Schema Summary

| Entity | Location | Format | Versioned | Owner |
|--------|----------|--------|-----------|-------|
| Aircraft Features | `01_raw/companies.csv` | CSV | N (source) | External |
| Preprocessed Features | `02_intermediate/` `03_primary/` | Parquet | Y | spaceflights/data_processing |
| Training Dataset | `04_feature/` `05_model_input/` | Parquet | Y | spaceflights/data_science |
| Trained Model | `06_models/*.pt` | PyTorch | Y | spaceflights/data_science |
| Model Metadata | `06_models/*.json` | JSON | Y | spaceflights/data_science |
| Predictions | `07_model_output/predictions.parquet` | Parquet | Y | spaceflights/data_science |
| Metrics | `07_model_output/metrics.json` | JSON | Y | spaceflights/data_science |

---

## Validation Rules & Constraints

### Input Validation (Aircraft Features)
- ✅ aircraft_id: non-null, unique
- ✅ passenger_capacity: int, 10–1000, non-null
- ✅ num_engines: int, 1–4, non-null
- ✅ crew_size: int, 2–20, non-null
- ✅ wingspace_m: float, 10.0–80.0, non-null
- ✅ length_m: float, 10.0–75.0, non-null
- ✅ range_km (target): float, 500–15000, non-null for training

### Normalization Validation (Preprocessed Features)
- ✅ All normalized features: μ ≈ 0, σ ≈ 1 (verify via mean/std on training set)
- ✅ Split assignment: all rows assigned to exactly one of {train, val, test}
- ✅ No data leakage: val/test sets normalized using training set statistics only

### Model Output Validation (Predictions)
- ✅ predicted_range_km: all positive, no nulls
- ✅ CI bounds: lower_ci < predicted < upper_ci for all rows
- ✅ Confidence: [0, 1] for all rows
- ✅ Residuals: absolute value ≤ 50% of predicted range (outliers flagged)

---

## Data Governance

| Field | Owner | SLA | Refresh Frequency |
|-------|-------|-----|-------------------|
| Aircraft Features | External data provider | Monthly validation | On-demand |
| Preprocessed Features | spaceflights/data_processing | N/A (intermediate) | Per training run |
| Trained Model | spaceflights/data_science | Retraining quarterly | As needed |
| Predictions | spaceflights/data_science | Batch run daily | Per request or scheduled |
| Evaluation Metrics | spaceflights/data_science | Dashboard updated daily | Continuous monitoring |

