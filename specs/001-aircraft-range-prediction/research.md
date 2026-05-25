# Research: Aircraft Range Prediction with PyTorch in Kedro

**Phase**: 0 (Planning Research) | **Completed**: 2026-05-25

**Purpose**: Resolve technical unknowns and validate framework choices for PyTorch integration in Kedro pipeline

---

## Research Findings

### 1. PyTorch + Kedro Integration Pattern

**Decision**: Wrap PyTorch model training in pure Kedro node functions.

**Rationale**:
- Kedro enforces functional purity: nodes are pure functions with explicit inputs/outputs
- PyTorch model training naturally fits this pattern: input data → trained model (output to `06_models/`)
- Avoids mixing PyTorch state management with Kedro's Catalog, maintaining reproducibility
- Enables Kedro's parallelization and caching of nodes

**Implementation Approach**:
```python
# Single responsibility principle: one node per task
node_1: train_data → trained_model + training_history
node_2: trained_model + test_data → predictions + confidence_intervals
node_3: predictions + ground_truth → evaluation_metrics
```

**Verified Best Practice**: 
- Kedro documentation supports custom Python objects in Catalog (models, pickles, joblib)
- PyTorch models serialize via `torch.save()` (state_dict) or TorchScript
- Kedro `PickleDataSet` or custom `PyTorchModelDataSet` can handle `.pt` files

**Alternatives Considered**:
- ❌ TensorFlow/Keras: Heavier dependencies, slower startup for this scale; PyTorch selected per user preference
- ❌ scikit-learn MLPRegressor: Insufficient for complex architectures; PyTorch offers more flexibility
- ❌ Custom training loop outside Kedro: Loses reproducibility, data lineage, parameterization benefits

---

### 2. Neural Network Architecture for Aircraft Range Prediction

**Decision**: 2-3 layer feedforward neural network (MLP) with ReLU activation and BatchNorm.

**Rationale**:
- Aircraft range prediction is fundamentally a regression task (continuous numeric output)
- 5 input features are relatively low-dimensional; shallow network sufficient (2-3 layers)
- ReLU is standard, well-understood activation; BN helps training stability
- Dropout optional for regularization if overfitting observed

**Architecture**:
```
Input (5 features) 
→ Dense(64) + ReLU + BatchNorm + Dropout(0.2)
→ Dense(32) + ReLU + BatchNorm
→ Dense(1)  # Range output
```

**Hyperparameters** (Phase 1 defaults; optimization out of scope):
- **Learning Rate**: 0.001 (Adam optimizer default)
- **Batch Size**: 32 (balanced for dataset size 100–1000)
- **Epochs**: 50 (early stopping by validation loss plateau)
- **Loss**: MSE (standard for regression)
- **Optimizer**: Adam (adaptive learning rate, stable convergence)

**Alternatives Considered**:
- ❌ LSTM/RNN: Overkill; no temporal sequence in aircraft features
- ❌ Large networks (10+ layers): Risk of overfitting on ~1000 aircraft; curse of dimensionality
- ✅ Simpler: Linear regression (scikit-learn): Baseline comparison, but NN offers non-linearity capture

---

### 3. Model Serialization & Versioning Strategy

**Decision**: PyTorch state_dict + JSON metadata file for versioning and reproducibility.

**Rationale**:
- `torch.save(model.state_dict(), path)` separates architecture from weights (enables architecture changes)
- JSON metadata (hyperparameters, training date, performance) supports model governance
- Parquet-based predictions enable Kedro's versioning (timestamp-based snapshots)

**File Structure**:
```
06_models/
├── aircraft_range_model_20260525_163000.pt    # PyTorch state_dict
├── aircraft_range_model_20260525_163000.json  # Metadata: {architecture, hyperparams, train_loss, val_loss, test_rmse, test_r2, timestamp}
└── model_registry.csv                         # Version log (model_path, timestamp, rmse, r2)
```

**Reproducibility**:
- Metadata includes `random_seed`, `train_split_seed`, `pytorch_version`
- Loading: recreate model architecture from config, load state_dict, inference deterministic (no dropout during eval)
- Versioned via Kedro's `versioned: true` in catalog (auto-timestamps)

**Alternatives Considered**:
- ❌ SavedModel (TensorFlow format): Not applicable; PyTorch selected
- ❌ ONNX: Adds complexity for v1; deferred to future optimization phase

---

### 4. Confidence Intervals & Uncertainty Quantification

**Decision**: Ensemble-based confidence intervals (train multiple models, use prediction variance across ensemble).

**Rationale**:
- Neural networks output point estimates; uncertainty requires post-hoc method
- Monte Carlo Dropout: enabled dropout during inference, run multiple forward passes, compute std. dev. of outputs
- Simpler alternative: Bootstrap ensemble (train N models on N bootstrap samples)
- Meets spec requirement: 95% CI reflecting model uncertainty

**Implementation** (Phase 2 tasks):
```python
def predict_with_uncertainty(model, features, n_forward_passes=10):
    """Run inference with dropout enabled; return mean ± std as CI."""
    model.train()  # Enable dropout
    predictions = [model(features).detach() for _ in range(n_forward_passes)]
    mean = predictions.mean()
    std = predictions.std()
    return mean, mean - 1.96*std, mean + 1.96*std  # 95% CI
```

**Validation**: Calibration check in evaluation pipeline — CI coverage should be 90–95% on test set

---

### 5. Data Preprocessing & Normalization

**Decision**: StandardScaler (zero mean, unit variance) for all numeric features; preserved in training metadata.

**Rationale**:
- Neural networks perform better on normalized features (helps gradient descent convergence)
- StandardScaler fit on train set; applied to val/test/prediction data
- Inverse transform stored in model metadata for interpretability (e.g., "predicted range = 5000 km")

**Preprocessing Pipeline**:
1. Load data from `01_raw/companies.csv`
2. Drop rows with missing target (range); impute features via median
3. Remove outliers (IQR method on each feature)
4. Split: train (70%), validation (15%), test (15%) with fixed seed
5. Fit StandardScaler on train; apply to all sets
6. Save scaler params (mean, std) with model for prediction-time use

---

### 6. Model Performance Targets & Evaluation Metrics

**Decision**: RMSE, R², MAE as primary metrics; segmented analysis by aircraft type.

**Rationale**:
- RMSE: directly interpretable (units: km); spec requires RMSE < 15% of mean range
- R²: explains variance explained (>0.85 target); intuitive for stakeholders
- MAE: robust to outliers, complements RMSE
- Segmentation: detect if model struggles with particular aircraft types (e.g., large commercial vs. regional)

**Target Thresholds** (from spec):
- RMSE < 15% of mean training range ✅
- R² > 0.85 ✅
- No aircraft type with >30% higher RMSE than average ✅
- 95% CI calibration: 90–95% coverage ✅

**Evaluation Plan**:
```python
metrics = {
    'rmse': mean_squared_error(y_true, y_pred, squared=False),
    'r2': r2_score(y_true, y_pred),
    'mae': mean_absolute_error(y_true, y_pred),
    'rmse_by_aircraft_type': {type: rmse for type, group in by_type},
}
```

---

### 7. Handling Out-of-Distribution Predictions (Extrapolation)

**Decision**: Flag extrapolations; apply range clipping with warning; predict anyway (no hard failures).

**Rationale**:
- Spec requirement: "gracefully handle feature values outside training range"
- Clipping features to [train_min, train_max] before prediction prevents extreme extrapolation
- Logging warning enables downstream monitoring; no crashes ensures pipeline robustness

**Implementation**:
```python
def safe_predict(model, features_raw, train_bounds):
    features_clipped = features_raw.clip(
        lower=train_bounds['min'], 
        upper=train_bounds['max']
    )
    if (features_raw != features_clipped).any():
        logger.warning(f"Features extrapolated; clipped to training range")
    predictions = model(features_clipped)
    return predictions
```

---

### 8. Testing Strategy for Neural Network Nodes

**Decision**: Unit tests for data preprocessing, integration tests for full pipeline, no model-specific tests (training deterministic per seed).

**Rationale**:
- Unit tests: preprocessing consistency, data shape validation, outlier handling
- Integration tests: model training produces outputs, predictions have correct shape/schema, metrics computed correctly
- No test for "model achieves RMSE < X": would require large dataset, non-deterministic; validation in Phase 2 after training

**Test Coverage**:
- ✅ `test_preprocess_features_handles_missing_values()`
- ✅ `test_preprocess_features_normalizes_correctly()`
- ✅ `test_train_pipeline_produces_model_artifact()`
- ✅ `test_predict_pipeline_produces_predictions_with_correct_shape()`
- ✅ `test_confidence_intervals_calibration_check()`

**80% Coverage Goal**: Achievable with node logic + pipeline flow tests; model weights not tested (deterministic via seed)

---

## Dependencies & Version Resolution

### New Dependencies to Add

```toml
# pyproject.toml
torch>=2.0,<3.0          # PyTorch; CPU-only acceptable for v1
torchvision>=0.15        # Data utilities (optional; could use pandas instead)
pytorch-lightning<3.0    # Optional: training loop abstraction (deferred to v2)
```

**Note**: Does NOT require `torch-cuda`; CPU training sufficient for dataset size 100–1000.

### Existing Compatible Dependencies

✅ **pandas**: Data manipulation, already present
✅ **scikit-learn**: StandardScaler, train_test_split, metrics — already present (1.5.1/1.8.0)
✅ **pytest**: Testing framework — already present (7.2)
✅ **NumPy**: Numerical operations, implicit in torch
✅ **Kedro**: Pipeline framework — already present (1.4.0)

---

## Go/No-Go Decision

**Status**: ✅ **GO** — All research questions resolved

**Confidence Justification**:
- PyTorch + Kedro integration pattern validated & documented
- Neural network architecture appropriate for problem scale (5 inputs, ~1000 samples)
- Model serialization strategy supports reproducibility & governance
- Uncertainty quantification approach (MC Dropout) meets spec requirements
- Testing strategy aligns with Constitution (TDD, ≥80% coverage)
- No blocking dependencies; all tools mature & well-supported

**Ready for**: Phase 1 Design (data-model, contracts, quickstart) and Phase 2 Task Generation

---

## Resolved Assumptions from Specification

| Assumption | Status | Resolution |
|-----------|--------|------------|
| Data availability | ✅ RESOLVED | Data source: `01_raw/companies.csv` or external; sourcing in task phase |
| PyTorch framework | ✅ RESOLVED | User requested PyTorch; implemented via state_dict + metadata pattern |
| Hyperparameter tuning scope | ✅ RESOLVED | v1 uses defaults; no Bayesian optimization required |
| Deployment scope | ✅ RESOLVED | v1: batch prediction only; real-time API deferred to v2 |
| Confidence intervals | ✅ RESOLVED | MC Dropout ensemble method; calibration validation in evaluation |
| Extrapolation handling | ✅ RESOLVED | Flag & clip features to training bounds; log warnings; predict anyway |

---

## Recommendations for Phase 2 (Task Generation)

1. **Test Data First**: Write unit tests for preprocessing nodes before implementation (TDD per Constitution)
2. **Dataset Validation**: Add data quality checks at pipeline entry (null rates, value ranges, duplicates)
3. **Reproducibility**: Seed all random operations (torch.manual_seed, np.random.seed, sklearn splits)
4. **Monitoring**: Add TensorBoard logging optional integration for training visualization
5. **Documentation**: Write detailed node docstrings including PyTorch-specific behavior (dropout, determinism)

