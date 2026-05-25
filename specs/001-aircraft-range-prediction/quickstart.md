# Quickstart: Aircraft Range Prediction Neural Network Pipeline

**Version**: 1.0 | **Phase**: 1 Design | **Created**: 2026-05-25

**Purpose**: Get the aircraft range prediction pipeline running end-to-end in development environment

---

## Prerequisites

- **Python**: 3.10+ (verified: `python --version`)
- **Git**: Repository cloned and on branch `001-aircraft-range-prediction`
- **Virtual Environment**: Created and activated
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # macOS/Linux
  # or: venv\Scripts\activate  # Windows
  ```

---

## Installation

### 1. Install Dependencies

Add PyTorch to `pyproject.toml` dependencies:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "torch>=2.0,<3.0",        # PyTorch (CPU or GPU compatible)
]
```

Then install the project in development mode:

```bash
cd /Users/tinamariathomas/workspace/tinamthomas/spaceflights
pip install -e ".[dev]"  # Installs with dev dependencies (pytest, ruff, etc.)
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import kedro; print(f'Kedro {kedro.__version__}')"
pytest --version
```

Expected output:
```
PyTorch 2.0.1
Kedro 1.4.0
pytest 7.2.x
```

---

## Data Setup

### 1. Prepare Raw Dataset

The pipeline expects aircraft data in `data/01_raw/companies.csv` with columns:
- `aircraft_id` (string): unique identifier
- `passenger_capacity` (int): seating capacity
- `num_engines` (int): number of engines
- `crew_size` (int): crew members
- `wingspace_m` (float): wing span in meters
- `length_m` (float): fuselage length in meters
- `range_km` (float): maximum range in kilometers

**Sample Data**:
```csv
aircraft_id,passenger_capacity,num_engines,crew_size,wingspace_m,length_m,range_km
A380,555,4,21,79.75,72.75,14800
B747-8,410,4,16,64.44,70.66,13450
B777-300,346,2,13,60.93,73.86,13650
A320,180,2,6,35.8,37.57,6300
737-800,189,2,6,35.79,39.5,5300
CRJ900,90,2,3,23.95,36.01,2700
Q400,74,2,2,27.38,32.84,1950
```

Place in: `data/01_raw/companies.csv` (or use existing data if available)

### 2. Configure Parameters

Update `conf/base/parameters_data_science.yml` (create if doesn't exist):

```yaml
# Training hyperparameters
train:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  random_seed: 42
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
  
# Model architecture
model:
  input_features: 5
  hidden_layers: [64, 32]
  activation: "relu"
  batch_norm: true
  dropout: 0.2
  output_features: 1  # Range prediction (regression)
  
# Optimizer settings
optimizer:
  name: "adam"
  betas: [0.9, 0.999]
  weight_decay: 1.0e-5
  
# Loss function
loss:
  name: "mse"
  
# Uncertainty quantification
uncertainty:
  method: "mc_dropout"
  n_forward_passes: 10
  ci_percentile: 95
```

### 3. Verify Catalog Configuration

Ensure `conf/base/catalog.yml` includes aircraft datasets:

```yaml
# Raw data (source)
aircraft_raw:
  type: pandas.CSVDataSet
  filepath: data/01_raw/companies.csv
  metadata:
    schema: {aircraft_id: str, passenger_capacity: int, num_engines: int, crew_size: int, wingspace_m: float, length_m: float, range_km: float}
    owner: external_data_provider
    refresh_frequency: monthly

# Cleaned data
aircraft_cleaned:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/aircraft_cleaned.parquet
  metadata:
    schema: {aircraft_id: str, passenger_capacity: int, num_engines: int, crew_size: int, wingspace_m: float, length_m: float, range_km: float}
    owner: spaceflights/data_processing
  
# Normalized & split
aircraft_train_test_split:
  type: pandas.ParquetDataSet
  filepath: data/05_model_input/aircraft_train_test_split.parquet
  versioned: true
  metadata:
    schema: {aircraft_id: str, passenger_capacity_normalized: float, num_engines_normalized: float, crew_size_normalized: float, wingspace_m_normalized: float, length_m_normalized: float, split: str, range_km: float}
    owner: spaceflights/data_science
    
# Model artifact
trained_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/aircraft_range_model.pt
  versioned: true
  metadata:
    owner: spaceflights/data_science
    
# Predictions
predictions:
  type: pandas.ParquetDataSet
  filepath: data/07_model_output/predictions.parquet
  versioned: true
  metadata:
    schema: {aircraft_id: str, predicted_range_km: float, range_lower_ci_km: float, range_upper_ci_km: float}
    owner: spaceflights/data_science
```

---

## Development Workflow

### Phase 1: Write Tests (TDD)

Per Spaceflights Constitution: **Test-First Development** — write failing tests before implementation.

```bash
# Create test file
cat > tests/pipelines/data_science/test_nodes.py << 'EOF'
import pytest
import pandas as pd
import torch
from spaceflights.pipelines.data_science.nodes import (
    preprocess_features,
    train_model,
    predict_with_confidence,
    evaluate_model,
)

def test_preprocess_features_normalizes_correctly():
    """Test that features are normalized to zero mean, unit variance."""
    # Sample data
    data = pd.DataFrame({
        'aircraft_id': ['A380', 'B747', '737'],
        'passenger_capacity': [555, 410, 189],
        'num_engines': [4, 4, 2],
        'crew_size': [21, 16, 6],
        'wingspace_m': [79.75, 64.44, 35.79],
        'length_m': [72.75, 70.66, 39.5],
        'range_km': [14800, 13450, 5300],
    })
    
    # Preprocess
    processed, scaler = preprocess_features(data)
    
    # Verify
    assert processed.shape == (3, 6)  # 5 features + split column
    # Normalized columns should be approximately zero mean
    normalized_cols = [col for col in processed.columns if 'normalized' in col]
    assert len(normalized_cols) == 5
    for col in normalized_cols:
        assert abs(processed[col].mean()) < 0.1, f"{col} not centered"
        
def test_train_model_produces_model_artifact():
    """Test that training pipeline produces a valid PyTorch model."""
    # Create toy dataset
    features = torch.randn(100, 5)
    target = torch.randn(100, 1)
    dataset = torch.utils.data.TensorDataset(features, target)
    
    # Train
    model, metadata = train_model(dataset, epochs=5, learning_rate=0.001)
    
    # Verify
    assert isinstance(model, torch.nn.Module)
    assert metadata['rmse'] > 0
    assert metadata['training_loss_history'] is not None
    
def test_predict_with_confidence_has_valid_ci():
    """Test that predictions include valid confidence intervals."""
    # ... similar setup ...
    predictions = predict_with_confidence(model, features, n_mc_samples=10)
    
    # Verify CI bounds
    assert all(predictions['range_lower_ci_km'] < predictions['predicted_range_km'])
    assert all(predictions['predicted_range_km'] < predictions['range_upper_ci_km'])
EOF

# Run tests (should fail; tests written before implementation)
pytest tests/pipelines/data_science/test_nodes.py -v
```

**Expected output**:
```
FAILED test_nodes.py::test_preprocess_features_normalizes_correctly - ModuleNotFoundError: No module named 'spaceflights.pipelines.data_science.nodes'
```

✅ Tests fail as expected (Red phase of Red-Green-Refactor)

### Phase 2: Implement Nodes (Green)

Create `src/spaceflights/pipelines/data_science/nodes.py`:

```python
import logging
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def preprocess_features(
    data: pd.DataFrame,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess aircraft features: clean, normalize, split.
    
    Args:
        data: Raw aircraft data with columns [aircraft_id, ..., range_km]
        random_seed: Reproducibility seed
        
    Returns:
        processed_data: Normalized features with split assignment
        scaler: Fitted StandardScaler for inference-time normalization
    """
    # Validation
    assert data.shape[0] > 0, "Empty dataset"
    assert 'range_km' in data.columns, "Missing target column: range_km"
    
    # Remove rows with missing values
    data = data.dropna()
    
    # Separate features and target
    feature_cols = ['passenger_capacity', 'num_engines', 'crew_size', 'wingspace_m', 'length_m']
    X = data[feature_cols]
    y = data[['range_km']]
    
    # Fit scaler on full data (or train subset for proper CV)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Create output dataframe
    output = data[['aircraft_id']].copy()
    for i, col in enumerate(feature_cols):
        output[f'{col}_normalized'] = X_normalized[:, i]
    output['range_km'] = y.values
    
    # Assign splits
    train_idx, other_idx = train_test_split(
        range(len(output)), test_size=0.3, random_state=random_seed
    )
    val_idx, test_idx = train_test_split(
        other_idx, test_size=0.5, random_state=random_seed
    )
    
    output.loc[train_idx, 'split'] = 'train'
    output.loc[val_idx, 'split'] = 'validation'
    output.loc[test_idx, 'split'] = 'test'
    
    logger.info(f"Preprocessed {len(output)} aircraft | train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return output, scaler

def train_model(
    data: pd.DataFrame,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    random_seed: int = 42
) -> Tuple[torch.nn.Module, Dict]:
    """
    Train PyTorch neural network to predict aircraft range.
    
    Args:
        data: Preprocessed data with normalized features and split assignment
        learning_rate: Adam optimizer learning rate
        batch_size: Training batch size
        epochs: Number of training epochs
        random_seed: Reproducibility seed
        
    Returns:
        model: Trained PyTorch model
        metadata: Training statistics and model info
    """
    # Set seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Prepare data
    train_data = data[data['split'] == 'train']
    val_data = data[data['split'] == 'validation']
    
    feature_cols = [col for col in train_data.columns if 'normalized' in col]
    X_train = torch.FloatTensor(train_data[feature_cols].values)
    y_train = torch.FloatTensor(train_data[['range_km']].values)
    
    X_val = torch.FloatTensor(val_data[feature_cols].values)
    y_val = torch.FloatTensor(val_data[['range_km']].values)
    
    # Define model
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 64),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(64),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(32),
        torch.nn.Linear(32, 1),
    )
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validate
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred, y_val)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f}")
    
    # Compute training metrics
    rmse = np.sqrt(np.mean(train_losses[-10:]))  # Average of last 10 epochs
    
    metadata = {
        'training_loss_history': train_losses,
        'validation_loss_history': val_losses,
        'rmse': rmse,
        'epochs': epochs,
        'learning_rate': learning_rate,
    }
    
    logger.info(f"Training complete | Final RMSE: {rmse:.2f}")
    
    return model, metadata

def predict_with_confidence(
    model: torch.nn.Module,
    features: pd.DataFrame,
    n_mc_samples: int = 10
) -> pd.DataFrame:
    """
    Generate predictions with confidence intervals using MC Dropout.
    
    Args:
        model: Trained PyTorch model
        features: Normalized feature dataframe
        n_mc_samples: Number of MC forward passes for uncertainty
        
    Returns:
        predictions: DataFrame with predicted_range_km and CI bounds
    """
    feature_cols = [col for col in features.columns if 'normalized' in col]
    X = torch.FloatTensor(features[feature_cols].values)
    
    # MC Dropout: enable dropout during inference
    predictions_list = []
    model.train()  # Enable dropout
    with torch.no_grad():
        for _ in range(n_mc_samples):
            y_pred = model(X)
            predictions_list.append(y_pred.numpy())
    
    predictions_array = np.array(predictions_list)  # Shape: (n_samples, batch_size, 1)
    mean_pred = predictions_array.mean(axis=0).squeeze()
    std_pred = predictions_array.std(axis=0).squeeze()
    
    # Build output
    output = features[['aircraft_id']].copy()
    output['predicted_range_km'] = mean_pred
    output['range_lower_ci_km'] = mean_pred - 1.96 * std_pred  # 95% CI
    output['range_upper_ci_km'] = mean_pred + 1.96 * std_pred
    output['prediction_confidence'] = 1.0 - (2 * std_pred / (mean_pred + 1e-6))  # Avoid div by zero
    
    logger.info(f"Generated predictions for {len(output)} aircraft")
    
    return output

def evaluate_model(
    predictions: pd.DataFrame,
    actual: pd.Series
) -> Dict:
    """
    Compute evaluation metrics: RMSE, R², MAE.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    y_pred = predictions['predicted_range_km'].values
    y_true = actual.values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {'rmse': rmse, 'r2': r2, 'mae': mae}
    logger.info(f"Evaluation metrics | RMSE={rmse:.2f} | R²={r2:.3f} | MAE={mae:.2f}")
    
    return metrics
```

### Phase 3: Re-run Tests (Green)

```bash
pytest tests/pipelines/data_science/test_nodes.py -v
```

**Expected**: ✅ All tests pass

### Phase 4: Run Full Pipeline

```bash
# Run data processing (cleanup, normalization)
kedro run --pipeline=data_processing

# Run data science (training + prediction + evaluation)
kedro run --pipeline=data_science

# Or run all pipelines
kedro run
```

### Phase 5: Linting & Code Quality

```bash
# Check code style
ruff check src/spaceflights/pipelines/data_science/ --line-length=100

# Run tests with coverage
pytest tests/ --cov=src/spaceflights --cov-report=html

# Check coverage report
open htmlcov/index.html  # Or view in browser
```

**Target**: ≥ 80% coverage ✅

---

## Verify Results

### 1. Check Generated Artifacts

```bash
# List generated files
find data/ -type f -name "*.parquet" -o -name "*.pt" -o -name "*.json" | sort

# Expected output
# data/02_intermediate/aircraft_cleaned.parquet
# data/05_model_input/aircraft_train_test_split.parquet
# data/06_models/aircraft_range_model_20260525_163000.pt
# data/06_models/aircraft_range_model_20260525_163000.json
# data/07_model_output/predictions.parquet
```

### 2. Inspect Predictions

```python
import pandas as pd

# Load predictions
preds = pd.read_parquet('data/07_model_output/predictions.parquet')
print(preds.head())
print(f"Shape: {preds.shape}")
print(f"Mean predicted range: {preds['predicted_range_km'].mean():.1f} km")
print(f"Mean CI width: {(preds['range_upper_ci_km'] - preds['range_lower_ci_km']).mean():.1f} km")
```

### 3. Review Training Metrics

```python
import json

with open('data/06_models/aircraft_range_model_20260525_163000.json') as f:
    metadata = json.load(f)

print(f"RMSE: {metadata['rmse']:.2f}")
print(f"R²: {metadata.get('r2', 'N/A')}")
print(f"Epochs: {metadata['epochs']}")
```

---

## Next Steps

1. **Phase 2 Implementation**: Follow TDD workflow; all tests must pass
2. **Integrate with Kedro Pipelines**: Wire nodes into `pipelines/data_science/pipeline.py`
3. **Generate Task List**: Run `/speckit.tasks` to create actionable development tasks
4. **Deploy Pipeline**: Set up GitHub Actions for continuous integration + testing

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: `pip install torch>=2.0`

### Issue: "No such file: data/01_raw/companies.csv"
**Solution**: Prepare raw data file (see Data Setup section above)

### Issue: "RuntimeError: CUDA out of memory"
**Solution**: Reduce batch_size in `parameters_data_science.yml` or use CPU: `torch.device('cpu')`

### Issue: "AssertionError: Feature values outside training range"
**Solution**: This is expected; features are clipped to training bounds with a warning logged

---

## Support

- **Kedro Docs**: https://docs.kedro.org/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Project README**: [../../README.md](../../README.md)
- **Constitution**: [../../../.specify/memory/constitution.md](../../../.specify/memory/constitution.md)

