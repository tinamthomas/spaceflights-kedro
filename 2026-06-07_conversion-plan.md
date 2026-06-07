# KEDRO CONVERSION - STATEMENT OF WORK
## PyTorch Neural Network Training Pipeline

**Conversion Date**: 2026-06-07  
**Source Notebook**: notebooks/simple.ipynb  
**Target Pipeline**: pytorch_neural_network  

---

## PROJECT DELIVERABLES

### DELIVERABLE 1: PIPELINE IMPLEMENTATION
**Pipeline**: pytorch_neural_network

1.1 **Task 1.1.1**: `prepare_synthetic_data()`
   - Input: None (generates synthetic data)
   - Output: `train_x`, `train_y` (PyTorch tensors as artifacts)
   - Description: Creates synthetic x and y tensors for training

1.2 **Task 1.2.1**: `create_model(params:model_config)`
   - Input: `params:model_config`
   - Output: `untrained_model`
   - Description: Initializes a linear neural network model with 1 hidden layer

1.3 **Task 1.3.1**: `train_model(untrained_model, train_x, train_y, params:training_config)`
   - Input: `untrained_model`, `train_x`, `train_y`, `params:training_config`
   - Output: `trained_model`
   - Description: Trains the model using SGD optimizer for specified epochs

1.4 **Task 1.4.1**: `evaluate_model(trained_model, train_x, train_y)`
   - Input: `trained_model`, `train_x`, `train_y`
   - Output: `training_metrics`
   - Description: Evaluates final model loss on training data

1.5 **Task 1.5.1**: `test_prediction(trained_model)`
   - Input: `trained_model`
   - Output: `sample_prediction`
   - Description: Tests model with sample input and returns prediction

---

### DELIVERABLE 2: DATA CATALOG CONFIGURATION

| Item | Dataset Name | Type | Path | Description |
|------|---|---|---|---|
| 2.1 | train_x | pickle.PickleDataset | data/05_model_input/train_x.pkl | Training input tensor |
| 2.2 | train_y | pickle.PickleDataset | data/05_model_input/train_y.pkl | Training target tensor |
| 2.3 | untrained_model | pickle.PickleDataset | data/06_models/untrained_model.pkl | Initialized model |
| 2.4 | trained_model | pickle.PickleDataset | data/06_models/trained_model.pkl | Trained model weights |
| 2.5 | training_metrics | json.JSONDataset | data/08_reporting/training_metrics.json | Final training loss |
| 2.6 | sample_prediction | json.JSONDataset | data/08_reporting/sample_prediction.json | Test prediction result |

---

### DELIVERABLE 3: PARAMETER CONFIGURATION

**Parameter Group**: `model_config`
- `input_size` (int): 1 - Input tensor dimension
- `hidden_size` (int): 1 - Hidden layer dimension
- `output_size` (int): 1 - Output tensor dimension
- `use_bias` (bool): False - Whether to use bias in linear layer

**Parameter Group**: `training_config`
- `epochs` (int): 150 - Number of training iterations (CONFIGURABLE - experimentation value)
- `learning_rate` (float): 0.01 - SGD learning rate (CONFIGURABLE - experimentation value)
- `loss_function` (str): "mse" - Loss function type (hardcoded)
- `optimizer_type` (str): "sgd" - Optimizer algorithm (hardcoded)

---

### DELIVERABLE 4: DEPENDENCIES

**Required Package**: `torch` (PyTorch 2.0+)

**Rationale**: 
- `import torch` detected in cells 4-9
- `import torch.nn as nn` detected for model definition
- No additional dependencies (no pandas, matplotlib, sklearn, etc.)

---

## PARAMETER DECISIONS

### Configurable Parameters (Make Experiments Easy)
- **epochs = 150**: ✅ Experimentation value - users will test different epoch counts
- **learning_rate = 0.01**: ✅ Experimentation value - critical for model convergence
- **input_size, hidden_size, output_size**: ✅ Architecture parameters users may adjust

### Hardcoded Constants (Keep in Code)
- **loss_function = "mse"**: ❌ Keep hardcoded - MSELoss is fixed for regression
- **optimizer_type = "sgd"**: ❌ Keep hardcoded - optimization algorithm choice
- **use_bias = False**: ❌ Keep hardcoded - structural constraint from notebook

---

## IMPLEMENTATION STANDARDS

### Node Requirements
- ✅ Pure functions only (no global state)
- ✅ Single output per node
- ✅ Type hints included
- ✅ Parameter groups used (params:model_config, params:training_config)

### Data Organization
```
data/05_model_input/   # train_x, train_y tensors
data/06_models/        # trained_model, untrained_model
data/08_reporting/     # training_metrics, sample_prediction
```

### Dataset Types (Kedro 1.0+)
- `pickle.PickleDataset` for PyTorch models and tensors
- `json.JSONDataset` for metrics and predictions

---

## QUALITY GATES

- [ ] 5 tasks implemented in pytorch_neural_network pipeline
- [ ] 6 datasets configured in catalog.yml
- [ ] 2 parameter groups defined in parameters.yml
- [ ] All nodes execute successfully
- [ ] All specified outputs generated
- [ ] Zero deviations from this SOW

---

## ACCEPTANCE CRITERIA

Success requires:
1. **Pipeline Execution**: `kedro run` completes without errors
2. **Output Verification**: All 6 datasets created at specified paths
3. **Data Integrity**: Model weights and predictions accessible
4. **Parameter Flexibility**: Both training_config parameters functional
5. **Reproducibility**: Same results with same parameters

