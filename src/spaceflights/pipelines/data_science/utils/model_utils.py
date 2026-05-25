"""
PyTorch Model Architecture and Serialization Utilities

Reference: specs/001-aircraft-range-prediction/plan.md
Constitution: Pure functions; explicit I/O contracts; docstrings required
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def build_pytorch_model(
    input_features: int,
    hidden_layers: List[int],
    activation: str = "relu",
    batch_norm: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """
    Build a PyTorch feedforward neural network for regression.

    Args:
        input_features: Number of input features (e.g., 5 for aircraft)
        hidden_layers: List of hidden layer sizes (e.g., [64, 32])
        activation: Activation function name ("relu", "tanh", "sigmoid")
        batch_norm: Whether to apply batch normalization after each layer
        dropout_rate: Dropout probability for regularization (0.0-1.0)

    Returns:
        torch.nn.Module: Sequential neural network model

    Example:
        >>> model = build_pytorch_model(
        ...     input_features=5,
        ...     hidden_layers=[64, 32],
        ...     activation="relu",
        ...     batch_norm=True,
        ...     dropout_rate=0.2
        ... )
        >>> print(model)
        Sequential(...)
    """
    if not hidden_layers:
        raise ValueError("hidden_layers cannot be empty")

    if activation not in ["relu", "tanh", "sigmoid"]:
        raise ValueError(f"activation must be 'relu', 'tanh', or 'sigmoid', got {activation}")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    # Select activation function
    activation_fn = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }[activation]

    layers: List[nn.Module] = []

    # Build hidden layers
    layer_sizes = [input_features] + hidden_layers

    for i in range(len(layer_sizes) - 1):
        in_features = layer_sizes[i]
        out_features = layer_sizes[i + 1]

        # Linear layer
        layers.append(nn.Linear(in_features, out_features))

        # Batch normalization (optional)
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))

        # Activation function
        layers.append(activation_fn)

        # Dropout for regularization & MC Dropout uncertainty
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))

    # Output layer (1 unit for regression: predicted_range_km)
    layers.append(nn.Linear(hidden_layers[-1], 1))

    model = nn.Sequential(*layers)

    logger.info(
        f"Built PyTorch model: input={input_features}, "
        f"hidden={hidden_layers}, output=1, dropout={dropout_rate}"
    )

    return model


def save_model(
    model: nn.Module,
    metadata: Dict[str, Any],
    model_path: Path,
    metadata_path: Path,
) -> None:
    """
    Serialize PyTorch model and metadata for reproducibility.

    Args:
        model: Trained torch.nn.Module
        metadata: Dict with training history, hyperparams, normalization params
        model_path: Path to save model state_dict (.pt file)
        metadata_path: Path to save metadata JSON

    Returns:
        None

    Raises:
        IOError: If unable to write files

    Example:
        >>> save_model(
        ...     model=trained_model,
        ...     metadata={"rmse": 450.0, "r2": 0.87, ...},
        ...     model_path=Path("models/aircraft_range_model.pt"),
        ...     metadata_path=Path("models/aircraft_range_model_metadata.json")
        ... )
    """
    # Ensure parent directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model state_dict (not entire model)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model state_dict to {model_path}")

    # Save metadata as JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)  # default=str for datetime serialization
    logger.info(f"Saved metadata to {metadata_path}")


def load_model(
    model_path: Path,
    metadata_path: Path,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load serialized PyTorch model and metadata.

    Args:
        model_path: Path to saved model state_dict (.pt file)
        metadata_path: Path to saved metadata JSON
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        Tuple of (loaded_model, metadata_dict)

    Raises:
        FileNotFoundError: If model or metadata file not found
        RuntimeError: If unable to load model state_dict

    Example:
        >>> model, metadata = load_model(
        ...     model_path=Path("models/aircraft_range_model.pt"),
        ...     metadata_path=Path("models/aircraft_range_model_metadata.json"),
        ...     device="cpu"
        ... )
        >>> print(f"Loaded model with R² = {metadata['r2']}")
    """
    # Validate files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata first (contains architecture info)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct model architecture from metadata
    model = build_pytorch_model(
        input_features=metadata["architecture"]["input_features"],
        hidden_layers=metadata["architecture"]["hidden_layers"],
        activation=metadata["architecture"].get("activation", "relu"),
        batch_norm=metadata["architecture"].get("batch_norm", True),
        dropout_rate=metadata["architecture"].get("dropout_rate", 0.2),
    )

    # Load state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode (important for Dropout behavior)

    logger.info(
        f"Loaded model from {model_path} and metadata from {metadata_path} "
        f"(device={device}, r2={metadata.get('r2', 'N/A')})"
    )

    return model, metadata


class PyTorchModelDataset:
    """
    Kedro-compatible dataset class for PyTorch models (optional enhancement).

    Can be used in catalog.yml for automatic model loading/saving:
    trained_model:
      type: pytorch.PyTorchModelDataset
      filepath: data/06_models/model.pt
      metadata_filepath: data/06_models/model_metadata.json
    """

    def __init__(self, filepath: str, metadata_filepath: str):
        self.filepath = Path(filepath)
        self.metadata_filepath = Path(metadata_filepath)

    def load(self) -> Tuple[nn.Module, Dict[str, Any]]:
        return load_model(self.filepath, self.metadata_filepath, device="cpu")

    def save(self, data: Tuple[nn.Module, Dict[str, Any]]) -> None:
        model, metadata = data
        save_model(model, metadata, self.filepath, self.metadata_filepath)
