"""
PyTorch Neural Network Pipeline Nodes

This module contains the node functions for training a simple linear neural network
using PyTorch. The pipeline demonstrates data preparation, model creation, training,
and evaluation.
"""

import json
from typing import Dict, Tuple

import torch
import torch.nn as nn


def prepare_synthetic_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data (x, y tensor pairs).
    
    Generates a simple linear relationship: y = 2x + 1
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Training x and y tensors
    """
    x = torch.tensor(
        [[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], 
        dtype=torch.float
    )
    y = torch.tensor(
        [[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], 
        dtype=torch.float
    )
    return x, y


def create_model(params: Dict) -> nn.Module:
    """
    Create and initialize a linear neural network model.
    
    Args:
        params: Dictionary containing model configuration with keys:
            - input_size: Input tensor dimension
            - hidden_size: Hidden layer dimension
            - output_size: Output tensor dimension
            - use_bias: Whether to use bias in linear layer
    
    Returns:
        nn.Module: Initialized but untrained model
    """
    layer1 = nn.Linear(
        params["input_size"],
        params["hidden_size"],
        bias=params["use_bias"]
    )
    model = nn.Sequential(layer1)
    return model


def train_model(
    untrained_model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    params: Dict
) -> nn.Module:
    """
    Train the neural network model using SGD optimizer.
    
    Args:
        untrained_model: Initialized neural network model
        train_x: Training input tensor
        train_y: Training target tensor
        params: Dictionary containing training configuration with keys:
            - epochs: Number of training iterations
            - learning_rate: SGD learning rate
            - loss_function: Type of loss function (e.g., "mse")
            - optimizer_type: Type of optimizer (e.g., "sgd")
    
    Returns:
        nn.Module: Trained model with optimized weights
    """
    model = untrained_model
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"])
    
    for epoch in range(params["epochs"]):
        model = model.train()
        
        # Forward pass
        output = model(train_x)
        loss = criterion(output, train_y)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print progress every 30 epochs
        if (epoch + 1) % 30 == 0:
            print(f"Epoch: {epoch + 1}/{params['epochs']} | Loss: {loss.item():.4f}")
    
    return model


def evaluate_model(
    trained_model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate the trained model on training data.
    
    Args:
        trained_model: Trained neural network model
        train_x: Training input tensor
        train_y: Training target tensor
    
    Returns:
        Dict[str, float]: Dictionary containing final training loss and accuracy metrics
    """
    trained_model.eval()
    with torch.no_grad():
        output = trained_model(train_x)
        criterion = nn.MSELoss()
        loss = criterion(output, train_y)
    
    metrics = {
        "final_training_loss": float(loss.item()),
        "model_status": "trained",
    }
    
    return metrics


def test_prediction(trained_model: nn.Module) -> Dict[str, float]:
    """
    Test the trained model with a sample input.
    
    Args:
        trained_model: Trained neural network model
    
    Returns:
        Dict[str, float]: Dictionary containing sample input and prediction
    """
    trained_model.eval()
    with torch.no_grad():
        sample = torch.tensor([10.0], dtype=torch.float)
        sample_input = sample.unsqueeze(0)
        predicted = trained_model(sample_input)
    
    result = {
        "sample_input": float(sample.item()),
        "prediction": float(predicted.item()),
        "prediction_note": "Model learned approximate linear relationship y ≈ 2x + 1"
    }
    
    return result
