"""
PyTorch Neural Network Pipeline

This pipeline orchestrates the training of a simple linear neural network:
1. Prepare synthetic data
2. Create model
3. Train model
4. Evaluate model
5. Test predictions
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    prepare_synthetic_data,
    create_model,
    train_model,
    evaluate_model,
    test_prediction,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the PyTorch neural network training pipeline.
    
    Returns:
        Pipeline: Configured pipeline with 5 nodes and their dependencies
    """
    return Pipeline(
        [
            node(
                func=prepare_synthetic_data,
                inputs=None,
                outputs=["train_x", "train_y"],
                name="prepare_synthetic_data_node",
            ),
            node(
                func=create_model,
                inputs="params:model_config",
                outputs="untrained_model",
                name="create_model_node",
            ),
            node(
                func=train_model,
                inputs=["untrained_model", "train_x", "train_y", "params:training_config"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "train_x", "train_y"],
                outputs="training_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=test_prediction,
                inputs="trained_model",
                outputs="sample_prediction",
                name="test_prediction_node",
            ),
        ]
    )
