from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model_input_table,
    prepare_features,
    process_companies,
    process_shuttles,
    predict_model,
    split_data,
    train_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                process_companies,
                inputs="companies",
                outputs="processed_companies",
                name="process_companies",
            ),
            node(
                process_shuttles,
                inputs="shuttles",
                outputs="processed_shuttles",
                name="process_shuttles",
            ),
            node(
                create_model_input_table,
                inputs=["processed_companies", "reviews", "processed_shuttles"],
                outputs="model_input_table",
                name="create_model_input_table",
            ),
            node(
                prepare_features,
                inputs="model_input_table",
                outputs=["X", "y"],
                name="prepare_features",
            ),
            node(
                split_data,
                inputs=["X", "y", "params:model_train"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model",
            ),
            node(
                predict_model,
                inputs=["model", "X_test"],
                outputs="y_pred",
                name="predict_model",
            ),
        ]
    )
