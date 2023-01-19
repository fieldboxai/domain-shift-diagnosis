"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.17.7
"""
from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import compute_model_outputs, train_sklearn_model


def create_pipeline(**kwargs: Any) -> Pipeline:
    return Pipeline(
        [
            node(
                train_sklearn_model,
                inputs=["X_source_noisy", "params:model_name", "params:random_state"],
                outputs="baseline_model",
            ),
            node(
                compute_model_outputs,
                inputs=["X_source_noisy", "X_source_true", "baseline_model"],
                outputs=[
                    "Z_source_pred",
                    "X_source_pred",
                    "C_pred",
                    "W_pred",
                    "mask_features",
                    "mse",
                    "mae",
                    "mse_test",
                    "mae_test",
                ],
            ),
        ]
    )
