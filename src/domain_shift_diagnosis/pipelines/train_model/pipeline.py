"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.17.7
"""
from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import compute_model_outputs, train_model


def create_pipeline(**kwargs: Any) -> Pipeline:
    return Pipeline(
        [
            node(
                train_model,
                inputs=[
                    "X_source_noisy",
                    "params:model_name",
                    "params:latent_dim",
                    "params:random_state",
                    "params:training_params",
                ],
                outputs=[
                    "model",
                    "model_parameters",
                    "mae",
                    "mse",
                    "beta",
                    "l1_decoder",
                ],
            ),
            node(
                compute_model_outputs,
                inputs=["X_source_noisy", "X_source_true", "model"],
                outputs=[
                    "Z_source_pred",
                    "X_source_pred",
                    "C_pred",
                    "W_pred",
                    "mask_features",
                    "latent_dim_pred",
                    "mae_test",
                    "mse_test",
                ],
            ),
        ]
    )
