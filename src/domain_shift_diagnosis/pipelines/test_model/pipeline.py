"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.17.7
"""
from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_concept_shift_detection_scores,
    compute_covariate_shift_detection_scores,
    compute_error_shifts,
    compute_latent_shifts,
    compute_pairwise_mapping_jaccard_score,
)


def create_pipeline(**kwargs: Any) -> Pipeline:
    return Pipeline(
        [
            node(
                compute_latent_shifts,
                inputs=[
                    "X_target_noisy",
                    "Z_source_pred",
                    "mask_features",
                    "model",
                ],
                outputs=[
                    "Z_target_pred",
                    "Z_shift",
                ],
            ),
            node(
                compute_error_shifts,
                inputs=[
                    "X_target_noisy",
                    "X_source_noisy",
                    "X_source_pred",
                    "model",
                ],
                outputs="error_shift",
            ),
            node(
                compute_pairwise_mapping_jaccard_score,
                inputs=["source_sampler", "W_pred"],
                outputs=["pairwise_mapping_jaccard_score", "mapping_recovery_score"],
            ),
            node(
                compute_covariate_shift_detection_scores,
                inputs="Z_shift",
                outputs="covariate_shift_detection_scores",
            ),
            node(
                compute_concept_shift_detection_scores,
                inputs="error_shift",
                outputs="concept_shift_detection_scores",
            ),
        ]
    )
