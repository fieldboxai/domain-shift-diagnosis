"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""
from typing import Any

from kedro.pipeline import Pipeline, node

from .nodes import plot_mask_features, plot_model_output, plot_shift_detection_scores


def create_pipeline(**kwargs: Any) -> Pipeline:
    return Pipeline(
        [
            node(
                plot_model_output,
                inputs=[
                    "C_pred",
                    "W_pred",
                    "Z_source_pred",
                    "pairwise_mapping_jaccard_score",
                ],
                outputs=[
                    "C_pred_heatmap",
                    "W_pred_heatmap",
                    "W_scores_heatmap",
                    "Z_source_pred_cov_heatmap",
                ],
            ),
            node(
                plot_shift_detection_scores,
                inputs=[
                    "covariate_shift_detection_scores",
                ],
                outputs="shift_detection_scores_heatmap",
            ),
            node(
                plot_mask_features,
                inputs="mask_features",
                outputs="mask_features_scatter_plot",
            ),
        ],
    )
