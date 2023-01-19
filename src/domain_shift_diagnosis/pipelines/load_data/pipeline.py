from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_source_data, load_target_data, plot_true_mapping


def create_pipeline(**kwargs: Any) -> Pipeline:
    return pipeline(
        [
            node(
                load_source_data,
                inputs=["params:source_data"],
                outputs=[
                    "Z_source_true",
                    "X_source_true",
                    "X_source_noisy",
                    "source_sampler",
                ],
            ),
            node(
                load_target_data,
                inputs=["params:source_data", "params:target_data"],
                outputs=["Z_target_true", "X_target_noisy"],
            ),
            node(
                plot_true_mapping,
                inputs=["source_sampler"],
                outputs=None
            )
        ]
    )
