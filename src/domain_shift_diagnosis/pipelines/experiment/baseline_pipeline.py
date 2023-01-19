"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""
from typing import Any

from kedro.config import ConfigLoader
from kedro.pipeline import Pipeline

from .. import (
    plot_results,
    test_baseline_model,
    train_baseline_model,
)


def create_pipeline(**kwargs: Any) -> Pipeline:

    return Pipeline(
        [
            train_baseline_model.create_pipeline(),
            test_baseline_model.create_pipeline(),
            plot_results.create_pipeline(),
        ],
    )
