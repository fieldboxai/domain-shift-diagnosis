"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from domain_shift_diagnosis.pipelines import (
    experiment,
    load_data
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    load_data_pipeline = load_data.create_pipeline()
    experiment_pipeline = experiment.create_pipeline()
    baseline_experiment_pipeline = experiment.create_baseline_pipeline()

    return {
        "__default__": load_data,
        "load_data": load_data_pipeline,
        "experiment": experiment_pipeline,
        "baseline_experiment": baseline_experiment_pipeline
    }
