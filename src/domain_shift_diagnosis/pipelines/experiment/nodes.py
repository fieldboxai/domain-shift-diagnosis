from typing import List

from kedro.pipeline import Pipeline

from .. import test_baseline_model, test_model, train_baseline_model, train_model


def update(model_name: str) -> List[Pipeline]:

    if model_name in ["PPCA", "SparsePCA16", "SparsePCA32"]:
        train_model_pipeline = train_baseline_model.create_pipeline()
        test_model_pipeline = test_baseline_model.create_pipeline()
    else:
        train_model_pipeline = train_model.create_pipeline()
        test_model_pipeline = test_model.create_pipeline()

    pipelines = [train_model_pipeline, test_model_pipeline]

    return pipelines
