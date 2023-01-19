from mlflow.tracking.client import MlflowClient
from tensorflow import keras

from domain_shift_diagnosis.models import model_registry


def load_model_from_mlflow(model_name: str) -> None:
    """Return the version of <model_name> with the highest mapping recovery
    scoere."""
    mlflow_client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    query = f"params.model_name = '{model_name}'"
    best_run = mlflow_client.search_runs(
        experiment_ids=["1"],
        filter_string=query,
        order_by=["metrics.mapping_recovery_score ASC"],
    )[0]
    model_path = best_run.info.artifact_uri + "/model/"
    model_class = model_registry[model_name]
    model = keras.models.load_model(
        model_path, custom_objects={model_class.__name__: model_class}
    )
    model.save("data/06_models/model")
