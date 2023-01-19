from pathlib import Path

from kedro.io.core import AbstractVersionedDataSet
from tensorflow import keras
import json

from domain_shift_diagnosis.models import model_registry


class ModelDataSet(AbstractVersionedDataSet):
    """Kedro dataset absraction for custom keras model"""

    def __init__(self, filepath: str):
        super().__init__(filepath, version=None)
        self._filepath = Path(filepath)

    def _load(self) -> keras.Model:
        model_parameters = json.load(open("data/06_models/model_parameters.json"))
        model_class = model_registry[model_parameters["model_name"]]
        load_path = self._get_load_path()
        return keras.models.load_model(
            load_path,
            custom_objects={
                model_class.__name__: model_class,
            },
        )

    def _save(self, model: keras.Model) -> None:
        save_path = self._get_save_path()
        model.save(save_path)

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path).exists()

    def _describe(self) -> dict:
        return dict(filepath=self._filepath)
