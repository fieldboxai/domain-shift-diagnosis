from typing import Dict

from tensorflow import keras

from .base_models import AbstractAutoEncoder, AbstractVariationalAutoEncoder
from .sparse_vae import SparseVAE
from .utils.losses import mean_absolute_error, mean_squared_error
from .variational_auto_encoder import VAE

model_registry: Dict[str, keras.Model] = {
    "LinearVAE": VAE,
    "LassoVAE": VAE,
    "SparseVAE": SparseVAE,
}

__all__ = [
    "LinearVAE",
    "SparseVAE",
    "AbstractAutoEncoder",
    "AbstractVariationalAutoEncoder",
    "model_registry",
    "mean_absolute_error",
    "mean_squared_error",
]
