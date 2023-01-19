from typing import Tuple, Union

import numpy as np
import pandas as pd
from kedro.config import ConfigLoader
from sklearn.decomposition import FactorAnalysis, SparsePCA

from .utils import compute_mask_from_weights

conf_path = "conf/"
conf_loader = ConfigLoader(conf_path)
parameters = conf_loader.get("parameters*/**",)

model_classes = {
    "SparsePCA16": SparsePCA,
    "SparsePCA32": SparsePCA,
    "PPCA16": FactorAnalysis,
    "PPCA32": FactorAnalysis,
}
model_params = {
    "SparsePCA16": {"alpha": 1, "n_components": 16},
    "SparsePCA32": {"alpha": 1, "n_components": 32},
    "PPCA16": {"n_components": 16},
    "PPCA32": {"n_components": 32},
}


def train_sklearn_model(
    X_source_noisy: pd.DataFrame, model_name: str, random_state: int
) -> Union[SparsePCA, FactorAnalysis]:
    """Train a sparse autoencoder model.

    Parameters
    ----------
    model_name : str
        Name of the model, also the key used in the model mapping.
    training_params: TrainingParams
        Parameters of the model.fit method.

    Returns
    -------
    keras.Model
        The trained model.
    float
        Mean absolute error value of the validation set.
    float
        Mean square error value of the validation set.
    float
        L1 Loss of the decoder weight.
    """

    model_class = model_classes[model_name]
    model_parameters = model_params[model_name]

    model = model_class(random_state=random_state, **model_parameters)
    model.fit(X_source_noisy.values)

    return model


def compute_model_outputs(
    X_source_noisy: pd.DataFrame,
    X_source_true: pd.DataFrame,
    model: Union[SparsePCA, FactorAnalysis],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
    float,
    float,
]:

    Z_source_pred = model.transform(X_source_noisy.values)
    X_source_pred = Z_source_pred @ model.components_
    error = X_source_noisy - X_source_pred

    weights = model.components_

    mask, mask_features, _ = compute_mask_from_weights(
        weights, compute_activity_mask=False
    )

    Z_source_pred_columns = np.array([f"Z{i}" for i in range(Z_source_pred.shape[1])])

    mask_active_factors = mask_features["is_factor_active"].values

    Z_source_pred = pd.DataFrame(
        Z_source_pred,
        index=X_source_true.index,
        columns=Z_source_pred_columns,
    )

    X_source_pred = pd.DataFrame(
        X_source_pred,
        index=X_source_true.index,
        columns=X_source_noisy.columns,
    )

    C_pred = pd.DataFrame(
        weights[mask_active_factors],
        index=Z_source_pred_columns[mask_active_factors],
        columns=X_source_true.columns,
    )

    W_pred = pd.DataFrame(
        mask,
        index=Z_source_pred_columns[mask_active_factors],
        columns=X_source_true.columns,
    )

    mse = np.mean(np.square(error.values))
    mae = np.mean(np.abs(error.values))

    error_test = X_source_pred.values - X_source_true.values
    mae_test = np.mean(np.square(error_test))
    mse_test = np.mean(np.abs(error_test))

    return (
        Z_source_pred,
        X_source_pred,
        C_pred,
        W_pred,
        mask_features,
        mse,
        mae,
        mse_test,
        mae_test,
    )
