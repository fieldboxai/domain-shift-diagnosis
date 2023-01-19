from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from kedro.config import ConfigLoader
from sklearn.model_selection import train_test_split
from tensorflow import keras

from ...models import model_registry

from .types import TrainingParams
from .utils import compute_mask_from_weights, parse_n_components

conf_path = "conf/"
conf_loader = ConfigLoader(conf_path)
parameters = conf_loader.get("parameters/models/*",)


def train_model(
    X_source_noisy: pd.DataFrame,
    model_name: str,
    latent_dim: int,
    random_state: int,
    training_params: TrainingParams,
) -> Tuple[keras.Model, Dict[str, Any], float, float, float, float]:
    """Train a VAE keras model.

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

    X_train, X_val = train_test_split(X_source_noisy.values, random_state=random_state)

    model_class = model_registry[model_name]
    model_parameters = parameters[model_name]
    model_parameters["random_state"] = random_state
    model_parameters["latent_dim"] = parse_n_components(X_source_noisy, latent_dim)
    model = model_class(input_dim=X_source_noisy.shape[1], **model_parameters)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_params["learning_rate"]),
        metrics={"mu": ["mse", "mae"]},
    )

    model.fit(
        X_train,
        X_val,
        tol=training_params["tol"],
        epochs=training_params["epochs"],
        batch_size=int(X_source_noisy.shape[0] * training_params["batch_size_ratio"]),
        patience=training_params["patience"],
        monitor=training_params["monitor"],
    )

    model_history = model.history.history
    mae = model_history["val_mu_mae"][-1]
    mse = model_history["val_mu_mse"][-1]

    beta = model_parameters.get("beta", 0)
    l1_decoder = model_parameters["decoder_params"]["mu_regressor_params"][
        "linear_params"
    ].get("l1_kernel", 0)

    model_parameters["model_name"] = model_name
    return model, model_parameters, mae, mse, beta, l1_decoder


def compute_model_outputs(
    X_source_noisy: pd.DataFrame,
    X_source_true: pd.DataFrame,
    model: keras.Model,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
    float,
    float,
]:

    Z_source_pred = model.transform(X_source_noisy.values)
    X_source_pred = model.inverse_transform(Z_source_pred)

    residuals_source_pred = X_source_noisy - X_source_pred

    mse_test = np.mean(np.square(residuals_source_pred.values))
    mae_test = np.mean(np.abs(residuals_source_pred.values))

    weights = model.get_factors_mapping()

    mask, mask_features, latent_dim_pred = compute_mask_from_weights(weights)

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
        columns=X_source_true.columns,
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

    return (
        Z_source_pred,
        X_source_pred,
        C_pred,
        W_pred,
        mask_features,
        latent_dim_pred,
        mae_test,
        mse_test,
    )
