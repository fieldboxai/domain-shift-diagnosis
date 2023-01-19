from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, SparsePCA

from ..test_model.utils import univariate_wasserstein_dist


def compute_latent_shifts(
    X_target_noisy: pd.DataFrame,
    Z_source_pred: pd.DataFrame,
    mask_features: pd.DataFrame,
    model: Union[SparsePCA, FactorAnalysis],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes univariate shifts in the latent space and the observable space."""
    mask_active_factors = mask_features["is_factor_active"].values

    Z_target_pred = pd.DataFrame(
        model.transform(X_target_noisy.values),
        columns=Z_source_pred.columns,
        index=X_target_noisy.index,
    )

    Z_shift = (
        Z_target_pred.loc[:, mask_active_factors]
        .groupby(
            [
                "Shift Type",
                "Shift Source",
                "Shift Intensity",
                "Gaussian Noise Std",
                "N Samples",
                "shift_id",
            ]
        )
        .apply(
            lambda df: pd.Series(
                univariate_wasserstein_dist(
                    Z_source_pred.values[:, mask_active_factors], df.values
                ),
                index=Z_target_pred.columns[mask_active_factors],
            )
        )
    )

    return Z_target_pred, Z_shift


def compute_error_shift(
    X_target_noisy: pd.DataFrame,
    X_source_noisy: pd.DataFrame,
    X_source_pred: pd.DataFrame,
    model: Union[SparsePCA, FactorAnalysis],
) -> pd.DataFrame:
    reconstruction_errors_source = np.square(X_source_pred - X_source_noisy.values)

    reconstruction_errors_target = np.square(
        (model.transform(X_target_noisy.values) @ model.components_ - X_target_noisy)
    )

    error_shift = reconstruction_errors_target.groupby(
        [
            "Shift Type",
            "Shift Source",
            "Shift Intensity",
            "Gaussian Noise Std",
            "N Samples",
            "shift_id",
        ]
    ).apply(
        lambda df: pd.Series(
            univariate_wasserstein_dist(reconstruction_errors_source.values, df.values),
            index=reconstruction_errors_target.columns,
        )
    )
    return error_shift
