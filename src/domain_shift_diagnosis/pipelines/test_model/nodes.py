from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tensorflow import keras


from sklearn.metrics import jaccard_score

from ...datasets import SourceDataSampler
from .utils import shift_detection_score, univariate_wasserstein_dist


def compute_latent_shifts(
    X_target_noisy: pd.DataFrame,
    Z_source_pred: pd.DataFrame,
    mask_features: pd.DataFrame,
    model: keras.Model,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes univariate wasserstein distance in the latent space"""
    mask_active_factors = mask_features["is_factor_active"].values

    Z_target_pred = pd.DataFrame(
        model.transform(X_target_noisy.values),
        columns=[f"Z{k}" for k in range(model.config["latent_dim"])],
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


def compute_error_shifts(
    X_target_noisy: pd.DataFrame,
    X_source_noisy: pd.DataFrame,
    X_source_pred: pd.DataFrame,
    model: keras.Model,
) -> pd.DataFrame:
    """Computes univariate wasserstein distance in the residual space"""
    reconstruction_errors_source = np.square(X_source_pred - X_source_noisy.values)

    reconstruction_errors_target = np.square(
        model.inverse_transform(model.transform(X_target_noisy.values)) - X_target_noisy
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


def compute_pairwise_mapping_jaccard_score(
    source_sampler: SourceDataSampler,
    W_pred: pd.DataFrame
) -> pd.DataFrame:
    W_true = source_sampler.W
    pairwise_mapping_jaccard_score = pd.DataFrame(
        data=cdist(W_true, W_pred, metric=jaccard_score),
        index=W_true.index,
        columns=W_pred.index,
    )
    pairwise_mapping_jaccard_score.index.name = "True Factors"
    pairwise_mapping_jaccard_score.columns.name = "Predicted Factors"

    # Mapping Recovery Score
    mapping_recovery_score = np.mean(pairwise_mapping_jaccard_score.max())
    return pairwise_mapping_jaccard_score, mapping_recovery_score


def compute_covariate_shift_detection_scores(Z_shift: pd.DataFrame) -> pd.DataFrame:

    Z_covariate_shift = Z_shift.loc["Covariate"]

    covariate_shift_detection_scores = shift_detection_score(Z_covariate_shift)
    covariate_shift_detection_scores.columns = covariate_shift_detection_scores.columns.astype(str)

    return covariate_shift_detection_scores


def compute_concept_shift_detection_scores(error_shift: pd.DataFrame) -> pd.DataFrame:

    error_concept_shift = error_shift.loc["Concept"]

    concept_shift_detection_scores = shift_detection_score(error_concept_shift)
    concept_shift_detection_scores.columns = concept_shift_detection_scores.columns.astype(str)

    return concept_shift_detection_scores
